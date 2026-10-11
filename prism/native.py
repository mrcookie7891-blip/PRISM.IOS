"""PRISM's own finite-context neural learner; all weights start random.

NumPy is numerical infrastructure only. No model or vocabulary is downloaded.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class CharacterTokenizer:
    symbols: tuple[str, ...]

    @classmethod
    def fit(cls, documents):
        return cls(tuple(sorted(set(''.join(documents)))))

    @property
    def size(self):
        return len(self.symbols) + 2  # 0 padding, 1 unknown

    def encode(self, text):
        lookup = {char: i + 2 for i, char in enumerate(self.symbols)}
        return np.array([lookup.get(char, 1) for char in text], dtype=np.int64)

    def decode(self, ids):
        return ''.join(self.symbols[i - 2] if i >= 2 else '\ufffd' if i == 1 else '' for i in ids)


@dataclass(frozen=True)
class Config:
    context: int = 32
    embedding: int = 16
    hidden: int = 128
    seed: int = 42

    def __post_init__(self):
        if min(self.context, self.embedding, self.hidden) < 1:
            raise ValueError('Model dimensions must be positive')


def examples(documents, tokenizer, context):
    """Contexts never cross document or split boundaries; build only token streams."""
    streams = [tokenizer.encode(doc) for doc in documents if doc]
    if not streams:
        raise ValueError('No nonempty documents')
    return streams


def batch_at(streams, indices, context):
    lengths = np.cumsum([len(s) for s in streams])
    docs = np.searchsorted(lengths, indices, side='right')
    starts = np.concatenate(([0], lengths[:-1]))
    x = np.zeros((len(indices), context), dtype=np.int64)
    y = np.empty(len(indices), dtype=np.int64)
    for row, (doc, index) in enumerate(zip(docs, indices)):
        pos = int(index - starts[doc])
        prior = streams[doc][max(0, pos - context):pos]
        if len(prior):
            x[row, -len(prior):] = prior
        y[row] = streams[doc][pos]
    return x, y


class NativeModel:
    def __init__(self, tokenizer, config=Config()):
        if not tokenizer.symbols:
            raise ValueError('Training vocabulary is empty')
        self.tokenizer, self.config = tokenizer, config
        self.metadata = {}
        rng = np.random.default_rng(config.seed)
        self.weights = {
            'embedding': (rng.normal(0, .1, (tokenizer.size, config.embedding))).astype('float32'),
            'w1': (rng.normal(0, 1 / math.sqrt(config.context * config.embedding),
                              (config.context * config.embedding, config.hidden))).astype('float32'),
            'b1': np.zeros(config.hidden, dtype='float32'),
            'w2': (rng.normal(0, 1 / math.sqrt(config.hidden),
                              (config.hidden, tokenizer.size))).astype('float32'),
            'b2': np.zeros(tokenizer.size, dtype='float32'),
        }

    def forward(self, x):
        w = self.weights
        flat = w['embedding'][x].reshape(len(x), -1)
        hidden = np.tanh(flat @ w['w1'] + w['b1'])
        logits = hidden @ w['w2'] + w['b2']
        logits[:, 0] = -1e9  # padding is never a prediction target
        return logits, (flat, hidden)

    def loss_and_gradients(self, x, y):
        logits, (flat, hidden) = self.forward(x)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = float(np.mean(-logits[np.arange(len(y)), y] + np.log(np.exp(logits).sum(axis=1))))
        delta = probs.copy()
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)
        dh = (delta @ self.weights['w2'].T) * (1 - hidden * hidden)
        grads = {'w2': hidden.T @ delta, 'b2': delta.sum(axis=0),
                 'w1': flat.T @ dh, 'b1': dh.sum(axis=0)}
        de = (dh @ self.weights['w1'].T).reshape(len(x), self.config.context, self.config.embedding)
        grads['embedding'] = np.zeros_like(self.weights['embedding'])
        np.add.at(grads['embedding'], x, de)
        return loss, grads

    def evaluate(self, streams, batch_size=256):
        total = sum(map(len, streams))
        loss, hits = 0.0, 0
        for start in range(0, total, batch_size):
            x, y = batch_at(streams, np.arange(start, min(total, start + batch_size)), self.config.context)
            logits, _ = self.forward(x)
            logits -= logits.max(axis=1, keepdims=True)
            loss += float((-logits[np.arange(len(y)), y] + np.log(np.exp(logits).sum(axis=1))).sum())
            hits += int((logits.argmax(axis=1) == y).sum())
        nll = loss / total
        return {'nll': nll, 'perplexity': math.exp(min(nll, 700)), 'top1_accuracy': hits / total, 'tokens': total}

    def generate(self, prompt, length=200, temperature=.8, seed=42):
        if length < 0 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('length and temperature must be nonnegative and finite')
        ids = list(self.tokenizer.encode(prompt))
        rng = np.random.default_rng(seed)
        result = []
        for _ in range(length):
            x = np.zeros((1, self.config.context), dtype=np.int64)
            prior = ids[-self.config.context:]
            if prior:
                x[0, -len(prior):] = prior
            logits, _ = self.forward(x)
            logits = logits[0].astype('float64')
            logits[:2] = -np.inf  # output only characters learned from training
            if temperature == 0:
                nxt = int(logits.argmax())
            else:
                logits = (logits - logits.max()) / temperature
                p = np.exp(logits)
                nxt = int(rng.choice(len(p), p=p / p.sum()))
            ids.append(nxt)
            result.append(nxt)
        return prompt + self.tokenizer.decode(result)

    def save(self, path, quantize=False, metadata=None):
        payload = {'manifest': np.array(json.dumps({'format': 'prism-native-v1',
            'config': asdict(self.config), 'symbols': self.tokenizer.symbols,
            'quantized': quantize, 'metadata': {**self.metadata, **(metadata or {})}}))}
        for name, weight in self.weights.items():
            if quantize:
                scale = max(float(np.max(np.abs(weight))) / 127, 1e-12)
                payload[name] = np.round(weight / scale).clip(-127, 127).astype('int8')
                payload[name + '_scale'] = np.array(scale, dtype='float32')
            else:
                payload[name] = weight
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive['manifest']))
            if manifest['format'] != 'prism-native-v1':
                raise ValueError('Unsupported PRISM model format')
            model = cls(CharacterTokenizer(tuple(manifest['symbols'])), Config(**manifest['config']))
            model.metadata = manifest.get('metadata', {})
            for name, initial in model.weights.items():
                weight = archive[name].astype('float32')
                if manifest['quantized']:
                    weight *= archive[name + '_scale']
                if weight.shape != initial.shape or not np.isfinite(weight).all():
                    raise ValueError('Invalid weights: ' + name)
                model.weights[name] = weight
            return model


def train(model, streams, steps=1000, batch_size=128, learning_rate=.003, seed=42, device='numpy'):
    """Adam with clipping. Optional Torch runs this exact network on CPU/CUDA/MPS."""
    if steps < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Positive training settings required')
    if device != 'numpy':
        from prism.torch_training import train_torch
        return train_torch(model, streams, steps, batch_size, learning_rate, seed, device)
    rng = np.random.default_rng(seed)
    total = sum(map(len, streams))
    first = {k: np.zeros_like(v) for k, v in model.weights.items()}
    second = {k: np.zeros_like(v) for k, v in model.weights.items()}
    losses = []
    for step in range(1, steps + 1):
        x, y = batch_at(streams, rng.integers(total, size=batch_size), model.config.context)
        loss, grads = model.loss_and_gradients(x, y)
        norm = math.sqrt(sum(float(np.sum(g.astype('float64') ** 2)) for g in grads.values()))
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise ValueError('Training diverged')
        for name, g in grads.items():
            g *= min(1.0, 1.0 / max(norm, 1e-12))
            first[name] = .9 * first[name] + .1 * g
            second[name] = .999 * second[name] + .001 * g * g
            model.weights[name] -= learning_rate * (first[name] / (1 - .9 ** step)) / (
                np.sqrt(second[name] / (1 - .999 ** step)) + 1e-8)
        if step == 1 or step % 100 == 0 or step == steps:
            losses.append({'step': step, 'train_nll': loss})
    return losses

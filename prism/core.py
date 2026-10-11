"""Small byte-context neural language model, implemented directly in NumPy."""
import json
import numpy as np


def split_data(data, context):
    # Split raw bytes BEFORE constructing windows, so no window crosses splits.
    a, b = int(len(data) * .8), int(len(data) * .9)
    parts = [np.frombuffer(p, dtype=np.uint8).copy() for p in
             (data[:a], data[a:b], data[b:])]
    if any(len(p) <= context for p in parts):
        raise ValueError('Corpus too small for three independent splits')
    return parts


def windows(data, positions, context):
    return data[positions[:, None] + np.arange(context)], data[positions + context]


class Model:
    def __init__(self, context=16, width=32, seed=0):
        if context < 1 or width < 1:
            raise ValueError('context and width must be positive')
        self.context = context
        rng = np.random.default_rng(seed)
        self.weights = {
            'embedding': rng.normal(0, .1, (256, width)).astype('float32'),
            'input': rng.normal(0, 1 / np.sqrt(context * width),
                                (context * width, width)).astype('float32'),
            'hidden_bias': np.zeros(width, dtype='float32'),
            'output': rng.normal(0, 1 / np.sqrt(width), (width, 256)).astype('float32'),
            'output_bias': np.zeros(256, dtype='float32')}

    def forward(self, x):
        w = self.weights
        flat = w['embedding'][x].reshape(len(x), -1)
        hidden = np.tanh(flat @ w['input'] + w['hidden_bias'])
        logits = hidden @ w['output'] + w['output_bias']
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return flat, hidden, probs

    def loss(self, x, y):
        return float(-np.log(self.forward(x)[2][np.arange(len(y)), y].clip(1e-30)).mean())

    def gradients(self, x, y):
        flat, hidden, probs = self.forward(x)
        probs[np.arange(len(y)), y] -= 1
        probs /= len(y)
        dh = (probs @ self.weights['output'].T) * (1 - hidden ** 2)
        de = (dh @ self.weights['input'].T).reshape(*x.shape, -1)
        embedding = np.zeros_like(self.weights['embedding'])
        np.add.at(embedding, x, de)
        return dict(embedding=embedding, input=flat.T @ dh,
                    hidden_bias=dh.sum(axis=0), output=hidden.T @ probs,
                    output_bias=probs.sum(axis=0))

    def step(self, x, y, rate=.1):
        grads = self.gradients(x, y)
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        for key, g in grads.items():
            self.weights[key] -= rate * g / max(1., norm)

    def save(self, path, metadata=None, quantized=False):
        arrays = {}
        for key, w in self.weights.items():
            if quantized:
                scale = max(float(np.abs(w).max()) / 127, 1e-12)
                arrays[key] = np.rint(w / scale).clip(-127, 127).astype('int8')
                arrays[key + '_scale'] = np.array(scale, dtype='float32')
            else:
                arrays[key] = w
        info = dict(version=1, context=self.context, quantized=quantized,
                    tokenizer='utf8-bytes', metadata=metadata or {})
        np.savez_compressed(path, info=json.dumps(info), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            info = json.loads(str(z['info']))
            if info['version'] != 1 or info['tokenizer'] != 'utf8-bytes':
                raise ValueError('Unsupported PRISM artifact')
            model = cls(info['context'], z['embedding'].shape[1])
            model.weights = {k: z[k].astype('float32') *
                             (float(z[k + '_scale']) if info['quantized'] else 1)
                             for k in model.weights}
        return model

    def generate(self, prompt, count=100, seed=0, temperature=1.):
        if temperature <= 0 or count < 0:
            raise ValueError('temperature must be positive and count nonnegative')
        rng = np.random.default_rng(seed)
        result = bytearray(prompt.encode('utf8'))
        for _ in range(count):
            context = (bytes(self.context) + result)[-self.context:]
            p = self.forward(np.frombuffer(context, dtype=np.uint8)[None])[2][0]
            logits = np.log(p.clip(1e-30)) / temperature
            p = np.exp(logits - logits.max()); p /= p.sum()
            result.append(int(rng.choice(256, p=p)))
        return result.decode('utf8', errors='replace')

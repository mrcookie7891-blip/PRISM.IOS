"""Optional accelerator training for PRISM's randomly initialized native network."""
import numpy as np
from prism.native import batch_at


def train_torch(model, streams, steps, batch_size, learning_rate, seed, device):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Install torch to use --device cpu/cuda/mps; NumPy inference needs no torch') from exc
    if device not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Device must be numpy, cpu, cuda, or mps')
    torch.manual_seed(seed)
    weights = {k: torch.tensor(v, device=device, requires_grad=True) for k, v in model.weights.items()}
    optimizer = torch.optim.Adam(list(weights.values()), lr=learning_rate)
    rng = np.random.default_rng(seed)
    total = sum(map(len, streams))
    losses = []
    for step in range(1, steps + 1):
        x, y = batch_at(streams, rng.integers(total, size=batch_size), model.config.context)
        x = torch.tensor(x, device=device)
        y = torch.tensor(y, device=device)
        flat = weights['embedding'][x].reshape(batch_size, -1)
        hidden = torch.tanh(flat @ weights['w1'] + weights['b1'])
        logits = hidden @ weights['w2'] + weights['b2']
        mask = torch.zeros(model.tokenizer.size, device=device)
        mask[0] = -1e9
        loss = torch.nn.functional.cross_entropy(logits + mask, y)
        if not torch.isfinite(loss):
            raise ValueError('Training diverged')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(weights.values()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            losses.append({'step': step, 'train_nll': float(loss.detach().cpu())})
    model.weights = {k: v.detach().cpu().numpy().copy() for k, v in weights.items()}
    return losses

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import Model, split_data, windows


def main():
    parser = argparse.ArgumentParser(description='PRISM: train your own weights from scratch')
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--corpus', type=Path, required=True)
    train.add_argument('--out', type=Path, default=Path('runs/native'))
    train.add_argument('--steps', type=int, default=2000)
    train.add_argument('--context', type=int, default=16)
    train.add_argument('--width', type=int, default=32)
    train.add_argument('--batch', type=int, default=64)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--rate', type=float, default=.1)
    generate = sub.add_parser('generate')
    generate.add_argument('--model', type=Path, required=True)
    generate.add_argument('--prompt', default='PRISM')
    generate.add_argument('--bytes', type=int, default=100)
    generate.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'generate':
        print(Model.load(args.model).generate(args.prompt, args.bytes, args.seed))
        return
    if args.steps < 1 or args.batch < 1 or args.rate <= 0:
        parser.error('steps, batch, and rate must be positive')
    raw = args.corpus.read_bytes()
    training, validation, test = split_data(raw, args.context)
    rng = np.random.default_rng(args.seed)
    model = Model(args.context, args.width, args.seed)
    def evaluate(part):
        positions = np.linspace(0, len(part) - args.context - 1,
                                min(1024, len(part) - args.context), dtype=int)
        losses = []
        for chunk in np.array_split(positions, max(1, (len(positions) + 63) // 64)):
            x, y = windows(part, chunk, args.context)
            losses.append((model.loss(x, y), len(chunk)))
        return sum(l * n for l, n in losses) / sum(n for _, n in losses)
    initial = evaluate(validation)
    best_loss, best = initial, {k: v.copy() for k, v in model.weights.items()}
    for step in range(1, args.steps + 1):
        pos = rng.integers(0, len(training) - args.context, args.batch)
        model.step(*windows(training, pos, args.context), rate=args.rate)
        if step % 100 == 0 or step == args.steps:
            loss = evaluate(validation)
            print(json.dumps(dict(step=step, validation_loss=loss)), flush=True)
            if loss < best_loss:
                best_loss, best = loss, {k: v.copy() for k, v in model.weights.items()}
    model.weights = best
    report = dict(seed=args.seed, steps=args.steps, corpus_sha256=hashlib.sha256(raw).hexdigest(),
                  split_bytes=[len(p) for p in (training, validation, test)],
                  initial_validation_loss=initial, validation_loss=best_loss,
                  test_loss=evaluate(test), parameters=sum(w.size for w in best.values()),
                  initialization='random; no pretrained weights')
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / 'model.npz', report)
    model.save(args.out / 'model-int8.npz', report, quantized=True)
    quantized = Model.load(args.out / 'model-int8.npz')
    model.weights = quantized.weights
    report['quantized_test_loss'] = evaluate(test)
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

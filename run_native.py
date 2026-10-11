#!/usr/bin/env python3
"""Train, export, or run PRISM's native model without any pretrained weights."""
import argparse
import json
from pathlib import Path
import numpy as np
from prism.dataset import split_documents
from prism.native import CharacterTokenizer, Config, NativeModel, examples, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('paths', nargs='+', type=Path)
    training.add_argument('--output', type=Path, default=Path('.prism_state/native.npz'))
    training.add_argument('--steps', type=int, default=1000)
    training.add_argument('--batch-size', type=int, default=128)
    training.add_argument('--context', type=int, default=32)
    training.add_argument('--embedding', type=int, default=16)
    training.add_argument('--hidden', type=int, default=128)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--learning-rate', type=float, default=.003)
    training.add_argument('--device', choices=['numpy', 'cpu', 'cuda', 'mps'], default='numpy')
    generation = commands.add_parser('generate')
    generation.add_argument('model', type=Path)
    generation.add_argument('--prompt', default='I am PRISM')
    generation.add_argument('--length', type=int, default=200)
    generation.add_argument('--temperature', type=float, default=.8)
    generation.add_argument('--seed', type=int, default=42)
    export = commands.add_parser('export')
    export.add_argument('model', type=Path)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--int8', action='store_true')
    args = parser.parse_args()
    if args.command == 'train' and (args.steps < 1 or args.batch_size < 1):
        parser.error("--steps and --batch-size must be positive")
    if args.command == 'generate':
        print(NativeModel.load(args.model).generate(args.prompt, args.length, args.temperature, args.seed))
        return
    if args.command == 'export':
        if args.output.resolve() == args.model.resolve():
            parser.error('Export output must differ from source checkpoint')
        NativeModel.load(args.model).save(args.output, quantize=args.int8,
                                         metadata={'source': str(args.model), 'purpose': 'local inference'})
        print(json.dumps({'output': str(args.output), 'bytes': args.output.stat().st_size}))
        return
    splits, provenance = split_documents(args.paths, args.seed)
    tokenizer = CharacterTokenizer.fit(splits['train'])
    model = NativeModel(tokenizer, Config(args.context, args.embedding, args.hidden, args.seed))
    streams = {k: examples(v, tokenizer, args.context) for k, v in splits.items()}
    before = model.evaluate(streams['validation'])
    # Validation selects checkpoints; test is measured only on the selected model.
    best_nll = before['nll']
    best_weights = {k: v.copy() for k, v in model.weights.items()}
    history = train(model, streams['train'], args.steps, args.batch_size,
                    args.learning_rate, args.seed, args.device)
    after = model.evaluate(streams['validation'])
    selected = after['nll'] < best_nll
    if not selected:
        model.weights = best_weights
    train_ids = np.concatenate(streams['train'])
    counts = np.bincount(train_ids, minlength=tokenizer.size).astype('float64') + 1
    counts[0] = 0
    probabilities = counts / counts.sum()
    test_ids = np.concatenate(streams['test'])
    baseline = float(-np.log(probabilities[test_ids]).mean())
    report = {'provenance': provenance, 'initialization': 'random', 'device': args.device,
              'parameters': sum(w.size for w in model.weights.values()),
              'training': {'steps': args.steps, 'batch_size': args.batch_size,
                           'learning_rate': args.learning_rate, 'seed': args.seed},
              'validation_before': before, 'validation_candidate': after,
              'trained_candidate_selected': selected, 'test': model.evaluate(streams['test']),
              'test_unigram_nll': baseline, 'training_history': history,
              'unknown_test_characters': int((test_ids == 1).sum())}
    model.save(args.output, metadata=report)
    report_path = args.output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

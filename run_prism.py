#!/usr/bin/env python3
"""Run the PRISM self-evolving mind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from letter_follow_mapper import load_corpus
from prism.evolution import EvolutionEngine


def collect_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.txt")))
        elif path.is_file():
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM: self-evolving AI that learns patterns and improves its own strategy."
    )
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--generations", "-g", type=int, default=5, help="Evolution cycles")
    parser.add_argument("--learn-only", action="store_true", help="Ingest data without evolution")
    parser.add_argument("--state", type=Path, default=Path(".prism_state/mind.json"))
    parser.add_argument("--think", metavar="SEED", help="Generate text from current mind")
    parser.add_argument("--status", action="store_true", help="Show saved mind status")
    args = parser.parse_args()

    engine = EvolutionEngine(state_path=args.state)

    if args.status:
        print(json.dumps(engine.mind.status(), indent=2))
        ideas = [
            {"pattern": i.pattern, "strength": round(i.strength, 2), "insight": i.insight}
            for i in engine.mind.knowledge.top_ideas(12)
        ]
        print(json.dumps({"top_ideas": ideas}, indent=2))
        return

    if args.think:
        print(engine.mind.think(args.think, length=80))
        return

    files = collect_files(args.paths)
    if not files:
        raise SystemExit("No input files found.")

    text = load_corpus(files)
    eval_slice = text[-min(len(text), 80_000) :]

    if args.learn_only:
        result = engine.learn_only(text)
        print(json.dumps(result, indent=2))
        return

    print(f"Training PRISM on {len(files)} files ({len(text):,} chars)...")
    results = engine.evolve_many(text, generations=args.generations, eval_text=eval_slice)

    for item in results:
        mark = "↑" if item.improved else "→"
        print(
            f"Gen {item.generation:>3} {mark}  fitness={item.fitness:.3f}  "
            f"intelligence={item.intelligence_score:.2f}  "
            f"context_order={item.genome.context_order}"
        )

    final = results[-1]
    print("\nNew ideas discovered:")
    for pattern in final.top_ideas:
        print(f"  • {pattern}")

    print("\nMind status:")
    print(json.dumps(engine.mind.status(), indent=2))
    print(f"\nState saved to {args.state}")


if __name__ == "__main__":
    main()

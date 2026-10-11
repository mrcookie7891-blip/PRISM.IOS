#!/usr/bin/env python3
"""Run the letter-follow mapper on corpus files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from letter_follow_mapper import ALPHABET, build_from_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map each letter to a 26-D [0,1] vector of follow frequencies."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("data")],
        help="Text files or directories (default: data/)",
    )
    parser.add_argument(
        "--letter",
        "-l",
        help="Show full profile for one letter (e.g. 't')",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=5,
        help="How many top followers to show per letter (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full model as JSON",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.txt")))
        elif path.is_file():
            files.append(path)

    if not files:
        raise SystemExit("No text files found.")

    model = build_from_files(files)

    if args.json:
        payload = {
            "total_bigrams": model.total_bigrams,
            "profiles": {ch: model.profiles[ch].as_dict() for ch in ALPHABET},
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Corpus files: {len(files)}")
    print(f"Letter bigrams analyzed: {model.total_bigrams:,}")
    print()

    letters = [args.letter.lower()] if args.letter else list(ALPHABET)
    for ch in letters:
        profile = model.profile_for(ch)
        if profile is None:
            continue
        top = profile.following_letters()[: args.top]
        nonzero = sum(1 for p in profile.probabilities if p > 0)
        print(f"'{ch}' -> {nonzero} possible followers (26-D vector, values in [0,1]):")
        for follower, prob in top:
            print(f"  {follower}: {prob:.4f}")
        if not top:
            print("  (never followed by another letter in corpus)")
        print()


if __name__ == "__main__":
    main()

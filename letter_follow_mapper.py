"""
Letter-follow frequency mapper.

Builds a multi-dimensional [0, 1] vector for each letter: dimension j is how
often letter j immediately follows that letter in the corpus (conditional
probability P(next = j | current = i)).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import string


ALPHABET = string.ascii_lowercase
LETTER_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}


def normalize_text(text: str) -> str:
    """Keep only ASCII letters, lowercased."""
    return "".join(ch.lower() for ch in text if ch.lower() in LETTER_INDEX)


@dataclass(frozen=True)
class FollowProfile:
    """Multi-dimensional [0, 1] point for one source letter."""

    letter: str
    probabilities: tuple[float, ...]  # len 26, index = following letter

    def as_dict(self) -> dict[str, float]:
        return {ALPHABET[i]: self.probabilities[i] for i in range(len(ALPHABET))}

    def following_letters(self, min_probability: float = 0.0) -> list[tuple[str, float]]:
        pairs = [(ALPHABET[i], p) for i, p in enumerate(self.probabilities) if p >= min_probability]
        return sorted(pairs, key=lambda item: (-item[1], item[0]))


@dataclass
class LetterFollowModel:
    """Transition model over the 26-letter alphabet."""

    profiles: dict[str, FollowProfile]
    total_bigrams: int

    def profile_for(self, letter: str) -> FollowProfile | None:
        letter = letter.lower()
        if letter not in LETTER_INDEX:
            return None
        return self.profiles.get(letter)

    def transition_matrix(self) -> list[list[float]]:
        """26x26 matrix: row = current letter, col = following letter."""
        return [
            list(self.profiles[ch].probabilities) if ch in self.profiles else [0.0] * 26
            for ch in ALPHABET
        ]

    def to_binary(self, threshold: float = 0.0) -> dict[str, tuple[int, ...]]:
        """Optional 0/1 view: 1 if follow probability >= threshold."""
        return {
            ch: tuple(1 if p >= threshold else 0 for p in profile.probabilities)
            for ch, profile in self.profiles.items()
        }


def _counts_to_probabilities(counts: dict[str, int]) -> tuple[float, ...]:
    total = sum(counts.get(ch, 0) for ch in ALPHABET)
    if total == 0:
        return tuple(0.0 for _ in ALPHABET)
    return tuple(counts.get(ch, 0) / total for ch in ALPHABET)


def build_letter_follow_model(text: str) -> LetterFollowModel:
    """
  Scan text and estimate P(next letter | current letter) for a-z.

  Each source letter maps to a 26-dimensional point in [0, 1] that sums to 1
  (or 0 if that letter never appears before another letter).
    """
    normalized = normalize_text(text)
    follow_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_bigrams = 0

    for i in range(len(normalized) - 1):
        current, nxt = normalized[i], normalized[i + 1]
        if current not in LETTER_INDEX or nxt not in LETTER_INDEX:
            continue
        follow_counts[current][nxt] += 1
        total_bigrams += 1

    profiles: dict[str, FollowProfile] = {}
    for ch in ALPHABET:
        probs = _counts_to_probabilities(follow_counts[ch])
        profiles[ch] = FollowProfile(letter=ch, probabilities=probs)

    return LetterFollowModel(profiles=profiles, total_bigrams=total_bigrams)


def load_corpus(paths: list[Path | str]) -> str:
    chunks: list[str] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        content = p.read_text(encoding="utf-8", errors="replace")
        if content:
            chunks.append(content)
    return "\n".join(chunks)


def build_from_files(paths: list[Path | str]) -> LetterFollowModel:
    return build_letter_follow_model(load_corpus(paths))

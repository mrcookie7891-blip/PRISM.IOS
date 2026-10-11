"""Evolvable learning strategy parameters."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import random


@dataclass
class Genome:
    """Controls how the mind learns, forgets, and discovers ideas."""

    context_order: int = 2  # 1-4: how many prior letters inform predictions
    learning_rate: float = 0.35  # weight given to new observations vs memory
    idea_min_support: int = 3  # min occurrences before an idea is kept
    novelty_bonus: float = 0.2  # reward for discovering rare-but-real patterns
    exploration: float = 0.1  # chance to try alternate context blending
    memory_decay: float = 0.02  # slow fade so old wrong ideas can be replaced

    def clamp(self) -> Genome:
        return Genome(
            context_order=max(1, min(4, self.context_order)),
            learning_rate=max(0.05, min(0.95, self.learning_rate)),
            idea_min_support=max(2, min(20, self.idea_min_support)),
            novelty_bonus=max(0.0, min(1.0, self.novelty_bonus)),
            exploration=max(0.0, min(0.5, self.exploration)),
            memory_decay=max(0.0, min(0.2, self.memory_decay)),
        )

    def mutate(self, rate: float = 0.3) -> Genome:
        g = self.clamp()

        def jitter(value: float, delta: float, low: float, high: float) -> float:
            if random.random() < rate:
                value += random.uniform(-delta, delta)
            return max(low, min(high, value))

        if random.random() < rate:
            g.context_order += random.choice([-1, 1])
        g.learning_rate = jitter(g.learning_rate, 0.15, 0.05, 0.95)
        if random.random() < rate:
            g.idea_min_support += random.choice([-1, 1])
        g.novelty_bonus = jitter(g.novelty_bonus, 0.15, 0.0, 1.0)
        g.exploration = jitter(g.exploration, 0.1, 0.0, 0.5)
        g.memory_decay = jitter(g.memory_decay, 0.05, 0.0, 0.2)
        return g.clamp()

    def crossover(self, other: Genome) -> Genome:
        return Genome(
            context_order=random.choice([self.context_order, other.context_order]),
            learning_rate=(self.learning_rate + other.learning_rate) / 2,
            idea_min_support=random.choice([self.idea_min_support, other.idea_min_support]),
            novelty_bonus=(self.novelty_bonus + other.novelty_bonus) / 2,
            exploration=(self.exploration + other.exploration) / 2,
            memory_decay=(self.memory_decay + other.memory_decay) / 2,
        ).clamp()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Genome:
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> Genome:
        return cls.from_dict(json.loads(raw))

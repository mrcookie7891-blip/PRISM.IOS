"""Persistent knowledge: transitions, ideas, and evolution history."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time

from prism.genome import Genome


@dataclass
class Idea:
    """A discovered pattern the mind considers worth remembering."""

    pattern: str
    strength: float
    generation: int
    connections: list[str] = field(default_factory=list)
    insight: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Idea:
        return cls(**data)


@dataclass
class KnowledgeBase:
    """Everything the mind has learned across its lifetime."""

    generation: int = 0
    fitness: float = 0.0
    intelligence_score: float = 0.0
    bytes_ingested: int = 0
    genome: Genome = field(default_factory=Genome)
    transitions: dict[str, dict[str, float]] = field(default_factory=dict)
    ideas: list[Idea] = field(default_factory=list)
    fitness_history: list[float] = field(default_factory=list)
    intelligence_history: list[float] = field(default_factory=list)

    def top_ideas(self, limit: int = 10) -> list[Idea]:
        return sorted(self.ideas, key=lambda idea: (-idea.strength, idea.pattern))[:limit]

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "fitness": self.fitness,
            "intelligence_score": self.intelligence_score,
            "bytes_ingested": self.bytes_ingested,
            "genome": self.genome.to_dict(),
            "transitions": self.transitions,
            "ideas": [idea.to_dict() for idea in self.ideas],
            "fitness_history": self.fitness_history,
            "intelligence_history": self.intelligence_history,
            "saved_at": time.time(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeBase:
        return cls(
            generation=data.get("generation", 0),
            fitness=data.get("fitness", 0.0),
            intelligence_score=data.get("intelligence_score", 0.0),
            bytes_ingested=data.get("bytes_ingested", 0),
            genome=Genome.from_dict(data.get("genome", {})),
            transitions=data.get("transitions", {}),
            ideas=[Idea.from_dict(item) for item in data.get("ideas", [])],
            fitness_history=data.get("fitness_history", []),
            intelligence_history=data.get("intelligence_history", []),
        )

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> KnowledgeBase:
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

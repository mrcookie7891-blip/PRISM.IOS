"""Evolution engine: the mind mutates its own learning strategy and keeps what works."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from prism.genome import Genome
from prism.knowledge import KnowledgeBase
from prism.mind import PrismMind


@dataclass
class EvolutionResult:
    generation: int
    fitness: float
    intelligence_score: float
    improved: bool
    genome: Genome
    top_ideas: list[str]


class EvolutionEngine:
    """
    Runs self-evolution cycles:

    1. Train on data
    2. Measure fitness (prediction + idea richness)
    3. Mutate genome and compete candidates
    4. Keep the smartest variant and persist state
    """

    def __init__(
        self,
        state_path: Path | str = ".prism_state/mind.json",
        population_size: int = 4,
        train_sample_chars: int = 200_000,
        eval_sample_chars: int = 40_000,
    ) -> None:
        self.state_path = Path(state_path)
        self.population_size = max(2, population_size)
        self.train_sample_chars = train_sample_chars
        self.eval_sample_chars = eval_sample_chars
        self.mind = PrismMind(KnowledgeBase.load(self.state_path))

    @staticmethod
    def _sample_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        third = max_chars // 3
        mid = len(text) // 2
        return text[:third] + text[mid : mid + third] + text[-third:]

    def _fitness(self, mind: PrismMind, train_text: str, eval_text: str) -> float:
        mind.ingest(train_text)
        stats = mind.evaluate(eval_text)
        idea_bonus = min(10.0, len(mind.knowledge.ideas) ** 0.5)
        ll_bonus = max(0.0, stats["log_likelihood"] + 4) * 5
        acc_bonus = stats["top1_accuracy"] * 30
        return ll_bonus + acc_bonus + idea_bonus

    def evolve_once(
        self,
        train_text: str,
        eval_text: str | None = None,
    ) -> EvolutionResult:
        eval_text = eval_text or train_text[-min(len(train_text), 50_000) :]
        train_sample = self._sample_text(train_text, self.train_sample_chars)
        eval_sample = self._sample_text(eval_text, self.eval_sample_chars)

        base = self.mind
        base_genome = base.genome
        base_candidate = base.clone_with_genome(base_genome)
        base_fitness = self._fitness(base_candidate, train_sample, eval_sample)

        candidates: list[tuple[float, PrismMind]] = [(base_fitness, base_candidate)]
        for _ in range(self.population_size - 1):
            mutant = base.clone_with_genome(base_genome.mutate())
            fit = self._fitness(mutant, train_sample, eval_sample)
            candidates.append((fit, mutant))

        if random.random() < 0.25:
            hybrid = base.clone_with_genome(
                base_genome.crossover(Genome().mutate()).mutate(rate=0.15)
            )
            candidates.append((self._fitness(hybrid, train_sample, eval_sample), hybrid))

        candidates.sort(key=lambda item: -item[0])
        best_fitness, winner = candidates[0]
        improved = best_fitness >= base.knowledge.fitness

        winner.ingest(train_text)
        winner.knowledge.generation = base.knowledge.generation + 1
        winner.knowledge.fitness = best_fitness
        eval_stats = winner.evaluate(self._sample_text(eval_text, self.eval_sample_chars))
        winner.knowledge.intelligence_score = winner.compute_intelligence(eval_stats)
        winner.knowledge.fitness_history.append(best_fitness)
        winner.knowledge.intelligence_history.append(winner.knowledge.intelligence_score)
        winner.knowledge.save(self.state_path)

        self.mind = winner
        return EvolutionResult(
            generation=winner.knowledge.generation,
            fitness=best_fitness,
            intelligence_score=winner.knowledge.intelligence_score,
            improved=improved,
            genome=winner.genome,
            top_ideas=[idea.pattern for idea in winner.knowledge.top_ideas(8)],
        )

    def evolve_many(
        self,
        train_text: str,
        generations: int = 5,
        eval_text: str | None = None,
    ) -> list[EvolutionResult]:
        results: list[EvolutionResult] = []
        for _ in range(generations):
            results.append(self.evolve_once(train_text, eval_text))
        return results

    def learn_only(self, text: str) -> dict:
        stats = self.mind.ingest(text)
        eval_stats = self.mind.evaluate(text)
        self.mind.knowledge.intelligence_score = self.mind.compute_intelligence(eval_stats)
        self.mind.knowledge.save(self.state_path)
        return {**stats, **eval_stats, "intelligence_score": self.mind.knowledge.intelligence_score}

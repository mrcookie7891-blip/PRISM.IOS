"""Compete learning strategies on held-out prediction, preserving the incumbent."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
from letter_follow_mapper import normalize_text
from prism.genome import Genome
from prism.knowledge import KnowledgeBase
from prism.mind import PrismMind


@dataclass
class EvolutionResult:
    generation: int
    fitness: float
    intelligence_score: float  # legacy field: prediction score only
    improved: bool
    genome: Genome
    top_ideas: list[str]


def held_out(train_text, eval_text=None):
    if eval_text is None:
        if len(normalize_text(train_text)) < 20:
            raise ValueError('Need at least 20 letters for a held-out split')
        cut = int(len(train_text) * .8)
        train_text, eval_text = train_text[:cut], train_text[cut:]
    elif not eval_text or normalize_text(eval_text) in normalize_text(train_text):
        raise ValueError('Evaluation text must be nonempty and excluded from training')
    if min(len(normalize_text(train_text)), len(normalize_text(eval_text))) < 3:
        raise ValueError('Training and evaluation require at least three letters each')
    return train_text, eval_text


class EvolutionEngine:
    def __init__(self, state_path='.prism_state/mind.json', population_size=4,
                 train_sample_chars=200_000, eval_sample_chars=40_000):
        if min(train_sample_chars, eval_sample_chars) < 3:
            raise ValueError('Sample limits must be at least three')
        self.state_path = Path(state_path)
        self.population_size = max(2, population_size)
        self.train_sample_chars = train_sample_chars
        self.eval_sample_chars = eval_sample_chars
        self.mind = PrismMind(KnowledgeBase.load(self.state_path))

    @staticmethod
    def _sample_text(text, max_chars):
        # Contiguous sampling avoids inventing transitions between distant chunks.
        return text[:max_chars]

    def _fitness(self, mind, train_text, eval_text):
        mind.ingest(train_text)
        return mind.evaluate(eval_text)['log_likelihood']

    def evolve_once(self, train_text, eval_text=None):
        train_text, eval_text = held_out(train_text, eval_text)
        train_sample = self._sample_text(train_text, self.train_sample_chars)
        eval_sample = self._sample_text(eval_text, self.eval_sample_chars)
        base = self.mind
        # Always compare on this same benchmark, not scores from prior datasets.
        incumbent = base.evaluate(eval_sample)['log_likelihood']
        candidates = [(incumbent, base.clone_with_genome(base.genome))]
        genomes = [base.genome] + [base.genome.mutate() for _ in range(self.population_size - 1)]
        if random.random() < .25:
            genomes.append(base.genome.crossover(Genome().mutate()).mutate(rate=.15))
        for genome in genomes:
            candidate = base.clone_with_genome(genome)
            candidates.append((self._fitness(candidate, train_sample, eval_sample), candidate))
        best_fitness, winner = max(candidates, key=lambda item: item[0])
        # Do not retrain after selection: persist exactly the evaluated model.
        winner.knowledge.generation = base.knowledge.generation + 1
        winner.knowledge.fitness = best_fitness
        winner.knowledge.intelligence_score = winner.compute_intelligence(winner.evaluate(eval_sample))
        winner.knowledge.fitness_history.append(best_fitness)
        winner.knowledge.intelligence_history.append(winner.knowledge.intelligence_score)
        winner.knowledge.save(self.state_path)
        self.mind = winner
        return EvolutionResult(winner.knowledge.generation, best_fitness,
            winner.knowledge.intelligence_score, best_fitness > incumbent,
            winner.genome, [idea.pattern for idea in winner.knowledge.top_ideas(8)])

    def evolve_many(self, train_text, generations=5, eval_text=None):
        return [self.evolve_once(train_text, eval_text) for _ in range(generations)]

    def learn_only(self, text, eval_text=None):
        training, validation = held_out(text, eval_text)
        stats = self.mind.ingest(training)
        eval_stats = self.mind.evaluate(validation)
        self.mind.knowledge.intelligence_score = self.mind.compute_intelligence(eval_stats)
        self.mind.knowledge.save(self.state_path)
        return {**stats, **eval_stats, 'prediction_score': self.mind.knowledge.intelligence_score}

"""Tests for the PRISM self-evolving mind."""

import tempfile
import unittest
from pathlib import Path

from prism.evolution import EvolutionEngine
from prism.mind import PrismMind
from prism.knowledge import KnowledgeBase


class PrismMindTests(unittest.TestCase):
    def test_ingest_and_predict(self) -> None:
        mind = PrismMind()
        mind.ingest("th" * 100 + "the " * 50)
        dist = mind.predict_next("t")
        self.assertGreater(dist.get("h", 0), dist.get("z", 0))

    def test_intelligence_grows_with_learning(self) -> None:
        mind = PrismMind()
        before = mind.compute_intelligence({"top1_accuracy": 0, "log_likelihood": -5})
        mind.ingest("the quick brown fox jumps over the lazy dog " * 20)
        after_stats = mind.evaluate("the quick brown fox jumps over the lazy dog " * 5)
        after = mind.compute_intelligence(after_stats)
        self.assertGreater(after, before)

    def test_evolution_improves_or_maintains_fitness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "mind.json"
            engine = EvolutionEngine(state_path=state, population_size=3)
            text = "abcdefghijklmnopqrstuvwxyz " * 30 + "the thee thou thread " * 40
            first = engine.evolve_once(text, eval_text=text)
            second = engine.evolve_once(text, eval_text=text)
            self.assertGreaterEqual(second.generation, first.generation)
            self.assertTrue(state.is_file())

    def test_persistence_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mind.json"
            mind = PrismMind()
            mind.ingest("evolving minds learn patterns")
            mind.knowledge.intelligence_score = 12.34
            mind.knowledge.save(path)
            loaded = KnowledgeBase.load(path)
            self.assertEqual(loaded.intelligence_score, 12.34)
            self.assertGreater(len(loaded.transitions), 0)


if __name__ == "__main__":
    unittest.main()

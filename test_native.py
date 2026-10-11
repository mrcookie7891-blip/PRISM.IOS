import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from prism.native import CharacterTokenizer, Config, NativeModel, batch_at, examples, train
from prism.dataset import split_documents
from prism.evolution import EvolutionEngine
from prism.genome import Genome
from prism.mind import PrismMind


class NativeTests(unittest.TestCase):
    def test_vocabulary_and_document_boundaries(self):
        tokenizer = CharacterTokenizer.fit(['ab', 'é!'])
        self.assertEqual(tokenizer.decode(tokenizer.encode('abé!')), 'abé!')
        self.assertEqual(tokenizer.encode('Ω')[0], 1)
        streams = examples(['ab', 'é!'], tokenizer, 3)
        x, y = batch_at(streams, np.array([0, 1, 2, 3]), 3)
        np.testing.assert_array_equal(x[2], [0, 0, 0])
        self.assertEqual(y[2], tokenizer.encode('é')[0])

    def test_gradients(self):
        model = NativeModel(CharacterTokenizer.fit(['abc']), Config(2, 3, 4, 3))
        x = np.array([[0, 2], [2, 3], [3, 2]])
        y = np.array([3, 4, 4])
        _, gradients = model.loss_and_gradients(x, y)
        for name, index in [('embedding', (2, 1)), ('w1', (1, 2)), ('w2', (1, 3)), ('b1', (1,)), ('b2', (3,))]:
            weight = model.weights[name]
            original, eps = float(weight[index]), .002
            weight[index] = original + eps
            plus = model.loss_and_gradients(x, y)[0]
            weight[index] = original - eps
            minus = model.loss_and_gradients(x, y)[0]
            weight[index] = original
            self.assertAlmostEqual(float(gradients[name][index]), (plus-minus)/(2*eps), delta=.001)

    def test_learning_export_and_generation(self):
        tokenizer = CharacterTokenizer.fit(['abcd'])
        model = NativeModel(tokenizer, Config(4, 4, 16))
        training = examples(['abcd' * 50], tokenizer, 4)
        validation = examples(['abcd' * 12], tokenizer, 4)
        initial = model.evaluate(validation)['nll']
        train(model, training, steps=150, batch_size=32, learning_rate=.01)
        final = model.evaluate(validation)['nll']
        self.assertLess(final, initial * .2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.npz'
            model.save(path)
            loaded = NativeModel.load(path)
            for key in model.weights:
                np.testing.assert_array_equal(model.weights[key], loaded.weights[key])
            model.save(path, quantize=True)
            quantized = NativeModel.load(path)
            self.assertLess(abs(quantized.evaluate(validation)['nll'] - final), .03)
            self.assertEqual(quantized.generate('a', 20, 0), model.generate('a', 20, 0))
        self.assertEqual(model.generate('a', 20, seed=7), model.generate('a', 20, seed=7))

    def test_splits_deduplicate_and_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.txt'
            path.write_text('\n'.join(['line %d' % i for i in range(30)] * 2))
            a, meta = split_documents([path], 10)
            b, _ = split_documents([path], 10)
            self.assertEqual(a, b)
            self.assertEqual(meta['unique_lines'], 30)
            self.assertFalse(set(a['train']) & set(a['validation']))
            self.assertFalse(set(a['test']) & set(a['train']))
            tokenizer = CharacterTokenizer.fit(a['train'])
            self.assertNotIn('Ω', tokenizer.symbols)

    def test_clone_isolation_and_score(self):
        mind = PrismMind()
        mind.ingest('the cat and the dog ' * 10)
        before = mind.knowledge.to_dict()
        cloned = mind.clone_with_genome(Genome(context_order=3))
        cloned.ingest('zebra zebra zebra')
        self.assertEqual(before['transitions'], mind.knowledge.transitions)
        self.assertEqual(before['ideas'], mind.knowledge.to_dict()['ideas'])
        stats = mind.evaluate('the mouse')
        score = mind.compute_intelligence(stats)
        mind.knowledge.generation += 100
        mind.knowledge.bytes_ingested += 10000000
        self.assertEqual(score, mind.compute_intelligence(stats))
        self.assertIn('the', [idea.pattern for idea in mind.knowledge.ideas])

    def test_evolution_rejects_leakage_and_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = EvolutionEngine(Path(directory) / 'mind.json')
            with self.assertRaises(ValueError):
                engine.evolve_once('the same data', 'the same data')
            engine.mind.ingest('abcd' * 50)
            before = engine.mind.evaluate('abcabc')['log_likelihood']
            with patch.object(engine, '_fitness', return_value=before - 1):
                result = engine.evolve_once('defdefdef', 'abcabc')
            self.assertEqual(result.fitness, before)
            self.assertFalse(result.improved)
            self.assertEqual(result.fitness, engine.mind.evaluate('abcabc')['log_likelihood'])


if __name__ == '__main__':
    unittest.main()

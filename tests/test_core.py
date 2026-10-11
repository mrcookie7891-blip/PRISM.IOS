import tempfile
import unittest
from pathlib import Path
import numpy as np
from prism.core import Model, split_data, windows


class CoreTests(unittest.TestCase):
    def test_gradients(self):
        model = Model(2, 3)
        x = np.array([[1, 2], [2, 3]]); y = np.array([3, 4])
        gradients = model.gradients(x, y)
        for key, index in [('embedding', (2, 1)), ('input', (1, 2)),
                           ('output', (1, 3)), ('hidden_bias', (1,)), ('output_bias', (3,))]:
            w = model.weights[key]; original = w[index]; eps = .002
            w[index] = original + eps; plus = model.loss(x, y)
            w[index] = original - eps; minus = model.loss(x, y)
            w[index] = original
            self.assertAlmostEqual(float(gradients[key][index]), (plus - minus) / (2 * eps), delta=.001)

    def test_learning_and_roundtrip(self):
        model = Model(4, 8)
        data = np.frombuffer(b'abcd' * 100, dtype=np.uint8)
        x, y = windows(data, np.arange(64), 4)
        initial = model.loss(x, y)
        for _ in range(150): model.step(x, y, .3)
        self.assertLess(model.loss(x, y), initial * .3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.npz'
            model.save(path)
            np.testing.assert_array_equal(model.forward(x)[2], Model.load(path).forward(x)[2])
            model.save(path, quantized=True)
            self.assertLess(abs(model.loss(x, y) - Model.load(path).loss(x, y)), .03)
        self.assertEqual(model.generate('a', 10, 42), model.generate('a', 10, 42))

    def test_split_boundaries(self):
        parts = split_data(b'a' * 80 + b'b' * 10 + b'c' * 10, 4)
        for part, expected in zip(parts, (97, 98, 99)):
            x, y = windows(part, np.arange(len(part) - 4), 4)
            self.assertTrue(np.all(x == expected) and np.all(y == expected))
        with self.assertRaises(ValueError): split_data(b'abc', 4)


if __name__ == '__main__': unittest.main()

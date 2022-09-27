import unittest
import numpy as np
from adaptive_neighbourhoods import distances


class TestDistances(unittest.TestCase):
    def test__inverse_multiquadric(self):
        result = distances.inverse_multiquadric(1, 2)
        self.assertAlmostEqual(result, 0.40824829046)
        result = distances.inverse_multiquadric(5, 5)
        self.assertEqual(result, 1.0)
        result = distances.inverse_multiquadric(
            np.array([1, 5]),
            np.array([2, 5]))
        self.assertTrue(np.allclose(result, np.array([0.40824829046, 1.0])))

    def test__norm(self):
        self.assertEqual(distances.norm(1, 1), 0)

    def test__inverse_quadric(self):
        self.assertTrue(distances.inverse_quadric(1, 2), 0.5)
        self.assertTrue(distances.inverse_quadric(1, 1), 0)


if __name__ == "__main__":
    unittest.main()

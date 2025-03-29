import unittest

import numpy as np

from utils import generate_states, hdp_states, infer_emissions


class TestGenerateStates(unittest.TestCase):
    def test_n(self):
        s, oracle, K, n, n_oracle = generate_states(T=100, alpha=0.1, beta=100, gamma=10, debug=False)

        n_debug = np.zeros_like(n)
        n_debug[np.diag_indices_from(n_debug)] += alpha
        np.add.at(n_debug, (s[:-1], s[1:]), 1)
        assert np.mean(n == n_debug) == 1
        assert n.shape[0] == K
        assert n_oracle.shape[0] == K

        self.assertEqual(np.mean(n == n_debug), 1)

    def test_K(self):
        self.assertEqual(1, 2)


if __name__ == '__main__':
    unittest.main()

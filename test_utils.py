import unittest

import numpy as np

from utils import validate_s, validate_oracle, count_n, count_n_oracle
from utils import generate_states, hdp_states, infer_emissions


class TestValidateOracle(unittest.TestCase):
    def test_not_np_array(self):
        oracle = [True, True, False, False]
        with self.assertRaises(TypeError) as context:
            validate_oracle(oracle)

        self.assertEqual(str(context.exception), "Oracle indicator sequence must be a numpy array.")

    def test_not_a_vector(self):
        oracle = np.zeros(shape=(0, 0))
        with self.assertRaises(ValueError) as context:
            validate_oracle(oracle)

        self.assertEqual(str(context.exception), "Oracle indicator sequence must be a vector.")

    def test_not_int64(self):
        oracle = np.zeros(shape=(0,), dtype=np.float64)
        with self.assertRaises(TypeError) as context:
            validate_oracle(oracle)

        self.assertEqual(str(context.exception), "Oracle indicator must be of type bool.")

    def test_valid_array(self):
        oracle = np.array([True, False, True, False], dtype=bool)
        validate_oracle(oracle)


class TestValidateS(unittest.TestCase):
    def test_not_np_array(self):
        s = [0, 0, 0, 0]
        with self.assertRaises(TypeError) as context:
            validate_s(s)

        self.assertEqual(str(context.exception), "Hidden state sequence must be a numpy array.")

    def test_not_a_vector(self):
        s = np.zeros(shape=(0, 0))
        with self.assertRaises(ValueError) as context:
            validate_s(s)

        self.assertEqual(str(context.exception), "Hidden state sequence must be a vector.")

    def test_not_int64(self):
        s = np.zeros(shape=(0,), dtype=np.float64)
        with self.assertRaises(TypeError) as context:
            validate_s(s)

        self.assertEqual(str(context.exception), "Hidden state sequence must be of type np.int64.")

    def test_min_state_not_0(self):
        s = np.array([11, 12, 13, 14], dtype=np.int64)
        with self.assertRaises(ValueError) as context:
            validate_s(s)

        self.assertEqual(str(context.exception), "Minimum hidden state element must be state 0, not state 11.")

    def test_min_state_not_negative(self):
        s = np.array([0, 0, 0, -1], dtype=np.int64)
        with self.assertRaises(ValueError) as context:
            validate_s(s)

        self.assertEqual(str(context.exception), "Minimum hidden state element must be state 0, not state -1.")

    def test_skipped_states(self):
        s = np.array([0, 1, 99, 100], dtype=np.int64)
        with self.assertRaises(ValueError) as context:
            validate_s(s)

        self.assertEqual(str(context.exception), "Hidden state elements must increase by 1 consecutively.")

    def test_valid_array(self):
        s = np.array([0, 1, 2, 3], dtype=np.int64)
        validate_s(s)


class TestCountN(unittest.TestCase):
    def test_self_transition(self):
        s = np.array([0] * 11, dtype=np.int64)  # Ten self-transitions from state 0 to state 0
        n = count_n(s, alpha=0)

        self.assertEqual(n, np.array([[10.0]]))

    def test_n_shape(self):
        for K in [1, 10]:
            s = np.arange(K)
            n = count_n(s, alpha=0)
            self.assertEquals(n.shape[0], K)
            self.assertEquals(n.shape[1], K)

    def test_incremental_states(self):
        K = 3
        s = np.arange(K)
        n = count_n(s, alpha=0)
        self.assertTrue(np.all(n == np.array([[0, 1.0, 0], [0, 0, 1.0], [0, 0, 0]])))

    def test_alpha(self):
        K = 100
        alpha = 5.0
        s = np.arange(K)
        n = count_n(s, alpha=alpha)
        self.assertTrue(np.all(np.diag(n) == alpha))

    def test_pattern_1(self):
        s = np.array([0, 0, 1, 2], dtype=np.int64)
        alpha = 1.74
        n = count_n(s, alpha=alpha)
        self.assertTrue(np.all(n == np.array([[1+alpha, 1, 0], [0, alpha, 1], [0, 0, alpha]])))

    def test_pattern_2(self):
        s = np.array([0, 0, 1, 2, 0, 1, 1, 1, 0, 2, 0, 2, 2, 1, 2], dtype=np.int64)
        alpha = 3.45
        n = count_n(s, alpha=alpha)
        self.assertTrue(np.all(n == np.array([[1+alpha, 2, 2], [1, 2+alpha, 2], [2, 1, 1+alpha]])))


class TestCountNOracle(unittest.TestCase):
    def test_different_lengths(self):
        s = np.arange(100)
        oracle = np.zeros(shape=101, dtype=bool)
        with self.assertRaises(ValueError) as context:
            count_n_oracle(s, oracle)

        self.assertEqual(str(context.exception), "Hidden state sequence and oracle indicator sequence must be the same length.")

    def test_n_oracle_shape(self):
        for K in [1, 10, 100]:
            s = np.arange(K)
            oracle = np.zeros(shape=K, dtype=bool)
            n_oracle = count_n_oracle(s, oracle)

            self.assertEqual(n_oracle.shape[0], K)

    def test_all_true(self):
        s = np.array([0, 1, 2, 3], dtype=np.int64)
        oracle = np.array([True, True, True, True], dtype=bool)
        n_oracle = count_n_oracle(s, oracle)
        self.assertTrue(np.all(n_oracle == np.array([1, 1, 1, 1])))

    def test_all_false(self):
        s = np.array([0, 1, 2, 3], dtype=np.int64)
        oracle = np.array([False, False, False, False], dtype=bool)
        n_oracle = count_n_oracle(s, oracle)
        self.assertTrue(np.all(n_oracle == np.array([0, 0, 0, 0])))

    def test_pattern_1(self):
        s = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        oracle = np.array([False, True, False, True, False], dtype=bool)
        n_oracle = count_n_oracle(s, oracle)
        self.assertTrue(np.all(n_oracle == np.array([2])))

    def test_pattern_2(self):
        s = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        oracle = np.array([False, True, False, True, False], dtype=bool)
        n_oracle = count_n_oracle(s, oracle)
        self.assertTrue(np.all(n_oracle == np.array([1, 1])))

    def test_patterns(self):
        for i in np.arange(1, 10):
            for j in np.arange(1, 10):
                s = np.array([0] * i + [1] * j, dtype=np.int64)
                oracle = np.array(s, dtype=bool)
                n_oracle = count_n_oracle(s, oracle)
                self.assertTrue(np.all(n_oracle == np.array([0, j])))

    def test_sums(self):
        for i in np.arange(1, 10):
            for j in np.arange(1, 10):
                s = np.array([0] * i + [1] * j, dtype=np.int64)
                oracle = np.array(s, dtype=bool)
                np.random.shuffle(s)
                np.random.shuffle(oracle)
                n_oracle = count_n_oracle(s, oracle)
                self.assertEqual(np.sum(oracle), np.sum(n_oracle))


if __name__ == '__main__':
    unittest.main()

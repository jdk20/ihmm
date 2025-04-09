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


class TestHDPStates(unittest.TestCase):
    def setUp(self):
        self.max_iter = 1000
        self.ni = [[0], [0, 0, 0, 0, 0], [1], [1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [11, 0, 11, 0, 11]]

    @staticmethod
    def dp_states_1(ni, i, alpha, beta):
        denominator = np.sum(ni) + beta + alpha
        ni[i] += alpha
        ni = np.append(ni, beta)
        p = ni/denominator

        return p

    @staticmethod
    def dp_states_2(n_oracle, gamma):
        denominator = np.sum(n_oracle) + gamma
        n_oracle = np.append(n_oracle, gamma)
        p = n_oracle/denominator

        return p

    def hdp_draws(self, alpha, beta, gamma, current_state, ni, no):
        K = len(ni)
        oracle_count = 0
        dp_1_counts = np.zeros(K + 1, dtype=np.float64)  # K states and beta
        dp_2_counts = np.zeros(K + 1, dtype=np.float64)  # K states and gamma
        for i in range(self.max_iter):
            n = np.zeros((K, K), dtype=np.float64)
            n_oracle = np.array(no)  # disable oracle transitions
            n[current_state, :] = ni
            next_state, is_oracle, _, _ = hdp_states(alpha, beta, gamma, current_state, n, n_oracle)

            if is_oracle:  # indicates second DP was called
                dp_1_counts[-1] += 1  # increments beta counter
                dp_2_counts[next_state] += 1
                oracle_count += 1
            else:
                dp_1_counts[next_state] += 1

        # Reset n and n_oracle
        n = np.zeros((K, K), dtype=np.float64)
        n[current_state, :] = ni
        n_oracle = np.array(no)

        # First DP
        p_simulated_1 = dp_1_counts/self.max_iter
        p_exact_1 = self.dp_states_1(n[current_state, :], current_state, alpha, beta)

        # Second DP
        p_simulated_2 = dp_2_counts/oracle_count
        p_exact_2 = self.dp_states_2(n_oracle, gamma)

        return p_exact_1, p_simulated_1, p_exact_2, p_simulated_2

    def test_zero_alpha(self):
        current_state = 0
        alpha, beta, gamma = 0, 1, 1

        for ni in self.ni:
            p_exact_1, p_simulated_1, _, _ = self.hdp_draws(alpha, beta, gamma, current_state, ni, [0] * len(ni))
            mae = np.mean(np.abs(p_exact_1 - p_simulated_1))
            self.assertLess(mae, 0.1, f"MAE: {mae}, exact: {p_exact_1}, simulated: {p_simulated_1}")

    def test_low_beta(self):
        current_state = 0
        alpha, beta, gamma = 1, 1, 1

        for ni in self.ni:
            p_exact_1, p_simulated_1, _, _ = self.hdp_draws(alpha, beta, gamma, current_state, ni, [0] * len(ni))
            mae = np.mean(np.abs(p_exact_1 - p_simulated_1))
            self.assertLess(mae, 0.1, f"MAE: {mae}, exact: {p_exact_1}, simulated: {p_simulated_1}")

    def test_self_transitions(self):
        alpha, beta, gamma = 8, 1, 1
        ni = [1, 2, 3, 4, 5]
        for current_state in [0, 1, 2, 3, 4]:
            p_exact_1, p_simulated_1, _, _ = self.hdp_draws(alpha, beta, gamma, current_state, ni, [0] * len(ni))
            mae = np.mean(np.abs(p_exact_1 - p_simulated_1))
            self.assertLess(mae, 0.1, f"MAE: {mae}, exact: {p_exact_1}, simulated: {p_simulated_1}")

    def test_high_beta(self):
        current_state = 0
        alpha, beta, gamma = 1, 1000, 1

        for ni in self.ni:
            p_exact_1, p_simulated_1, _, _ = self.hdp_draws(alpha, beta, gamma, current_state, ni, [0] * len(ni))
            mae = np.mean(np.abs(p_exact_1 - p_simulated_1))
            self.assertLess(mae, 0.1, f"MAE: {mae}, exact: {p_exact_1}, simulated: {p_simulated_1}")

    def test_low_gamma(self):
        current_state = 0
        alpha, beta, gamma = 1, 100, 1

        for ni in self.ni:
            _, _, p_exact_2, p_simulated_2 = self.hdp_draws(alpha, beta, gamma, current_state, ni, [0] * len(ni))
            mae = np.mean(np.abs(p_exact_2 - p_simulated_2))
            self.assertLess(mae, 0.1, f"MAE: {mae}, exact: {p_exact_2}, simulated: {p_simulated_2}")

    def test_high_gamma(self):
        current_state = 0
        alpha, beta, gamma = 1, 100, 10000

        for ni in self.ni:
            _, _, p_exact_2, p_simulated_2 = self.hdp_draws(alpha, beta, gamma, current_state, ni, [0] * len(ni))
            mae = np.mean(np.abs(p_exact_2 - p_simulated_2))
            self.assertLess(mae, 0.1, f"MAE: {mae}, exact: {p_exact_2}, simulated: {p_simulated_2}")


class TestGenerateStates(unittest.TestCase):
    def setUp(self):
        self.T, self.alpha, self.beta, self.gamma = 1, 1, 1, 1
        self.s = np.array([0, 1, 2, 3], dtype=np.int64)
        self.oracle = np.array([True, True, True, True], dtype=bool)
        self.n_oracle = count_n_oracle(self.s, self.oracle)
        self.n = count_n(self.s, alpha=self.alpha)

    def test_valid_args_none(self):
        generate_states(self.T, self.alpha, self.beta, self.gamma)

    def test_valid_args_s(self):
        generate_states(self.T, self.alpha, self.beta, self.gamma, s=self.s)

    def test_valid_args_s_oracle(self):
        generate_states(self.T, self.alpha, self.beta, self.gamma, s=self.s, oracle=self.oracle)

    def test_valid_args_s_n(self):
        generate_states(self.T, self.alpha, self.beta, self.gamma, s=self.s, n=self.n)

    def test_valid_args_s_oracle_n(self):
        generate_states(self.T, self.alpha, self.beta, self.gamma, s=self.s, oracle=self.oracle, n=self.n)

    def test_valid_args_s_oracle_n_oracle(self):
        generate_states(self.T, self.alpha, self.beta, self.gamma, s=self.s,
                        oracle=self.oracle, n_oracle=self.n_oracle)

    def test_valid_args_all(self):
        generate_states(self.T, self.alpha, self.beta, self.gamma, s=self.s,
                        oracle=self.oracle, n=self.n, n_oracle=self.n_oracle)

    def test_invalid_args_no_s(self):
        with self.assertRaises(ValueError) as context:
            generate_states(self.T, self.alpha, self.beta, self.gamma, oracle=self.oracle)
        self.assertEqual(str(context.exception), "Additional arguments were given when s was not.")

        with self.assertRaises(ValueError) as context:
            generate_states(self.T, self.alpha, self.beta, self.gamma, n_oracle=self.n_oracle)
        self.assertEqual(str(context.exception), "Additional arguments were given when s was not.")

        with self.assertRaises(ValueError) as context:
            generate_states(self.T, self.alpha, self.beta, self.gamma, n=self.n)
        self.assertEqual(str(context.exception), "Additional arguments were given when s was not.")

    def test_invalid_args_s_n_oracle(self):
        with self.assertRaises(ValueError) as context:
            generate_states(self.T, self.alpha, self.beta, self.gamma, s=self.s, n_oracle=self.n_oracle)
        self.assertEqual(str(context.exception), "n_oracle was given when oracle was not.")


if __name__ == '__main__':
    unittest.main()

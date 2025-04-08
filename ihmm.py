import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from utils import hp_optimization_equations, hdp_states, infer_emissions, generate_states, count_n_oracle


# Import hyperparameter optimization equations
hp_opt = hp_optimization_equations()

# 1. Given observation sequence (emitted symbols) y_t, infer T and Q
y = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1] * 30  # ABCDEFEDCB
y = np.array(list(y), dtype=np.int64)
T = y.size

assert np.min(np.unique(y)) == 0, f"Symbol index should start at 0: {np.unique(y)}"
assert np.mean(np.diff(np.unique(y)) == 1) == 1.0, f"Symbols are non-incremental: {np.unique(y)}."

# 2. Set hyperparameters to initial values
# Gamma priors
vague_prior = 0.001
a_alpha, b_alpha, a_beta, b_beta, a_gamma, b_gamma, a_beta_e, b_beta_e, a_gamma_e, b_gamma_e = [vague_prior] * 10

# State HDP
alpha, beta, gamma = 0.1, 2, 10

# Observation HDP
beta_e, gamma_e = 1, 1

# 3. Generate initial hidden state sequence s, n, n_oracle and K
s, oracle, n, n_oracle = generate_states(T, alpha, beta, gamma, debug=True)

# 4. Infer m and m_oracle (using beta_e)
m, m_oracle, oracle_e = infer_emissions(y, s, beta_e)
assert np.sum(m_oracle) == np.sum(oracle_e)

K = np.max(s) + 1
Q = np.max(y) + 1
for iterations in range(3):
    # Gibbs sweep over T
    for t in range(1, T-1):
        current_state = s[t]
        previous_state = s[t-1]
        next_state = s[t+1]

        s[t] = -1  # until replaced by new_state draw

        # Remove s[t] from n, m, n_oracle, m_oracle
        if oracle[t]:
            n_oracle[current_state] -= 1
            oracle[t] = False

        if oracle_e[t]:
            m_oracle[y[t]] -= 1
            oracle_e[t] = False

        n[previous_state, current_state] -= 1  # transition into
        n[current_state, next_state] -= 1  # transition out of
        m[current_state, y[t]] -= 1

        # Shrink K, n, n_oracle, and m
        if current_state not in s:
            assert n_oracle[current_state] == 0
            # assert np.sum(np.delete(n[current_state, :], current_state)) == 0
            # assert np.sum(np.delete(n[:, current_state], current_state)) == 0
            assert np.sum(m[current_state, :]) == 0
            assert n[current_state, current_state] == alpha
            assert n.shape[0] == K
            assert n_oracle.shape[0] == K
            assert m.shape[0] == K

            K -= 1
            n_oracle = np.delete(n_oracle, current_state)
            n = np.delete(np.delete(n, current_state, 0), current_state, 1)
            m = np.delete(m, current_state, 0)
            s[s > current_state] -= 1

            assert n.shape[0] == K
            assert n_oracle.shape[0] == K
            assert m.shape[0] == K
            assert np.mean(n_oracle > 0) == 1.0
            assert np.mean(np.sum(m, axis=1) > 0) == 1.0
            assert np.mean(np.diff(np.unique(s)) == 1) == 1.0

        # Generate new s[t]
        new_state, is_oracle, n, n_oracle = hdp_states(alpha, beta, gamma, current_state, n, n_oracle, debug=True)
        new_K = n_oracle.shape[0]

        s[t] = new_state
        n[previous_state, new_state] += 1  # transition into
        n[new_state, next_state] += 1  # transition out of
        oracle[t] = is_oracle

        if new_K > K:
            m = np.pad(m, pad_width=[(0, 1), (0, 0)], mode='constant')

        K = new_K

        if m[new_state, y[t]] == 0:  # oracle: existing symbol
            m_oracle[y[t]] += 1
            m[new_state, y[t]] += 1
            oracle_e[t] = True
        elif m[new_state, y[t]] > 0:  # existing emission or oracle: existing symbol
            p_oracle = beta_e/(np.sum(m[new_state, :]) + beta_e)
            p = np.random.rand()
            if p <= p_oracle:  # oracle: existing symbol
                m_oracle[y[t]] += 1
                oracle_e[t] = True
            m[new_state, y[t]] += 1

    assert np.sum(oracle) == np.sum(n_oracle)
    assert np.sum(oracle_e) == np.sum(m_oracle)

    # Update hyperparameters
    kappa = np.sum(n > 0, axis=1)
    K_e = np.sum(m > 0, axis=1)
    T_o = np.sum(oracle)
    T_o_e = np.sum(oracle_e)

    for eq in range(5):
        if eq == 0:
            values = (beta, n, K, a_alpha, b_alpha)
        elif eq == 1:
            values = (alpha, n, kappa, K, a_beta, b_beta)
        elif eq == 2:
            values = (m, K_e, K, Q, a_beta_e, b_beta_e)
        elif eq == 3:
            values = (T_o, K, a_gamma, b_gamma)
        elif eq == 4:
            values = (T_o_e, K, a_gamma_e, b_gamma_e)

        r, output = optimize.newton(hp_opt[eq]['g'],
                                    0,
                                    fprime=hp_opt[eq]['g_prime'],
                                    args=values,
                                    full_output=True,
                                    maxiter=1000)
        r = np.exp(r)

        if eq == 0:
            alpha = r
            print(f'alpha = {alpha}')
        elif eq == 1:
            beta = r
            print(f'beta = {beta}')
        elif eq == 2:
            beta_e = r
            print(f'beta_e = {beta_e}')
        elif eq == 3:
            gamma = r
            print(f'gamma = {gamma}')
        elif eq == 4:
            gamma_e = r
            print(f'gamma_e = {gamma_e}')


"""
# Figure 1
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
T = 250
s, _, _, _, _ = generate_states(T, alpha=0.1, beta=1000, gamma=100, debug=True)
axs[0, 0].stairs(s, color='k')
s, _, _, _, _ = generate_states(T, alpha=0, beta=0.1, gamma=100, debug=True)
axs[0, 1].stairs(s, color='k')
s, _, _, _, _ = generate_states(T, alpha=8, beta=2, gamma=2, debug=True)
axs[1, 0].stairs(s, color='k')
s, _, _, _, _ = generate_states(T, alpha=1, beta=1, gamma=10000, debug=True)
axs[1, 1].stairs(s, color='k')
plt.show()
"""

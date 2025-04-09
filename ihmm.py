import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from utils import (hp_optimization_equations, hdp_states, infer_emissions, generate_states, count_n_oracle, count_n,
                   validate_s)


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
s, oracle, n, n_oracle = generate_states(T, alpha, beta, gamma)

# 4. Infer m and m_oracle (using beta_e)
m, m_oracle, oracle_e = infer_emissions(y, s, beta_e)
assert np.sum(m_oracle) == np.sum(oracle_e)

K = np.max(s) + 1
Q = np.max(y) + 1

mass_n = (K * alpha) + (s.size - 1)
mass_m = y.size
assert np.mean(count_n(s, alpha) == n) == 1.0
assert np.sum(oracle) == np.sum(n_oracle)
assert np.sum(oracle_e) == np.sum(m_oracle)

for iteration in range(100):
    print(K)

    # Gibbs sweep over T
    for t in range(0, T):  # range(1, T-1)
        K = n_oracle.size
        if t > 0:
            previous_state = s[t-1]
        else:
            previous_state = np.nan

        current_state = s[t]
        current_obs = y[t]

        if t < T-1:
            next_state = s[t+1]
        else:
            next_state = np.nan

        assert n.shape[0] == K
        assert n_oracle.shape[0] == K
        assert m.shape[0] == K

        # --------------------------------------------------------------------------------------------------------------
        # Remove transition and emission counts for the current state
        # --------------------------------------------------------------------------------------------------------------
        # Modify n
        if not np.isnan(previous_state):
            n[previous_state, current_state] -= 1

        if not np.isnan(next_state):
            n[current_state, next_state] -= 1

        # Modify m
        m[current_state, current_obs] -= 1

        # Modify n_oracle
        if oracle[t]:  # oracle transition or new state
            oracle[t] = False
            assert n_oracle[current_state] != 0
            n_oracle[current_state] -= 1

        # Modify m_oracle
        if oracle_e[t]:  # oracle emission or new observation
            oracle_e[t] = False
            assert m_oracle[current_obs] != 0
            m_oracle[current_obs] -= 1

        assert np.sum(oracle) == np.sum(n_oracle)
        assert np.sum(oracle_e) == np.sum(m_oracle)

        # --------------------------------------------------------------------------------------------------------------
        # HDP sample
        # --------------------------------------------------------------------------------------------------------------
        is_oracle_e = bool(np.random.randint(2))
        new_state, is_oracle, _, _ = hdp_states(alpha, beta, gamma, current_state, n.copy(), n_oracle.copy())
        s[t] = new_state

        # Expand n, n_oracle
        if new_state == K:
            # print(f'State {new_state} added')
            K += 1
            is_oracle = True
            n = np.pad(n, pad_width=[(0, 1), (0, 1)], mode='constant')
            n[new_state, new_state] = alpha  # self-transition point mass
            mass_n += alpha
            n_oracle = np.append(n_oracle, 0)
            m = np.pad(m, pad_width=[(0, 1), (0, 0)], mode='constant')
        else:
            is_oracle = bool(np.random.randint(2))

        # --------------------------------------------------------------------------------------------------------------
        # Add transition and emission counts for the new state
        # --------------------------------------------------------------------------------------------------------------
        # Modify n
        if not np.isnan(previous_state):
            n[previous_state, new_state] += 1

        if not np.isnan(next_state):
            n[new_state, next_state] += 1

        # Modify m
        m[new_state, current_obs] += 1

        # Modify n_oracle
        if is_oracle:  # oracle transition or new state
            oracle[t] = True
            n_oracle[new_state] += 1

        # Modify m_oracle
        if is_oracle_e:
            oracle_e[t] = True
            m_oracle[current_obs] += 1

        # --------------------------------------------------------------------------------------------------------------
        # Clean up non-existant states
        # --------------------------------------------------------------------------------------------------------------
        if current_state not in s:
            assert n_oracle[current_state] == 0
            assert np.sum(np.delete(n[current_state, :], current_state)) == 0
            assert np.sum(np.delete(n[:, current_state], current_state)) == 0
            assert np.sum(m[current_state, :]) == 0
            assert np.allclose(n[current_state, current_state], alpha)
            # print(f'State {current_state} eliminated')
            K -= 1
            n_oracle = np.delete(n_oracle, current_state)
            n = np.delete(np.delete(n, current_state, 0), current_state, 1)
            m = np.delete(m, current_state, 0)
            s[s > current_state] -= 1
            mass_n -= alpha

            assert np.allclose(np.diff(np.unique(s)), 1)

        for state in np.unique(s):
            if state not in s:
                raise ValueError(f'Unaccounted elimination of {state}')

        # --------------------------------------------------------------------------------------------------------------
        # Debug
        # --------------------------------------------------------------------------------------------------------------
        # Check for zero-sum transition/emission counts
        n_debug = np.zeros(shape=(K, K), dtype=np.float64)
        n_debug[np.diag_indices_from(n_debug)] += alpha
        np.add.at(n_debug, (s[:-1], s[1:]), 1)

        m_debug = np.zeros(shape=(K, Q), dtype=np.float64)
        np.add.at(m_debug, (s, y), 1)

        n_oracle_debug = np.zeros(shape=K, dtype=np.int64)
        for k in range(K):
            n_oracle_debug[k] = np.sum(np.logical_and(s == k, oracle))

        m_oracle_debug = np.zeros(shape=Q, dtype=np.int64)
        for q in range(Q):
            m_oracle_debug[q] = np.sum(np.logical_and(y == q, oracle_e))

        assert np.allclose(mass_n, np.sum(n))
        assert np.allclose(mass_m, np.sum(m))
        assert np.allclose(n - n_debug, 0.0)
        assert np.allclose(m - m_debug, 0)
        assert np.allclose(n_oracle - n_oracle_debug, 0)
        assert np.allclose(m_oracle - m_oracle_debug, 0)

    # ------------------------------------------------------------------------------------------------------------------
    # Update hyperparameters
    # ------------------------------------------------------------------------------------------------------------------
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
            idx = np.diag_indices(n.shape[0])
            n[idx] -= alpha
            n[idx] += r
            alpha = r
            mass_n = (K * alpha) + (s.size - 1)
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
    print(f"Step: {t}")
    print(f"Previous state: {previous_state}")
    print(f"Current state: {current_state}")
    print(f"Next state: {next_state}")
    print(f"n[{previous_state}, {current_state}]: {n[previous_state, current_state]}")
    print(f"n[{current_state}, {next_state}]: {n[current_state, next_state]}")

    s[t] = -1

    # Remove s[t] from n, m, n_oracle, m_oracle
    if oracle[t]:
        n_oracle[current_state] -= 1
        oracle[t] = False

    if oracle_e[t]:
        m_oracle[y[t]] -= 1
        oracle_e[t] = False

    assert n[previous_state, current_state] - 1 >= 0
    assert n[current_state, next_state] - 1 >= 0

    n_before = n.copy()
    n[previous_state, current_state] -= 1  # transition into
    n[current_state, next_state] -= 1  # transition out of
    m[current_state, y[t]] -= 1
    assert np.allclose(np.sum(n_before) - np.sum(n), 2)

    # Shrink K, n, n_oracle, and m
    if current_state not in s:
        assert n_oracle[current_state] == 0
        assert np.sum(np.delete(n[current_state, :], current_state)) == 0
        assert np.sum(np.delete(n[:, current_state], current_state)) == 0
        assert np.sum(m[current_state, :]) == 0
        assert np.allclose(n[current_state, current_state], alpha)
        assert n.shape[0] == K
        assert n_oracle.shape[0] == K
        assert m.shape[0] == K

        print(f"Removing state {current_state}, shrinking from {K} states to {K-1} states")
        K -= 1
        n_oracle = np.delete(n_oracle, current_state)
        n = np.delete(np.delete(n, current_state, 0), current_state, 1)
        m = np.delete(m, current_state, 0)
        s[s > current_state] -= 1

        # assert np.mean(n_oracle > 0) == 1.0
        assert np.mean(np.sum(m, axis=1) > 0) == 1.0
        assert np.mean(np.diff(np.unique(s)) == 1) == 1.0
        assert n.shape[0] == K
        assert n_oracle.shape[0] == K
        assert m.shape[0] == K

    # Generate new s[t]
    new_state, is_oracle, _, _ = hdp_states(alpha, beta, gamma, current_state, n, n_oracle)
    s[t] = new_state
    validate_s(s)

    new_K = np.max(s)+1
    if new_K > K:
        print(f"Expanding states from {K} states to {new_K} states")
        m = np.pad(m, pad_width=[(0, 1), (0, 0)], mode='constant')
        n = np.pad(n, pad_width=[(0, 1), (0, 1)], mode='constant')
        n_oracle = np.append(n_oracle, 1)

    n[previous_state, new_state] += 1  # transition into
    n[new_state, next_state] += 1  # transition out of
    oracle[t] = is_oracle

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

    print('')

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
        idx = np.diag_indices(n.shape[0])
        n[idx] -= alpha
        n[idx] += r
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

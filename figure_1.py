import pickle
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from utils import hp_optimization_equations


def hdp_states(current_state, n, n_oracle, K, alpha, beta, gamma, debug=False):
    rng = np.random.default_rng()  # 1337+8
    is_oracle = False
    p = np.zeros(shape=1 + K, dtype=np.float64)  # oracle and all other transitions

    if n.size == 0:  # special-case when no states exist
        no_states = True
        p[0] = 1.0  # force oracle ignoring alpha/alpha+beta self-transition probability
    else:
        no_states = False
        n_existing = n[current_state, :]  # transitions to existing states (including self)
        nc = np.sum(n_existing) + beta + alpha  # normalizing constant
        p[0] = beta / nc  # oracle

    for k in range(K):
        if k == current_state:
            p[k + 1] = (n_existing[k] + alpha) / nc  # self-transition
        else:
            p[k + 1] = n_existing[k] / nc  # transition

    if debug:
        assert np.allclose(np.sum(p), 1.0), f'p={np.sum(p)}'
    choice = rng.choice(a=1 + K, size=1, p=p)

    # Oracle
    if choice == 0:
        p_oracle = np.zeros(shape=1 + K, dtype=np.float64)  # new state and all other transitions
        nc_oracle = np.sum(n_oracle) + gamma  # normalizing constant

        p_oracle[0] = gamma / nc_oracle  # new state

        for k in range(K):
            p_oracle[k + 1] = n_oracle[k] / nc_oracle

        if debug:
            assert np.allclose(np.sum(p_oracle), 1.0)
        choice_oracle = rng.choice(a=1 + K, size=1, p=p_oracle)
        # Oracle New State
        if choice_oracle == 0:
            K += 1  # increase state counter
            next_state = K - 1  # for zero-index

            # Expand n and n_oracle
            n_oracle = np.append(n_oracle, 0)
            n = np.pad(n, pad_width=[(0, 1), (0, 1)], mode='constant')

            # Add alpha to self-transition prob for new state
            n[next_state, next_state] += alpha
            if debug:
                txt = 'Oracle New State'
        # Oracle Transition to Self/Existing State
        else:
            next_state = choice_oracle[0] - 1
            if debug:
                if current_state == next_state:
                    txt = 'Oracle Self-Transition to State'
                else:
                    txt = 'Oracle Existing Transition to State'

        n_oracle[next_state] += 1  # increase state transition via oracle
        is_oracle = True
        if debug:
            p_txt = p_oracle[choice_oracle][0]

    # Non-Oracle Transition to Self/Existing State
    else:
        next_state = choice[0] - 1
        if debug:
            if current_state == next_state:
                txt = 'Self-Transition to State'
            else:
                txt = 'Existing Transition to State'
            p_txt = p[choice][0]

    if not no_states:
        n[current_state, next_state] += 1

    if debug:
        print(f'{txt} {next_state} from {current_state} with p={p_txt}')

    return next_state, is_oracle, n, n_oracle, K


def generate_states(T: int,
                    alpha: float,
                    beta: float,
                    gamma: float,
                    K: int = 0,
                    s = np.empty(shape=0, dtype=np.int64),
                    n = np.zeros(shape=(0, 0), dtype=np.float64),
                    n_oracle = np.zeros(shape=0, dtype=np.int64),
                    debug: bool = False):

    oracle = np.zeros(shape=T, dtype=bool)
    current_state, n_existing, nc = None, None, None

    for t in range(T):
        next_state, is_oracle, n, n_oracle, K = hdp_states(current_state, n, n_oracle, K,
                                                           alpha, beta, gamma, debug=debug)
        oracle[t] = is_oracle

        # Append next state to state sequence and update counts
        s = np.append(s, next_state)
        current_state = next_state

        if debug:
            n_debug = np.zeros_like(n)
            n_debug[np.diag_indices_from(n_debug)] += alpha
            np.add.at(n_debug, (s[:-1], s[1:]), 1)
            assert np.mean(n == n_debug) == 1
            assert n.shape[0] == K
            assert n_oracle.shape[0] == K

    return s, n, n_oracle, oracle, K


def infer_emissions(y, s, K, Q, beta_e):
    # 4. Generate m using s, K, Q
    m_debug = np.zeros(shape=(K, Q), dtype=np.float64)
    np.add.at(m_debug, (s, y), 1)

    # 5. Generate m_oracle
    m_oracle = np.zeros(shape=Q, dtype=np.int64)
    m = np.zeros(shape=(K, Q), dtype=np.float64)
    oracle_e = np.zeros_like(y, dtype=bool)  # indicator for Gibbs sweep

    symbols = []
    t = 0
    for i, q in zip(s, y):
        if q not in symbols:  # oracle: new symbol
            symbols.append(q)
            m_oracle[q] += 1
            m[i, q] += 1
            oracle_e[t] = True
        elif m[i, q] == 0:  # oracle: existing symbol
            m_oracle[q] += 1
            m[i, q] += 1
            oracle_e[t] = True
        elif m[i, q] > 0:  # existing emission or oracle: existing symbol
            p_oracle = beta_e/(np.sum(m[i, :]) + beta_e)
            p = np.random.rand()
            if p <= p_oracle:  # oracle: existing symbol
                m_oracle[q] += 1
                oracle_e[t] = True
            m[i, q] += 1

        t += 1

    assert np.mean(m == m_debug) == 1

    return m, m_oracle, oracle_e


# Import hyperparameter optimization equations
hp_opt = hp_optimization_equations()

# 1. Given observation sequence (emitted symbols) y_t, infer T and Q
y = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1] * 30  # ABCDEFEDCB
y = np.array(list(y), dtype=np.int64)
T = y.size
Q = np.unique(y).size  # unique symbols

assert np.min(np.unique(y)) == 0, f"Symbol index should start at 0: {np.unique(y)}"
assert np.mean(np.diff(np.unique(y)) == 1) == 1.0, f"Symbols are non-incremental: {np.unique(y)}."

# 2. Set hyperparameters to initial values
vague_prior = 0.001
a_alpha, b_alpha, a_beta, b_beta, a_beta_e, b_beta_e, a_gamma, b_gamma, a_gamma_e, b_gamma_e = [vague_prior] * 10
alpha, beta, gamma = 0.1, 100, 10
beta_e, gamma_e = 1, 1

# 3. Generate initial hidden state sequence s_t, n, n_oracle and K
s, n, n_oracle, oracle, K = generate_states(T, alpha, beta, gamma, debug=False)

# 4. Infer m and m_oracle (using beta_e)
m, m_oracle, oracle_e = infer_emissions(y, s, K, Q, beta_e)
assert np.sum(m_oracle) == np.sum(oracle_e)

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
            assert np.sum(np.delete(n[:, current_state], current_state)) == 0
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
        new_state, is_oracle, n, n_oracle, new_K = hdp_states(current_state, n, n_oracle, K,
                                                              alpha, beta, gamma, debug=True)

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

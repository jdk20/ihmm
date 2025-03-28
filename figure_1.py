import numpy as np
import matplotlib.pyplot as plt


def generate_states(s, n, n_oracle, oracle, K, T, alpha, beta, gamma, debug=False):
    current_state, n_existing, nc = None, None, None
    rng = np.random.default_rng()  # 1337+8

    for t in range(T):
        p = np.zeros(shape=1+K, dtype=np.float64)  # oracle and all other transitions

        if s.size == 0:  # special-case when no states exist
            p[0] = 1.0  # force oracle ignoring alpha/alpha+beta self-transition probability
        else:
            current_state = s[-1]  # current state (used for self-transitions)
            n_existing = n[current_state, :]  # transitions to existing states (including self)
            nc = np.sum(n_existing) + beta + alpha  # normalizing constant
            p[0] = beta/nc  # oracle

        for k in range(K):
            if k == current_state:
                p[k+1] = (n_existing[k] + alpha)/nc  # self-transition
            else:
                p[k+1] = n_existing[k]/nc  # transition

        if debug:
            assert np.allclose(np.sum(p), 1.0), f'p={np.sum(p)}'
        choice = rng.choice(a=1+K, size=1, p=p)

        # Oracle
        if choice == 0:
            p_oracle = np.zeros(shape=1+K, dtype=np.float64)  # new state and all other transitions
            nc_oracle = np.sum(n_oracle) + gamma  # normalizing constant

            p_oracle[0] = gamma/nc_oracle  # new state

            for k in range(K):
                p_oracle[k+1] = n_oracle[k]/nc_oracle

            if debug:
                assert np.allclose(np.sum(p_oracle), 1.0)
            choice_oracle = rng.choice(a=1+K, size=1, p=p_oracle)
            # Oracle New State
            if choice_oracle == 0:
                K += 1  # increase state counter
                next_state = K-1  # for zero-index

                # Expand n and n_oracle
                n_oracle = np.append(n_oracle, 0)
                n = np.pad(n, pad_width=[(0, 1), (0, 1)], mode='constant')

                # Add alpha to self-transition prob for new state
                n[next_state, next_state] += alpha
                if debug:
                    txt = 'Oracle New State'
            # Oracle Transition to Self/Existing State
            else:
                next_state = choice_oracle[0]-1
                if debug:
                    if current_state == next_state:
                        txt = 'Oracle Self-Transition to State'
                    else:
                        txt = 'Oracle Existing Transition to State'

            n_oracle[next_state] += 1  # increase state transition via oracle
            oracle[t] = True
            if debug:
                p_txt = p_oracle[choice_oracle][0]

        # Non-Oracle Transition to Self/Existing State
        else:
            next_state = choice[0]-1
            if debug:
                if current_state == next_state:
                    txt = 'Self-Transition to State'
                else:
                    txt = 'Existing Transition to State'
                p_txt = p[choice][0]

        # Append next state to state sequence and update counts
        s = np.append(s, next_state)
        if t > 0:
            n[current_state, next_state] += 1

        if debug:
            print(f't={t}: {txt} {next_state} from {current_state} with p={p_txt}')
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


# 1. Given observation sequence (emitted symbols) y_t, infer T and Q
y = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1] * 30  # ABCDEFGFEDCB
y = np.array(list(y))
T = len(y)
Q = len(np.unique(y))  # unique symbols

# Create empty hidden state sequence
K = 0
s = np.empty(shape=0, dtype=np.int64)
n = np.zeros(shape=(K, K), dtype=np.float64)
n_oracle = np.zeros(shape=K, dtype=np.int64)
oracle = np.zeros(shape=T, dtype=bool)

# 2. Set hyperparameters to initial values
alpha, beta, gamma = 0.1, 100, 10
beta_e, gamma_e = 1, 1

# 3. Generate initial hidden state sequence s_t, n, n_oracle and K
s, n, n_oracle, oracle, K = generate_states(s, n, n_oracle, oracle, K,
                                            T=T, alpha=alpha, beta=beta, gamma=gamma, debug=True)

# 4. Infer m and m_oracle (using beta_e)
m, m_oracle, oracle_e = infer_emissions(y, s, K, Q, beta_e)
assert np.sum(m_oracle) == np.sum(oracle_e)

# 5. Gibbs sweep over T
for t in range(1, T-1):
    # Remove s[t] from n, m, n_oracle, m_oracle
    if oracle[t]:
        n_oracle[s[t]] -= 1
        oracle[t] = False

    if oracle_e[t]:
        m_oracle[y[t]] -= 1
        oracle_e[t] = False

    n[s[t-1], s[t]] -= 1  # transition into
    n[s[t], s[t+1]] -= 1  # transition out of
    m[s[t], y[t]] -= 1

    # Shrink n and n_oracle if needed

    # Generate new s[t]
    new_state, _, _, is_oracle, _ = generate_states(np.array([s[t-1]]), n.copy(), n_oracle.copy(), oracle.copy(), K,
                                                    T=1, alpha=alpha, beta=beta, gamma=gamma, debug=False)
    new_state = new_state[-1]
    is_oracle = is_oracle[-1]

    s[t] = new_state
    n[s[t-1], s[t]] += 1  # transition into
    n[s[t], s[t+1]] += 1  # transition out of

    if is_oracle:
        oracle[t] = True

    if m[s[t], y[t]] == 0:  # oracle: existing symbol
        m_oracle[y[t]] += 1
        m[s[t], y[t]] += 1
        oracle_e[t] = True
    elif m[s[t], y[t]] > 0:  # existing emission or oracle: existing symbol
        p_oracle = beta_e/(np.sum(m[s[t], :]) + beta_e)
        p = np.random.rand()
        if p <= p_oracle:  # oracle: existing symbol
            m_oracle[y[t]] += 1
            oracle_e[t] = True
        m[s[t], y[t]] += 1

"""
# Figure 1
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
s, _, _, _ = generate_states(T=250, alpha=0.1, beta=1000, gamma=100, debug=True)
axs[0, 0].stairs(s, color='k')

s, _, _, _ = generate_states(T=250, alpha=0, beta=0.1, gamma=100, debug=True)
axs[0, 1].stairs(s, color='k')

s, _, _, _ = generate_states(T=250, alpha=8, beta=2, gamma=2, debug=True)
axs[1, 0].stairs(s, color='k')

s, _, _, _ = generate_states(T=250, alpha=1, beta=1, gamma=10000, debug=True)
axs[1, 1].stairs(s, color='k')

plt.show()
"""

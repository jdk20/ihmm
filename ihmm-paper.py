import sympy as sp
from sympy.stats import Gamma, density
import numpy as np
import matplotlib.pyplot as plt


def ihmm(alpha, beta, gamma, T):
    K, current_state, n_existing, nc = 0, None, None, None
    s = np.empty(shape=0, dtype=np.int64)
    n = np.empty(shape=(K, K), dtype=np.float64)
    n_oracle = np.empty(shape=K, dtype=np.int64)
    rng = np.random.default_rng()

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

        assert np.allclose(np.sum(p), 1.0), f'p={np.sum(p)}'
        choice = rng.choice(a=1+K, size=1, p=p)

        # Oracle
        if choice == 0:
            p_oracle = np.zeros(shape=1+K, dtype=np.float64)  # new state and all other transitions
            nc_oracle = np.sum(n_oracle) + gamma  # normalizing constant

            p_oracle[0] = gamma/nc_oracle  # new state

            for k in range(K):
                p_oracle[k+1] = n_oracle[k]/nc_oracle

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
                txt = 'Oracle New State'
            # Oracle Transition to Self/Existing State
            else:
                next_state = choice_oracle[0]-1
                if current_state == next_state:
                    txt = 'Oracle Self-Transition to State'
                else:
                    txt = 'Oracle Existing Transition to State'

            n_oracle[next_state] += 1  # increase state transition via oracle
            p_txt = p_oracle[choice_oracle][0]

        # Non-Oracle Transition to Self/Existing State
        else:
            next_state = choice[0]-1
            if current_state == next_state:
                txt = 'Self-Transition to State'
            else:
                txt = 'Existing Transition to State'
            p_txt = p[choice][0]

        # Append next state to state sequence and update counts
        s = np.append(s, next_state)
        if t > 0:
            n[current_state, next_state] += 1

        print(f't={t}: {txt} {next_state} from {current_state} with p={p_txt}')

        # Debug
        m = np.zeros_like(n)
        m[np.diag_indices_from(m)] += alpha
        np.add.at(m, (s[:-1], s[1:]), 1)
        assert np.mean(n == m) == 1

        assert n.shape[0] == K
        assert n_oracle.shape[0] == K

    return s, n, n_oracle


# Figure 1
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
"""
s, n, n_oracle = ihmm(alpha=0.1, beta=1000, gamma=100, T=250)
axs[0, 0].stairs(s, color='k')

s = ihmm(alpha=0, beta=0.1, gamma=100, T=250)
axs[0, 1].stairs(s, color='k')
"""

s, n, n_oracle = ihmm(alpha=8, beta=2, gamma=2, T=250)
axs[1, 0].stairs(s, color='k')

"""
s = ihmm(alpha=1, beta=1, gamma=10000, T=250)
axs[1, 1].stairs(s, color='k')

plt.show()
"""

import numpy as np


def hdp_hidden(current_state, n, n_oracle, alpha, beta, gamma, verbose=True):
    """
    n:
    n_oracle
    alpha:
    beta:
    gamma:
    """
    K = n.shape[0]
    p = np.zeros(shape=1+K, dtype=np.float64)  # oracle and all other transitions

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
            next_state = K - 1  # for zero-index

            # Expand n and n_oracle
            n_oracle = np.append(n_oracle, 0)
            n = np.pad(n, pad_width=[(0, 1), (0, 1)], mode='constant')

            # Add alpha to self-transition prob for new state
            n[next_state, next_state] += alpha
            txt = 'Oracle New State'
        # Oracle Transition to Self/Existing State
        else:
            next_state = choice_oracle[0] - 1
            if current_state == next_state:
                txt = 'Oracle Self-Transition to State'
            else:
                txt = 'Oracle Existing Transition to State'

        n_oracle[next_state] += 1  # increase state transition via oracle
        p_txt = p_oracle[choice_oracle][0]

    # Non-Oracle Transition to Self/Existing State
    else:
        next_state = choice[0] - 1
        if current_state == next_state:
            txt = 'Self-Transition to State'
        else:
            txt = 'Existing Transition to State'
        p_txt = p[choice][0]

    # Append next state to state sequence and update counts
    n[current_state, next_state] += 1

    if verbose:
        print(f'{txt} {next_state} from {current_state} with p={p_txt}')

    assert n.shape[0] == K
    assert n_oracle.shape[0] == K

    return next_state, n, n_oracle


def hdp_emission(current_emission, m, m_oracle, beta_e, gamma_e, verbose=True):
    """
    m:
    m_oracle
    beta_e:
    gamma_e:
    """
    K = m.shape[0]
    p = np.zeros(shape=1+K, dtype=np.float64)  # oracle and all other transitions

    m_existing = m[current_state, :]  # transitions to existing states (including self)
    nc = np.sum(m_existing) + beta_e  # normalizing constant
    p[0] = beta_e/nc  # oracle

    for k in range(K):
        p[k+1] = m_existing[k]/nc  # transition

    assert np.allclose(np.sum(p), 1.0), f'p={np.sum(p)}'
    choice = rng.choice(a=1+K, size=1, p=p)

    # Oracle
    if choice == 0:
        p_oracle = np.zeros(shape=1+K, dtype=np.float64)  # new state and all other transitions
        nc_oracle = np.sum(m_oracle) + gamma_e  # normalizing constant

        p_oracle[0] = gamma_e/nc_oracle  # new state

        for k in range(K):
            p_oracle[k+1] = m_oracle[k]/nc_oracle

        assert np.allclose(np.sum(p_oracle), 1.0)
        choice_oracle = rng.choice(a=1+K, size=1, p=p_oracle)
        # Oracle New Emission
        if choice_oracle == 0:
            K += 1  # increase state counter
            next_emission = K - 1  # for zero-index

            # Expand n and n_oracle
            m_oracle = np.append(m_oracle, 0)
            m = np.pad(n, pad_width=[(0, 1), (0, 1)], mode='constant')

            txt = 'Oracle New Emission'
        # Oracle Transition to Self/Existing State
        else:
            next_emission = choice_oracle[0] - 1
            txt = 'Oracle Existing Transition to State'

        m_oracle[next_emission] += 1  # increase state transition via oracle
        p_txt = p_oracle[choice_oracle][0]

    # Non-Oracle Transition to Self/Existing State
    else:
        next_emission = choice[0] - 1
        txt = 'Existing Transition to State'
        p_txt = p[choice][0]

    # Append next state to state sequence and update counts
    m[current_emission, next_emission] += 1

    if verbose:
        print(f'{txt} {next_emission} from {current_emission} with p={p_txt}')

    assert m.shape[0] == K
    assert m_oracle.shape[0] == K

    return next_emission, m, m_oracle


rng = np.random.default_rng(1337+8)

alpha, beta, gamma = 2, 2, 8
K = 10  # hidden states
T = 250  # timesteps
s = np.random.randint(K, size=T, dtype=np.int64)

n = np.zeros(shape=(K, K), dtype=np.float64)
n_oracle = np.zeros(shape=K, dtype=np.int64)

# Create n
np.add.at(n, (s[:-1], s[1:]), 1)

current_state = s[-1]

next_state, n, n_oracle = hdp_hidden(current_state, n, n_oracle, alpha, beta, gamma)

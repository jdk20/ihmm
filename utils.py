import numpy as np

from pickle import load
from os.path import join
from sympy import diff, exp, lambdify, symbols, IndexedBase


def hp_optimization_equations(mainpath: str = ''):
    # ------------------------------------------------------------------------------------------------------------------
    # Hyperparameter optimisation (Section 4.2)
    # ------------------------------------------------------------------------------------------------------------------
    # Symbols
    # Hyperparmeters
    # gamma is given the symbol gamma^s in order to not conflict with the sp.gamma() function
    alpha, beta, gamma, beta_e, gamma_e = symbols('alpha beta gamma^s beta^e gamma^e', positive=True)

    # Prior: Gamma distribution
    # a: shape parameter = 0.001
    # b: rate parameter = 0.001
    a_alpha, b_alpha = symbols('a_alpha b_alpha', positive=True)
    a_beta, b_beta = symbols('a_beta b_beta', positive=True)
    a_beta_e, b_beta_e = symbols('a_beta_e b_beta_e', positive=True)
    a_gamma, b_gamma = symbols('a_gamma b_gamma', positive=True)
    a_gamma_e, b_gamma_e = symbols('a_gamma_e b_gamma_e', positive=True)

    # Liklihood symbols
    # T_o: number of times oracle has been used for transitions
    # T_o_e: number of times oracle has been used for emissions
    T_o, T_o_e = symbols('T^o T^o^e', positive=True)

    # Liklihood vectors and matrices
    # i...K states
    # j...K states
    # q...Q emissions
    i, j, K, q, Q = symbols('i j K q Q', positive=True, integer=True)
    # n[i,j]: state transition matrix of shape [K,K]
    # m[i,q]: observation emission matrix of shape [K,Q]
    # kappa[i]: vector of shape [K], renamed from K(i) in paper in order to not conflict with integer K
    # number of possible transitions from state i, including itself
    # K_e[i]: vector of shape [K], number of possible emissions from state i
    n = IndexedBase('n', positive=True)  # from i..K, i..K
    m = IndexedBase('m', positive=True)  # from i..K, q..Q
    kappa = IndexedBase('Kappa', positive=True)  # from i..K
    K_e = IndexedBase('K^e', positive=True)  # from i..K

    # Newton-Rapheson helper
    z = symbols('z', real=True)

    equations = {}
    for eq in range(5):
        with open(join(mainpath, f'equation_{eq}.pkl'), 'rb') as file:
            expr = load(file)

        if eq == 0:
            arg = alpha
            args = (beta, n, K, a_alpha, b_alpha)
        elif eq == 1:
            arg = beta
            args = (alpha, n, kappa, K, a_beta, b_beta)
        elif eq == 2:
            arg = beta_e
            args = (m, K_e, K, Q, a_beta_e, b_beta_e)
        elif eq == 3:
            arg = gamma
            args = (T_o, K, a_gamma, b_gamma)
        elif eq == 4:
            arg = gamma_e
            args = (T_o_e, K, a_gamma_e, b_gamma_e)

        score = expr.subs(arg, exp(z))
        g = lambdify((z,) + args, score, modules=['numpy', 'scipy'])
        g_prime = lambdify((z,) + args, diff(score, z), modules=['numpy', 'scipy'])

        equations[eq] = {}
        equations[eq]['g'] = g
        equations[eq]['g_prime'] = g_prime

    return equations


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

        p_oracle[0] = gamma/nc_oracle  # new state

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
                    s: np.ndarray = np.empty(shape=0, dtype=np.int64),
                    n: np.ndarray = np.zeros(shape=(0, 0), dtype=np.float64),
                    n_oracle: np.ndarray = np.zeros(shape=0, dtype=np.int64),
                    debug: bool = False):

    if n.shape[0] != K or n.shape[1] != K:
        raise ValueError(f'Transition matrix n should be of shape ({K}, {K}) but is instead '
                         f'of shape ({n.shape[0]}, {n.shape[0]})')

    if n_oracle.size != K:
        raise ValueError(f'Oracle vector n_oracle should be of length ({K}) but is instead of length ({n_oracle.size})')

    if s.size > 0:
        unique_states = np.unique(s)
        if unique_states.size != K:
            raise ValueError(f'Number of hidden states ({unique_states.size}) does not match K = {K}')

        if np.min(unique_states) != 0:
            raise ValueError(f"Hidden state sequence should contain state 0: {unique_states}")

        if np.mean(np.diff(unique_states) == 1) != 1.0:
            raise ValueError(f"Hidden states are non-incremental: {unique_states}.")

    oracle = np.zeros(shape=T, dtype=bool)
    current_state, n_existing, nc = None, None, None

    for t in range(T):
        next_state, is_oracle, n, n_oracle, K = hdp_states(current_state, n, n_oracle, K,
                                                           alpha, beta, gamma, debug=debug)
        oracle[t] = is_oracle

        # Append next state to state sequence and update counts
        s = np.append(s, next_state)
        current_state = next_state

        n_debug = np.zeros_like(n)
        n_debug[np.diag_indices_from(n_debug)] += alpha
        np.add.at(n_debug, (s[:-1], s[1:]), 1)
        assert np.mean(n == n_debug) == 1
        assert n.shape[0] == K
        assert n_oracle.shape[0] == K

    return s, oracle, K, n, n_oracle


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

import numpy as np

from typing import Optional, Tuple
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

    arg, args = None, None
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


def hdp_states(alpha, beta, gamma, current_state, n, n_oracle):
    if alpha < 0:
        raise ValueError(f'Self-transition hyperparameter alpha >= 0.')

    if beta <= 0:
        raise ValueError(f'Transition density hyperparameter beta > 0.')

    if gamma <= 0:
        raise ValueError(f'Scale hyperparameter gamma > 0.')

    rng = np.random.default_rng()

    K = n.shape[1]
    is_oracle = False
    n_existing, nc, txt, p_txt = None, None, None, None
    p = np.zeros(shape=1+K, dtype=np.float64)  # oracle and all other transitions

    if n.size == 0:  # special-case when no states exist
        no_states = True
        p[0] = 1.0  # force oracle ignoring alpha/alpha+beta self-transition probability
    else:
        no_states = False
        n_existing = n[current_state, :]  # transitions to existing states (including self)
        nc = np.sum(n_existing) + beta + alpha  # normalizing constant
        p[0] = beta/nc  # oracle

    for k in range(K):
        if k == current_state:
            p[k+1] = (n_existing[k] + alpha)/nc  # self-transition
        else:
            p[k+1] = n_existing[k]/nc  # transition

    choice = rng.choice(a=1+K, size=1, p=p)

    # Oracle
    if choice == 0:
        p_oracle = np.zeros(shape=1 + K, dtype=np.float64)  # new state and all other transitions
        nc_oracle = np.sum(n_oracle) + gamma  # normalizing constant

        p_oracle[0] = gamma/nc_oracle  # new state

        for k in range(K):
            p_oracle[k+1] = n_oracle[k]/nc_oracle

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

        # Oracle Transition to Self/Existing State
        else:
            next_state = choice_oracle[0] - 1

        n_oracle[next_state] += 1  # increase state transition via oracle
        is_oracle = True

    # Non-Oracle Transition to Self/Existing State
    else:
        next_state = choice[0] - 1

    if not no_states:
        n[current_state, next_state] += 1

    return next_state, is_oracle, n, n_oracle


def generate_states(T: int,
                    alpha: float,
                    beta: float,
                    gamma: float,
                    s: Optional[np.ndarray] = None,
                    oracle: Optional[np.ndarray] = None,
                    n: Optional[np.ndarray] = None,
                    n_oracle: Optional[np.ndarray] = None,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates a hidden state sequence governed by the hyperparameters alpha, beta, and gamma using the infinite
    hidden markov model (iHMM). Optionally, this function can accept a preexisting hidden state sequence, transition
    matrix, and oracle vectors and extend them by T draws.

    See: Beal, M., Ghahramani, Z. and Rasmussen, C., 2001. The Infinite Hidden Markov Model.

    Args:
        T: Number of draws from the iHMM.
        alpha: Self-transition hyperparameter.
        beta: Density of the transition matrix n hyperparameter.
        gamma: Size (number of states K) of the transition matrix n hyperparameter.
        s: Hidden state seqeuence, where a value of 0 indicates state 0.
        oracle: Oracle indicator vector, where True at draw t means the oracle was used to generate the state at draw t.
        n: Hidden state transtion matrix containing counts of size (K, K), where n[i, j] is the count of transitions
        from state i to state j.
        n_oracle: Oracle vector of length (K) where n_oracle[i] is the count of the number of times the oracle was used
        to generate state i.

    Returns:
        s: in place modification
        oracle: in place modification
        n: in place modification
        n_oracle: in place modification

    Raises:
        ValueError: Compares the sizes of s and oracle, and the shapes of n, and n_oracle with K.
    """
    if T < 0:
        raise ValueError(f'Number of draws T={T} must be positive.')

    if s is None:
        if any(_ is not None for _ in (oracle, n, n_oracle)):
            raise ValueError(f'Additional arguments were given when s was not.')
        K = 0
        s = np.zeros(0, dtype=np.int64)
        oracle = np.zeros(0, dtype=bool)
        n = np.zeros((K, K), dtype=np.float64)
        n_oracle = np.zeros(K, dtype=np.int64)
        current_state = None
    else:
        K = np.max(s) + 1
        current_state = s[-1]
        n = count_n(s, alpha) if n is None else n
        if oracle is None:
            if n_oracle is not None:
                raise ValueError(f'n_oracle was given when oracle was not.')
            oracle = np.zeros_like(s, dtype=bool)
            n_oracle = np.zeros(K, dtype=np.int64)
        else:
            n_oracle = count_n_oracle(s, oracle) if n_oracle is None else n_oracle

    if n.shape[0] != K or n.shape[1] != K:
        raise ValueError(f'Transition matrix n should be of shape ({K}, {K}) but is instead '
                         f'of shape ({n.shape[0]}, {n.shape[0]})')

    if n_oracle.size != K:
        raise ValueError(f'Oracle vector n_oracle should be of length ({K}) but is instead of length ({n_oracle.size})')

    if np.sum(oracle) != np.sum(n_oracle):
        raise ValueError(f'Mismatch in oracle use between the oracle vector ({np.sum(n_oracle)}) '
                         f'and oracle indicator vector ({np.sum(oracle)}).')

    if s.size > 0:
        validate_s(s)

        if np.unique(s).size != K:  # edge case
            raise ValueError(f'Number of hidden states ({np.unique(s).size}) does not match K = {K}')

        if s.size != oracle.size:
            raise ValueError(f"Hidden state vector (t = {s.size}) and oracle indicator vector (t = {oracle.size}) "
                             f"should be same size.")

        if np.mean(n == count_n(s, alpha)) != 1.0:
            raise ValueError(f'Transition matrix n does not match transitions found in hidden state sequence s.')

    # Preallocate
    _s = np.zeros(shape=T, dtype=np.int64)
    _oracle = np.zeros(shape=T, dtype=bool)

    for t in range(T):
        next_state, is_oracle, n, n_oracle = hdp_states(alpha, beta, gamma, current_state, n, n_oracle)

        # Append next state to state sequence
        _s[t] = next_state
        _oracle[t] = is_oracle

        current_state = next_state

    s = np.concatenate((s, _s))
    oracle = np.concatenate((oracle, _oracle))

    if np.unique(s).size != n.shape[0]:  # edge case
        raise ValueError(f'Number of hidden states ({np.unique(s).size}) does not match K = {n.shape[0]}')

    if s.size != oracle.size:
        raise ValueError(f"Hidden state vector (t = {s.size}) and oracle indicator vector (t = {oracle.size}) "
                         f"should be same size.")

    if np.mean(n == count_n(s, alpha)) != 1.0:
        raise ValueError(f'Transition matrix n does not match transitions found in hidden state sequence s.')

    return s, oracle, n, n_oracle


def count_n(s: np.ndarray,
            alpha: float):
    validate_s(s)

    K = np.max(s) + 1
    n = np.zeros(shape=(K, K), dtype=np.float64)
    n[np.diag_indices_from(n)] += alpha
    np.add.at(n, (s[:-1], s[1:]), 1)

    return n


def count_n_oracle(s: np.ndarray,
                   oracle: np.ndarray):
    validate_s(s)
    validate_oracle(oracle)

    if s.shape[0] != oracle.shape[0]:
        raise ValueError('Hidden state sequence and oracle indicator sequence must be the same length.')

    K = np.max(s) + 1
    n_oracle = np.zeros(shape=K, dtype=np.int64)

    for k in range(K):
        n_oracle[k] = np.sum(np.logical_and(s == k, oracle))

    return n_oracle


def validate_oracle(oracle: np.ndarray):
    if not isinstance(oracle, np.ndarray):
        raise TypeError(f"Oracle indicator sequence must be a numpy array.")

    if oracle.ndim != 1:
        raise ValueError(f"Oracle indicator sequence must be a vector.")

    if oracle.dtype != bool:
        raise TypeError(f"Oracle indicator must be of type bool.")


def validate_s(s: np.ndarray):
    if not isinstance(s, np.ndarray):
        raise TypeError(f"Hidden state sequence must be a numpy array.")

    if s.ndim != 1:
        raise ValueError(f"Hidden state sequence must be a vector.")

    if s.dtype != np.int64:
        raise TypeError(f"Hidden state sequence must be of type np.int64.")

    if s.size > 0:
        if np.min(s) != 0:
            raise ValueError(f"Minimum hidden state element must be state 0, not state {np.min(s)}.")

        if not np.all(np.diff(np.unique(s)) == 1):
            raise ValueError(f"Hidden state elements must increase by 1 consecutively.")

        if np.max(s) + 1 != np.unique(s).size:  # might be unreachable
            raise ValueError(f"Number of hidden state elements does not match number in hidden state sequence.")


def infer_emissions(y, s, beta_e):
    # 4. Generate m using s, K, Q
    K = np.max(s) + 1
    Q = np.max(y) + 1
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

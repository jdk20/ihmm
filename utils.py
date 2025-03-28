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

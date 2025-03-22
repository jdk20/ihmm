import re
import pickle
from typing import BinaryIO

import numpy as np
import sympy as sp

from scipy.stats import gamma as gamma_dist
from scipy import optimize


def gamma_pdf(x_: sp.Expr, shape_: sp.Expr, rate_or_scale: sp.Expr, form: str = 'rate') -> sp.Expr:
    """
    Gamma distribution (shape-rate or shape-scale) form. All inputs are positive real numbers.

    Parameters
    x_ (sp.Expr): argument
    shape_ (sp.Expr): shape
    rate_or_scale (sp.Expr): rate (inverse-scale) or scale

    Returns
    sp.Expr: Gamma distribution probability density function

    Raises
    ValueError: If form parameter is not 'rate' or 'scale'
    """

    if not x_.is_positive:
        raise ValueError(f"Argument x needs to be positive.")

    if not shape_.is_positive:
        raise ValueError(f"Parameter shape needs to be positive.")

    if not rate_or_scale.is_positive:
        raise ValueError(f"Parameter rate or scale needs to be positive.")

    if form == 'rate':
        return (rate_or_scale**shape_/sp.gamma(shape_)) * x_**(shape_-1) * sp.exp(-rate_or_scale*x_)
    elif form == 'scale':
        return (1/(sp.gamma(shape_)*rate_or_scale**shape_)) * x_**(shape_-1) * sp.exp(-x_/rate_or_scale)
    else:
        raise ValueError(f"Parameter form={form} must be either 'rate' or 'scale'.")


# ----------------------------------------------------------------------------------------------------------------------
# "Vague" Gamma priors, p(beta) = Gamma(beta | a, lambda) = Gamma(beta | a, theta)
# ----------------------------------------------------------------------------------------------------------------------
# x: argument
# a: shape, vague = 0.001
# lambda: rate (also known as inverse-scale), vague = 0.001
# theta: scale, vague = 1000
x, a, lambda_, theta = sp.symbols('x a lambda theta', positive=True)

# Gamma distribution (shape-rate form, also known as shape-inverse scale form) Section 4.2, footnote 5
gamma_inverse_scale = gamma_pdf(x, a, lambda_, form='rate')

# Alternative representation, Gamma distribution (shape-scale form)
gamma_scale = gamma_pdf(x, a, theta, form='scale')

# ----------------------------------------------------------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------------------------------------------------------
assert gamma_scale.subs({theta: 1/lambda_}) - gamma_inverse_scale == 0  # convert rate to scale

# use arg, shape, rate, scale to not collide with symbols x, a, lambda, theta
n_iterations = 100
for arg, shape, rate in np.random.rand(n_iterations, 3):
    scale = 1/rate  # inverse-scale

    value_1 = float(gamma_inverse_scale.subs({x: arg, a: shape, lambda_: rate}).evalf())
    value_2 = float(gamma_scale.subs({x: arg, a: shape, theta: scale}).evalf())
    value_3 = gamma_dist.pdf(arg, a=shape, scale=scale)  # compare with scipy implementation of Gamma PDF

    assert np.isclose(value_1, value_2)
    assert np.isclose(value_2, value_3)

# Gamma PDF Comparison with scipy
shape, rate = 9, 2
f = sp.lambdify((x, a, lambda_), gamma_inverse_scale, modules='numpy')
args = np.linspace(1e-6, 20, 100)
assert np.isclose(np.linalg.norm(f(args, shape, rate) - gamma_dist.pdf(args, a=shape, scale=1/rate)), 0)

# Prior score functions
score_gamma_inverse_scale = sp.apart(sp.simplify(sp.diff(sp.log(gamma_inverse_scale), x)), x)

# ----------------------------------------------------------------------------------------------------------------------
# Hyperparameter optimisation (Section 4.2)
# ----------------------------------------------------------------------------------------------------------------------
# Symbols
# Hyperparmeters
# gamma is given the symbol gamma^s in order to not conflict with the sp.gamma() function
alpha, beta, gamma, beta_e, gamma_e = sp.symbols('alpha beta gamma^s beta^e gamma^e', positive=True)

# Prior: Gamma distribution
# a: shape parameter = 0.001
# b: rate parameter = 0.001
a_alpha, b_alpha = sp.symbols('a_alpha b_alpha', positive=True)
a_beta, b_beta = sp.symbols('a_beta b_beta', positive=True)
a_beta_e, b_beta_e = sp.symbols('a_beta_e b_beta_e', positive=True)
a_gamma, b_gamma = sp.symbols('a_gamma b_gamma', positive=True)
a_gamma_e, b_gamma_e = sp.symbols('a_gamma_e b_gamma_e', positive=True)

# Liklihood symbols
# T_o: number of times oracle has been used for transitions
# T_o_e: number of times oracle has been used for emissions
T_o, T_o_e = sp.symbols('T^o T^o^e', positive=True)

# Liklihood vectors and matrices
# i...K states
# j...K states
# q...Q emissions
i, j, K, q, Q = sp.symbols('i j K q Q', positive=True, integer=True)
# n[i,j]: state transition matrix of shape [K,K]
# m[i,q]: observation emission matrix of shape [K,Q]
# kappa[i]: vector of shape [K], renamed from K(i) in paper in order to not conflict with integer K
# number of possible transitions from state i, including itself
# K_e[i]: vector of shape [K], number of possible emissions from state i
n = sp.IndexedBase('n', positive=True)  # from i..K, i..K
m = sp.IndexedBase('m', positive=True)  # from i..K, q..Q
kappa = sp.IndexedBase('Kappa', positive=True)  # from i..K
K_e = sp.IndexedBase('K^e', positive=True)  # from i..K

# Newton-Rapheson helper
z = sp.Symbol('z', real=True)

# ----------------------------------------------------------------------------------------------------------------------
# Prior equations
# ----------------------------------------------------------------------------------------------------------------------
prior_1_alpha = gamma_pdf(alpha, a_alpha, b_alpha)
prior_1_beta = gamma_pdf(beta, a_beta, b_beta)
prior_2 = gamma_pdf(beta_e, a_beta_e, b_beta_e)
prior_3 = gamma_pdf(gamma, a_gamma, b_gamma)
prior_4 = gamma_pdf(gamma_e, a_gamma_e, b_gamma_e)

# ----------------------------------------------------------------------------------------------------------------------
# Liklihood equations
# ----------------------------------------------------------------------------------------------------------------------
# Dirichlet-multinomial distributions
# We use zero-indexing to match numpy conventions

# Section 4.2 Top
term_1 = (beta**(kappa[i]-1) * sp.gamma(alpha + beta))/sp.gamma(alpha)
term_2 = sp.gamma(n[i, i] + alpha)/sp.gamma(sp.Sum(n[i, j], (j, 0, K-1)) + alpha + beta)
liklihood_1 = sp.Product(term_1*term_2, (i, 0, K-1))

# Section 4.2 Middle
term_1 = (beta_e**K_e[i] * sp.gamma(beta_e))/sp.gamma(sp.Sum(m[i, q], (q, 0, Q-1)) + beta_e)
liklihood_2 = sp.Product(term_1, (i, 0, K-1))

# Section 4.2 Bottom left
liklihood_3 = (gamma**K * sp.gamma(gamma))/sp.gamma(T_o + gamma)

# Section 4.2 Bottom Right
# There appears to be a mistake where the paper references gamma**K_e instead of gamma_e**K (K_e does not exist)
liklihood_4 = (gamma_e**K * sp.gamma(gamma_e))/sp.gamma(T_o_e + gamma_e)

# ----------------------------------------------------------------------------------------------------------------------
# Posterior equations
# ----------------------------------------------------------------------------------------------------------------------
posterior_1 = prior_1_alpha * prior_1_beta * liklihood_1  # unnormalized posterior P(alpha, beta | s)
posterior_2 = prior_2 * liklihood_2  # unnormalized posterior P(beta_e | s, y)
posterior_3 = prior_3 * liklihood_3  # unnormalized posterior P(gamma | s)
posterior_4 = prior_4 * liklihood_4  # unnormalized posterior P(gamma_e | s, y)

# need explicit expand_log instead of simplify
log_posterior_1 = sp.expand_log(sp.log(posterior_1))
log_posterior_2 = sp.expand_log(sp.log(posterior_2))
log_posterior_3 = sp.expand_log(sp.log(posterior_3))
log_posterior_4 = sp.expand_log(sp.log(posterior_4))

# Score functions (technically the derivative of the log-posterior)
score_1_alpha = sp.diff(log_posterior_1, alpha)
score_1_beta = sp.diff(log_posterior_1, beta)
score_2 = sp.diff(log_posterior_2, beta_e)
score_3 = sp.diff(log_posterior_3, gamma)
score_4 = sp.diff(log_posterior_4, gamma_e)

# ----------------------------------------------------------------------------------------------------------------------
# Section 4.2 Equation Unit Testing
# ----------------------------------------------------------------------------------------------------------------------
term_1 = K_e[i]/beta_e + sp.digamma(beta_e) - sp.digamma(sp.Sum(m[i, q], (q, 0, Q-1)) + beta_e)
score_2_paper = sp.Sum(term_1, (i, 0, K-1)) - b_beta_e + (a_beta_e - 1)/beta_e

assert sp.simplify(score_2 - score_2_paper) == 0

pattern = r"\\operatorname\{polygamma\}\{\\left\(0,\s*([^)]+?)\s*\\right\)\}"
replacement = r"\\Psi{(\1)}"
for _ in [liklihood_1, liklihood_2, liklihood_3, liklihood_4, score_1_alpha, score_1_beta, score_2, score_3, score_4]:
    print('\\begin{equation}')
    print('\t', re.sub(pattern, replacement, sp.latex(_)))
    print('\\end{equation}')
    print('')

# ----------------------------------------------------------------------------------------------------------------------
# Newton Method
# ----------------------------------------------------------------------------------------------------------------------
# Newton's method (with parameter = exp(z) to keep the parameter > 0)
# sp.exp(x) -> sp.Min(sp.exp(x), 50)
# Want to avoid exp overflow and underflow

vague_gamma = 0.001
expr, arg, values = None, None, ()
for eq in range(5):
    if eq == 0:
        expr = score_1_alpha
        arg = alpha
        args = (beta, n, K, a_alpha, b_alpha)
        values = (1, np.ones((3, 3)), 3, vague_gamma, vague_gamma)
    elif eq == 1:
        expr = score_1_beta
        arg = beta
        args = (alpha, n, kappa, K, a_beta, b_beta)
        values = (1, np.eye(3), 2*np.ones(3), 3, vague_gamma, vague_gamma)
    elif eq == 2:
        expr = score_2
        arg = beta_e
        args = (m, K_e, K, Q, a_beta_e, b_beta_e)
        values = (np.ones((3, 5)), 2*np.ones(5), 3, 5, vague_gamma, vague_gamma)  # bete_e ** K_e can be unstable
    elif eq == 3:
        expr = score_3
        arg = gamma
        args = (T_o, K, a_gamma, b_gamma)
        values = (1, 3, vague_gamma, vague_gamma)
    elif eq == 4:
        expr = score_4
        arg = gamma_e
        args = (T_o_e, K, a_gamma_e, b_gamma_e)
        values = (1, 3, vague_gamma, vague_gamma)

    score = expr.subs(arg, sp.exp(z))
    f = sp.lambdify((arg,) + args, expr, modules=['numpy', 'scipy'])
    g = sp.lambdify((z,) + args, score, modules=['numpy', 'scipy'])
    g_prime = sp.lambdify((z,) + args, sp.diff(score, z), modules=['numpy', 'scipy'])

    print(f"Equation {eq}")
    print(f"g={g(*(0,) + values)}")
    print(f"g'={g_prime(*(0,) + values)}")

    r, output = optimize.newton(g, 0, fprime=g_prime,
                                args=values,
                                full_output=True,
                                maxiter=1000)

    print(f"z={r}")
    print(f"exp(z)={np.exp(r)}")
    print(f"g(z)={g(*(r,) + values)}")
    print(f"f(exp(z))={f(*(np.exp(r),) + values)}")
    print(f'Iterations: {output.iterations}')
    print('')

    assert np.isclose(g(*(r,) + values), 0)
    assert np.isclose(f(*(np.exp(r),) + values), 0)

    with open(f'equation_{eq}.pkl', 'wb') as p:
        pickle.dump(score, p)

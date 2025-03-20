import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.stats import gamma


def gamma_pdf(x_, shape_, rate_):
    # Gamma distribution (shape-rate/inverse-scale) form. All inputs are positive real numbers.
    # x_: argument
    # shape_: shape
    # rate_: rate or inverse-scale
    return (rate_**shape_/sp.gamma(shape_)) * x_**(shape_-1) * sp.exp(-rate_*x_)


# sp.init_printing()
# "Vague" Gamma priors, p(beta) = Gamma(beta | a, lambda) = Gamma(beta | a, theta)
# beta: argument
# a: shape
# lambda: rate or inverse-scale
# theta: scale
beta = sp.Symbol('beta', positive=True)
a, lambda_, theta = sp.symbols('a lambda theta', positive=True)

# Gamma distribution (shape-rate/inverse-scale form) Section 4.2, footnote 5
gamma_inverse_scale = (lambda_**a/sp.gamma(a)) * beta**(a-1) * sp.exp(-lambda_*beta)
gamma_inverse_scale_fn = gamma_pdf(beta, a, lambda_)

# Alternative representation, Gamma distribution (shape-scale form)
gamma_scale = (1/(sp.gamma(a)*theta**a)) * beta**(a-1) * sp.exp(-beta/theta)

# Unit tests
assert gamma_scale.subs({theta: 1/lambda_}) - gamma_inverse_scale == 0
assert gamma_scale.subs({theta: 1/lambda_}) - gamma_inverse_scale_fn == 0
n_iterations = 100
for x, shape, rate in np.random.rand(n_iterations, 3):
    scale = 1/rate  # inverse-scale

    value_1 = float(gamma_inverse_scale.subs({beta: x, a: shape, lambda_: rate}).evalf())
    value_2 = float(gamma_inverse_scale_fn.subs({beta: x, a: shape, lambda_: rate}).evalf())
    value_3 = float(gamma_scale.subs({beta: x, a: shape, theta: scale}).evalf())
    value_4 = gamma.pdf(x, a=shape, scale=scale)

    assert np.isclose(value_1, value_2)
    assert np.isclose(value_2, value_3)
    assert np.isclose(value_1, value_4)

# Gamma PDF
f = sp.lambdify((beta, a, lambda_), gamma_inverse_scale, modules='numpy')
x = np.linspace(1e-6, 20, 100)
shape, rate = 9, 2
# plt.plot(x, f(x, shape, rate), x, gamma.pdf(x, a=shape, scale=1/rate))
# plt.show()

# Prior score functions
score_gamma_inverse_scale = sp.apart(sp.simplify(sp.diff(sp.log(gamma_inverse_scale), beta)), beta)

# Section 4.2
# Prior symbols
a_alpha, b_alpha, a_beta, b_beta, a_beta_e, b_beta_e, a_gamma, b_gamma, a_gamma_e, b_gamma_e = (
    sp.symbols('a_alpha b_alpha a_beta b_beta a_beta_e b_beta_e a_gamma b_gamma a_gamma_e b_gamma_e', positive=True))

# Liklihood symbols
alpha, beta, beta_e, gamma, gamma_e = sp.symbols('alpha beta beta^e gamma gamma^e', positive=True)
T_o, T_o_e = sp.symbols('T^o T^o^e', positive=True)

# Liklihood vectors and matrices
i, j, q, K, Q = sp.symbols('i j q K Q', integer=True, positive=True)
n = sp.IndexedBase('n', positive=True)  # from i..K, i..K
m = sp.IndexedBase('m', positive=True)  # from i..K, q..Q
KK = sp.IndexedBase('Kappa', positive=True)  # from i..K
K_e = sp.IndexedBase('K^e', positive=True)  # from i..K

# Liklihood equations
liklihood_1 = sp.Product(((beta**(KK[i]-1) * sp.gamma(alpha + beta))/sp.gamma(alpha))*(sp.gamma(n[i, i] + alpha)/sp.gamma(sp.Sum(n[i, j], (j, 1, K)) + alpha + beta)), (i, 1, K))
liklihood_2 = sp.Product((beta_e**K_e[i] * sp.gamma(beta_e))/sp.gamma(sp.Sum(m[i, q], (q, 1, Q)) + beta_e), (i, 1, K))  # P(s,y | beta)
liklihood_3 = (gamma**K * sp.gamma(gamma))/sp.gamma(T_o + gamma)  # P(gamma | s)
liklihood_4 = (gamma_e**K * sp.gamma(gamma_e))/sp.gamma(T_o_e + gamma_e)  # P(gamma_e | s,y)

# Posterior equations (priors called directly)
posterior_1 = gamma_pdf(alpha, a_alpha, b_alpha) * gamma_pdf(beta, a_beta, b_beta) * liklihood_1
posterior_2 = gamma_pdf(beta_e, a_beta_e, b_beta_e) * liklihood_2  # unnormalized posterior P(beta | s,y)
posterior_3 = gamma_pdf(gamma, a_gamma, b_gamma) * liklihood_3
posterior_4 = gamma_pdf(gamma_e, a_gamma_e, b_gamma_e) * liklihood_4

# need explicit expand_log instead of simplify
log_posterior_1 = sp.expand_log(sp.log(posterior_1))
log_posterior_2 = sp.expand_log(sp.log(posterior_2))
log_posterior_3 = sp.expand_log(sp.log(posterior_3))
log_posterior_4 = sp.expand_log(sp.log(posterior_4))

score_posterior_1_alpha = sp.diff(log_posterior_1, alpha)
score_posterior_1_beta = sp.diff(log_posterior_1, beta)
score_posterior_2 = sp.diff(log_posterior_2, beta_e)
score_posterior_3 = sp.diff(log_posterior_3, gamma)
score_posterior_4 = sp.diff(log_posterior_4, gamma_e)

# use polygamma from scipy
f_2 = sp.lambdify((a_beta_e, b_beta_e, beta_e, m, Q, K_e, K), score_posterior_2, modules=['numpy', 'scipy'])
f_2(1, 1, 1, np.ones((4, 4)), 3, [1, 1, 1, 1], 3)

# Section 4.2 Equation Unit Testing
score_posterior_gt = sp.Sum(K_e[i]/beta_e + sp.digamma(beta_e) -
                            sp.digamma(sp.Sum(m[i, q], (q, 1, Q)) + beta_e),
                            (i, 1, K)) - b_beta_e + (a_beta_e - 1)/beta_e

assert sp.simplify(score_posterior_2 - score_posterior_gt) == 0

pattern = r"\\operatorname\{polygamma\}\{\\left\(0,\s*([^)]+?)\s*\\right\)\}"
replacement = r"\\Psi{(\1)}"
for _ in [liklihood_1, liklihood_2, liklihood_3, liklihood_4, score_posterior_1_alpha, score_posterior_1_beta,
          score_posterior_2, score_posterior_3, score_posterior_4]:
    print('\\begin{equation}')
    print('\t', re.sub(pattern, replacement, sp.latex(_)))
    print('\\end{equation}')
    print('')

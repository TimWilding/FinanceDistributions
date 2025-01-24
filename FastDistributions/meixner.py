"""
Meixner distribution has additional skewness and kurtosis parameters and has been used to model financial returns.
The Meixner distribution is a generalisation of the https://en.wikipedia.org/wiki/Hyperbolic_secant_distribution
Hyperbolic Secant Distribution. Miexner was a theoretical physicist.
There is a general discussion of the distribution at https://reference.wolfram.com/language/ref/MeixnerDistribution.html
The Meixner distribution has the interesting property that it is closed under addition
Meixner(a, b, m1, d1) + Meixner(a, b, m2, d2) = Meixner*a, b, m1+m2, d1+d2)
"""

import numpy as np
import pybobyqa
from scipy.stats import uniform, rv_continuous, FitError
from scipy.special import gamma, gammaln, psi, polygamma
from .stat_functions import _basestats

LOC_VAR = 3
SCALE_VAR = 0
SKEW_VAR = 1
SHAPE_VAR = 2


def meixner_loglikelihood(
    x, alpha: float = 1.0, beta: float = 0.0, delta: float = 1.0, μ: float = 0.0
):
    """
    Calculate the log of the probability density of the Meixner distribution.

    Parameters:
        x (float or np.ndarray): The point(s) at which to calculate the log-PDF.
        alpha (float): Scale parameter (alpha > 0).
        beta (float): Skewness parameter (|beta| < pi).
        delta (float): Shape parameter (delta > 0, controls kurtosis).
        μ (float): Location parameter.

    Returns:
        float or np.ndarray: The log-PDF value(s) at x.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not (-np.pi < beta < np.pi):
        raise ValueError("beta must satisfy -pi < beta < pi.")
    if delta <= 0:
        raise ValueError("delta must be positive.")

    # Core computations
    log_pre_factor = (
        (2 * delta) * np.log(2 * np.cos(beta / 2))
        - np.log(2 * alpha)
        - gammaln(2 * delta)
    )
    log_exp_factor = beta * (x - μ) / alpha
    arg = delta + 1j * (x - μ) / alpha
    # the mpmath gamma function can 
    # calculate gamma to a higher precision
    # it may be worth switching here.
    log_gamma_term = 2 * np.log(
#        np.abs(gamma(arg))
        np.maximum(np.abs(gamma(arg)), 1e-300)
    )  # Log of the squared modulus of Gamma function

    # Log-PDF value
    log_pdf = log_pre_factor + log_exp_factor + log_gamma_term - np.log(np.pi)
    return log_pdf


def _meixner_log_pdf_gradient(x, alpha, beta, delta, mu):
    # TODO - test this function properly and debug
    """

    Compute the gradient of the logarithm of the Meixner PDF with respect to parameters.

    Parameters:
        x (float or np.ndarray): Point(s) at which to calculate the gradient.
        alpha (float): Scale parameter (alpha > 0).
        beta (float): Skewness parameter (|beta| < pi).
        delta (float): Shape parameter (delta > 0).
        mu (float): Location parameter.

    Returns:
        tuple: Gradients with respect to (alpha, beta, delta, mu).
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not (-np.pi < beta < np.pi):
        raise ValueError("beta must satisfy -pi < beta < pi.")
    if delta <= 0:
        raise ValueError("delta must be positive.")

    z = delta + 1j * (x - mu) / alpha
    cos_beta_half = np.cos(beta / 2)
    tan_beta_half = np.tan(beta / 2)
    digamma_z = psi(z)

    # Gradient with respect to alpha
    grad_alpha = (
        -1 / alpha
        + beta * (x - mu) / alpha**2
        - 2 * np.imag(digamma_z) * (x - mu) / alpha**2
    )

    # Gradient with respect to beta
    grad_beta = -2 * delta * tan_beta_half + (x - mu) / alpha

    # Gradient with respect to delta
    grad_delta = 2 * np.log(2 * cos_beta_half) - psi(2 * delta) + 2 * np.real(digamma_z)

    # Gradient with respect to mu
    grad_mu = -beta / alpha - 2 * np.imag(digamma_z) / alpha

    return grad_alpha, grad_beta, grad_delta, grad_mu


# def _meixner_log_pdf_hessian(x, alpha, beta, delta, mu):
#     # TODO - Test this function properly and debug
#     """
#     Compute the Hessian of the logarithm of the Meixner PDF with respect to parameters.

#     Parameters:
#         x (float or np.ndarray): Point(s) at which to calculate the Hessian.
#         alpha (float): Scale parameter (alpha > 0).
#         beta (float): Skewness parameter (|beta| < pi).
#         delta (float): Shape parameter (delta > 0).
#         mu (float): Location parameter.

#     Returns:
#         np.ndarray: The 4x4 Hessian matrix.
#     """
#     if alpha <= 0:
#         raise ValueError("alpha must be positive.")
#     if not (-np.pi < beta < np.pi):
#         raise ValueError("beta must satisfy -pi < beta < pi.")
#     if delta <= 0:
#         raise ValueError("delta must be positive.")

#     z = delta + 1j * (x - mu) / alpha
#     cos_beta_half = np.cos(beta / 2)
#     tan_beta_half = np.tan(beta / 2)
#     sec_beta_half_sq = 1 / np.cos(beta / 2)**2
#     digamma_z = psi(z)
#     trigamma_z = polygamma(1, z)  # psi'(z)

#     # Precompute terms
#     x_mu_alpha = (x - mu) / alpha
#     imag_digamma = np.imag(digamma_z)
#     real_digamma = np.real(digamma_z)

#     # Second derivatives
#     hessian = np.zeros((4, 4))

#     # (1, 1): alpha-alpha
#     hessian[0, 0] = 1 / alpha**2 - 2 * beta * x_mu_alpha / alpha**3 \
#                     - 2 * imag_digamma * x_mu_alpha / alpha**3 \
#                     - 2 * np.imag(trigamma_z) * (x - mu)**2 / alpha**4

#     # (2, 2): beta-beta
#     hessian[1, 1] = -delta * sec_beta_half_sq

#     # (3, 3): delta-delta
#     hessian[2, 2] = -polygamma(1, 2 * delta) + 2 * np.real(trigamma_z)

#     # (4, 4): mu-mu
#     hessian[3, 3] = -2 * np.imag(trigamma_z) / alpha**2

#     # Mixed terms
#     # (1, 4): alpha-mu
#     hessian[0, 3] = beta / alpha**2 + 2 * imag_digamma / alpha**2 - 2 * np.imag(trigamma_z) * (x - mu) / alpha**3
#     hessian[3, 0] = hessian[0, 3]  # Symmetry

#     # (1, 2): alpha-beta
#     hessian[0, 1] = hessian[1, 0] = 0  # No cross-derivative dependence

#     # (2, 4): beta-mu
#     hessian[1, 3] = hessian[3, 1] = 1 / alpha

#     # (2, 3): beta-delta
#     hessian[1, 2] = hessian[2, 1] = 0  # No direct dependence

#     # (3, 4): delta-mu
#     hessian[2, 3] = hessian[3, 2] = 2 * np.imag(trigamma_z) / alpha

#     return hessian


class Meixner(rv_continuous):
    """
    Class to represent the Meixner distribution using the same form as the
    scipy.stats.distribution objects
    """

    def __init__(self, beta, delta, loc: float = 0.0, scale: float = 1.0):
        super().__init__(self)
        self.alpha = scale
        self.beta = beta
        self.delta = delta
        self.mu = loc

    def _pdf(self, x):
        pd = np.exp(self._logpdf(x))
        if isinstance(pd, (float, np.float64)):
        # Handle single float
            return 0.0 if np.isnan(pd) else pd
        elif isinstance(pd, np.ndarray):
        # Handle numpy array
            return np.nan_to_num(pd, nan=0.0)
        else:
            raise TypeError('Must be a float or numpy array')

    def _logpdf(self, x):
        return meixner_loglikelihood(x, self.alpha, self.beta, self.delta, self.mu)

    def _var(self):
        var = 0.5 * self.alpha**2 * self.delta / (np.cos(self.beta / 2) ** 2)
        return var

    def std(self):
        return np.sqrt(self.var())

    def _mean(self):
        mean = self.mu + self.alpha * self.delta * np.tan(self.beta / 2)
        return mean
    
    def _stats(self):
        return _basestats(self)


    #    def grad_pdf(self, x):
    #        return meixner_log_pdf_gradient(x, self.alpha, self.beta, self.delta, self.mu)#
    #
    #    def hessian_pdf(self, x):
    #        return meixner_log_pdf_hessian(x, self.alpha, self.beta, self.delta, self.mu)
    @staticmethod
    def fit(x, prob=None, display_progress=True):
        """
        Uses MLE to return Generalised Skew-T parameters this is
        set to be similar to fit in the scipy.stats package so
        it should raises a TypeError or ValueError if the input
        is invalid and a scipy.stats.FitError if fitting fails
        or the fit produced would be invalid
        Returns
        -------
        parameter_tuple : tuple of floats
            Estimates for any shape parameters (if applicable), followed by
            those for location and scale. For most random variables, shape
            statistics will be returned, but there are exceptions (e.g.
            ``norm``).
        """
        sol = _meixner_fit(x, prob, display_progress)
        return (sol.x[SCALE_VAR], sol.x[SKEW_VAR], sol.x[SHAPE_VAR], sol.x[LOC_VAR])

    @staticmethod
    def fitclass(x, prob=None, display_progress=True):
        """
        Uses MLE to return Generalised Skew-T class this is
        set to be similar to fit in the scipy.stats package so
        it should raises a TypeError or ValueError if the input
        is invalid and a scipy.stats.FitError if fitting fails
        or the fit produced would be invalid
        Returns
        -------
        parameter_tuple : tuple of floats
            Estimates for any shape parameters (if applicable), followed by
            those for location and scale. For most random variables, shape
            statistics will be returned, but there are exceptions (e.g.
            ``norm``).
        """
        sol = _meixner_fit(x, prob, display_progress)
        return Meixner(
            sol.x[SKEW_VAR], sol.x[SHAPE_VAR], sol.x[LOC_VAR], sol.x[SCALE_VAR]
        )


def _meixner_fit(returns_data, prob=None, display_progress=True):
    """
        Routine to fit a generalised skew-t distribution to a set of data sets with
        weights using the BOBYQA algorithm of Powell
    (https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf)
        This routine does not require derivatives
    """

    n = returns_data.shape[0]
    if prob is None:
        wt_prob = np.ones(n) / n
    else:
        wt_prob = prob
    mean_dist = np.average(returns_data, weights=wt_prob)
    sd_dist = np.sqrt(np.cov(returns_data, aweights=wt_prob))

    # Initialise NLopt engine to use BOBYQA algorithm

    # Set up starting parameters & upper & lower bounds
    # Note that this assumes that
    # SCALE_VAR = 0, SKEW_VAR=1, SHAPE_VAR=2, and LOC_VAR=3
    init_x = np.array([sd_dist, 1.0, 1.0, mean_dist])
    EPS = 1e-8
    lower = np.array(
        [
            1e-8 * sd_dist,
            -np.pi+EPS,
            EPS,
            -100 * sd_dist,
        ]
    )
    upper = np.array([10 * sd_dist, np.pi-EPS, 1000, 100 * sd_dist])

    # NLopt function must return a gradient even if algorithm is derivative-free
    # - this function will return an empty gradient
    ll_func = lambda x: -np.sum(
        wt_prob
        * meixner_loglikelihood(
            returns_data,
            x[SCALE_VAR],
            x[SKEW_VAR],
            x[SHAPE_VAR],
            x[LOC_VAR],
        )
    )
    if display_progress:
        print("Fitting Generalised Skew-T Distribution")
        print("=======================================")
        print("Initial Log Likelihood")
        print(ll_func(init_x))
    soln = pybobyqa.solve(ll_func, init_x, bounds=(lower, upper))
    #  https://nlopt.readthedocs.io/en/latest/NLopt_Reference/ says that ROUNDOFF_LIMITED results are usually useful so I'm going to assume they are
    #  print("")
    #  print("Solution xmin = %s" % str(soln.x))
    #  print("Objective value f(xmin) = %.10g" % (soln.fun))
    #  print("Needed %g objective evaluations" % soln.nfev)
    #  print("Exit flag = %g" % soln.status)
    #  print(soln.message)
    if display_progress:
        print(soln)
    if soln.flag == soln.EXIT_INPUT_ERROR:
        raise ValueError("input value problem in fitting routine")
    if (soln.flag < soln.EXIT_SUCCESS) & (soln.flag != soln.EXIT_LINALG_ERROR):
        print(soln)
        raise FitError("Fitting error in Generalised Skew-T fitting")
    return soln

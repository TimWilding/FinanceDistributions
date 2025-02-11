"""
Entropy Distributions are an extension of the Normal distribution that rely on the maximum 
entropy property. If all we know about a distribution is its first two moments, then 
a Normal distribution is the distribution with the maximum entropy. This can be extended
to the situation with higher moments and has solutions where the pdf is the exponential
of a polynomial function.
"""

import numpy as np
from typing import Any
from scipy.special import roots_legendre
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.stats import rv_continuous, FitError



def gauss_legendre_sample(n):
    """
    This routine returns a sample of x values and weights
    to use when calculating the integral of a function between
    -infinity and +infinity. The integral is approximated
    by the sum of the weights * the function value at each x

    For a function f, the integral is approx:
        np.sum(x_wts * f(x_vals))

    Args:
      n: an integer to determine the number of Legendre polynomials

    Returns:
      x_vals: numpy array containing points for function evaluation
      x_wts:  numpy array containing weights for each point
    """
    t, w = roots_legendre(n)
    # define x_transform and dxdt_inv
    # x_transform converts -1 to +1 to -infinity to +infinity
    def x_transform(t):
        return t / (1 - t**2)
    def dxdt_inv(t):
        return (1 + t**2) / (1 - t**2) ** 2
    x_vals = x_transform(t)
    x_wts = w * dxdt_inv(t)
    return x_vals, x_wts


def vandermonde_matrix(x, m=4, k=0):
    """
    Constructs a Vandermonde matrix. Each column of the
    vandermonde matrix contains the values of x raised to the
    power i

    Args:
       x: A 1-D NumPy array.
       m: Maximum power to use in the matrix (defaults to 4).
       k: Minimum power to use in the matrix (defaults to 0)

    Returns:
       A Vandermonde matrix.
    """
    van_mat = np.column_stack([x**i for i in range(k, m + 1)])
    return van_mat


def safe_log_likelihoods(ƛ, x_vals, max_val=100.0):
    """
    Constructs an array of log-likelihoods with a capped
    maximum value. The log-likelihood takes a polynomial
    form with the coefficients given in ƛ

    Args:
       ƛ: a  1-D numpy array of coefficients for the entropy pdf
       x_vals: A 1-D NumPy array.
       max_val: cap to use for log-likelihood.

    Returns:
       an array of capped log-likelihoods.

    """
    ll = -np.polyval(ƛ[::-1], x_vals)
    ll[ll > max_val] = max_val
    return ll


class EntropyDistribution(rv_continuous):
    """
    Class to represent Entropy Distributions using the same form as the
    scipy.stats.distribution objects. Entropy Distributions have a polynomial
    log-likelihood and characterise the returns distribution that maximises
    entropy given knowledge of a set number of moments. In the case of 2 moments,
    this is the classical Normal distribution, but this routine can handle
    distributions that have a higher number of moments such as Skew & Kurtosis
    and is designed to handle situations such as financial data which are known
    to have several higher moments.
    """

    def __init__(self, ƛ=None, n=200):
        self.ƛ = ƛ
        (x, w) = gauss_legendre_sample(n)
        self.x = x
        self.w = w
        self.prob_sum = 0.0
        self.prob_sum = np.log(np.sum(self.w * self._pdf(x)))
        self._calc_stats()
        super().__init__(self)

    def _calc_stats(self):
        prob_vals = self._pdf(self.x)
        self._mu = np.sum(self.x * prob_vals * self.w)
        self._sigma = np.sum((self.x - self._mu) ** 2 * prob_vals * self.w)
        mom_3 = np.sum((self.x - self._mu) ** 3 * prob_vals * self.w)
        mom_4 = np.sum((self.x - self._mu) ** 4 * prob_vals * self.w)
        self._skew = mom_3 / self._sigma ** (3 / 2)
        self._kurt = (mom_4 / (self._sigma**2)) - 3

    #       print(f'Prob Integral = {np.sum(prob_vals*self.w)}')
    #       print(f'Mean = {self._mu}')
    #       print(f'Sigma = {self._sigma}')
    #       print(f'Skew = {self._skew}')
    #       print(f'Kurt = {self._kurt}')

    def _logpdf(self, x, max_val=100.0):
        ƛ = np.hstack([self.prob_sum, self.ƛ])
        p = -np.polyval(ƛ[::-1], x)
        if isinstance(p, np.ndarray):
            p[p > max_val] = max_val
        else:
            if p > max_val:
                p = max_val
        return p

    def _pdf(self, x):
        return np.exp(self._logpdf(x))

    def _var(self):
        return self._sigma**2

    def std(self):
        return self._sigma

    def _mean(self):
        return self._mu

    def _stats(self: Any) -> None:
        return None

    @staticmethod
    def fit(
        x,
        prob=None,
        display_progress=True,
        x0=np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        max_iter=50000,
        x_toler=1e-10,
    ):
        """
        Uses MLE to return Entropy Distribution parameters this is
        set to be similar to fit in the scipy.stats package so
        it should raises a TypeError or ValueError if the input
        is invalid and a scipy.stats.FitError if fitting fails
        or the fit produced would be invalid
        Returns
        -------
        parameter_tuple : tuple of floats
            Estimates for any shape parameters (if applicable)
        """
        sol = _entropy_fit(x, prob, x0, display_progress, max_iter, x_toler)
        return sol.x

    @staticmethod
    def fitclass(
        x,
        prob=None,
        display_progress=True,
        x0=np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        max_iter=50000,
        x_toler=1e-10,
    ):
        """
        Uses MLE to return Entropy Distribution class this is
        set to be similar to fit in the scipy.stats package so
        it should raises a TypeError or ValueError if the input
        is invalid and a scipy.stats.FitError if fitting fails
        or the fit produced would be invalid
        Returns
        -------
        parameter_tuple : tuple of floats
            Estimates for any shape parameters.
        """
        sol = _entropy_fit(x, prob, x0, display_progress, max_iter, x_toler)
        return EntropyDistribution(sol.x[1::])


class EntropyDistFit:
    """
    Class used with cyipopt or any optimisation routine to fit an Entropy Distribution to a set of returns.
    This class calculates parameters for the objective function that is the log
    likelihood of the returns data.
    It returns the objective value, the gradient, the constraint function, the
    jacobian and a problem Hessian
    objective value = log-likelihood using a polynomial objective function
    gradient = gradient of log-likelihood (a weighted sum of the vandermonde matrix)
    constraint = integral of pdf = 1.0 (using Gauss-Legendre Quadrature)
    jacobian = gradient of constraint
    hessian = hessian of the objective function and problem constraints

    """

    def __init__(self, ret_data, wt_data=None, num_legendre=200, npoly=5):
        self.ret_data = ret_data
        if wt_data is None:
            self.wt_data = np.ones(ret_data.shape[0]) / ret_data.shape[0]
        else:
            self.wt_data = wt_data
        (x_vals, x_wts) = gauss_legendre_sample(num_legendre)
        self.x_vals = x_vals
        self.x_wts = x_wts
        self.npoly = npoly
        # Precompute constant terms in the optimisation problem
        self.grad = self.wt_data.T @ vandermonde_matrix(self.ret_data, npoly - 1)
        # Calculate the vandermonde matrix for the Gauss-Legendre quadrature
        self.v_xvals = vandermonde_matrix(self.x_vals, npoly - 1)

    def objective(self, ƛ):
        #
        # The callback for calculating the objective
        #
        return np.sum(self.wt_data * np.polyval(ƛ[::-1], self.ret_data))

    def gradient(self, ƛ):
        #
        # The callback for calculating the gradient
        #
        # m = ƛ.shape[0]-1
        # g = vandermonde_matrix(self.ret_data, m)
        # return self.wt_data.T @ g
        return self.grad

    def constraints(self, ƛ):
        #
        # The callback for calculating the constraints
        #
        ll = safe_log_likelihoods(ƛ, self.x_vals)
        pdf_vals = np.exp(ll)
        return np.array(np.sum(self.x_wts * pdf_vals))  # experiment without np.array

    def jacobian(self, ƛ):
        #
        # The callback for calculating the Jacobian
        #
        m = ƛ.shape[0] - 1
        #  V = vandermonde_matrix(self.x_vals, m)
        ll = safe_log_likelihoods(ƛ, self.x_vals)
        pdf_vals = np.exp(ll)
        g_vals = self.v_xvals.T * pdf_vals
        return -g_vals @ self.x_wts

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #
        return np.nonzero(np.tril(np.ones((self.npoly, self.npoly))))

    def objhession(self, ƛ):
        """
        Hessian of the unconstrained negative log-likelihood function
        given that ƛ[0] is the normalisation coefficient for the probability distribution
        and ƛ[1:] are the parameters of the entropy distribution
        """
        m = ƛ.shape[0] - 1
        H = np.zeros((m + 1, m + 1))
        return H

    def _constrainthessian(self, ƛ, v):
        m = ƛ.shape[0] - 1
        ll = safe_log_likelihoods(ƛ, self.x_vals)
        pdf_vals = np.exp(ll)
        wt_s = np.sqrt(pdf_vals * self.x_wts)
        wt_g = self.v_xvals.T * wt_s
        hess = wt_g @ wt_g.T
        return hess * v

    def hessian(self, ƛ, lagrange, obj_factor):
        """
        Hessian of the Lagrangian function
        """
        #
        # The callback for calculating the Hessian
        #
        m = ƛ.shape[0] - 1
        obj_hess = np.zeros((m + 1, m + 1))

        ll = safe_log_likelihoods(ƛ, self.x_vals)
        pdf_vals = np.exp(ll)
        wt_s = np.sqrt(pdf_vals * self.x_wts)
        #       g = vandermonde_matrix(self.x_vals, m)
        wt_g = self.v_xvals.T * wt_s
        con_hess = wt_g @ wt_g.T

        hess = obj_factor * obj_hess + lagrange[0] * con_hess
        row, col = self.hessianstructure()

        return hess[row, col]


def _entropy_fit(
    returns_data,
    prob=None,
    x0=np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    display_progress=True,
    max_iter=50000,
    x_toler=1e-10,
):

    ent_fit = EntropyDistFit(returns_data, npoly=x0.shape[0])

    init_x = x0.copy()
    init_x[0] = np.log(ent_fit.constraints(init_x))

    n = init_x.shape[0]
    if n > 4:
        init_x[n - 1] = 1e-2

    lb = np.array([-100.0] * n)
    lb[n - 1] = 1e-8
    ub = np.array([100.0] * n)
    bds = Bounds(lb, ub, keep_feasible=True)

    #  init_x = np.array([1.0, 0.0, 1.])#, 0.0, 0.0])
    if display_progress:
        print("Adjusted Probability Sum")
        print(ent_fit.constraints(init_x))
    npts = returns_data.shape[0]
    if prob is None:
        wt_prob = np.ones(npts) / npts
    else:
        wt_prob = prob
    # mean_dist = np.average(returns_data, weights=wt_prob)
    # var_dist = np.cov(returns_data, aweights=wt_prob)
    nlc = NonlinearConstraint(
        ent_fit.constraints, 1, 1, jac=ent_fit.jacobian, hess=ent_fit._constrainthessian
    )
    if display_progress:
        print("Initial Log Likelihood")
        print(ent_fit.objective(init_x))
    res = minimize(
        ent_fit.objective,
        init_x,
        method="trust-constr",
        jac=ent_fit.gradient,
        hess=ent_fit.objhession,
        constraints=nlc,
        bounds=bds,
        options={
            "gtol": 1e-8,
            "xtol": x_toler,
            "disp": display_progress,
            "maxiter": max_iter,
        },
    )

    if display_progress:
        print("Final Log Likelihood")
        print(ent_fit.objective(res.x))
    if not res.success:
        print("Fitting of Entropy Distribution failed")
        print(res)
        raise FitError("Fitting error in Entropy Distribution fitting")
    return res

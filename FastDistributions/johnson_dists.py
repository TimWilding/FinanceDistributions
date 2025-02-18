"""
Implementation of the Johnson SU distribution and fitting routine
The Johnson distribution is a four-parameter family of continuous probability distributions
defined by four parameters. The distribution is a transformed version of the Normal
distribution using an arcsinh transformation. The Johnson SU distribution is an
unbounded version of the Johnson SB distribution.
"""

import numpy as np
import pybobyqa
from scipy.stats import rv_continuous, FitError
from scipy.special import erf, erfinv
from typing import Any


def log_pdf_johnson_sb(x, gamma, delta, xi, lambd):
    """
    Computes the log of the PDF for the Johnson SB distribution.

    Parameters:
        x (float or np.ndarray): Value(s) where the PDF is evaluated.
        gamma (float): Shape parameter.
        delta (float): Shape parameter.
        xi (float): Location parameter.
        lambd (float): Scale parameter.

    Returns:
        float or np.ndarray: Log of the PDF at x.
    """
    if np.any(x <= xi) or np.any(x >= xi + lambd):
        raise ValueError("x must be in the range (xi, xi + lambd)")

    z = (x - xi) / (lambd - x + xi)
    transformed = gamma + delta * np.log(z)

    log_pdf = (
        -0.5 * transformed**2
        - np.log(lambd)
        - np.log(x - xi)
        - np.log(lambd - x + xi)
        - 0.5 * np.log(2 * np.pi)
    )
    return log_pdf


def log_pdf_johnson_su(x, gamma, delta, xi, lambd):
    """
    Computes the log of the PDF for the Johnson SU distribution.

    Parameters:
        x (float or np.ndarray): Value(s) where the PDF is evaluated.
        gamma (float): Shape parameter.
        delta (float): Shape parameter.
        xi (float): Location parameter.
        lambd (float): Scale parameter.

    Returns:
        float or np.ndarray: Log of the PDF at x.
    """
    if lambd <= 0:
        raise ValueError("lambd must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")

    z = (x - xi) / lambd
    sinh_inv = np.arcsinh(z)

    transformed = gamma + delta * sinh_inv

    log_pdf = (
        np.log(delta)
        - 0.5 * transformed**2
        - np.log(lambd)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(1 + z**2)
    )
    return log_pdf


class JohnsonSU(rv_continuous):
    """
    Class to represent the JohnsonSU distribution using the same form as the
    scipy.stats.distribution objects. The distribution is characterised by:
            gamma (float): Shape parameter.
            delta (float): Shape parameter.
            xi (float): Location parameter.
            lambd (float): Scale parameter.
    """

    def __init__(self, gamma, delta, loc: float = 0.0, scale: float = 1.0):
        #        gamma (float): Shape parameter.
        #        delta (float): Shape parameter.
        #        xi (float): Location parameter.
        #        lambd (float): Scale parameter.
        super().__init__(self)
        self.gamma = gamma
        self.delta = delta
        self.lambd = scale
        self.xi = loc

    def _pdf(self, x):
        pd = np.exp(self._logpdf(x))
        if isinstance(pd, (float, np.float64)):
            # Handle single float
            return 0.0 if np.isnan(pd) else pd
        elif isinstance(pd, np.ndarray):
            # Handle numpy array
            return np.nan_to_num(pd, nan=0.0)
        else:
            raise TypeError("Must be a float or numpy array")

    def _logpdf(self, x):
        return log_pdf_johnson_su(x, self.gamma, self.delta, self.xi, self.lambd)

    def _omegastats(self):
        m = np.exp(1.0 / self.delta**2)
        omega = self.gamma / self.delta
        return m, omega

    def _var(self):
        m, omega = self._omegastats()
        var = (self.lambd**2 / 2) * (m - 1) * (m * np.cosh(2 * omega) + 1)
        return var

    def std(self):
        return np.sqrt(self.var())

    def _mean(self):
        m, omega = self._omegastats()
        mean = self.xi - self.lambd * np.sqrt(m) * np.sinh(omega)
        return mean

    def _skew(self):
        m, omega = self._omegastats()
        skew = (
            -self.lambd**3
            * np.sqrt(m)
            * (m - 1) ** 2
            * (m * (m + 2) * np.sinh(3 * omega) + 3 * np.sinh(omega))
            / (4 * self._var() ** 1.5)
        )
        return skew

    def _kurtosis(self):
        m, omega = self._omegastats()
        kurt = (
            self.lambd**4
            * (m - 1) ** 2
            * (
                m**2 * (m**4 + 2 * m**3 + 3 * m**2 - 3) * np.cosh(4 * omega)
                + 4 * m**2 * (m + 2) * np.cosh(2 * omega)
                + 3 * (2 * m + 1)
            )
            / (8 * self._var()**2)
        )
        return kurt

    def _cdf(self: Any, x: Any) -> Any:
        z = (x - self.xi) / self.lambd
        sinh_inv = np.arcsinh(z)
        transformed = self.gamma + self.delta * sinh_inv
        cdf = 0.5 * (1 + erf(transformed / np.sqrt(2)))
        return cdf

    def _ppf(self, q, *args, **kwargs):
        z = np.sqrt(2) * erfinv(2 * q - 1)
        m = (z - self.gamma) / self.delta
        return self.xi + self.lambd * np.sinh(m)

    def _stats(self):
        mean = self._mean()
        variance = self._var()
    # Johnson SU skew and kurtosis are complex; you can use numerical methods
        skew = self._skew()
        kurt = self._kurtosis()
        return mean, variance, skew, kurt

    @staticmethod
    def fit(returns_data, prob=None, display_progress=True):
        # TODO: change the name of this function since fit is a method of the class
        """
        Fit a Johnson SU distribution to a set of data sets with weights
        Uses the BOBYQA algorithm of Powell.
        Parameters:
            returns_data (np.ndarray): Data to fit the distribution to
            prob (np.ndarray): Weights for the data
            display_progress (bool): Whether to display progress
        Returns:
            JohnsonSU: Tuple contining parameters for Johnson SU distribution
        """
        return _johnson_su_fit(returns_data, prob, display_progress)

    @staticmethod
    def fitclass(returns_data, prob=None, display_progress=True):
        """
        Fit a Johnson SU distribution to a set of data sets with weights
        Uses the BOBYQA algorithm of Powell.
        Parameters:
            returns_data (np.ndarray): Data to fit the distribution to
            prob (np.ndarray): Weights for the data
            display_progress (bool): Whether to display progress
        Returns:
            JohnsonSU: Instance of JohnsonSU class fitted to the data
        """
        sol = _johnson_su_fit(returns_data, prob, display_progress)
        return JohnsonSU(sol.x[0], sol.x[1], sol.x[2], sol.x[3])


def _johnson_su_fit(returns_data, prob=None, display_progress=True):
    """
        Routine to fit a Johnson SU distribution to a set of data sets with
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
    init_x = np.array(
        [
            0.0,
            1.0,
            mean_dist,
            sd_dist,
        ]
    )
    eps = 1e-8
    lower = np.array(
        [
            -10,
            eps,
            -100 * sd_dist,
            1e-8 * sd_dist,
        ]
    )
    upper = np.array(
        [
            10,
            10,
            100 * sd_dist,
            10 * sd_dist,
        ]
    )

    def ll_func(x):
        return -np.sum(
            wt_prob
            * log_pdf_johnson_su(
                returns_data,
                x[0],
                x[1],
                x[2],
                x[3],
            )
        )

    if display_progress:
        print("Fitting Johnson SU Distribution")
        print("=======================================")
        print("Initial Log Likelihood")
        print(ll_func(init_x))
    soln = pybobyqa.solve(ll_func, init_x, bounds=(lower, upper))
    #  https://nlopt.readthedocs.io/en/latest/NLopt_Reference/ says that ROUNDOFF_LIMITED results
    #  are usually useful so I'm going to assume they are
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
        raise FitError("Fitting error in Johnson SU fitting")
    return soln

"""
Implementation of generalised Skew T Log Likelihood
(see https://en.wikipedia.org/wiki/Skewed_generalized_t_distribution)
"""

import math
import numpy as np
import pybobyqa
from scipy.special import gamma, loggamma
from scipy.stats import uniform, rv_continuous, FitError, beta
from .stat_functions import _basestats

LOC_VAR = 0
SCALE_VAR = 1
SKEW_VAR = 2  # -1 < skew_var< -1
K_VAR = 3  # K>0
N_VAR = 4  # n>2


def var_adjustment_sged(ƛ, pinv):
    a = (1 + 3 * (ƛ**2)) * math.pi * gamma(3 * pinv)
    b = 4 * (0.5 * pinv) * ƛ * gamma(0.5 + pinv) * np.sqrt(gamma(pinv))
    return np.sqrt(np.maximum((a - b**2) / (math.pi * gamma(pinv)), 1e-20))


def var_adjustment_sgt(ƛ, pinv, q):

    if (pinv < 5) & (q < 5):
        a_adj = gamma(3 * pinv) * gamma(q - 2 * pinv) / gamma(pinv) / gamma(q)
        b_adj = gamma(2 * pinv) * gamma(q - pinv) / gamma(pinv) / gamma(q)
    else:
        a_adj = (
            loggamma(3 * pinv) + loggamma(q - 2 * pinv) - loggamma(pinv) - loggamma(q)
        )
        a_adj = np.exp(a_adj)
        b_adj = loggamma(2 * pinv) + loggamma(q - pinv) - loggamma(pinv) - loggamma(q)
        b_adj = np.exp(b_adj)

    a = (
        1 + 3 * (ƛ**2)
    ) * a_adj  # * gamma(3*pinv) * gamma(q - 2*pinv) / gamma(pinv) / gamma(q)
    b = 2 * ƛ * b_adj  # gamma(2*pinv)*gamma(q-pinv) / gamma(pinv) / gamma(q)
    return np.sqrt(np.maximum(a - b**2, 1e-20))


def s_lambda(ƛ, pinv, q):
    """
    Calcualte S(lambda) from Theodassiou (eqn 4)
    """

    if (pinv < 5) & (q < 5):
        a_adj = (
            gamma(2 * pinv) ** 2
            * gamma(q - pinv) ** 2
            / gamma(pinv)
            / gamma(q)
            / gamma(3 * pinv)
            / gamma(q - 2 * pinv)
        )
    else:
        a_adj = (
            2 * loggamma(2 * pinv)
            + 2 * loggamma(q - pinv)
            - loggamma(pinv)
            - loggamma(q)
            - loggamma(3 * pinv)
            - loggamma(q - 2 * pinv)
        )
        a_adj = np.exp(a_adj)

    s_lambda = (
        1 + 3 * (ƛ**2) - 4 * (ƛ**2) * a_adj
    )  
    return np.sqrt(np.maximum(s_lambda, 1e-20))


def generalised_skewt_loglikelihood(x, μ, σ, ƛ, k, n):
    """
    This function matches https://en.wikipedia.org/wiki/Skewed_generalized_t_distribution
    but doesn't seem to match Theodossiou (1998). Written to match Theodossiou (1998). There
    seems to be an error in the Wiki page!
    Valid Ranges
          -1 < ƛ < 1
      n > 2
      k > 0
      σ > 0
    Typical Distributions
      k = 2, Hansen Skew T
      n -> ∞, Skewed Generalised Error Distribution
      k = 1,  n -> ∞, Skewed Laplace Distribution
      k = 2,  n -> ∞, Skewed Normal
    """
    p = k
    q = n / k
    pinv = 1.0 / p
    ε = x - μ

    if p > 1e4:
        # This doesn't appear to be a good approx
        return np.log(
            uniform.pdf(x, μ - np.sqrt(3) * σ * (1 - ƛ), 2 * np.sqrt(3) * σ * (1 + ƛ))
        )

    if q > 1e4:
        v = 1.0 / var_adjustment_sged(ƛ, pinv)
        d = np.abs(ε) ** p / (v * σ * (1 + ƛ * np.sign(ε))) ** p

        # Use the Skew Generalised Error Distribution
        ll = np.log(p) - np.log(2) - np.log(σ) - loggamma(pinv) - d
        return ll

    # t = var_adjustment_sgt(ƛ, pinv, q)
    # if isnan(t)
    #     println("t is nan")
    # end
    v = q ** (-pinv) / var_adjustment_sgt(ƛ, pinv, q)

    d = (np.abs(ε) / (v * σ * (1 + ƛ * np.sign(ε)))) ** p

    # see https://math.stackexchange.com/questions/3922187/what-is-the-log-of-the-beta-function-how-can-it-be-simplified
    ll = (
        np.log(p)
        + loggamma(pinv + q)
        - np.log(2)
        - np.log(v)
        - np.log(σ)
        - (pinv * np.log(q))
    )
    ll = ll - loggamma(pinv) - loggamma(q) - (pinv + q) * np.log(1 + (d / q))
    return ll


class GeneralisedSkewT(rv_continuous):
    """
    Class to represent Generalised Skew T distribution using the same form as the
    scipy.stats.distribution objects.
    """

    def __init__(
        self,
        ƛ: float = 0,
        k: float = 2,
        n: float = 10000,
        loc: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__(self)
        self.μ = loc
        self.σ = scale
        self.ƛ = ƛ
        self.k = k
        self.n = n

    def _logpdf(self, x):
        return generalised_skewt_loglikelihood(
            x, self.μ, self.σ, self.ƛ, self.k, self.n
        )

    def _pdf(self, x):
        return np.exp(self._logpdf(x))

    # TODO: Implemeent skewness and kurtosis calculations
    # see  Theodossiou (1998)

    def _var(self):
        if self.n <= 2:
            return np.Inf
        return self.σ**2

    def std(self):
        return np.sqrt(self.var())

    def _mean(self):
        # Matches equation 7 from 
        # theodassiou
        p = self.k
        q = self.n / self.k
        p_inv = 1.0 / p
        v = 2.0 / s_lambda(self.ƛ, p_inv, q)
        if (p_inv < 5) & (q < 5):
            mean_adj = (
                loggamma(2 * p_inv)
                + loggamma(q - p_inv)
                -0.5 * loggamma(p_inv)
                -0.5 * loggamma(q)
                -0.5 * loggamma(3*p_inv)
                -0.5 * loggamma(q-2*p_inv) 
            )
            mean_adj = np.exp(mean_adj)
        else:
            mean_adj = (
            gamma(2 * p_inv)
            * gamma(q - p_inv)
            / np.sqrt(gamma(p_inv))
            / np.sqrt(gamma(q))
            / np.sqrt(gamma(3*p_inv))
            / np.sqrt(gamma(q-2*p_inv))
            )

        lam_adj = (
            self.ƛ
            * v
            * mean_adj
        )
        return self.μ + lam_adj * self.σ

    def _skewkurt(self):
        #        raise NotImplemented
        # Currently trying to match equation 9 from
        # theodassiou - typo in equation
        p = self.k
        q = self.n / self.k
        p_inv = 1.0 / p
        s_l = 1.0 / s_lambda(self.ƛ, p_inv, q)


        mean_adj = (
            gamma(2 * p_inv)
            * gamma(q - p_inv)
            / np.sqrt(gamma(p_inv))
            / np.sqrt(gamma(q))
            / np.sqrt(gamma(3*p_inv))
            / np.sqrt(gamma(q-2*p_inv))
            )

        m1 = (
            self.ƛ
            * 2 * s_l
            * mean_adj
        )

        m2 = (1 + 3*self.ƛ**2)*s_l**2

        m3 = (
            4
            * self.ƛ
            * (1 + self.ƛ**2) # typo in equ 9 - see eqn 5 for real value
            * s_l**3
            * gamma(4 * p_inv)
            * gamma(q - 3 * p_inv)
            * np.sqrt(gamma(p_inv))
            * np.sqrt(gamma(q))
            / np.sqrt(gamma(3 * p_inv) ** 3) 
            / np.sqrt(gamma(q - 2 * p_inv) ** 3)
        )

        m_adj = (
            gamma(5 * p_inv)
            * gamma(q - 4 * p_inv)
            * gamma(p_inv)
            * gamma(q)
            / gamma(3 * p_inv) ** 2
            / gamma(q - 2 * p_inv) ** 2
        )
        m4 = (1 + 10 * self.ƛ**2 + 5 * self.ƛ**4) * s_l**4 * m_adj

        mu = self._mean()
        m3 = m3  #- 3 * (mu / self.σ) - (mu / self.σ) ** 3
        skew = m3 - 3*m1*m2 + 2*m1**3
        kurt = (
                m4 
                - 4*m1*m3
                + 6*m1**2*m2
                - 3*m1**4
                )
    
        return skew, kurt



    def _stats(self):
        mean = self._mean()
        variance = self._var()
        skew, kurt = self._skewkurt()
        return mean, variance, skew, kurt

    def _ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles).
        """
        pr = q.copy()
        p = self.k
        qloc = self.n / self.k
        p_inv = 1.0 / p
        ƛ = self.ƛ * np.ones(pr.shape)
        σ = self.σ / (qloc ** (p_inv) * var_adjustment_sgt(ƛ, p_inv, qloc))
        out_sign = np.ones(pr.shape)

        flip = pr > (1 - ƛ) / 2

        pr[flip] = 1.0 - pr[flip]
        ƛ[flip] = -ƛ[flip]
        out_sign[flip] = -1
        out = (
            σ
            * (ƛ - 1)
            * (1 / (qloc * beta(p_inv, qloc).ppf(1 - 2 * pr / (1 - ƛ))) - 1 / qloc)
            ** (-p_inv)
        )
        out = out_sign * out + self.μ
        return out

    def _cdf(self, x):
        """
        Cumulative distribution function.
        """
        x_e = x - self.μ
        p = self.k
        q = self.n / self.k
        p_inv = 1.0 / p

        flip = x_e > 0

        if isinstance(x, (float, np.float64)):
            ƛ = self.ƛ
            # Handle single float
            if flip:
                ƛ = -ƛ
                x_e = -x_e
        elif isinstance(x, np.ndarray):
            # Handle numpy array
            ƛ = self.ƛ * np.ones(x.shape)
            ƛ[flip] = -ƛ[flip]
            x_e[flip] = -x_e[flip]
        else:
            raise TypeError("Must be a float or numpy array")

        σ = self.σ / (q ** (p_inv) * var_adjustment_sgt(ƛ, p_inv, q))

        out = (1 - ƛ) / 2 + ((ƛ - 1) / 2) * beta(p_inv, q).cdf(
            1 / (1 + q * (σ * (1 - ƛ) / (-x_e)) ** p)
        )

        if isinstance(x, (float, np.float64)):
            # Handle single float
            if flip:
                out = 1 - out
        else:
            out[flip] = 1 - out[flip]
        return out

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
        sol = _gen_skewt_fit(x, prob, display_progress)
        return (
            sol.x[SKEW_VAR],
            np.tan(sol.x[K_VAR]),
            np.tan(sol.x[N_VAR]),
            sol.x[LOC_VAR],
            sol.x[SCALE_VAR],
        )

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
        sol = _gen_skewt_fit(x, prob, display_progress)
        return GeneralisedSkewT(
            sol.x[SKEW_VAR],
            np.tan(sol.x[K_VAR]),
            np.tan(sol.x[N_VAR]),
            sol.x[LOC_VAR],
            sol.x[SCALE_VAR],
        )


def _gen_skewt_fit(returns_data, prob=None, display_progress=True):
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
    init_x = np.array([mean_dist, sd_dist, 0.0, np.arctan(2.0), np.arctan(8.0)])
    lower = np.array(
        [
            -10 * sd_dist,
            1e-8 * sd_dist,
            -0.9999,
            0.001 * math.pi / 2,
            0.001 * math.pi / 2,
        ]
    )
    upper = np.array(
        [10 * sd_dist, 5 * sd_dist, 0.9999, 0.999 * math.pi / 2, 0.999 * math.pi / 2]
    )

    # NLopt function must return a gradient even if algorithm is derivative-free
    # - this function will return an empty gradient
    def ll_func(x):
        return -np.sum(
            wt_prob
            * generalised_skewt_loglikelihood(
                returns_data,
                x[LOC_VAR],
                x[SCALE_VAR],
                x[SKEW_VAR],
                np.tan(x[K_VAR]),
                np.tan(x[N_VAR]),
            )
        )

    if display_progress:
        print("Fitting Generalised Skew-T Distribution")
        print("=======================================")
        print("Initial Log Likelihood")
        print(ll_func(init_x))
    soln = pybobyqa.solve(ll_func, init_x, bounds=(lower, upper))
    #  https://nlopt.readthedocs.io/en/latest/NLopt_Reference/ says that
    #  ROUNDOFF_LIMITED results are usually useful so I'm going to assume they are
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

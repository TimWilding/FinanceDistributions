"""
Suite of routines used to fit a covariance matrix using a Multivariate
T-Distribution using the EM Algorithm.
"""

import math
import time
from functools import partial

import numpy as np
import scipy.optimize as sopt
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.special import gamma
from scipy.stats import chi2

from .correl_calcs import mahal_dist
from .stat_functions import _print_progress
from .regress_calcs import wls_regress

MIN_DOF = 0.2
MAX_DOF = 1000
MIN_TAU = 0.01
MAX_TAU = 100


def _wt_stats(tau, returns_data):
    """
    returns weighted sample mean and covariance
    matrix using returns_data & tau
    """
    wt_returns = returns_data.T * tau
    samp_ave = np.sum(wt_returns, axis=1) / np.sum(tau)
    samp_covar = wt_returns @ returns_data / np.sum(tau)
    samp_covar = samp_covar - np.outer(samp_ave, samp_ave)
    return samp_ave, samp_covar


class TDist:
    """
    Multivariate T-Distribution routines
    """

    def __init__(self, average, covariance, nu):
        self.average = average
        self.covariance = covariance
        self.nu = nu

    def mahal_dist(self, returns_data):
        """
        Calculate the Mahalanobis Distance using a sample
        """
        return mahal_dist(self.average, self.covariance, returns_data)

    def cumdf(self, returns_data):
        """
        Calculate the cumulative density function
        """
        mahal_distances, _ = self.mahal_dist(returns_data)
        return TDist.cdf(
            self.nu,
            mahal_distances,
            returns_data.shape[1] * np.ones(returns_data.shape[0]),
        )

    def simulate(self, num_obs, rng_sim=None):
        """
        generate T-Dist random numbers with a positive semi-definite covariance matrix a
        and a given mean
        """

        rng = rng_sim
        if rng is None:
            rng = np.random.default_rng()

        num_var = self.covariance.shape[0]

        # Use eigen-decomposition to work out the square root of the
        # covariance matrix
        w, u = eigh(
            self.covariance
        )  # use the eigen-decomposition so that the results are robust
        w = np.sqrt(np.maximum(w, 0))
        sig_root = w * u

        # generate a set of independent normal random numbers
        norm_rnd = rng.normal(0.0, 1.0, (num_obs, num_var))
        # multiply by the square root of the covariance matrix
        y = norm_rnd @ sig_root.T

        # generate the gamma random numbers to rescale the results
        gamma_param = 0.5 * self.nu
        v_scale = rng.gamma(
            size=(num_obs, 1), scale=1.0 / gamma_param, shape=gamma_param
        )
        y = np.multiply(y, np.sqrt(1 / v_scale))

        # add the average to the results
        y = y + np.outer(np.ones((num_obs)), self.average)
        return y

    @staticmethod
    def prob_cdf(v, mahl_dist, no_stocks, nu):
        """Calculate the probability of the Mahalanobis distance exceeds this value
        given the degrees of freedom"""
        # This calculation has been verified by Monte Carlo simulation
        gamma_param = 0.5 * nu
        gam_shape = gamma_param
        gam_scale = 1.0 / gamma_param

        def gampdf(t):
            return (
                t ** (gam_shape - 1.0)
                * math.exp(-t / gam_scale)
                / (gamma(gam_shape) * gam_scale**gam_shape)
            )

        return gampdf(v) * chi2.cdf(mahl_dist * v, no_stocks)

    @staticmethod
    def cdf(nu, mahl_dist, no_stocks):
        """Work out the cumulative distribution function for the Mahlanobis Distance"""
        # This effectively uses MC integration to calculate the cumulative distribution function
        # Step 1 - work out num_samples estimates of the variance scale
        # Step 2 - calculate Chi-sq probability given the new variance scale
        #
        # Use numerical integration to work out the probability

        # gampdf = prob. distribution function
        # f is the weighted chi-squared function
        n_obs = mahl_dist.shape[0]
        if no_stocks.shape[0] != n_obs:
            raise ValueError(
                "No. of stocks and mahlanobis distance must be same dimensions"
            )
        cdf = np.zeros((n_obs))
        for i in range(0, n_obs):
            if nu < 5000:
                cdf[i] = quad(
                    partial(
                        TDist.prob_cdf,
                        mahl_dist=mahl_dist[i],
                        no_stocks=no_stocks[i],
                        nu=nu,
                    ),
                    0,
                    np.inf,
                )[0]
            else:
                cdf[i] = chi2.cdf(mahl_dist[i], no_stocks[i])
        return cdf

    @staticmethod
    def llcalc(degrees_of_freedom, mahl_dist, no_stock, log_cov_det):
        """returns the period-by-period log-likelihood for the T-distribution
        factor model given a set of intermediate calculations, used to
        fit the degrees of freedom
        https://en.wikipedia.org/wiki/Multivariate_t-distribution"""

        lg_df_2 = math.lgamma(degrees_of_freedom / 2)
        gamma_v = np.vectorize(lambda x: math.lgamma(x) - lg_df_2)
        nu_adj_var = 0.5 * (degrees_of_freedom + no_stock)
        ln_nu_dist = nu_adj_var * np.log(1 + mahl_dist / degrees_of_freedom)
        lg_gam = gamma_v(nu_adj_var)
        try:
            lg_gam = lg_gam - 0.5 * no_stock * math.log(degrees_of_freedom)
        except ValueError:
            print("No Stock")
            print(no_stock)
            print("Mahl Dist")
            print(mahl_dist)
            print("Log Cov Det")
            print(log_cov_det)
            print("DoF")
            print(degrees_of_freedom)
        log_likelihood = (
            lg_gam - 0.5 * no_stock * math.log(math.pi) - 0.5 * log_cov_det - ln_nu_dist
        )

        return log_likelihood

    @staticmethod
    def optimisedegreesoffreedom(
        nu, mahl_dist, no_stock, log_cov_det, min_dof=None, max_dof=None
    ):
        """calculate the degrees of freedom parameter that maximimises the likelihood"""
        ll_current = -np.sum(TDist.llcalc(nu, mahl_dist, no_stock, log_cov_det))

        def fp(p):
            return -np.sum(TDist.llcalc(p, mahl_dist, no_stock, log_cov_det))

        new_min = min_dof
        new_max = max_dof
        fun_brack = (min_dof, nu, max_dof)

        # Maybe work out the gradient before calling
        try:
            res = sopt.brent(fp, brack=fun_brack, full_output=True)
        except ValueError:
            try:
                res = sopt.brent(fp, brack=(MIN_DOF, nu, MAX_DOF), full_output=True)
            except ValueError:
                res = sopt.fminbound(fp, MIN_DOF, MAX_DOF, full_output=True)

        nu_ret = nu
        if res[1] < ll_current:
            nu_ret = res[0]
            new_max = min(nu_ret + 2 * abs(nu_ret - nu), MAX_DOF)
            new_min = max(nu_ret - 2 * abs(nu_ret - nu), MIN_DOF)
        return (nu_ret, new_min, new_max)

    @staticmethod
    def em_fit(returns_data, max_iters=100, tol=1e-10, display_progress=True, dof=-1.0):
        """
        Use the EM Algorithm to fit a multivariate T-distribution to the returns
        """
        nobs = returns_data.shape[0]
        nvar = returns_data.shape[1]
        start = time.time()
        tau = np.ones(nobs)
        mahal_distances = np.ones(nobs)

        nu = dof
        fit_dof = False
        if nu < 0:
            fit_dof = True
            nu = 8.0

        # Initialise with the sample mean & covariance
        samp_ave, samp_covar = _wt_stats(tau, returns_data)
        log_likelihood = []
        prev_ll = 0.0
        ll = 0.0
        delta_ll = 0.0
        _print_progress(
            display_progress,
            "Iteration    Nu      DeltaTau        LL            LL_target",
        )

        for iter_num in range(1, max_iters + 1):

            # Expected Sufficient Stats
            # tau = a weights scale
            # Calculate tau using mahalanobis distances etc

            mahal_distances, log_cov_det = mahal_dist(
                samp_ave, samp_covar, returns_data
            )

            if fit_dof:
                nu, _, _ = TDist.optimisedegreesoffreedom(
                    nu, mahal_distances, nvar, log_cov_det, MIN_DOF, MAX_DOF
                )

            prev_ll = ll
            ll = np.sum(TDist.llcalc(nu, mahal_distances, nvar, log_cov_det))
            log_likelihood.append(ll)
            ll_target = ll

            if iter_num > 1:
                delta_ll_prev = delta_ll
                delta_ll = ll - prev_ll
                if iter_num > 2:
                    alpha_k = delta_ll / delta_ll_prev
                    ll_target = log_likelihood[iter_num - 2] + delta_ll / max(
                        (1 - alpha_k), 0.001
                    )

            # Now we have the Mahlanobis distance calculate
            # the weighting scheme
            tau_prev = tau
            tau = (nu + nvar) / (nu + mahal_distances)
            tau[tau < MIN_TAU] = MIN_TAU
            tau[tau > MAX_TAU] = MAX_TAU

            delta_tau = np.max(np.abs(tau - tau_prev))
            _print_progress(
                display_progress,
                f"{iter_num:04d}      {nu:7.2f} {delta_tau:7.2f}"
                f"            {ll:7.2f}      {ll_target:7.2f}",
            )

            # Maximum Likelihood
            samp_ave, samp_covar = _wt_stats(tau, returns_data)
            if iter_num > 2:
                if np.abs(ll - ll_target) < tol:
                    break

        _print_progress(display_progress, f"Total time taken: {time.time()-start} s")

        return (samp_ave, samp_covar, nu, tau)

    @staticmethod
    def regress(y, X, max_iters=100, tol=1e-10, display_progress=True, dof=-1.0):
        """
        Use the EM Algorithm to fit a linear model with
        T-distributed residuals to the dataset
        y = X @ b + e

        Inputs:
        =============
        y: dependent variable
        X: independent variables
        max_iters: maximum number of iterations
        tol: tolerance for convergence
        display_progress: whether to print progress
        dof: degrees of freedom for the T-distribution, if negative
             then it will be fitted

        Returns:
        =============
        b_hat: estimated regression coefficients
        s: estimated scale of the residuals
        nu: estimated degrees of freedom for the T-distribution
        """
        nobs = y.shape[0]
        start = time.time()
        tau = np.ones(nobs)
        nu = dof
        fit_dof = False
        if nu < 0:
            fit_dof = True
            nu = 8.0

        # Initialise with the OLS regression parameters
        tau = np.ones(nobs)
        log_likelihood = []
        prev_ll = 0.0
        ll = 0.0
        delta_ll = 0.0
        _print_progress(
            display_progress,
            "Iteration    Nu      DeltaTau        LL            LL_target",
        )

        for iter_num in range(1, max_iters + 1):

            # M-step calculate the regression coefficients given the
            # current weighting

            b_hat, s = wls_regress(y, X, tau)
            if X.ndim == 1: 
                e = y - X[:, np.newaxis] @ b_hat
            else:
                e = y - X @ b_hat
    
            mah_dist = (e * e) / (s * s)

            if fit_dof:
                nu, _, _ = TDist.optimisedegreesoffreedom(
                    nu, mah_dist, 1, 2*np.log(s), MIN_DOF, MAX_DOF
                )

            prev_ll = ll

            ll = np.sum(TDist.llcalc(nu, mah_dist, 1, np.log(s)))
            log_likelihood.append(ll)
            ll_target = ll
            # Confirm the calculation using the scipy stats t-distribution
            #       log_likelihoods = t.logpdf(e, df=nu, loc=0, scale=np.sqrt(s))
            #       total_log_likelihood = np.sum(log_likelihoods)
            #        print(total_log_likelihood)
            if iter_num > 1:
                delta_ll_prev = delta_ll
                delta_ll = ll - prev_ll
                if delta_ll_prev <= 1e-10:
                    delta_ll_prev = 1e-10
                if iter_num > 2:
                    alpha_k = delta_ll / delta_ll_prev
                    ll_target = log_likelihood[iter_num - 2] + delta_ll / max(
                        (1 - alpha_k), 0.001
                    )
            # E-Step - calculate the expected value of the weighting scheme
            # In this case, a variance scale used to weight the residual variance
            # values

            tau_prev = tau
            tau = (nu + 1) / (nu + mah_dist)
            tau[tau < MIN_TAU] = MIN_TAU
            tau[tau > MAX_TAU] = MAX_TAU

            delta_tau = np.max(np.abs(tau - tau_prev))
            _print_progress(
                display_progress,
                f"{iter_num:04d}      {nu:7.2f} {delta_tau:7.2f}"
                f"            {ll:7.2f}      {ll_target:7.2f}",
            )

            if iter_num > 2:
                if np.abs(ll - ll_target) < tol:
                    break

        _print_progress(display_progress, f"Total time taken: {time.time()-start} s")

        return (b_hat, s, nu, log_likelihood)

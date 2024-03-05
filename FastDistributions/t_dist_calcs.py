import math
import time
import numpy as np
import scipy.optimize as sopt
from scipy.special import gamma
from scipy.stats import chi2
from scipy.integrate import quad


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

    @staticmethod
    def cdf(nu, mahl_dist, no_stocks):
        """Work out the cumulative distribution function for the Mahlanobis Distance"""
        # This effectively uses MC integration to calculate the cumulative distribution function
        # Step 1 - work out num_samples estimates of the variance scale
        # Step 2 - calculate Chi-sq probability given the new variance scale
        #
        # Use numerical integration to work out the probability

        gamma_param = 0.5 * nu
        gam_shape = gamma_param
        gam_scale = 1.0 / gamma_param
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
                gampdf = (
                    lambda v: v ** (gam_shape - 1.0)
                    * math.exp(-v / gam_scale)
                    / (gamma(gam_shape) * gam_scale**gam_shape)
                )
                f = lambda v: gampdf(v) * chi2.cdf(mahl_dist[i] * v, no_stocks[i])
                cdf[i] = quad(f, 0, np.Inf)[0]
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
        fp = lambda p: -np.sum(TDist.llcalc(p, mahl_dist, no_stock, log_cov_det))
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

    def em_fit(returns_data, max_iters=100, tol=1e-10, display_progress=True, dof=-1.0):
        """
        Use the EM Algorithm to fit a multivariate T-distribution to the returns
        """
        nobs = returns_data.shape[0]
        nvar = returns_data.shape[1]
        start = time.time()
        tau = np.ones(nobs)
        mahl_dist = np.ones(nobs)

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


        for iter_num in range(1, max_iters + 1):

            # Expected Sufficient Stats
            # tau = a weights scale
            # Calculate tau using mahl distances etc
            (U, w, VT) = np.linalg.svd(samp_covar)
            log_cov_det = np.sum(np.log(w.real))
            samp_excess = returns_data - samp_ave
            samp_excess_v = samp_excess @ VT.T
            s_temp_v = samp_excess_v / w
            mahl_dist = np.sum(s_temp_v * samp_excess_v, axis=1)

            if fit_dof:
                nu, _, _ = TDist.optimisedegreesoffreedom(
                    nu, mahl_dist, nvar, log_cov_det, MIN_DOF, MAX_DOF
                )

            prev_ll = ll
            ll = np.sum(TDist.llcalc(nu, mahl_dist, nvar, log_cov_det))
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
            tau = (nu + nvar) / (nu + mahl_dist)
            tau[tau < MIN_TAU] = MIN_TAU
            tau[tau > MAX_TAU] = MAX_TAU

            delta_tau = np.max(np.abs(tau - tau_prev))
            if display_progress:
                print(
                    f"Iteration {iter_num}: nu={nu:7.2f}, delta tau={delta_tau:7.2f}, ll={ll:7.2f}, ll_target={ll_target:7.2f}"
                )

            # Maximum Likelihood
            samp_ave, samp_covar = _wt_stats(tau, returns_data)
            if iter_num > 2:
                if np.abs(ll - ll_target) < tol:
                    break

        if display_progress:
            print(f"Total time taken: {time.time()-start} s")

        return (samp_ave, samp_covar, nu)

"""
Routines to assist with the use of Black-Litterman when building
quadratic optimised portfolios. For details of the techniques
see Satchell & Scowcroft (2000) - A demystification of the Black-Litterman
model
"""

import numpy as np
from scipy.linalg import inv
from scipy.stats import chi2


def black_litterman_stats(t, Sigma, P, q, Omega, tau=1.0, delta=5.0, exp_returns=False):
    """
    uses Black-Litterman approach to return the covariance matrix
    and expected returns given a set of views and a vector of means
    and a covariance matrix

    These calculations are based on a prior distribution of returns
     = N(mu, Sigma), and Bayesian updating.

    If tau<>1.0 is used, then the calculations can be changed to the
    prior distribution of expected returns

    These inputs can be used to build a portfolio using optimisation
    with user-defined constraints

    Input
    ===========
    t = nassets x 1 vector of model portfolio weights
    Sigma = nassets x nassets covariance matrix of returns
    P = nviews x nassets matrix of portfolios used to construct views
    q = nviews x 1  vector of expected view returns
    Omega = nviews x nviews covariance matrix of view returns
    tau = scalar containing scaling parameter for covariance matrix
    delta = scalar containing scaling parameter for expected returns
    exp_returns = use the Bayesian model as a prior for expected returns

    Output
    ===========
    m = nassets x 1 vector of expected returns
    H = nassets x nassets covariance matrix of returns
    """
    pi = delta * Sigma @ t  # calculate equilibrium expected returns

    # Build the updated variance matrix using the Bayesian updating formula
    # Note that if we are assuming the prior distribution of expected returns
    # then we should add the covariance matrix back in
    # Sigma_hat = H + Sigma
    H = inv(inv(tau * Sigma) + P.T @ inv(Omega) @ P)
    # Build the updated forecast of expected returns using the Bayesian
    # updating formula
    m = H @ (inv(tau * Sigma) @ pi + P.T @ inv(Omega) @ q)
    if exp_returns:
        H = H + Sigma
    return (m, H)


def calc_delta_kl(t, mu, Sigma):
    """
    Use the Kulback-Liebler Divergence to calculate the
    optimal delta from a model portfolio

    Inputs
    ========
    t = model portfolio
    mu = expected returns
    Sigma = covariance matrix

    Outputs
    ========
    delta = risk aversion for optimal portfolio
    """
    delta = t.T @ mu / (t.T @ Sigma @ t)
    return delta


def theils_view_compatibility(q, pi, Sigma_mu, P, Omega):
    """
    Return Theil's (1971) measure of view compatibility
    Calculation taken from Jay Walters comprehensive review of
    Black-Litterman

    Inputs
    ============
    q = nviews x 1  vector of expected view returns
    pi = nassets x 1 vector of equilibrium expected returns
    Sigma_mu = nassets x nassets covariance matrix of expected mean returns
    P = nassets x nviews matrix of portfolios used to construct views
    Omega = nviews x nviews covariance matrix of view returns

    Output
    ===========
    eps_hat = Theil's measure
    p_theil = probability of Theil's measure
    dptdq = sensitivity of probability of Theil's measure to view
    """
    # Calculate the difference between the views and the expected returns
    q_diff = q - P @ pi
    # Use the covariance matrix to test the difference
    # this is effectively a Mahlanobis distance
    # Note could use Cholesky decomp and then solve
    # problems as required to fix some numerical stability issues
    cov_mat_inv = inv(Omega + P @ Sigma_mu @ P.T)
    eps_hat = q_diff @ cov_mat_inv @ q_diff
    p_theil = 1 - chi2.cdf(eps_hat, q.shape[0])
    dptdq = -2 * chi2.pdf(eps_hat, q.shape[0]) * cov_mat_inv @ q_diff
    return (eps_hat, p_theil, dptdq)


def fusai_meucci_consistency(pi_hat, pi, Sigma_mu, P, Omega):
    """
    Calculate Fusai & Meucci's measure of consistency. This
    measure is the Mahlanobis distance of the Black-Litterman
    forecast returns from the equilibrium returns.

    Inputs
    ============
    pi_hat = nassets x 1 vector of Black-Litterman expected returns
    pi = nassets x 1 vector of equilibrium expected returns
    Sigma_mu = nassets x nassets covariance matrix of expected mean returns

    Outputs
    ============
    M_q = Fusai Meucci measure of consistency (Mahlanobis Distance)
    P_M_q = Probability of M_q
    dPdq = sensitivity of M_q to individual view
    """
    M_q = (pi_hat - pi).T @ inv(Sigma_mu) @ (pi_hat - pi)
    P_M_q = 1 - chi2.cdf(M_q, pi_hat.shape[0])
    dMdq = 2 * (pi_hat - pi) @ P.T @ inv(P @ Sigma_mu @ P.T + Omega)
    dPdq = -chi2.pdf(M_q, pi_hat.shape[0]) * dMdq
    return M_q, P_M_q, dPdq


def theils_source(Sigma_mu, P, Omega):
    """
    Determine the contribution to the posterior precision of the prior and the views
    this sums to 1 across all views and the prior
    Inputs
    ============
    Sigma_mu = nassets x nassets covariance matrix of expected mean returns
    P = nassets x nviews matrix of portfolios used to construct views
    Omega = nviews x nviews covariance matrix of view returns
    Outputs
    ============
    theta_prior = posterior precision of the prior
    theta_views = posterior precision of the views
    theta_prior + sum(theta_views) = 1.0
    """
    n_assets = P.shape[1]
    n_views = P.shape[0]
    H_inv = inv(inv(Sigma_mu) + P.T @ inv(Omega) @ P)
    theta_prior = (1 / n_assets) * np.trace(inv(Sigma_mu) @ H_inv)
    omega_p = inv(Omega) @ P
    theta_views = np.zeros(n_views)
    for i in range(n_views):
        P_slice = np.outer(P[i, :], omega_p[i, :])
        theta_views_mat = (1 / n_assets) * P_slice @ H_inv
        theta_views[i] = np.trace(theta_views_mat)
    return theta_prior, theta_views


def he_litterman_lambda(t, Sigma, P, q, Omega, tau=1.0, delta=5.0, exp_returns=False):
    """
    The Black-Litterman method returns a portfolio that is a combination of the
    equilibrium portfolio and a linear combination of the view portfolios.
    He & Litterman (2005) calculate Lambda - the coefficients used to combine
    those view portfolios. This routine calculates Lambda and the sensitivity
    of the Lambda coefficient to the forecast returns of the view portfolios

    Input
    ==========
    t = nassets x 1 vector of model portfolio weights
    Sigma = nassets x nassets covariance matrix of returns
    P = nviews x nassets matrix of portfolios used to construct views
    q = nviews x 1  vector of expected view returns
    Omega = nviews x nviews covariance matrix of view returns
    tau = scalar containing scaling parameter for covariance matrix
    delta = scalar containing scaling parameter for expected returns

    Outputs
    ==========
    he_lit_lam = q x 1 vector used to construct linear combination
    of view portfolios
    d_lam_dq = sensitivity of he_lit_lam to view returns
    """
    f = 0.0
    if exp_returns:
        f = 1.0
    pi = delta * Sigma @ t  # calculate equilibrium expected returns
    q_eq = P @ pi

    B = (tau / (f + tau)) ** 2 * inv(Omega + (f / (f + tau)) * P @ Sigma @ P.T)
    view_cov = P @ Sigma @ P.T
    A = (tau / (f + tau)) * inv(view_cov + Omega)
    A = A + (tau / (f + tau)) ** 2 * inv(
        Omega + (f / (f + tau)) * view_cov
    ) @ view_cov @ inv(view_cov + Omega)
    #    pi_hat, Sigma_hat = black_litterman_stats(t, Sigma, P, q, Omega, tau, delta)
    he_lit_lam = ((f + tau) / delta) * (A @ (q - q_eq) + B @ q_eq)
    d_lam_dq = ((f + tau) / delta) * A
    return he_lit_lam, d_lam_dq


def reverse_optimise(t, sigma, delta):
    """
    Calculate the returns of the portfolio that would make
    the given portfolio mean-variance optimal with a given
    risk tolerance.
    Inputs
    ========
    t = model portfolio
    sigma = covariance matrix
    delta = risk aversion coefficient
    Outputs
    =======
    pi = expected returns
    """
    pi = delta * sigma @ t
    return pi


def unconstrained_optimal_portfolio(sigma, mu, delta):
    """
    Calculate the unconstrained optimal portfolio given
    covariance matrix, expected returns, and risk aversion
    coefficient.
    Inputs
    ========
    sigma = covariance matrix
    mu = expected returns
    delta = risk aversion coefficient
    Outputs
    =======
    x = unconstrained optimal portfolio
    """
    x = (inv(sigma) @ mu) / delta
    return x


def braga_natale_measure(t, sigma, P, q, omega, tau=1.0, delta=1.0, exp_returns=False):
    """
    Calculate the tracking error volatility of the Black-Litterman portfolio
    and the sensitivity of the tracking error to the view
    """
    pi_hat, Sigma_hat = black_litterman_stats(
        t, sigma, P, q, omega, tau, delta, exp_returns
    )
    w_bl = unconstrained_optimal_portfolio(Sigma_hat, pi_hat, delta)
    _, dlamdq = he_litterman_lambda(t, sigma, P, q, omega, tau, delta, exp_returns)

    te = np.sqrt((w_bl - t) @ sigma @ (w_bl - t))
    dtedw = sigma @ (w_bl - t) / te
    f = 0.0
    if exp_returns:
        f = 1.0
    dtedq = dtedw @ P.T @ dlamdq / (f + tau)
    return te, dtedq

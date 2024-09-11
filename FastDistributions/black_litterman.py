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
    Calculate Fusai & Meucci's measure of consistency
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

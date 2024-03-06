"""
Various correlation calcs adjusting for serial correlation
"""

import numpy as np
import scipy.linalg as sla


def ols_regress(Y, X):
    """
    OLS Regression - very quick routine
    Y - n x p matrix of returns
    X - n x q matrix of factor returns
    Output
    =====
    B0 - 1 x p matrix of estimated average returns
    Bx - q x p matrix of estimated factor loadings
    Ee - n x p matrix of residuals
    se - p array of estimated residual variances
    """
    # OLS estimated parameters
    n = X.shape[0]
    G = np.hstack([np.ones((n, 1)), X])
    # OLS regression
    c_delta, low = sla.cho_factor(G.T @ G, lower=True)
    Be = sla.cho_solve((c_delta, low), G.T @ Y, overwrite_b=False, check_finite=False)
    Ee = Y - G @ Be  # estimated residuals
    se = np.var(Ee, axis=0)  # est. res. var

    Bx = Be[1:, :]  # est. betas
    B0 = Be[0:1, :]  # est. alphas
    return (B0, Bx, Ee, se)


def adjusted_correl(X, lag=1):
    """Calculate a revised estimate of the covariance matrix
    accounting for the correlationg at a lag. This code is very simple and
    doesn't account for any autocorrelation in the returns of a single
    series. See Scholes & Williams 1977 for details of this method"""
    T = X.shape[0]
    p = X.shape[1]
    corr_mat = 0.5 * np.eye(p)
    for i in range(p):
        Z = np.zeros((T - 2 * lag, 1 + 2 * lag))

        Z[:, 0] = X[lag : T - lag, i]  # Synchronous returns of first index

        for j in range(1, lag + 1):
            Z[:, 2 * j - 1] = X[(lag + j) : (T - lag + j), i]  # j-lagged returns
            Z[:, 2 * j] = X[(lag - j) : T - lag - j, i]  # j-leading returns

        # Fill in the lower-diagonal
        if i < p - 1:
            Y = X[lag : T - lag, i + 1 :]
            _, B, _, _ = ols_regress(Y, Z)
            B_true = np.sum(B, axis=0)
            sig_y = np.std(Y, axis=0)
            sig_x = np.std(Z[:, 0])
            correl = sig_x * (B_true / sig_y)
            corr_mat[i + 1 : p, i] = correl
    return nearest_pos_def(corr_mat + corr_mat.T)


def newey_adj_cov(X, lag=1, demean=True):
    """
    Use the Newey-West (1987) adjusted covariance matrix
    to produce a robust estimate of the covariance
    matrix adjusted for serial correlation
    Inputs
    X = t x n matrix of observations
        t = number of samples
        n = number of variables
    lag = number of lags to use for estimator
        = -1 use estimate of lags
    Outputs
    n x n covariance matrix
    """
    if demean:
        X = X - np.mean(X, axis=0)
    gamma = np.dot(X.T, X)

    # Use Bartlett Weighting Scheme
    t = X.shape[0]
    p = lag
    if lag == -1:
        # Determine the number of lags to use
        # based on the sample size
        # Stock and Watson (2003) formula
        p = int(0.75 * t ** (1 / 3))
    for j in range(1, p + 1):
        w = 1 - (float(j) / float(p + 1))
        gamma_lag = np.dot(X[0 : t - j, :].T, X[j:t, :])
        gamma = gamma + w * (gamma_lag + gamma_lag.T)
    return gamma / t


def newey_adj_corr(X, lag=1):
    """
    Use the Newey-West adjusted covariance matrix
    to return a correlation matrix
    """
    s = nearest_pos_def(newey_adj_cov(X, lag))
    return corr_conv(s)


# see https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
# the nearestPD routine is a copy and paste from there
def nearest_pos_def(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def corr_conv(cov_mat):
    """
    converts covariance matrix to correlation matrix
    entry should be square positive semi-definite covariance
    matrix
    """

    sig = np.diag(cov_mat)

    corr_mat = cov_mat / np.sqrt(np.outer(sig, sig))

    return corr_mat


def mahal_dist(samp_ave, samp_covar, returns_data):
    """
    Calculate the Mahalanobis Distance using a sample
    average, covariance for a set of returns data
    see https://en.wikipedia.org/wiki/Mahalanobis_distance
    The Mahalanobis distance should be distributed with
    a Chi-Squared distribution with degrees of freedom
    equal to p where p is the number of assets
    Inputs
    =====================================
    samp_ave - p vector of sample averages
    samp_covar - p x p covariance matrix
    returns_data - n x p matrix of returns
    Outputs
    =====================================
    mahl_dist - n vector of Mahlanobis Distances
    log_cov_det - log of the determinant of covariance
                  matrix
    """
    (_, w, VT) = np.linalg.svd(samp_covar)
    log_cov_det = np.sum(np.log(w.real))
    # Use robust inversion by ignoring small
    # singular values
    s = np.copy(w.real)
    rcond = 1e-15  # maybe should be a settable param?
    cutoff = rcond * np.amax(s, axis=-1, keepdims=True)
    large = s > cutoff
    s[~large] = 0
    s = np.divide(1, s, where=large, out=s)
    samp_excess = returns_data - samp_ave
    samp_excess_v = samp_excess @ VT.T
    s_temp_v = samp_excess_v * s
    mahal_distances = np.sum(s_temp_v * samp_excess_v, axis=1)
    return (mahal_distances, log_cov_det)

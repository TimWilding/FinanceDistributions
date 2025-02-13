"""
Contains Mean-Variance Optimisation and other variants of portfolio optimisation routines
"""

import cvxpy as cp
import numpy as np
import pandas as pd
# from typing import Tuple


def get_weighted_stats(sample_weights, sample_returns, annualise=52):
    """
    Get weighted covariance matrix and mean for sample stats
    """
    cov = annualise * np.cov(sample_returns.T, aweights=sample_weights)
    means = annualise * np.average(sample_returns, weights=sample_weights, axis=0)
    return (means, cov)


def get_pf_stats(pf_weights, expected_returns, covariances):
    """
    Return the mean, st. dev., and risk allocation for a portfolio given
    the sample weights and returns
    """
    mu = expected_returns
    sigma = covariances
    mu_pf = mu @ pf_weights
    sigma_pf = pf_weights.T @ sigma @ pf_weights
    alloc = np.multiply(pf_weights, sigma @ pf_weights) / sigma_pf
    return (mu_pf, np.sqrt(sigma_pf), alloc)


def get_optimal_sharpe_pf(expected_returns, covariances):
    """
    Use the typical Cornuejols recipe for a constrained Sharpe PF Optimisation with a quadratic programming
    routine - in this case, the constraint is that
    all assets have positive weights
    (see, e.g., https://quant.stackexchange.com/questions/39137/how-can-i-find-the-portfolio-with-maximum-sharpe-ratio-using-lagrange-multipli/39157#39157)
    """  # noqa: E501
    mu = expected_returns
    sigma = covariances
    num_assets = mu.shape[0]
    x = cp.Variable(num_assets)
    kappa = cp.Variable()  # extra slack variable to rescale the portfolio
    constraints = [cp.sum(x) == kappa, x >= 0, mu @ x == 1.0, kappa >= 0]
    # Define the objective function
    objective = cp.Minimize(cp.quad_form(x, sigma))

    # Define the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=True)

    # Print the solution
    #    print("Optimal portfolio:", x.value)
    return x.value / kappa.value


def get_robust_pf(
    risk_tol, kappa, expected_returns, covariances, report_progress=False
):
    """
    This is a robust asset allocation mechanism that uses
    sum(mu*w^(1/kappa)) - risk_tol*w'*sigma*w to
    build a robust portfolio

    Since limit(k->infty w^(1/k)) = 1 + ln(w)/k, we replace the limit with
    log. For this portfolio, one point along the efficient frontier corresponds
    to the risk-parity portfolio

    Note, at the moment, this formulation only allows for positive positions
    The formulation needs to be slightly adjusted to allow for negative positions


    """
    if kappa < 1.0:
        raise ValueError("kappa must be >=1.0")

    mu = expected_returns
    sigma = covariances

    num_assets = mu.shape[0]

    x = cp.Variable(num_assets)

    constraints = [cp.sum(x) == 1.0, x >= 0]
    # Define the objective function
    if kappa > 50:
        objective = cp.Maximize(
            cp.matmul(mu, cp.log(x)) - risk_tol * cp.quad_form(x, sigma)
        )
    else:
        objective = cp.Maximize(
            cp.matmul(mu, cp.power(x, 1 / kappa)) - risk_tol * cp.quad_form(x, sigma)
        )

    # Define the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=report_progress)

    # Print the solution
    if report_progress:
        print("Solution")
        print("========")
        print(x)
        print("Optimal portfolio:", x.value)
        print("Expected Return = ", x.value.T @ mu)
        print("Volatility = ", x.value.T @ sigma @ x.value)
        allocation = mu.T @ np.power(x.value, 1 / kappa)
        if kappa > 50:
            allocation = mu.T @ np.log(x.value)
        print("Expected Allocation = ", allocation)
    return x.value


def get_mv_pf(risk_tol, expected_returns, covariances, report_progress=False):
    """
    This is a robust asset allocation mechanism that uses
    sum(mu*w) - risk_tol*w'*sigma*w to
    build a robust portfolio

    """
    mu = expected_returns
    sigma = covariances
    num_assets = mu.shape[0]

    x = cp.Variable(num_assets)

    constraints = [cp.sum(x) == 1.0, x >= 0]
    # Define the objective function
    objective = cp.Maximize(mu @ x - risk_tol * cp.quad_form(x, sigma))
    #  objective = cp.Maximize(mu @ x - risk_tol*cp.quad_form(x, sigma))

    # Define the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=report_progress)

    # Print the solution
    #    print("Optimal portfolio:", x.value)
    return x.value


def get_risk_parity_pf(covariances, risk_budgets=None, report_progress=False):
    """
    Build a risk-parity portfolio using the mean-variance formulation
    see https://quant.stackexchange.com/questions/3114/how-to-construct-a-risk-parity-portfolio
    """
    # Define the mean and covariance matrix
    sigma = covariances
    num_assets = covariances.shape[0]
    x = cp.Variable(num_assets)
    budget_constraints = np.ones(num_assets) / num_assets
    if risk_budgets is not None:
        if risk_budgets.shape[0] != num_assets:
            raise ValueError("wrong shape risk budgets")
        budget_constraints = risk_budgets

    # kappa is an arbitrary constant - it ends up being effectively ignored
    # because we rescale the output values. Let's try and pick a good value so
    # that our values are around 0.0 to 1.0
    kappa = -np.log(num_assets)
    # Define the constraints
    constraints = [cp.matmul(budget_constraints, cp.log(x)) >= kappa, x >= 0]

    # Define the objective function
    objective = cp.Minimize(cp.quad_form(x, sigma))

    # Define the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=report_progress)
    risk_parity_pf = x.value / np.sum(x.value)
    if report_progress:
        marginals = sigma @ risk_parity_pf
        risk_contrib = risk_parity_pf * marginals
        df = pd.DataFrame(
            {
                "RiskMarg": sigma @ risk_parity_pf,
                "Weights": risk_parity_pf,
                "RiskContrib": risk_contrib,
                "RiskProportion": risk_contrib / np.sum(risk_contrib),
            }
        )
        print(df)
    return risk_parity_pf

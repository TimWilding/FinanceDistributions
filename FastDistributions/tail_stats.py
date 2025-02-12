"""
module containing useful tail statistics such as expected shortfall 
"""

import numpy as np
from scipy import stats
from scipy.stats import rv_continuous
from scipy.integrate import quad
import warnings


def expected_shortfall(distribution: rv_continuous, quantile: float, *args, **kwargs):
    """
    Calculate the Expected Shortfall (ES) for a given scipy.stats distribution. Please
    note that this function may not be valid for distributions with particularly heavy tails
    So, for example, the Levy-Stable distributioni with an alpha <1 would not have a finite
    expected shortfall.

    Parameters:
        distribution (rv_continuous): A scipy.stats continuous distribution object.
        quantile (float): The quantile level (e.g., 0.95 for the 95th percentile).
        *args: Positional arguments for the distribution (e.g., shape parameters).
        **kwargs: Keyword arguments for the distribution (e.g., loc and scale parameters).

    Returns:
        float: The Expected Shortfall at the specified quantile.
        float: The Value at Risk (VaR) at the specified quantile.
    """
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between 0 and 1.")

    # Calculate the Value at Risk (VaR) at the given quantile
    var = distribution.ppf(1 - quantile, *args, **kwargs)

    # Define a function to integrate the tail of the distribution
    def tail_expectation(x):
        return x * distribution.pdf(x, *args, **kwargs)

    # Compute the expected shortfall as the conditional expectation
    es, _ = quad(tail_expectation, -np.inf, var)
    tail_prob = 1 - quantile
    return es / tail_prob, var


def sample_expected_shortfall(returns: np.array, quantile: float):
    """
    Calculate the Expected Shortfall (ES) for a given set of returns.

    Parameters:
        distribution (rv_continuous): A scipy.stats continuous distribution object.
        quantile (float): The quantile level (e.g., 0.95 for the 95th percentile).

    Returns:
        float: The Expected Shortfall at the specified quantile.
        float: The Value at Risk (VaR) at the specified quantile.
    """
    threshold = np.quantile(returns, 1 - quantile)
    negative_returns = returns[returns < threshold]
    return np.mean(negative_returns), threshold


def fit_tail_model(returns: np.ndarray, threshold: float = 0.95):
    """
    Comprehensive EVT analysis for tail behavior using
    Pickands-Balkema-De Haan (GPD) distribution.
    See https://en.wikipedia.org/wiki/Pickands%E2%80%93Balkema%E2%80%93De_Haan_theorem
    for a discussion of the asymptotic tail behaviour of a distribution
    We can get some useful financial stats using alpha as a percentile:
        var = stats.genpareto.ppf(alpha, *gpd_params)
        expected shortfall = var / (1 - ξ) if ξ < 1 else infinity
    Note that alpha needs to be adjusted for the tail distribution.
    Parameters:
    -----------
    returns : np.ndarray
        Array of log returns
    threshold : float
        Threshold quantile for tail analysis
        >0.5 for left tail
        <0.5 for right tail

    Returns:
    --------
        ξ = shape parameter of gpd (tail index)
        β = scale parameter of gpd
        threshold_value = value of threshold quantile
    """
    threshold_value = np.quantile(returns, 1 - threshold)
    if threshold < 0.5:
        excess = returns[returns > threshold_value] - threshold_value
    else:
        excess = threshold_value - returns[returns < threshold_value]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # fit a generalised pareto distribution with a location
        # parameter of 0 (floc=0)
        gpd_params = stats.genpareto.fit(excess, floc=0)

    # Extract parameters
    ξ = gpd_params[0]  # shape parameter (tail index)
    β = gpd_params[2]  # scale parameter

    return ξ, β, threshold_value


def expected_shortfall_tail_model(alpha, ξ, β, threshold_value, quantile: float):
    """
    Calculate the Expected Shortfall (ES) for a given scipy.stats distribution. Please
    note that this function may not be valid for distributions with particularly heavy tails
    So, for example, the Levy-Stable distributioni with an alpha <1 would not have a finite
    expected shortfall.

    Parameters:
        alpha - the quantile cutoff used to build the  tail model
        ξ = shape parameter of gpd (tail index)
        β = scale parameter of gpd
        threshold_value = value of threshold quantile
        quantile (float): The quantile level (e.g., 0.95 for the 95th percentile).


    Returns:
        float: The Expected Shortfall at the specified quantile.
        float: The Value at Risk (VaR) at the specified quantile.
    """
    if not alpha <= quantile < 1:
        raise ValueError("Quantile must be between alpha and 1.")

    gp = stats.genpareto(ξ, 0, β)
    # Calculate the Value at Risk (VaR) at the given quantile
    var = gp.ppf((quantile - alpha) / (1 - alpha))

    # Define a function to integrate the tail of the distribution
    def tail_expectation(x):
        return x * gp.pdf(x)

    # Compute the expected shortfall as the conditional expectation
    es, _ = quad(tail_expectation, var, np.inf)
    tail_prob = (1 - quantile) / (1 - alpha)
    es = es / tail_prob
    return threshold_value - es, threshold_value - var
    

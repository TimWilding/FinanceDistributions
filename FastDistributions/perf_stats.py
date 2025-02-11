"""
Performance statistics such as the Sharpe Ratio, the Probabilistic Sharpe Ratio, and the Omega Ratio
"""

import numpy as np
import scipy


def sharpe_ratio(excess_returns, periods_in_year=1):
    """
    Calculate Sharpe Ratio from Time Series of Excess Returns
    Report the variance of the estimate of Sharpe ratio taken
    from Bailey & de Prado (2012) under the assumption of Normal
    returns (original calculation by Jobson & Korkie)

    Parameters
    ======
    excess_returns:
       numpy array containing returns

    periods_in_year:
       sampling frequency (252=daily, 52=weekly, 12=monthly)

    Outputs
    =======
    Sharpe Ratio:
      annualised ratio of mean returns to standard deviation of returns

    mu:
      annualised mean of returns

    sigma:
      annualised standard deviation of returns

    sr_sd:
      annualised standard deviation of Sharpe Ratio under Normal distribution
    """
    mu = np.mean(excess_returns)
    sigma = np.std(excess_returns)

    sr = mu / sigma
    n = excess_returns.shape[0]
    V_g = (1 + 0.5 * sr**2) / n

    mu_annual = periods_in_year * mu
    sigma_annual = np.sqrt(periods_in_year) * sigma
    sr_annual = mu_annual / sigma_annual
    sr_sd = np.sqrt(periods_in_year * V_g)

    return (sr_annual, mu_annual, sigma_annual, sr_sd)


def psr(sr, skew, kurt, srb, periods_in_year, n):
    """
    Use the series statistics to determine the
    Probabilistic Sharpe Ratio for a given SR
    Bailey & de Prado (2012) "The Sharpe Ratio Efficient Frontier".
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643)

    Parameters
    =========
    sr:
         Sharpe Ratio for a particular fund

    skew:
         Skewness for a particular fund

    kurt:
         Kurtosis for a particular fund

    periods_in_year:
         sampling frequency for fund data

    n:
         number of fund data points

    sharpe_ratio_benchmark:
         Sharpe Ratio we are testing ourselves against

    periods_in_year:
         sampling frequency (252=daily, 52=weekly, 12=monthly)

    Returns
    =======
    P(SR>benchmark_sharpe_ratio):
         probability of SR being better than benchmark
    """
    V_g = (1 + (1 / 2) * sr**2 - skew * sr + ((kurt - 3) / 4) * sr**2) / (n - 1)
    srb = srb / np.sqrt(periods_in_year)
    prob_sr = scipy.stats.norm.cdf((sr - srb) / np.sqrt(V_g))
    return prob_sr


def probabilistic_sharpe_ratio(
    excess_returns, sharpe_ratio_benchmark=0, periods_in_year=1
):
    """
    Use the variance of the estimate of Sharpe ratio taken
    from Opdyke (2007) "Comparing Sharpe Ratios: So Where Are the P-Values"
    to calculate the probability that the sample Sharpe Ratio
    is better than the benchmark Sharpe Ratio. The sample
    Sharpe ratio asymptotically converges to a Normal distribution
    with a variance that is a function of the first 4 moments
    of the excess returns sample.

    This is the Probabilistic Sharpe Ratio proposed by
    Bailey & de Prado (2012) "The Sharpe Ratio Efficient Frontier".
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643)
    Empirically, this improves on a traditional Sharpe Ratio by adjusting
    for the length of history and the non-Normality of returns.

    Parameters
    ==========
    excess_returns:
         vector of excess returns for a particular fund

    sharpe_ratio_benchmark:
         Sharpe Ratio we are testing ourselves against

    periods_in_year:
         sampling frequency (252=daily, 52=weekly, 12=monthly)

    Returns
    =======
    P(SR>benchmark_sharpe_ratio):
         probability of SR being better than benchmark

    Annualised Sharpe Ratio:
         annualised Sharpe Ratio
    """
    mu = np.mean(excess_returns)
    sigma = np.std(excess_returns)
    skew = scipy.stats.skew(excess_returns)
    kurt = scipy.stats.kurtosis(excess_returns, fisher=False)
    sr = mu / sigma
    n = excess_returns.shape[0]
    V_g = (1 + (1 / 2) * sr**2 - skew * sr + ((kurt - 3) / 4) * sr**2) / (n - 1)
    srb = sharpe_ratio_benchmark / np.sqrt(periods_in_year)
    prob_sr = scipy.stats.norm.cdf((sr - srb) / np.sqrt(V_g))
    return (prob_sr, np.sqrt(periods_in_year) * sr, np.sqrt(periods_in_year * V_g))


def omega_ratio(returns, returns_threshold=0, periods_in_year=12, geom_returns=True):
    """
    Calculate the Omega Ratio (https://en.wikipedia.org/wiki/Omega_ratio)
    using the time series of excess returns.
    Omega_ratio = A/B, where A = integral of (1-CDF(x)) from threshold to infinity
                             B = integral of CDF(x) from -infinity to threshold

    Note that these integrals are approximately the sum of the returns over the
    threshold / the sum of the returns under the threshold


    Parameters
    ==========

    returns:
         time series of excess returns

    returns_threshold:
         annual returns threshold for the Omega ratio

    periods_in_year:
         number of time periods in year (12 = monthly data, 52 = weekly, 252=daily)

    Returns
    =======
    omega:
       omega ratio

    """

    excess_returns_threshold = returns_threshold / np.sqrt(periods_in_year)
    if not geom_returns:
        excess_returns = (1 + returns_threshold) ** np.sqrt(1 / periods_in_year) - 1

    excess_returns = returns - excess_returns_threshold
    positive_sum = np.sum(excess_returns[excess_returns > 0])
    negative_sum = -np.sum(excess_returns[excess_returns < 0])
    omega = positive_sum / negative_sum
    return omega


def min_track_record_length(sr, sr_std, sharpe_ratio_benchmark=0.0, prob=0.95):
    """
    Calculate the minimum track record length required to have a certain probability
    of beating the benchmark (minTRL) given a certain Sharpe ratio and Standard Deviation
    This can be used to determine how long we need to watch a fund before we can
    be certain that it will beat our benchmark.

    Parameters
    ==========

    sr:
        Sharpe ratio expressed in the same frequency as the other parameters.

    sr_std:
        Standard deviation of the Estimated sharpe ratio,
        expressed in the same frequency as the other parameters.

    sr_benchmark:
        Benchmark sharpe ratio expressed in the same frequency as the other parameters.
        By default set to zero (comparing against no investment skill).

    prob:
        Confidence level used for calculating the minTRL.
        Between 0 and 1, by default=0.95


    Returns
    =======
    minTRL:
       minimum number of returns neede to get P(SR*)>prob

    Notes
    =====
    minTRL = minimum of returns/samples needed (with same SR and SR_STD) to accomplish a PSR(SR*) > `prob`
    PSR(SR*) = probability that SR^ > SR*
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR* = `sr_benchmark`


    """
    min_trl = (
        1
        + (sr_std**2)
        * (scipy.stats.norm.ppf(prob) / (sr - sharpe_ratio_benchmark)) ** 2
    )
    return min_trl

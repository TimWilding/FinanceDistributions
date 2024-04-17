"""
PRIIPS calculation functions for category 2 PRIIPS.
"""

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

#  see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hermitenorm.html
from scipy.special import hermitenorm
from .stat_functions import parallel_bootstrap


def Cornish_Fisher_percentile(
    mu, sigma, skew, kurt, holding_period, pctile, periods_in_year
):
    """
    Use the Cornish-Fisher approximation to calculate the percentile of
    the distribution using the first 4 moments
    see https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion
    for discussion
    mu - average return
    sigma - standard deviation
    skew - skewness
    kurt - kurtosis
    holdingperiod = number of years for recommended holding period
    pctile = percentile of the distribution
    periodsinYear = number of periodsin the year (256 = days, 52 = weeks, 12 = months)

    The PRIIPS documentation expands out the formula for the Cornish-Fisher expansion
    and quotes specific coefficients for the skewness, kurtosis and other parameters
    at set percentile points to be used in the calculation of different performance
    statistics.
    See, for example, page 45-46 of
    https://www.esma.europa.eu/sites/default/files/library/jc_2016_21_final_draft_rts_priips_kid_report.pdf
    The formulas here are equivalent, but have more significant figures in the coefficients.
    If you're really concerned about this, feel free to type them out!
    """
    n = holding_period * periods_in_year
    z_alpha = norm.ppf(pctile)
    w = z_alpha + (hermitenorm(2)(z_alpha) / 6.0) * (skew / np.sqrt(n))
    w = w + (hermitenorm(3)(z_alpha) / 24.0) * (kurt / n)
    w = w - ((2 * hermitenorm(3)(z_alpha) + hermitenorm(1)(z_alpha)) / 36.0) * (
        skew * skew / n
    )
    x = mu * n + np.sqrt(n) * sigma * w - 0.5 * sigma * sigma * n
    return x


def convert_VaR_to_volatility(value_at_risk, pctile, holding_period):
    """
    PRIIPS recommended formula for converting a Value At Risk number to
    a volatility
    - see page 31 of
    https://www.esma.europa.eu/sites/default/files/library/jc_2016_21_final_draft_rts_priips_kid_report.pdf
    """
    z_alpha = norm.ppf(pctile)
    vol = np.sqrt(z_alpha * z_alpha - 2 * value_at_risk) + z_alpha
    vol = vol / np.sqrt(holding_period)
    return vol


def volatility_to_MRM_class(VaR_equivalent_vol: float) -> int:
    """
    PRIIPS function for converting a volatility to a
    market risk class
    - see table on page 29 of
    https://www.esma.europa.eu/sites/default/files/library/jc_2016_21_final_draft_rts_priips_kid_report.pdf
    """
    if VaR_equivalent_vol < 0.005:
        return 1
    if VaR_equivalent_vol < 0.05:
        return 2
    if VaR_equivalent_vol < 0.12:
        return 3
    if VaR_equivalent_vol < 0.2:
        return 4
    if VaR_equivalent_vol < 0.3:
        return 5
    if VaR_equivalent_vol < 0.8:
        return 6
    return 7


def get_stress_window(
    periods_in_year: int, holding_period: float, returns, use_2020: bool = True
):
    """
    Return the size of the window used to calculate the stress volatility.
    This is a function of both data frequency and holding period. The size of the
    window is smaller if the holding period is less than a year. The PRIIPS guidelines
    state that you require certain amounts of data depending on the data frequency
    so this function will raise an Exception if there is insufficient data
    see p.74 of https://www.eiopa.europa.eu/document/download/51861f2c-84a1-4c51-a891-3cd5c47db80c_en?filename=Final%20report%20on%20draft%20RTS%20to%20amend%20PRIIPs%20KID.pdf

    Note that there are suggestions tha the calculations have changed in the 2020 recommendations
    this is to prevent the unfavourable scenario being below the stress scenario
    """
    n = returns.shape[0]
    if use_2020:
        print(
            "Code not adjusted for new recommended stress windows - proceed with caution"
        )
    if periods_in_year == 12:
        if n < 60:
            raise Exception(
                "Insufficient data for PRIIPS calculation - 5 years monthly required"
            )
        if holding_period < 1.0:
            return 6
        else:
            return 12
    if periods_in_year == 52:
        if n < 208:
            raise Exception(
                "Insufficient data for PRIIPS calculation - 4 years weekly required"
            )
        if holding_period < 1.0:
            return 8
        else:
            return 16
    # otherwise, assume daily data
    if n < 2 * periods_in_year:
        raise Exception(
            "Insufficient data for PRIIPS calculation - 2 years daily required"
        )
    if holding_period < 1.0:
        return 21

    return 63


def calc_sigma_stress(
    returns, periods_in_year: int, holding_period: float, use_2020: bool
):
    """
    Return the 90th percentile of the rolling volatility
    for use in the stress scenario calculation
    """
    rolling_window = get_stress_window(
        periods_in_year, holding_period, returns, use_2020
    )
    rolling_vols = np.std(
        sliding_window_view(returns, window_shape=rolling_window, axis=0), axis=2
    )
    if (holding_period > 1) & use_2020:
        return np.percentile(rolling_vols, 95, axis=0)
    else:
        return np.percentile(rolling_vols, 90, axis=0)


def calc_moments(returns):
    """
    calc_moments calculates the first 4 moments of an array
    of returns data
    returns is an nxp array of data
    returns 1xp array of moments
    """
    t = returns.shape[0]
    p = returns.shape[1]

    mom_0 = t * np.ones((1, p))  # count of the number of observations
    mom_1 = (
        np.sum(returns, axis=0) / t
    )  # mean of all of the observed returns in the sample

    excess_returns = returns - np.ones((t, 1)) * mom_1
    mom_2 = np.sum(excess_returns**2, axis=0) / mom_0
    mom_3 = np.sum(excess_returns**3, axis=0) / mom_0
    mom_4 = np.sum(excess_returns**4, axis=0) / mom_0

    sigma = np.sqrt(mom_2)  # St. Dev. Estimate
    skew = mom_3 / (sigma**3)  # Skewness estimat
    kurt = (mom_4 / (sigma**4)) - 3  # Kurtosis estimate

    return (mom_1, sigma, skew, kurt)


def PRIIPS_stats(returns, holding_period=5, periods_in_year=256):
    """
    Use a sample of returns to calculate the PRIIPS statistics for a category
    2 fund. The returns should be geometric returns - ln(p_t+1/p_t)

    returns - n x p array of geometric returns
                         (n = number of time periods, p = number of funds)
    holding_period = recommended holding period in years
    periods_in_year = number of periods in year
                          (12 = monthly data, 52 = weekly, 256 = daily)

    The function first calculates the sample moments and then uses the
    Cornish_Fisher_percentile function to convert that to a set of performances
    over the relevant holding period

    The function returns:
     - 10th Percentile - Unfavourable Scenario Outcome
     - 50th Percentile - Moderate Scenario Outcome
     - 90th Percentile - Favourable Scenario Outcome
     - VaR equivalent volatility
     - Market Risk Class

    The performance outcome shows the value of 1 Euro invested in the fund in
    each of the three scenarios.
    This function does not currently calculate the stressed outcomes. Those
    require a rolling volatility
    """
    if returns.shape[0] < 1:
        raise Exception("Insufficient data for calculation of PRIIPS statistics")
    mu, sigma, skew, kurt = calc_moments(returns)

    sigma_stress, stress_val = get_stress_outcome(
        returns, holding_period, periods_in_year, False, skew, kurt
    )

    def local_cf_pctile(x, y, z):
        return Cornish_Fisher_percentile(
            x, y, skew, kurt, holding_period, z, periods_in_year
        )

    vol = convert_VaR_to_volatility(
        local_cf_pctile(0.0, sigma, 0.025), 0.025, holding_period
    )

    return (
        np.exp(local_cf_pctile(mu, sigma, 0.1)),
        np.exp(local_cf_pctile(mu, sigma, 0.5)),
        np.exp(local_cf_pctile(mu, sigma, 0.9)),
        vol,
        np.vectorize(volatility_to_MRM_class)(vol),
        sigma_stress,
        np.exp(stress_val),
    )


def get_stress_outcome(
    returns,
    holding_period: int,
    periods_in_year: int,
    use_2020: bool,
    skew: float,
    kurt: float,
):
    sigma_stress = calc_sigma_stress(returns, periods_in_year, holding_period, use_2020)
    # Stress scenario percentile under the old regulations
    # Annex IV Article 10 Sub D prior to January 1st 2023
    # see, e.g. https://www.handbook.fca.org.uk/techstandards/PRIIPs/2017/reg_del_2017_653_oj/annex04.html?date=2021-01-01
    stress_pctile = 0.1
    if use_2020:
        # Stress scenario percentile under the new regulations
        # Annex IV Article 18 Sub D as of January 1st 2023
        # see, e.g.
        stress_pctile = 0.05
    if holding_period < 1.0:
        stress_pctile = 0.01
    stress_val = Cornish_Fisher_percentile(
        0.0, sigma_stress, skew, kurt, holding_period, stress_pctile, periods_in_year
    )
    return sigma_stress, stress_val


def PRIIPS_stats_2020(returns, holding_period=5, periods_in_year=256):
    """
    This is the revised PRIIPS methodology presented on page 22 in
    https://www.esma.europa.eu/sites/default/files/library/jc_2020_66_final_report_on_draft_rts_to_amend_the_priips_kid.pdf

    The function returns:
     - Unfavourable Scenario Outcome - worst performance over holding period
     - Moderate Scenario Outcome - average performance over holding period
     - Favourable Scenario Outcome - best performance over holding period
     - VaR equivalent volatility
     - Market Risk Class

     Note that the stress performance percentile has changed
    """
    if returns.shape[0] < 1:
        raise Exception("Insufficient data for calculation of PRIIPS statistics")

    mu, sigma, skew, kurt = calc_moments(returns)
    sigma_stress, stress_val = get_stress_outcome(
        returns, holding_period, periods_in_year, True, skew, kurt
    )

    def local_cf_pctile(x, y, z):
        return Cornish_Fisher_percentile(
            x, y, skew, kurt, holding_period, z, periods_in_year
        )

    vol = convert_VaR_to_volatility(
        local_cf_pctile(0.0, sigma, 0.025), 0.025, holding_period
    )

    returns_df = pd.DataFrame(returns)

    # Define the rolling window size (5 years in this case, assuming 252 trading days per year)
    # This assumption is actually pretty dubious because there are actually fewer trading days
    # in a year
    # here is a good description of the new
    # https://www.deloitte.com/lu/en/Industries/investment-management/blogs/priips-rts-calculation-methodology-for-performance-scenarios.html
    # here are the actual rules https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02021R2268-20220714&from=EN 
    # Q&A at https://www.esma.europa.eu/sites/default/files/library/jc_2017_49_jc_priips_qa.pdf
    window_size = holding_period * periods_in_year

    # within the time period specified in point 6 of this annex, identification 
    # of all overlapping sub intervals individually equal in length to the duration 
    # of the recommended holding period, and which start or end in each of the months, or
    # at each of the valuation dates fro PRIIPS with a monthly valuation frequency, which
    # are contained within that period.
    # So, this seems terribly ambiguous - we can take the rolling 5 year periods (after all,
    # they start or end in each of the monehts)
    # Alternatively, we can move forward on a monthly basis and only take the month end
    # So, I'm left with a question - do the periods include a rolling daily 5 year period
    # or do we only sample that rolling daily period on a monthly basis
    # let's suppose we have a 10 year sample starting 1st Jan 2013
    # do we take 
    # 1st Jan 2013 - 1st Jan 2018, 2nd Jan 2013 - 2nd Jan 2018, 3rd Jan 2013 - 3rd Jan 2018,... (sampled daily)
    # or
    # 1st Jan 2013 - 1st Jan 2018, 1st Feb 2013 - 1st Feb 2018,  1st March 2013 - 1st March 2018,... (sampled monthly)
    # https://www.esma.europa.eu/sites/default/files/library/jc_2017_49_jc_priips_qa.pdf p34 clarifies that this 
    # is meant to be monthly  

    rolling_sum = returns_df.rolling(window=window_size).sum()

    # Calculate the worst performance within each rolling window
    # reshape to make same shape as older method
    worst_performance = np.exp(rolling_sum.min().values).reshape(1, -1)
    best_performance = np.exp(rolling_sum.max().values).reshape(1, -1)

    return (
        worst_performance,
        np.exp(mu * holding_period * periods_in_year),
        best_performance,
        vol,
        np.vectorize(volatility_to_MRM_class)(vol),
        sigma_stress,
        np.exp(stress_val),
    )


def PRIIPS_stats_array(
    Z,
    column_names,
    holding_period,
    periods_in_year,
    use_new: bool = False,
    sample_name=None,
) -> pd.DataFrame:
    """
    Converts the results of a call to a function that returns PRIIPS statistics
    to a dataframe that can be used in the resultant analysis
    """
    if use_new:
        s = PRIIPS_stats_2020(Z / 100.0, holding_period, periods_in_year)
    else:
        s = PRIIPS_stats(Z / 100.0, holding_period, periods_in_year)

    dict_results = {
        "Identifier": column_names,
        "Unfavourable": np.squeeze(s[0]),
        "Moderate": np.squeeze(s[1]),
        "Favourable": np.squeeze(s[2]),
        "VaREquivalentVolatility": np.squeeze(s[3]),
        "SummaryRiskIndicator": np.squeeze(s[4]),
        "StressVolatility": np.squeeze(s[5]),
        "StressOutcome": np.squeeze(s[6]),
    }

    if sample_name is not None:
        dict_results["Sample"] = [sample_name] * Z.shape[1]

    return pd.DataFrame(dict_results)


def PRIIPS_stats_df(
    sample_df: pd.DataFrame,
    holding_period=5,
    periods_in_year=256,
    sample_name=None,
    use_new: bool = False,
    calc_stress: bool = False,
    index_field="Index",
    date_field="Date",
    return_field="LogReturn",
) -> pd.DataFrame:
    """
    Returns the PRIIPS stats in an easy-to-use dataframe
    """
    pivoted_df = sample_df.pivot(
        index=date_field, columns=index_field, values=return_field
    )
    Y = pivoted_df.values
    mask = np.logical_not(np.any(np.isnan(Y), axis=1))
    Z = Y[mask, :]  # only use columns of Y that don't contain any NaNs

    return PRIIPS_stats_array(
        Z, pivoted_df.columns, holding_period, periods_in_year, use_new, sample_name
    )


def PRIIPS_stats_bootstrap(
    sample_df: pd.DataFrame,
    holding_period=5,
    periods_in_year=256,
    nbs=1000,
    use_new=False,
    index_field="Index",
    date_field="Date",
    return_field="LogReturn",
) -> pd.DataFrame:
    """
    Performs a bootstrap and
    returns the PRIIPS stats in an easy-to-use dataframe
    SHOULD I GET RID OF THIS
    """
    pivoted_df = sample_df.pivot(
        index=date_field, columns=index_field, values=return_field
    )
    Y = pivoted_df.values
    mask = np.logical_not(np.any(np.isnan(Y), axis=1))
    Z = Y[mask, :]  # only use columns of Y that don't contain any NaNs
    boot_fn = lambda x: PRIIPS_stats_array(
        x, pivoted_df.columns, holding_period, periods_in_year, use_new, None
    )

    return parallel_bootstrap(Z, boot_fn, nbs, include_sample=False)

"""
Routine to do a parallel bootstrap
"""

from multiprocessing.pool import ThreadPool
import time
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import rv_continuous
import pandas as pd
import numpy as np


def _array_slice_sample(a, sample_idx, axis=0):
    """
    returns a sample along the axis chosen
    mostly inspired by
    https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
    """
    return a[(slice(None),) * (axis % a.ndim) + (sample_idx,)]


def _bootstrap_sample(ret_data, trial_name, fn):
    """
    wrapper function for supplied bootstrap
    """
    x = fn(ret_data)
    x["trial"] = trial_name
    return x


def _print_progress(display_progress, msg_str):
    if display_progress:
        print(msg_str)


def parallel_bootstrap(
    ret_data,
    fn,
    nbs: int = 1000,
    include_sample: bool = True,
    threads: int = 16,
    nskip: int = 100,
    axis=0,
    display_progress=True,
    sample_size=None,
) -> pd.DataFrame:
    """
    Quick and dirty code to bootstrap a function that returns a
    dictionary of results. The function should return a
    dictionary that includes a field called 'trial'
    see https://stackoverflow.com/questions/352670/weighted-random-selection-with-and-without-replacement#353576
    - arguments
    - ret_data = array of data supplied for function
    - fn = function that operates on an array of data and returns
           a dictionary of values or a dataframe
    - include_sample = return the function value for the sample
    - threads = number of threads to use
    - nskip = report every time nskip threads complete
    - axis = axis to use for random slicing if multi-dimensional array
    - sample_size = number of samples to use in the boostrap sub-sample
    """
    start = time.time()
    lst_bs = []
    lst_results = []
    n = ret_data.shape[axis]
    samp_size = n
    if sample_size is not None:
        samp_size = sample_size
    if include_sample:
        dict_opt = fn(ret_data)
        dict_opt["trial"] = "Sample Optimal"
        lst_bs.append(dict_opt)

    pool = ThreadPool(processes=threads)
    _print_progress(display_progress, "Starting bootstrap threads")

    for i in range(nbs):
        # Create BS pointers into dataset with size n
        sample_idx = np.random.choice(range(n), size=samp_size, replace=True)
        ret_sample = _array_slice_sample(ret_data, sample_idx, axis)
        lst_results.append(
            pool.apply_async(_bootstrap_sample, (ret_sample, f"Simulation {i+1}", fn))
        )

    elapsed_time = time.time() - start
    _print_progress(
        display_progress, f"Started bootstrap iterations after {elapsed_time:.2f} s"
    )
    cur_time = time.time()
    i = 0
    concat = False
    for result in lst_results:
        samp_result = result.get()
        if isinstance(samp_result, pd.DataFrame):
            concat = True
        lst_bs.append(samp_result)
        i += 1
        if (i % nskip == 0) & (i > 0):
            elapsed_time = time.time() - cur_time
            _print_progress(
                display_progress,
                f"Completed {i}/{nbs} iterations ({elapsed_time:.2f} s)",
            )
            cur_time = time.time()

    _print_progress(
        display_progress, f"Bootstrap Completed = {(time.time()-start):.2f} s"
    )
    df_out = []
    if concat:
        df_out = pd.concat(lst_bs)
    else:
        df_out = pd.DataFrame(lst_bs)
    return df_out


def rolling_backtest_date_function(
    df_history: pd.DataFrame,
    back_fn,
    rolling_window_years: float = 10.0,
    rolling_start_years: float = 8.0,
    sample_freq=relativedelta(days=1),
    skip_periods: int = 500,
) -> pd.DataFrame:
    """
    backtest function that calculates rolling daily statistics
    df_history = DataFrame containing daily returns data for the funds
    back_fn = function using start and end date that returns a dataframe of results
    rolling_window_years = size of the data sample in years
    rolling_start_years  = first date to calculate the PRIIPS performance stats
    """

    # Find the latest date in the DataFrame
    latest_date = df_history["Date"].max()

    # Set the initial rolling window end date to the latest date
    rolling_window_end = latest_date

    # Define the number of rolling windows you want (e.g., 5 years with 5-year gaps)

    start_date = latest_date - relativedelta(years=rolling_start_years)
    lst_stats = []
    print("Running Backtest")
    print(f"Start Date = {start_date:%Y-%m-%d}")
    print(f"End Date   = {latest_date:%Y-%m-%d}")
    day_count = 0
    start_time = time.time()
    while start_date <= latest_date:

        # Calculate the start and end dates for the rolling window
        # Wrap in try catch block to handle leap years
        try:
            rolling_window_start = start_date - relativedelta(
                years=rolling_window_years
            )  # 10 years minus leap year days
        except Exception as e:
            temp_date = start_date - timedelta(days=1.0)
            rolling_window_start = temp_date - relativedelta(
                years=rolling_window_years
            )  # 10 years minus leap year days

        rolling_window_end = start_date

        # Append the subset to the rolling_subsets list
        df_stats = back_fn(rolling_window_start, rolling_window_end)

        df_stats["Date"] = start_date
        lst_stats.append(df_stats)

        # Move the rolling window back by 5 years for the next iteration
        day_count += 1
        if day_count % skip_periods == 0:
            print(f"Completed {day_count:<6d} periods - {start_date:%Y-%m-%d}")
        start_date += sample_freq  # 1 DAY

    # Your rolling_subset DataFrame will now contain data for each rolling window
    df = pd.concat(lst_stats)
    run_time = time.time() - start_time
    print(f"No of Periods = {day_count:<6d}")
    print(f"Runtime (sec)    = {run_time:<11.2f}")
    return df


def _backtest_fn(df: pd.DataFrame, start_date, end_date, back_fn):
    """
    Used to create a subset of data with the correct timeframe and
    pass that data to the calculation function
    back_fn - function that takes a dataframe and returns a dataframe
    start_date - start of rolling window frame
    end_date - end of rolling window frame
    df - dataframe with date field that will be filtered using the
    window periods
    """
    df_subset = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Append the subset to the rolling_subsets list
    df_stats = back_fn(df_subset)
    return df_stats


def rolling_backtest(
    df_history: pd.DataFrame,
    back_fn,
    rolling_window_years: float = 10.0,
    rolling_start_years: float = 8.0,
    sample_freq=relativedelta(days=1),
    skip_periods: int = 500,
) -> pd.DataFrame:
    """
    backtest function that calculates rolling daily statistics
    df_history = DataFrame containing daily returns data for the funds
    back_fn = function using start and end date that returns a dataframe of results
    rolling_window_years = size of the data sample in years
    rolling_start_years  = first date to calculate the PRIIPS performance stats
    sample_freq = frequency of the sample
    skip_periods = number of periods to skip before printing progress
    """

    def sample_fn(x, y):
        return _backtest_fn(df_history, x, y, back_fn)

    return rolling_backtest_date_function(
        df_history,
        sample_fn,
        rolling_window_years=rolling_window_years,
        rolling_start_years=rolling_start_years,
        sample_freq=sample_freq,
        skip_periods=skip_periods,
    )


def _basestats(self):
    # Return mean, variance, skewness, and kurtosis analytically
    mean = self._mean()
    variance = self._var()
    # Johnson SU skew and kurtosis are complex; you can use numerical methods
    skew = None
    kurt = None
    return mean, variance, skew, kurt


def edf_stats(x, dist: rv_continuous):
    """
    Empirical Distribution Function statistics for distribution testing.
    Returns a variety of statistics comparing the empirical distribution
    function with the theoretical distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data.
    dist : rv continuos derived class

    Returns
    -------
    result :
    A2 : Anderson-Darling Statistic
    W2: Cramer-Von Mises Statistic
    K: Kolgomorov-Smirnov Statistic
    U2: Watson Statistic
    V: Kuiper Statistic

    See Also
    --------
    kstest : The Kolmogorov-Smirnov test for goodness-of-fit.

    Notes
    -----
    No critical values provided currently
    Examples
    --------
    Test the null hypothesis that a random sample was drawn from a particular
    distribution (with unspecified mean and standard deviation).

    Note should fit the distribution and calculate the critical values
    accordingly


    """  # numpy/numpydoc#87  # noqa: E501
    y = np.sort(x)
    n = len(y)
    cdf = dist.cdf(y)
    eps = 1e-15
    logcdf = np.log(np.maximum(cdf, eps))
    cdf_mean = np.sum(cdf) / n
    logsf = np.log(np.maximum(1 - cdf, eps))  # log of survival function
    i = np.arange(1, n + 1)
    a_squared = -n - np.sum(
        (2 * i - 1.0) / n * (logcdf + logsf[::-1]), axis=0
    )  # Anderson-Darling
    w_squared = np.sum((cdf - ((2 * i - 1.0) / (2 * n))) ** 2) + (
        1 / (12 * n)
    )  # Cramer-Von Mises
    u_squared = w_squared - n * (cdf_mean - 0.5) ** 2  # Watson

    # Kolmogorov-Smirnov style stats
    d_plus = np.max(((i / n) - cdf))
    d_minus = np.max((cdf - ((i - 1) / n)))

    kuiper_v = d_plus + d_minus  # Kuiper
    k_s_stat = max(d_plus, d_minus)  # Kolgomorov-Smirnov
    return a_squared, w_squared, k_s_stat, u_squared, kuiper_v


def reject_block_sample(dist, gen_dist, size=None, random_state=None, max_ratio=2.7, max_samples=100):
    """
    Use the rejection sampling method to generate samples from a
    distribution that is difficult to sample from directly.

    dist = scipy.stats distribution object
    gen_dist = scipy.stats distribution object used to generate samples
    size = number of samples to generate
    random_state = random number generator
    max_ratio = maximum ratio of p(x)/q(x) for all x
    """

    def pdf_ratio_fn(x):
        return -dist._pdf(x) / gen_dist.pdf(x)

    M = 1.1 * max_ratio  # Safety margin factor (should be >= max(p(x)/q(x)))
    # find max p/q
    samples = np.array([])
    tot_size = np.prod(size)
    count = 0
    max_pq = 1.0
    generated_samples = 0
    tot_samples = 0
    while (generated_samples < tot_size) & (tot_samples < max_samples*tot_size):

        # Sample from proposal distribution
        # must use size=1 to get a scalarS

        num_to_generate = tot_size - generated_samples
        tot_samples = tot_samples + num_to_generate

        x = gen_dist.rvs(size=num_to_generate, random_state=random_state)
        u = random_state.rand(num_to_generate)

        pdf_ratio = -pdf_ratio_fn(x)
        # Compute acceptance probability
        accept_prob = pdf_ratio / M

        if np.any(accept_prob > 1):
            print(f"x = {x[accept_prob > 1]}, pdf_ratio = {pdf_ratio[accept_prob > 1]}, M={M}")
            count = count + len(x[accept_prob > 1])

        if np.any(pdf_ratio > max_pq):
            max_pq = np.max(pdf_ratio)

        samples = np.concatenate((samples, x[u < accept_prob]))
        generated_samples = len(samples)
    if tot_samples > max_samples*tot_size:
        print(f"Exceeded maximum number of samples = {max_samples}")
    if count > 0:
        print("Value of M set too low in rejection block sampling routine")
    return samples

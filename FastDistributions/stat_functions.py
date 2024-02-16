"""
Routine to do a parallel bootstrap
"""
from multiprocessing.pool import ThreadPool
import datetime
from datetime import timedelta
import time
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
    x['trial'] = trial_name
    return x

def parallel_bootstrap(ret_data, fn, nbs:int=1000,
                       include_sample:bool=True, threads:int=16,
                       nskip:int=100, axis=0)-> pd.DataFrame:
    """
    Quick and dirty code to bootstrap a function that returns a 
    dictionary of results. The function should return a 
    dictionary that includes a field called 'trial'
    see https://stackoverflow.com/questions/352670/weighted-random-selection-with-and-without-replacement#353576
    - arguments
    - ret_data = array of data supplied for function
    - fn = function that operates on an array of data and returns a dictionary of values
    - include_sample = return the function value for the sample
    - threads = number of threads to use
    - nskip = report every time nskip threads complete
    - axis = axis to use for random slicing if multi-dimensional array
    """
    start = time.time()
    lst_bs = []
    lst_results = []
    n = ret_data.shape[axis]
    if include_sample:
        dict_opt = fn(ret_data)
        dict_opt['trial'] = 'Sample Optimal'
        lst_bs.append(dict_opt)

    pool = ThreadPool(processes=threads)
    print('Starting bootstrap threads')
    for i in range(nbs):
        # Create BS pointers into dataset with size n
        sample_idx = np.random.choice(range(n), size=n, replace=True)
        ret_sample = _array_slice_sample(ret_data, sample_idx, axis)
        lst_results.append(pool.apply_async(_bootstrap_sample,
                           (ret_sample, f'Simulation {i+1}', fn)))

    elapsed_time = time.time() - start
    print(f'Started bootstrap iterations after {elapsed_time:.2f} s')
    cur_time = time.time()
    i = 0
    concat=False
    for result in lst_results:
        samp_result = result.get()
        if isinstance(samp_result, pd.DataFrame):
            concat=True
        lst_bs.append(samp_result)
        i += 1
        if (i % nskip == 0) & (i>0):
            elapsed_time = time.time() - cur_time
            print(f'Completed {i}/{nbs} iterations ({elapsed_time:.2f} s)')
            cur_time = time.time()

    print(f'Bootstrap Completed = {(time.time()-start):.2f} s')
    df_out = []
    if concat:
        df_out = pd.concat(lst_bs)
    else:
        df_out = pd.DataFrame(lst_bs)
    return df_out



def rolling_backtest(df_history, back_fn,
                     rolling_window_years=10,
                     rolling_start_years=8):
    """
    backtest function that calculates rolling statistics
    df_history = DataFrame containing returns data for the funds
    back_fn = function that calculates the PRIIPS performance stats
    rolling_window_years = size of the data sample in years
    holding_period = recommended holding period used for calculation
    rolling_start_years  = first date to calculate the PRIIPS performance stats
    """
    NUM_DAYS = 500
# Find the latest date in the DataFrame
    latest_date = df_history['Date'].max()

# Set the initial rolling window end date to the latest date
    rolling_window_end = latest_date

# Create an empty DataFrame to store the rolling subset
    rolling_subset = pd.DataFrame(columns=['Index', 'Date', 'LogReturn'])

# Define the number of rolling windows you want (e.g., 5 years with 5-year gaps)

    start_date = latest_date.replace(year=latest_date.year - rolling_start_years)
    lst_stats = []
    print('Running Backtest')
    print('Start Date = {0:%Y-%m-%d}'.format(start_date))
    print('End Date   = {0:%Y-%m-%d}'.format(latest_date))
    day_count = 0
    start_time = time.time()
    while start_date <= latest_date:

    # Calculate the start and end dates for the rolling window
    # Wrap in try catch block to handle leap years
        try:
            rolling_window_start = start_date.replace(year=start_date.year - rolling_window_years) # 10 years minus leap year days
        except:
            temp_date = start_date - timedelta(days=1.0)
            rolling_window_start = temp_date.replace(year=temp_date.year - rolling_window_years) # 10 years minus leap year days

        rolling_window_end = start_date

    # Create a subset of data within the rolling window
        df_subset = df_history[(df_history['Date'] >= rolling_window_start) &
                                     (df_history['Date'] <= rolling_window_end)]

    # Append the subset to the rolling_subsets list
        df_stats = _bootstrap_sample(df_subset, start_date, back_fn)
        df_stats['Date'] = start_date
        lst_stats.append(df_stats)

    # Move the rolling window back by 5 years for the next iteration
        day_count += 1
        if day_count % NUM_DAYS == 0:
             print('Completed {0:<6d} days - {1:%Y-%m-%d}'.format(day_count, start_date))
        start_date += timedelta(days=1)  # 1 DAY

# Your rolling_subset DataFrame will now contain data for each rolling window
    df = pd.concat(lst_stats)
    run_time = time.time() - start_time
    print('No of Days = {:<6d}'.format(day_count))
    print('Runtime (sec)    = {:<11.2f}'.format(run_time))
    return df

"""
Routine to do a parallel bootstrap
"""
from multiprocessing.pool import ThreadPool
import time
import pandas as pd
import numpy as np

def _bootstrap_sample(ret_data, trial_name, fn):
    """
    wrapper function for supplied bootstrap
    """
    x = fn(ret_data)
    x['trial'] = trial_name
    return x

def parallel_bootstrap(ret_data, fn, nbs:int=1000,
                       include_sample:bool=True, threads:int=16,
                       nskip:int=100)-> pd.DataFrame:
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
    """
    start = time.time()
    lst_bs = []
    lst_results = []
    n = ret_data.shape[0]
    if include_sample:
        dict_opt = fn(ret_data)
        dict_opt['trial'] = 'Sample Optimal'
        lst_bs.append(dict_opt)

    pool = ThreadPool(processes=threads)
    print('Starting bootstrap threads')
    for i in range(nbs):
        # Create BS pointers into dataset with size n
        sample_idx = np.random.choice(range(n), size=n, replace=True)
        ret_sample = ret_data[sample_idx]
        lst_results.append(pool.apply_async(_bootstrap_sample,
                           (ret_sample, f'Simulation {i+1}', fn)))

    elapsed_time = time.time() - start
    print(f'Started bootstrap iterations after {elapsed_time:.2f} s')
    cur_time = time.time()
    i = 0
    for result in lst_results:
        dict_results = result.get()
        lst_bs.append(dict_results)
        i += 1
        if (i % nskip == 0) & (i>0):
            elapsed_time = time.time() - cur_time
            print(f'Completed {i}/{nbs} iterations ({elapsed_time:.2f} s)')
            cur_time = time.time()

    print(f'Bootstrap Completed = {(time.time()-start):.2f} s')
    return pd.DataFrame(lst_bs)

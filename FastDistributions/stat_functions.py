"""
Routine to do a parallel bootstrap
"""
from multiprocessing.pool import ThreadPool
import pandas as pd
import numpy as np
import time

def _bootstrap_sample(ret_data, trial_name, fn):
     x = fn(ret_data)
     x['trial'] = trial_name
     return x

def parallel_bootstrap(ret_data, fn, nbs=1000, include_sample=True, threads=16):
    """
     Quick and dirty code to bootstrap a function that returns a dictionary
     of results. The function should return a dictionary that includes a 
     field called 'Trial'
    """
    start = time.time()
    lst_bs = []
    lst_results = []
    n = ret_data.shape[0]
    if include_sample:
        x_opt = fn(ret_data)
        x_opt['trial'] = 'Sample Optimal'
        lst_bs.append(x_opt)

    pool = ThreadPool(processes=threads)
    print('Starting bootstrap threads')
    for i in range(nbs):
        pb = np.random.choice(range(n), size=n, replace=True) # Create BS pointers into dataset with size n
        ret_sample = ret_data[pb]
        lst_results.append(pool.apply_async(_bootstrap_sample, (ret_sample, 'Simulation {0}'.format(i+1), fn)))
    
    elapsed_time = time.time() - start
    print(f'Started bootstrap iterations after {elapsed_time} s') 
    
    i = 0
    for result in lst_results:
        dict_results = result.get()
        lst_bs.append(dict_results)
        i += 1
        if (i % 100 == 0) & (i>0):
            elapsed_time = time.time() - start
            print(f'Completed {i}th iterations ({elapsed_time} s)')
    
    print('Bootstrap Completed = {0} s'.format(time.time()-start))
    return pd.DataFrame(lst_bs)
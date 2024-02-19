import FastDistributions as fd
from scipy.stats import norm

def norm_fn(ret_data):
    """
    Used for testing the backtest function
    """
    norm_fit = norm.fit(ret_data)
    norm_ll = np.sum(norm(norm_fit[0], norm_fit[1]).logpdf(ret_data))
    x = {
         'norm_loc' : norm_fit[0],
         'norm_scale' : norm_fit[1],
         'norm_ll'    : norm_ll
        }
    return x

def test_bootstrap():
   lst_indices = [('^GSPC', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')]
   df_ret = fd.download_yahoo_returns(lst_indices)
   sp_ret = df_ret[df_ret.Ticker=='^GSPC']['LogReturn'].values
   df_bs = fd.parallel_bootstrap(sp_ret, norm_fn, nskip=50, nbs=1000)
   assert(len(df_bs.index)==1000)
   return


import FastDistributions as fd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import levy_stable, norm

# Set matplotlib to plot the way I like
sns.set_theme() # use the sns theme for plotting - just more attractive!
#plt.rcParams['figure.dpi'] = 360
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['ytick.labelsize'] = 8.0
plt.rcParams['xtick.labelsize'] = 8.0
plt.rcParams['legend.fontsize'] = 8.0
plt.rcParams["font.family"] = 'sans-serif' #'Humor Sans' #

def boot_fn(ret_data):
    norm_fit = norm.fit(ret_data)
    norm_ll = np.sum(norm(norm_fit[0], norm_fit[1]).logpdf(ret_data))
    gsfit = fd.GeneralisedSkewT.fit(ret_data, display_progress=False)
    gs = fd.GeneralisedSkewT(gsfit[0], gsfit[1], gsfit[2], gsfit[3], gsfit[4])
    gs_ll = np.sum(gs.logpdf(ret_data))
    lsfit = fd.LevyStableInterp.fit(ret_data, display_progress=False)
    ls = levy_stable(lsfit[0], lsfit[1], lsfit[2], lsfit[3])
    ls_ll = np.sum(ls.logpdf(ret_data))
    x = {
         'norm_loc'   : norm_fit[0],
         'norm_scale' : norm_fit[1],
         'norm_ll'    : norm_ll,
         'gs_loc'     : gsfit[3],
         'gs_scale'   : gsfit[4],
         'gs_lambda'  :  gsfit[0],
         'gs_k'       : gsfit[1],
         'gs_n'       : gsfit[2],
         'gs_ll'      : gs_ll,
         'ls_alpha'   : lsfit[0],
         'ls_beta'    : lsfit[1],
         'ls_loc'     : lsfit[2],
         'ls_scale'   : lsfit[3],
         'ls_ll'      : ls_ll
        }
    lst_pctiles =  [0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9]
    data_pctiles = np.percentile(ret_data, lst_pctiles)
    for pctile, pct_val in zip(lst_pctiles, data_pctiles):
        x[f'{pctile}th_pctile'] = pct_val

    gs_pctiles = gs.ppf(np.array(lst_pctiles)/100.0)
    for pctile, gs_pctile in zip(lst_pctiles, list(gs_pctiles)):
        x[f'gs_{pctile}th_pctile'] = gs_pctile

    ls_pctiles = ls.ppf(np.array(lst_pctiles)/100.0)
    for pctile, ls_pctile in zip(lst_pctiles, list(ls_pctiles)):
        x[f'ls_{pctile}th_pctile'] = ls_pctile

    return x

def norm_fn(ret_data):
    norm_fit = norm.fit(ret_data)
    norm_ll = np.sum(norm(norm_fit[0], norm_fit[1]).logpdf(ret_data))
    x = {
         'norm_loc' : norm_fit[0],
         'norm_scale' : norm_fit[1],
         'norm_ll'    : norm_ll
        }
    return x

lst_indices = [('^GSPC', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')]
df_ret = fd.download_yahoo_returns(lst_indices)

sp_ret = df_ret[df_ret.Ticker=='^GSPC']['LogReturn'].values

df_bs = fd.parallel_bootstrap(sp_ret, norm_fn, nskip=50, nbs=1000)
df_bs.to_csv('full_bootstrap.csv')


print('Finished testing')
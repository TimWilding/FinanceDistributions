import FastDistributions as fd
from matplotlib import pyplot as plt
import seaborn as sns


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

def gs_fn(ret_data):
    gsfit = fd.GeneralisedSkewT.fit(ret_data, display_progress=False)
    lsfit = fd.LevyStableInterp.fit(ret_data, display_progress=False)
    x = {
         'gs_loc' : gsfit[3],
         'gs_scale' : gsfit[4],
         'gs_lambda' : gsfit[0],
         'gs_k' : gsfit[1],
         'gs_n' : gsfit[2],
         'ls_alpha' : lsfit[0],
         'ls_beta'  : lsfit[1],
         'ls_loc'   : lsfit[2],
         'ls_scale' : lsfit[3]
        }
    return x


lst_indices = [('^GSPC', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')]
df_ret = fd.download_yahoo_returns(lst_indices)

sp_ret = df_ret[df_ret.Ticker=='^GSPC']['LogReturn'].values

df_bs = fd.parallel_bootstrap(sp_ret, gs_fn, nskip=50, nbs=1000)
df_bs.to_csv('gsd_bootstrap.csv')


print('Finished testing')
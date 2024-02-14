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


# alpha = stability parameter (0, 2]
#beta = skewness parameter [-1, 1]
#ls = levy_stable(2.0, 0.0, loc=0, scale=1.0 )
#ls_new = levy_stable(1.5, -0.15, loc=0, scale=1.0 ) # alpha, beta close to FTSE 100
#plot_function(lambda x : ls.pdf(x), title='Levy Stable', fn_2=lambda x : ls_new.pdf(x))
gsd = fd.GeneralisedSkewT(0, 1.0, 0.2, 1, 1000)
gsd_skew = fd.GeneralisedSkewT(0,1.0, 0.2, 2.0, 1000)
print(gsd.pdf(-1.0))
lst_indices = [('^GSPC', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')]
df_ret = fd.download_yahoo_returns(lst_indices)

sp_ret = df_ret[df_ret.Ticker=='^GSPC']['LogReturn'].values

ls = fd.LevyStableInterp.fit(sp_ret)
gs = fd.GeneralisedSkewT.fit(sp_ret)
#plot_function(lambda x : gsd.pdf(x), title='Skewed Lapace', fn_2=lambda x : gsd_skew.pdf(x))
dict_pdf = {
            'Levy-Stable' : [ls.pdf, 'b-'],
            'Generalised SkewT' : [gs.pdf, 'k-']
            }
fd.plot_hist_fit(sp_ret, 'SP 500', dict_pdf, 50, log_scale=True)
dict_ppf = {
            'Levy-Stable' : [ls.ppf, 'bo'],
            'Generalised SkewT' : [gs.ppf, 'go']
            }
fd.plot_qq(sp_ret, 'SP500 Returns', dict_ppf, nbins=500)

print('Finished testing')
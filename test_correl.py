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
BACKTEST_YEARS = 15 # Number of years to use for backtest
WINDOW_YEARS = 2 # Number of years to use for calculation of correlation
# Indices to use for backtest - adding VWRA.L changes the dataframe and breaks everything! These aren't total returns
# but I don't "think" that should significantly impact correlations.

def calc_correl(df):
    """
    given a dataframe with columns - Identifier, Date, and LogReturn
    calculate a correlation matrix for each of the identifiers.
    Return the lower-diagonal elements of the correlation matrix
    """

    pivot_df = df.pivot(index='Date', columns='Name', values='LogReturn')
    pivot_df = pivot_df.dropna()
    correlation_df = pivot_df.corr()
    lower_triangle = correlation_df.where(np.tril(np.ones(correlation_df.shape), k=-1).astype(bool))
    lower_triangle = lower_triangle.rename_axis('Names').stack().reset_index()
    lower_triangle.columns = ['Name1', 'Name2', 'correlation']
    return lower_triangle.sort_values(by=['Name1', 'Name2'], ascending=[True, True])

lst_indices = [('^GSPC', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')] #, ('VWRA.L', 'Vanguard FTSE All-World')
#lst_indices = [('^SP500TR', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')] #, ('VWRA.L', 'Vanguard FTSE All-World')
df_prices = fd.download_yahoo_returns(lst_indices)

df_month = df_prices.groupby(['Name', 'EndOfMonth']).agg({'LogReturn':'sum'}).reset_index().rename({'EndOfMonth' : 'Date'})
df_month.columns = ['Name', 'Date', 'LogReturn']
df_roll_month_correls = fd.rolling_backtest(df_month, calc_correl, WINDOW_YEARS, BACKTEST_YEARS)

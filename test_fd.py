import datetime
import FastDistributions as fd
import numpy as np
from scipy.stats import norm
import pandas as pd
from pandas import read_csv


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
    nbs = 100
    lst_indices = [('^GSPC', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')]
    df_ret = fd.download_yahoo_returns(lst_indices)
    sp_ret = df_ret[df_ret.Ticker=='^GSPC']['LogReturn'].values
    df_bs = fd.parallel_bootstrap(sp_ret, norm_fn, nskip=10, nbs=nbs)
    assert(len(df_bs.index)==nbs+1)
    return

def test_priips():

    MIN_DATE_FILTER = datetime.datetime(2004, 1, 1) # beginning of sample 'Jan 2004' - Jan 7 results in similar
    df_price_history = fd.calculate_returns(read_csv('https://raw.githubusercontent.com/TimWilding/FinanceDataSet/main/PRIIPS_Data_Set.csv', parse_dates=[1]),
                                       'Index', 'Date')


    df_price_history = df_price_history[df_price_history.Date>=MIN_DATE_FILTER] # Note - this removes NaNs from initial price points in LogReturn column

    back_fn = lambda x : fd.PRIIPS_stats_df(x, use_new=False, holding_period=5)
    df_backtest_old = fd.rolling_backtest(df_price_history, back_fn, rolling_window_years=5, rolling_start_years=15)
    df_sample_5y = df_price_history[(df_price_history.Date>datetime.datetime(2018, 10, 27))
                                & (df_price_history.Date<=datetime.datetime(2023, 10, 27))]   
    df_bs = fd.PRIIPS_stats_bootstrap(df_sample_5y)
    print('Finished')
    return


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

def test_dists():
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

    ls = fd.LevyStableInterp.fitclass(sp_ret)
    gs = fd.GeneralisedSkewT.fitclass(sp_ret)
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
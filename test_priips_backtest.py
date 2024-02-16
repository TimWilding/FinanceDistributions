import datetime
import FastDistributions as fd
from datetime import timedelta
import pandas as pd
from pandas import read_csv
import numpy as np
import datetime
import seaborn as sns
import time
import matplotlib.pyplot as plt
# https://github.com/topics/free-fonts
sns.set_theme() # use the sns theme for plotting - just more attractive!
#plt.rcParams['figure.dpi'] = 360
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.labelsize'] = 9.0
plt.rcParams['ytick.labelsize'] = 9.0
plt.rcParams['xtick.labelsize'] = 9.0
plt.rcParams['legend.fontsize'] = 9.0
plt.rcParams["font.family"] = 'sans-serif' #'Humor Sans' #
# ['Liberation Mono', 'DejaVu Sans', 'STIXSizeFiveSym', 'cmmi10', 'cmb10', 'STIXSizeThreeSym', 'cmtt10', 'DejaVu Serif', 'STIXSizeFourSym', 'Liberation Serif', 'Liberation Sans Narrow', 'cmr10', 'cmsy10', 'STIXNonUnicode', 'DejaVu Serif Display', 'cmex10', 'STIXSizeOneSym', 'STIXSizeTwoSym', 'DejaVu Sans Mono', 'Humor Sans', 'Liberation Sans', 'STIXGeneral', 'cmss10', 'DejaVu Sans Display']



ANNUALISE = 256 # Number of trading days in the year for annualising = 52*5 Actual number of days often less
# For example, PRIIPS uses 256

# Total Period
MAX_DATE_FILTER = datetime.datetime(2023, 10, 27) # end of sample 'March 2022'
MIN_DATE_FILTER = datetime.datetime(2004, 1, 1) # beginning of sample 'Jan 2004' - Jan 7 results in similar

def plot_stats_col(df, col_name, axis_label, chart_title):
    """
    Plot each of the indices on chart from the statistics daata set
    """
    df_sort = df.sort_values(['Identifier', 'Date'])

# Plot the time series for each asset using Seaborn
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_sort, x='Date', y=col_name, hue='Identifier')
    plt.ylabel(axis_label)
    plt.title(chart_title)

# Display the plot
    plt.show()

date_ranges = {
                '2004-2010' : (datetime.datetime(2004, 1, 1), datetime.datetime(2010, 12, 31)),
                '2010-2016' : (datetime.datetime(2011, 1, 1), datetime.datetime(2016, 12, 31)),
                '2017-2023' : (datetime.datetime(2017, 1, 1), datetime.datetime(2023, 10, 27))
                }
# Read in the price history
df_price_history = fd.calculate_returns(read_csv('https://raw.githubusercontent.com/TimWilding/FinanceDataSet/main/PRIIPS_Data_Set.csv', parse_dates=[1]),
                                       'Index', 'Date')


df_price_history = df_price_history[df_price_history.Date>=MIN_DATE_FILTER] # Note - this removes NaNs from initial price points in LogReturn column

back_fn = lambda x : fd.PRIIPS_stats_df(x, use_new=False, holding_period=5)
df_backtest_old = fd.rolling_backtest(df_price_history, back_fn, rolling_window_years=5, rolling_start_years=15)
plot_stats_col(df_backtest_old, 'Favourable', 'Performance', 'Favourable (Orig)')

df_sample_5y = df_price_history[(df_price_history.Date>datetime.datetime(2018, 10, 27))
                                & (df_price_history.Date<=datetime.datetime(2023, 10, 27))]
df_bs = fd.PRIIPS_stats_bootstrap(df_sample_5y)
print('Finished')
from scipy.stats import levy_stable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_hist_fit(data, series_name, dict_fn, nbins=20,
                  log_scale=False, ylim=None):
    """
    Plots a histogram of some data and then plots several functions over it.
    Those functions are typically distributions fitted to the data such as the
    t-distribution
    """
# Plot the histogram
    plt.hist(data, bins=nbins, density=True, alpha=0.7, color='blue', label='Histogram')
 
    x = np.linspace(data.min(), data.max(), 500)
    y_max = 0.0
    y_min = 10000
    for fn_name in dict_fn:
        f_pdf = dict_fn[fn_name][0](x)
        if np.max(f_pdf) > y_max:
            y_max = np.max(f_pdf)
        if np.min(f_pdf) < y_min:
            y_min = np.min(f_pdf)
        plt.plot(x, f_pdf, dict_fn[fn_name][1], label=fn_name)


# Add labels and legend
    plt.xlabel('Log-Return')
    plt.ylabel('Probability Density')
    plt.title(f'{series_name} Histogram')
    plt.legend()
    if log_scale:
        plt.yscale('log')
        if ylim is None:
            plt.ylim([0.8*y_min, 1.2*y_max])
        else:
            plt.ylim(ylim)

# Show the plot
    plt.show()

def plot_function(fn, x_lim=[-6, 6], y_lim=None, n=10000, title='Function', fn_2=None):
    """
    Takes a function and produces a graph using the limits
    given
    fn = univariate function
    x_lim = min and max of function range
    y_lim = min and max of function values (should I set a default)
    n = number of points to use to evaluate function
    """
    t_min = x_lim[0]
    t_max = x_lim[1]
# Used t_vals & y_vals to separate ourselves from the y & t symbols used to build the solution to the ODE
    t_vals = np.linspace(t_min, t_max, n) # build a grid of t values to use for calculating the function values
    y_vals = fn(t_vals) # Apply the function to the grid of t values to get a python array of function values

# pass t_vals and y_vals to the plotting routine
    plt.plot(t_vals,y_vals,
             linestyle='-')
    if fn_2 is not None:
        y_2_vals = fn_2(t_vals)
        plt.plot(t_vals, y_2_vals, linestyle='-.')
    plt.xlabel("Value of x")
    plt.ylabel("Value of function")
    plt.title(title)
    if y_lim is None:
        y_max = 1.1*np.max(y_vals) - 0.1*np.min(y_vals)
        y_min = 1.1*np.min(y_vals) - 0.1*np.max(y_vals)
    else:
        y_min = y_lim[0]
        y_max = y_lim[1]
    plt.ylim([y_min, y_max])
    plt.yticks(np.arange(y_min, y_max, (y_max - y_min)/10.0)) # plot tick marks every 0.1 along the axis
    plt.xlim([t_min, t_max])
    plt.show()
    
def plot_qq(data, series_name, dict_fn, nbins=500,
            show_45=True, xlim=None, show_labels=False):
    """
    see https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot
    Plot a q-q plot for several distributions using the functions defined
    in the dictionary. 
       - data = sample data
       - series_name = name of series used to construct plot title
       - dict_fn = dictionary of functions used to calculate the distribution
                   quantiles (typically, dist.ppf for sci.stats distributions)
       - nbins = number of percentile points to use
       - show_45 = show line at 45 degrees angle 
    """
    n = data.shape[0]
    pctiles = 100*np.linspace(0.5/nbins, (nbins-0.5)/nbins, nbins) # see pctiles
    data_pctiles = np.percentile(data, pctiles)
    x_min = np.min(data_pctiles)
    x_max = np.max(data_pctiles) 
    for fn_name in dict_fn:
        fn_pctiles = dict_fn[fn_name][0](pctiles/100.0)
        if np.min(fn_pctiles) < x_min:
            x_min = np.min(fn_pctiles)
        if np.max(fn_pctiles) > x_max:
            x_max = np.max(fn_pctiles)
        plt.plot(fn_pctiles, data_pctiles,  dict_fn[fn_name][1], label=fn_name, 
                 markerfacecolor='None', markersize=4)
        if show_labels:
            for i in range(nbins):
                 if xlim is None:
                     plt.text(data_pctiles[i], fn_pctiles[i], f'{pctiles[i]:0.2f}',
                              fontsize=6, ha='right', va='bottom')  # Adjust fontsize and position as needed
                 else:
                     if (data_pctiles[i] < xlim[1]) & (data_pctiles[i] > xlim[0]):
                         plt.text(fn_pctiles[i], data_pctiles[i], f'{pctiles[i]:0.2f}%',
                                  fontsize=6, ha='right')#, va='bottom')  # ha='right') Adjust fontsize and position as needed
            
    
    f = 0.05
    x_min = x_min - f*(x_max - x_min)
    x_max = x_max + f*(x_max - x_min)
    if show_45:
        sline = np.linspace(x_min, x_max, 100)
        plt.plot(sline, sline, 'k--')
    plt.xlabel('Theoretical Percentiles')
    plt.ylabel('Sample Percentiles')
    plt.title(f'{series_name} Q-Q Plot')
    if xlim is None:
        plt.xlim([x_min, x_max])
        plt.ylim([x_min, x_max])
    else:
        plt.xlim(xlim)
        plt.ylim(xlim)
    plt.legend()
    plt.show()

def plot_indexed_prices(df, col_name='LogReturn', axis_label='Value',
                        id_field='Index', date_field='Date'):
    """
    Plot each of the indices on chart for the performance dataset
    Performance is indexed to 100 on the first day
    """
    df_sort = df.sort_values([id_field, date_field])
    df_sort['CumLogRet'] = df.groupby(id_field)[col_name].cumsum()
    df_sort['IndexedPrice'] = df_sort.groupby(id_field)['CumLogRet'].transform(lambda x: 100*np.exp((x -x.iloc[0] + 1.0)/100.0))

# Plot the time series for each asset using Seaborn
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df_sort, x=date_field, y='IndexedPrice', hue=id_field)
    plt.ylabel(axis_label)

# Display the plot
    plt.show()
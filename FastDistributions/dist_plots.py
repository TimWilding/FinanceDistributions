"""
Useful statistical plots such as:
 plot_qq - generalised qq plot that can take multiple distributions,
 plot_indexed_prices - plot all returns referenced to 100 on start date
 plot_hist_fit - plot a histogram with the theoretical fit
 plot_ks - CDFs with Kolmogorov-Smirnov test applied
 plot_function - plot a generic function on a chart
 plot_log_function - plots dictionary of functions on a log-log scale
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
from scipy.stats import chi2, rankdata, kstest, ecdf


def plot_hist_fit(
    data,
    series_name,
    dict_fn,
    nbins=20,
    log_scale=False,
    ylim=None,
    xlim=None,
    xlabel="Log-Return",
    ax=None,
):
    """
    Plots a histogram of some data and then plots several functions over it.
    Those functions are typically distributions fitted to the data such as the
    t-distribution
    Inputs
      - data - data to histogram
      - series_name - name of the series to plot
      - dict_fn - dictionary of theoretical fits - each fit is a tuple containing
                  theoretical function and a line marker
      Optional
      - log_scale - plot the probability distribution on a log-scale
      - nbins - number of bins for the histogram
      - ylim - maximum value of pdf
      - xlabel - quantity we are histogramming
    """
    # Plot the histogram
    show = False
    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True

    ax.hist(data, bins=nbins, density=True, alpha=0.7, color="blue", label="Histogram")
    x_vals = np.linspace(data.min(), data.max(), 500)
    y_max = 0.0
    y_min = 10000
    for fn_name in dict_fn:
        f_pdf = dict_fn[fn_name][0](x_vals)
        if np.max(f_pdf) > y_max:
            y_max = np.max(f_pdf)
        if np.min(f_pdf) < y_min:
            y_min = np.min(f_pdf)
        ax.plot(x_vals, f_pdf, dict_fn[fn_name][1], label=fn_name)

    # Add labels and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    ax.set_title(f"{series_name}")
    ax.legend()
    if log_scale:
        ax.set_yscale("log")
        if ylim is None:
            ax.set_ylim([0.8 * y_min, 1.2 * y_max])
        else:
            ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    # Show the plot
    if show:
        plt.show()


def plot_reg_errors(
    coefficients1,
    errors1,
    coefficients2,
    errors2,
    cols,
    title="Regression Coefficients with Standard Errors",
    watermark=None,
):
    """
    Plot a comparison of regression coefficients from two models using the data
    in coefficients1 and coefficients2
    """
    (_, ax) = plt.subplots(figsize=(10, 5))
    # Define the data

    # Set the number of variables
    num_vars = len(coefficients1)

    # Define the positions for the bar chart
    bar_width = 0.35

    # Define the positions for the bar chart
    positions1 = np.arange(num_vars)
    positions2 = [pos + bar_width for pos in positions1]
    # Create the bar chart
    _ = ax.bar(
        positions1,
        coefficients1,
        yerr=errors1,
        capsize=3,
        width=bar_width,
        label="Daily",
        color="b",
        alpha=0.6,
        ecolor="b",
        snap=False,
    )
    _ = ax.bar(
        positions2,
        coefficients2,
        yerr=errors2,
        capsize=3,
        width=bar_width,
        label="Weekly",
        color="r",
        alpha=0.6,
        ecolor="r",
        snap=False,
    )

    # Add labels and title
    # plt.xlabel('Features')
    ax.set_ylabel("Coefficients")
    ax.set_title(title)
    if watermark is not None:
        watermark_img = plt.imread(watermark)  # Load your image filedollarimage.jpg
        xdata = ax.get_lines()[0].get_xdata()
        ydata = ax.get_lines()[0].get_ydata()
        ax.imshow(
            watermark_img,
            extent=[xdata[0] - 0.5, xdata[-1] + 0.5, ydata[0], ydata[-1]],
            aspect="auto",
            alpha=0.3,
            zorder=0,
        )
    ax.set_xticks(
        positions1 + bar_width / 2, cols, rotation=270
    )  # Update with your variable names
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_mahal_cdf(chi_sqrd, num_assets, ax=None):
    """
    Plot empirical percent rank of mahalanobis distance vs
    theoretical from Normal distribution
    """
    show = False
    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    n = chi_sqrd.shape[0]
    cdf = chi2.cdf(chi_sqrd, num_assets).reshape(n)
    pct_rank = (rankdata(chi_sqrd / num_assets)) / (n - 1)
    ax.scatter(pct_rank, cdf)
    ax.set_title("Mahalanobis Distance")
    ax.set_xlabel("Percent Rank")
    ax.set_ylabel("Chi-Squared Probability")
    sline = np.linspace(0.0, 1.0, 100)
    ax.plot(sline, sline, "k--")
    if show:
        plt.show()


def plot_mahal_dist(mahal_dist, dates, num_assets, title, cutoff=0.95, ax=None):
    """
    Plot the Mahalanobis Distance on a chart
    mahal_dist - vector of Mahalanobis distances
    dates - dates used for calculation of distances
    num_assets - number of assets
    title - title of chart
    cutoff - plot horizontal line at chi-squared cutoff
    ax - axis if needed
    """
    show = False
    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    chisq_cutoff = chi2.ppf(cutoff, num_assets)
    ax.plot(dates, mahal_dist, "r-")
    ax.axhline(y=chisq_cutoff, color="gray", linestyle="--")
    ax.set_ylim([0.0, np.max(mahal_dist)])
    ax.set_title(title)
    if show:
        plt.show()


def plot_multi_function(
    dict_fn,
    x_lim=None,
    y_lim=None,
    n=10000,
    title="Function",
    x_label="x",
    y_label="y",
    y_log_scale=False,
    ax=None,
):
    """
    Takes a function and produces a graph using the limits
    given
    fn = univariate function
    x_lim = min and max of function range
    y_lim = min and max of function values (should I set a default)
    n = number of points to use to evaluate function
    """
    show = False
    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    if x_lim is None:
        x_lim = [-6, 6]
    t_min = x_lim[0]
    t_max = x_lim[1]
    # Used t_vals & y_vals to separate ourselves from the y & t
    # symbols used to build the solution to the ODE
    t_vals = np.linspace(
        t_min, t_max, n
    )  # build a grid of t values to use for calculating the function values
    # y_vals = fn(t_vals) # Apply the function to the grid of t values
    # to get a python array of function values
    y_max = 0.0
    y_min = 0.0

    for fn_name in dict_fn:
        y_vals = dict_fn[fn_name][0](t_vals)
        if np.max(y_vals) > y_max:
            y_max = np.max(y_vals)
        if np.min(y_vals) < y_min:
            y_min = np.min(y_vals)
        ax.plot(t_vals, y_vals, dict_fn[fn_name][1], label=fn_name)

    if y_lim is None:
        y_max = 1.1 * y_max - 0.1 * y_min
        y_min = 0  # 1.1*np.min(y_vals) - 0.1*np.max(y_vals)
    else:
        y_min = y_lim[0]
        y_max = y_lim[1]
    if y_log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_ylim([y_min, y_max])
    # plt.yticks(np.arange(y_min, y_max, (y_max - y_min)/10.0))
    # plot tick marks every 0.1 along the axis
    ax.set_xlim([t_min, t_max])
    ax.legend()
    if show:
        plt.show()


def plot_function(
    fn, x_lim=None, y_lim=None, n_pts=10000, title="Function", fn_2=None
):
    """
    Takes a function and produces a graph using the limits
    given
    fn = univariate function
    x_lim = min and max of function range
    y_lim = min and max of function values (should I set a default)
    n_pts = number of points to use to evaluate function
    """
    if x_lim is None:
        x_lim = [-6, 6]
    t_min = x_lim[0]
    t_max = x_lim[1]
    # build a grid of t values to use for calculating the function values
    t_vals = np.linspace(t_min, t_max, n_pts)
    # Apply the function to the grid of t values to get a python array of function values
    y_vals = fn(t_vals)

    # pass t_vals and y_vals to the plotting routine
    plt.plot(t_vals, y_vals, linestyle="-")
    if fn_2 is not None:
        y_2_vals = fn_2(t_vals)
        plt.plot(t_vals, y_2_vals, linestyle="-.")
    plt.xlabel("Value of x")
    plt.ylabel("Value of function")
    plt.title(title)
    if y_lim is None:
        y_max = 1.1 * np.max(y_vals) - 0.1 * np.min(y_vals)
        y_min = 1.1 * np.min(y_vals) - 0.1 * np.max(y_vals)
    else:
        y_min = y_lim[0]
        y_max = y_lim[1]
    plt.ylim([y_min, y_max])
    # plot tick marks every 0.1 along the axis
    plt.yticks(np.arange(y_min, y_max, (y_max - y_min) / 10.0))
    plt.xlim([t_min, t_max])
    plt.show()


def _show_qq_labels(ax, nbins, data_pctiles, fn_pctiles, pctiles, xlim):
    """
    Show labels on a q-q plot - this is a helper function
    """
    for i in range(nbins):
        if xlim is None:
            # Adjust fontsize and position as needed
            ax.text(
                data_pctiles[i],
                fn_pctiles[i],
                f"{pctiles[i]:0.2f}",
                fontsize=6,
                ha="right",
                va="bottom",
            )
        else:
            if (
                (data_pctiles[i] < xlim[1])
                & (data_pctiles[i] > xlim[0])
                & (fn_pctiles[i] < xlim[1])
                & (fn_pctiles[i] > xlim[0])
            ):
                ax.text(
                    fn_pctiles[i],
                    data_pctiles[i],
                    f"{pctiles[i]:0.2f}%",
                    fontsize=6,
                    ha="right",
                )


def plot_qq(
    data,
    series_name,
    dict_fn,
    nbins=500,
    show_45=True,
    xlim=None,
    show_labels=False,
    ax=None,
):
    """
    see https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot
    Plot a q-q plot for several distributions using the functions defined
    in the dictionary.
       - data = numpy array containing sample data
       - series_name = name of series used to construct plot title
       - dict_fn = dictionary of functions used to calculate the distribution
                   quantiles (typically, dist.ppf for sci.stats distributions)
       - nbins = number of percentile points to use
       - show_45 = show line at 45 degrees angle
    """
    pctiles = 100 * np.linspace(
        0.5 / nbins, (nbins - 0.5) / nbins, nbins
    )  # see pctiles
    data_pctiles = np.percentile(data, pctiles)
    x_min = np.min(data_pctiles)
    x_max = np.max(data_pctiles)

    show = False
    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True

    for fn_name in dict_fn:
        fn_pctiles = dict_fn[fn_name][0](pctiles / 100.0)
        if np.min(fn_pctiles) < x_min:
            x_min = np.min(fn_pctiles)
        if np.max(fn_pctiles) > x_max:
            x_max = np.max(fn_pctiles)
        ax.plot(
            fn_pctiles,
            data_pctiles,
            dict_fn[fn_name][1],
            label=fn_name,
            markerfacecolor="None",
            markersize=4,
        )
        if show_labels:
            _show_qq_labels(ax, nbins, data_pctiles, fn_pctiles, pctiles, xlim)
    pad_val = 0.05
    x_min = x_min - pad_val * (x_max - x_min)
    x_max = x_max + pad_val * (x_max - x_min)
    if show_45:
        sline = np.linspace(x_min, x_max, 100)
        ax.plot(sline, sline, "k--")
    ax.set_xlabel("Theoretical Percentiles")
    ax.set_ylabel("Sample Percentiles")
    ax.set_title(f"{series_name} Q-Q Plot")
    if xlim is None:
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([x_min, x_max])
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
    ax.legend()
    if show:
        plt.show()


def _calc_cum_fn(ret_data):
    """
    Function to calculate performance from log returns with first
    date indexed to 100.
    """
    return 100 * np.exp((ret_data - ret_data.iloc[0] + 1.0) / 100.0)


def plot_indexed_prices(
    df_returns: pd.DataFrame,
    log_return_col="LogReturn",
    axis_label="Value",
    id_field="Index",
    date_field="Date",
    log_scale=False,
    match_return=None,
):
    """
    Plot each of the indices on chart for a performance dataset containing
    log returns.
    Performance is indexed to 100 on the first date of the dataset
    - df_returns - dataframe containing at least three columns - id, date, and log return
    - log_return_col - column containing log returns
    - id_field used to separate out the price series
    - date_field contains the date
    - col_name = column containing log returns
    - log_scale = use a log scale for the plot
    - match_return = identifier so that all returns are scaled the same
    """
    if match_return is not None:
        sum_returns = df_returns.groupby("Ticker")[log_return_col].sum()
        df_returns["Scaled_Return"] = df_returns.apply(
            lambda row: row[log_return_col]
            * sum_returns.loc[match_return]
            / (sum_returns.loc[row["Ticker"]]),
            axis=1,
        )
    df_sort = df_returns.sort_values([id_field, date_field])

    if match_return is not None:
        df_sort["CumLogRet"] = df_returns.groupby(id_field)["Scaled_Return"].cumsum()
    else:
        df_sort["CumLogRet"] = df_returns.groupby(id_field)[log_return_col].cumsum()

    df_sort["IndexedPrice"] = df_sort.groupby(id_field)["CumLogRet"].transform(
        _calc_cum_fn
    )

    # Plot the time series for each asset using Seaborn
    sns.lineplot(data=df_sort, x=date_field, y="IndexedPrice", hue=id_field)
    plt.ylabel(axis_label)
    if log_scale:
        plt.yscale("log")

    # Display the plot
    plt.show()


def _plot_log_cdf_function(
    dict_fn,
    ret_data,
    left_tail=False,
    ref_fn=None,
    ret_marker="kx",
    x_lim=None,
    y_lim=None,
    n_pts=100,
    title="CDF",
):
    """
    Takes a function and produces a log-log graph using the limits
    given
    dict_fn = dictionary of cdf functions
    ret_data = sample data for comparison with empirical cdf function
    left_tail = look at PR(X<-x) instead of PR(X>x)
    ref_fn = comparison function for behaviour in tail
    x_lim = log10 values of min and max of function range
    y_lim = min and max of function values (should I set a default)
    n = number of points to use to evaluate cumulative density function
    """
    sample_pdf = scipy.stats.ecdf(ret_data)
    if x_lim is None:
        x_lim = [-3, 2]
    t_min = x_lim[0]
    t_max = x_lim[1]
    # build a grid of t values to use for calculating the function values
    t_vals = np.logspace(t_min, t_max, n_pts)
    for fn_name in dict_fn:
        if left_tail:
            f_pdf = dict_fn[fn_name][0](-t_vals)
        else:
            f_pdf = 1 - dict_fn[fn_name][0](t_vals)

        #        if np.max(f_pdf) > y_max:
        #            y_max = np.max(f_pdf)
        #        if np.min(f_pdf) < y_min:
        #            y_min = np.min(f_pdf)
        plt.plot(t_vals, f_pdf, dict_fn[fn_name][1], label=fn_name)
    if ref_fn is not None:
        f_pdf = ref_fn[0](t_vals)
        plt.plot(t_vals, f_pdf, ref_fn[1], label=ref_fn[2])

    if left_tail:
        f_pdf = sample_pdf.cdf.evaluate(-t_vals)
        plt.plot(t_vals, f_pdf, ret_marker, label="Sample")
    else:
        f_pdf = 1 - sample_pdf.cdf.evaluate(t_vals)
        plt.plot(t_vals, f_pdf, ret_marker, label="Sample")

    plt.xlabel("x")
    if left_tail:
        plt.ylabel("Pr(X<-x)")
    else:
        plt.ylabel("Pr(X>x)")
    plt.title(title)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.xlim([10**t_min, 10**t_max])
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()


def plot_log_cdf(
    dict_dist,
    ret_data,
    ref_fn=None,
    x_lim=None,
    y_lim=None,
    n_pts=25,
    ret_marker="kx",
    series_name="Daily Returns",
):
    """
    Plot the tails of the CDF function - left and right tail
     - dict_dist - a dictionary containing a set of distributions each distribution is characterised
                   by a two-entry list containing the cdf function and the marker used on the chart
                   e.g. {'Normal': [norm_fit.cdf, 'r-']} would draw the fit of the
                   normal distribution using a red line
     - ret_data - sample data used for comparison
     - ref_fn - useful to plot a reference function on the chart such as a power law to compare tail
                behaviour with the ideal
     - xlim - log range to span 10^xlim(0) to 10^xlim(1)
     - ylim - y range to span
     - ret_marker - marker to use for sample data CDF
     - series name - used in title of the CDF plots
    """
    if x_lim is None:
        x_lim = [-3, 2]
    _, _ = plt.subplots(nrows=1, ncols=2)

    plt.subplot(1, 2, 1)

    _plot_log_cdf_function(
        dict_dist,
        ret_data,
        False,
        ref_fn,
        x_lim=x_lim,
        ret_marker=ret_marker,
        y_lim=y_lim,
        n_pts=n_pts,
        title=f"{series_name} - Positive CDF",
    )

    plt.subplot(1, 2, 2)
    _plot_log_cdf_function(
        dict_dist,
        ret_data,
        True,
        ref_fn,
        x_lim=x_lim,
        ret_marker=ret_marker,
        y_lim=y_lim,
        n_pts=n_pts,
        title=f"{series_name} - Negative CDF",
    )
    plt.show()


def plot_ks(index_ret, dist, dist_name, index_name, xlabel='LogReturn', ax=None):
    """
    Plot the Kolmogorov-Smirnov test for a distribution. The
    Kolmogorov-Smirnov test is a non-parametric test that compares
    the empirical distribution function of the sample data with the
    cumulative distribution function of the distribution.

    Inputs
    =========
     - index_ret - sample data
     - dist - distribution object
     - dist_name - name of the distribution
     - index_name - name of the index
    """
    ks_res = kstest(index_ret, dist.cdf)
    show = False
    if ax is None:
        ax = plt.gca()
        show = True
    x_range = np.quantile(index_ret, [0.001, 0.999])
    try:
        x_min = np.maximum(x_range[1], dist.ppf(0.999))  # index_ret.min()
        x_max = np.minimum(x_range[0], dist.ppf(0.001))  # index_ret.max()
    except ValueError:
        x_min = x_range[1]
        x_max = x_range[0]

    x_vals = np.linspace(x_min, x_max, 200)
    cdf = dist.cdf(x_vals)
    e_cdf = ecdf(index_ret).cdf.evaluate(x_vals)
    ax.plot(x_vals, e_cdf, 'r-', label="Sample")
    ax.plot(x_vals, cdf, 'b-', label=dist_name)
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative Probability')
    ax.legend(loc='upper left')
    ax.axvline(ks_res.statistic_location, 0.0, 1.0, color='black', linestyle='--')
    ax.set_title(f'{index_name} - {dist_name}')
    textstr = '\n'.join((
        f'KS Stat={ks_res.statistic:.2f}',
        f'Location={ks_res.statistic_location:.2f}',
        f'P-Value={ks_res.pvalue:.2f}'))

    # Add text box to the plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='black')
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=6,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    if show:
        plt.show()
    return ax

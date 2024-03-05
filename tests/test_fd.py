"""
Testing module for the FinanceDistributions package
"""
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from .context import FastDistributions as fd


def plot_stats_col(df, col_name, axis_label, chart_title):
    """
    Plot each of the indices on chart from the statistics daata set
    """
    df_sort = df.sort_values(["Identifier", "Date"])

    # Plot the time series for each asset using Seaborn
    _, _ = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_sort, x="Date", y=col_name, hue="Identifier")
    plt.ylabel(axis_label)
    plt.title(chart_title)

    # Display the plot
    plt.show()


def norm_fn(ret_data):
    """
    Used for testing the backtest function
    """
    norm_fit = norm.fit(ret_data)
    norm_ll = np.sum(norm(norm_fit[0], norm_fit[1]).logpdf(ret_data))
    x = {"norm_loc": norm_fit[0], "norm_scale": norm_fit[1], "norm_ll": norm_ll}
    return x


def test_bootstrap():
    """
    Test the parallel bootstrap routine by fitting several normal distributions
    to a data sample downloaded by Yahoo Finance
    """
    nbs = 100
    lst_indices = [("^GSPC", "SP 500"), ("^FTSE", "FTSE 100"), ("^N225", "Nikkei 225")]
    df_ret = fd.download_yahoo_returns(lst_indices)
    sp_ret = df_ret[df_ret.Ticker == "^GSPC"]["LogReturn"].values
    df_bs = fd.parallel_bootstrap(sp_ret, norm_fn, nskip=10, nbs=nbs)
    assert len(df_bs.index) == nbs + 1
    return


def test_priips():
    """
    Test the PRIIPS calculation functions on a data set contained in Github
    """
    MIN_DATE_FILTER = datetime.datetime(
        2004, 1, 1
    )  # beginning of sample 'Jan 2004' - Jan 7 results in similar
    df_price_history = fd.calculate_returns(
        read_csv(
            "https://raw.githubusercontent.com/TimWilding/FinanceDataSet/main/PRIIPS_Data_Set.csv",
            parse_dates=[1],
        ),
        "Index",
        "Date",
    )

    df_price_history = df_price_history[
        df_price_history.Date >= MIN_DATE_FILTER
    ]  # Note - this removes NaNs from initial price points in LogReturn column

    df_backtest_old = fd.rolling_backtest(
        df_price_history,
        lambda x: fd.PRIIPS_stats_df(x, use_new=False, holding_period=5),
        rolling_window_years=5,
        rolling_start_years=15,
    )

    plot_stats_col(df_backtest_old, "Favourable", "Performance", "Favourable (Orig)")
    df_sample_5y = df_price_history[
        (df_price_history.Date > datetime.datetime(2018, 10, 27))
        & (df_price_history.Date <= datetime.datetime(2023, 10, 27))
    ]
    df_bs = fd.PRIIPS_stats_bootstrap(df_sample_5y)
    print("Finished")
    return


def calc_correl(df, lag=1):
    """
    given a dataframe with columns - Identifier, Date, and LogReturn
    calculate a correlation matrix for each of the identifiers.
    Return the lower-diagonal elements of the correlation matrix
    """

    pivot_df = df.pivot(index="Date", columns="Name", values="LogReturn")
    pivot_df = pivot_df.dropna()
    correlation_df = pivot_df.corr()
    lower_triangle = correlation_df.where(
        np.tril(np.ones(correlation_df.shape), k=-1).astype(bool)
    )
    lower_triangle = lower_triangle.rename_axis("Names").stack().reset_index()
    lower_triangle.columns = ["Name1", "Name2", "correlation"]
    return lower_triangle.sort_values(by=["Name1", "Name2"], ascending=[True, True])


def calc_adj_corr(df, lag=1):
    pivot_df = df.pivot(index="Date", columns="Name", values="LogReturn")
    pivot_df = pivot_df.dropna()
    lst_names = df["Name"].unique()
    X = pivot_df[lst_names].values
    corr_mat = fd.newey_adj_corr(X, lag)
    df_corr = pd.DataFrame(corr_mat, columns=lst_names)
    df_corr["Name"] = lst_names
    df_corr.set_index("Name", inplace=True)
    lower_triangle = df_corr.where(np.tril(np.ones(df_corr.shape), k=-1).astype(bool))
    lower_triangle = lower_triangle.rename_axis("Names").stack().reset_index()
    lower_triangle.columns = ["Name1", "Name2", "correlation"]
    return lower_triangle.sort_values(by=["Name1", "Name2"], ascending=[True, True])


def test_correl():
    """
    Rolling backtest of correlations
    """
    BACKTEST_YEARS = 15  # Number of years to use for backtest
    WINDOW_YEARS = 2  # Number of years to use for calculation of correlation
    lst_indices = [
        ("^GSPC", "SP 500"),
        ("^FTSE", "FTSE 100"),
        ("^N225", "Nikkei 225"),
    ]  # , ('VWRA.L', 'Vanguard FTSE All-World')
    # lst_indices = [('^SP500TR', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')] #, ('VWRA.L', 'Vanguard FTSE All-World')
    df_prices = fd.download_yahoo_returns(lst_indices)

    df_month = (
        df_prices.groupby(["Name", "EndOfMonth"])
        .agg({"LogReturn": "sum"})
        .reset_index()
        .rename({"EndOfMonth": "Date"})
    )
    df_month.columns = ["Name", "Date", "LogReturn"]
    df_roll_month_correls = fd.rolling_backtest(
        df_month, calc_adj_corr, WINDOW_YEARS, BACKTEST_YEARS
    )


def test_robust_correl():
    """
    Test robust correlation
    """
    lst_indices = [
        ("^GSPC", "SP 500"),
        ("^FTSE", "FTSE 100"),
        ("^N225", "Nikkei 225"),
    ]  # , ('VWRA.L', 'Vanguard FTSE All-World')
    # lst_indices = [('^SP500TR', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')] #, ('VWRA.L', 'Vanguard FTSE All-World')
    df_prices = fd.download_yahoo_returns(lst_indices, download_period="5y")
    pivot_df = df_prices.pivot(index="Date", columns="Name", values="LogReturn")
    pivot_df = pivot_df.dropna()
    (samp_ave, samp_covar, nu, _) = fd.TDist.em_fit(pivot_df.values, dof=-8.0)
    samp_corr = fd.corr_conv(samp_covar)
    (_, norm_cov, _, _) = fd.TDist.em_fit(pivot_df.values, dof=1000)
    norm_corr = fd.corr_conv(norm_cov)
    act_corr = pivot_df.corr()
    print("Finished")


def test_dists():
    """
    Test the distribution fitting code using data downloaded from Yahoo Finance
    """
    # alpha = stability parameter (0, 2]
    # beta = skewness parameter [-1, 1]
    # ls = levy_stable(2.0, 0.0, loc=0, scale=1.0 )
    # ls_new = levy_stable(1.5, -0.15, loc=0, scale=1.0 ) # alpha, beta close to FTSE 100
    # plot_function(lambda x : ls.pdf(x), title='Levy Stable', fn_2=lambda x : ls_new.pdf(x))
    gsd = fd.GeneralisedSkewT(0, 1.0, 0.2, 1, 1000)
    gsd_skew = fd.GeneralisedSkewT(0, 1.0, 0.2, 2.0, 1000)
    print(gsd.pdf(-1.0))
    lst_indices = [("^GSPC", "SP 500"), ("^FTSE", "FTSE 100"), ("^N225", "Nikkei 225")]
    df_ret = fd.download_yahoo_returns(lst_indices)

    sp_ret = df_ret[df_ret.Ticker == "^GSPC"]["LogReturn"].values

    ls_fit = fd.LevyStableInterp.fitclass(sp_ret)
    gs_fit = fd.GeneralisedSkewT.fitclass(sp_ret)
    norm_mod = norm.fit(sp_ret)
    norm_fit = norm(norm_mod[0], norm_mod[1])

    lst_dist = {
        "Levy-Stable": [ls_fit.cdf, "r-"],
        "Normal": [norm_fit.cdf, "b-"],
        "Generalised SkewT": [gs_fit.cdf, "g-"],
    }
    ref_fn = [lambda x: 0.01 * (x) ** (-1.6), "y--", "Power Law"]
    fd.plot_log_cdf(
        lst_dist, sp_ret, ref_fn, x_lim=[-0.5, 1.5], y_lim=[1e-5, 1], n_pts=25
    )
    dict_pdf = {
        "Levy-Stable": [ls_fit.pdf, "b-"],
        "Generalised SkewT": [gs_fit.pdf, "k-"],
    }
    fd.plot_hist_fit(sp_ret, "SP 500", dict_pdf, 50, log_scale=True)
    dict_ppf = {
        "Levy-Stable": [ls_fit.ppf, "bo"],
        "Generalised SkewT": [gs_fit.ppf, "go"],
    }
    fd.plot_qq(sp_ret, "SP500 Returns", dict_ppf, nbins=500)

    print("Finished testing")


if __name__ == "__main__":
    test_dists()

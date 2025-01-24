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
import FastDistributions as fd



def plot_stats_col(df, col_name, axis_label, chart_title, id_field="Identifier"):
    """
    Plot each of the indices on chart from the statistics daata set
    """
    df_sort = df.sort_values([id_field, "Date"])

    # Plot the time series for each asset using Seaborn
    _, _ = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_sort, x="Date", y=col_name, hue=id_field)
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
    df_ret = fd.get_test_data()
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
    df_price_history = fd.get_test_data()

    df_price_history = df_price_history[
        df_price_history.Date >= MIN_DATE_FILTER
    ]  # Note - this removes NaNs from initial price points in LogReturn column
    # Remove Bitcoin because it doesn't have enough history
    df_price_history = df_price_history[
        df_price_history["Ticker"].isin(["^AEX", "^FTSE", "^N225", "^GSPC"])
    ]
    
    df_sample_5y = df_price_history[
        (df_price_history.Date > datetime.datetime(2018, 10, 27))
        & (df_price_history.Date < datetime.datetime(2023, 10, 27))
    ]
    df_s = fd.PRIIPS_stats_df(df_sample_5y, index_field="Ticker")

    df_backtest_old = fd.rolling_backtest(
        df_price_history,
        lambda x: fd.PRIIPS_stats_df(
            x, use_new=False, holding_period=5, index_field="Ticker"
        ),
        rolling_window_years=5,
        rolling_start_years=15,
    )

    df_backtest_old.to_csv("PRIIPS_Backtest.csv")
    plot_stats_col(df_backtest_old, "Favourable", "Performance", "Favourable (Orig)")

    df_bs = fd.PRIIPS_stats_bootstrap(df_sample_5y, index_field="Ticker")
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
    df_prices = fd.get_test_data()

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


def test_new_priips_aex():
    df_prices = fd.get_test_data()
    df_prices = df_prices[df_prices.Ticker == "^AEX"]
    df_backtest = fd.rolling_backtest(
        df_prices,
        lambda x: fd.PRIIPS_stats_df(
            x, use_new=True, holding_period=5, index_field="Ticker"
        ),
        rolling_window_years=10,
        rolling_start_years=15,
    )
    df_backtest.to_csv("AEX_backtest.csv")
    plot_stats_col(df_backtest, "Moderate", "Performance", "Moderate (Orig)")


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
    df_prices = fd.get_test_data(5)
    pivot_df = df_prices.pivot(index="Date", columns="Name", values="LogReturn")
    pivot_df = pivot_df.dropna()
    (samp_ave, samp_covar, nu, _) = fd.TDist.em_fit(pivot_df.values, dof=-8.0)
    samp_corr = fd.corr_conv(samp_covar)
    (_, norm_cov, _, _) = fd.TDist.em_fit(pivot_df.values, dof=1000)
    norm_corr = fd.corr_conv(norm_cov)
    act_corr = pivot_df.corr()
    print("Finished")


def test_yf():
    """
    Test the Yahoo Finance Download
    """
    lst_indices = [
        ("^GSPC", "SP 500"),
        ("^FTSE", "FTSE 100"),
        ("^N225", "Nikkei 225"),
        ("BTC-USD", "Bitcoin"),
    ]
    df_download = fd.download_yahoo_returns(lst_indices)
    df_download.to_csv("asset_returns.csv")


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
    # lst_indices = [("^GSPC", "SP 500"), ("^FTSE", "FTSE 100"), ("^N225", "Nikkei 225"), ('WFBIX', 'US Aggregate Bond Index')]
    df_ret = fd.get_test_data()

    sp_ret = df_ret[df_ret.Ticker == "^GSPC"]["LogReturn"].values

    ls_fit = fd.LevyStableInterp.fitclass(sp_ret)
    gs_fit = fd.GeneralisedSkewT.fitclass(sp_ret)
    ent_fit = fd.EntropyDistribution.fitclass(sp_ret)
    mei_fit = fd.Meixner.fitclass(sp_ret)


    norm_mod = norm.fit(sp_ret)
    norm_fit = norm(norm_mod[0], norm_mod[1])

    lst_dist = {
        "Levy-Stable": [ls_fit.cdf, "r-"],
        "Normal": [norm_fit.cdf, "b-"],
        "Generalised SkewT": [gs_fit.cdf, "g-"],
        "Entropy Distribution": [ent_fit.cdf, "k-"],
        "Meixner Distribution": [mei_fit.cdf, "y-"],
    }
    ref_fn = [lambda x: 0.01 * (x) ** (-1.6), "y--", "Power Law"]
    fd.plot_log_cdf(
        lst_dist, sp_ret, ref_fn, x_lim=[-0.5, 1.5], y_lim=[1e-5, 1], n_pts=25
    )
    dict_pdf = {
        "Levy-Stable": [ls_fit.pdf, "b-"],
        "Generalised SkewT": [gs_fit.pdf, "k-"],
        "Entropy Distribution": [ent_fit.pdf, "r-"],
        "Meixner Distribution": [mei_fit.pdf, "y-"],
    }
    fd.plot_hist_fit(sp_ret, "SP 500", dict_pdf, 50, log_scale=True)
    dict_ppf = {
        "Levy-Stable": [ls_fit.ppf, "bo"],
        "Generalised SkewT": [gs_fit.ppf, "go"],
        "Entropy Distribution": [ent_fit.ppf, "ro"],
        "Meixner Distribution": [mei_fit.ppf, "yo"],
        }
    fd.plot_qq(sp_ret, "SP500 Returns", dict_ppf, nbins=500)

    print("Finished testing")


def test_cached_read():
    df_prices = fd.read_cached_excel('https://www.eurekahedge.com/Indices/ExportIndexReturnsToExcel?IndexType=Eurekahedge&IndexId=640', header=3)
    df_prices['LogReturn'] = np.log((100 + df_prices['PercentReturn'])/100)
    df_prices = df_prices.dropna()
    print(fd.probabilistic_sharpe_ratio(df_prices['LogReturn'].values))
    print("Finished testing cached Excel read")


def test_regress():
    df_prices = fd.get_test_data(5)
    pivot_df = df_prices.pivot(index="Date", columns="Name", values="LogReturn")
    pivot_df = pivot_df.dropna()
    df = fd.sample_regress(pivot_df, "SP 500", True, True)
    print("Finished testing regression")

def test_risk_parity():
    """This is a test on a known problem"""
    sigma = np.vstack([np.array((1.0000, 0.0015, -0.0119)),
                   np.array((0.0015, 1.0000, -0.0308)),
                   np.array((-0.0119, -0.0308, 1.0000))])
    risk_budget = np.array((0.1594, 0.0126, 0.8280))
    ans = np.array([0.2798628, 0.08774909, 0.63238811])
    rpp = fd.get_risk_parity_pf(sigma,
                                risk_budgets=risk_budget)
    np.testing.assert_allclose(rpp, ans, rtol=1e-4)
    # assert rpp.risk_concentration.evaluate() < 1e-9



def test_black_litterman():
    # Data given in He & Litterman 1999 for illustrative calculations
    lst_countries =['Australia','Canada','France','Germany','Japan','UK','USA']

# Table 1 - Correlations
    correl_hel = np.array([[1, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
                       [0.488, 1, 0.664, 0.655, 0.31, 0.608, 0.779],
                       [0.478, 0.664, 1, 0.861, 0.355, 0.783, 0.668],
                       [0.515, 0.655, 0.861, 1, 0.354, 0.777, 0.653],
                       [0.439, 0.31, 0.355, 0.354, 1, 0.405, 0.306],
                       [0.512, 0.608, 0.783, 0.777, 0.405, 1, 0.652],
                       [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1]])

# Table 2 - volatilities and market cap weights
    sigma_hel = np.array([16, 20.3, 24.8, 27.1, 21, 20, 18.7])/100
    w_hel = np.array([1.6, 2.2, 5.2, 5.5, 11.6, 12.4, 61.5])/100

# View Portfolios
    P = np.array([[0.0, 0.0,   -29.5, 100.0, 0.0, -70.5, 0.0],
                 [0.0, 100.0, 0.0, 0.0,   0.0, 0.0, -100.0]])/100


# Calculate the covariance matrix from the data
    cov_hel = fd.cov_from_correl(correl_hel, sigma_hel)

# Calculate the equilibrium returns from the equilibrium weights and the covariance
# matrix
    delta = 2.5
    tau_hel = 0.05
    q = np.array([5, 3])/100
    Omega = np.array([[0.021, 0.0], 
                      [0.0, 0.017]])*tau_hel
    pi = fd.reverse_optimise(w_hel, cov_hel, delta) # calculate equilibrium returns
    print(pi)
    pi_hat, sigma_hat = fd.black_litterman_stats(w_hel, cov_hel, P, q, Omega, tau_hel, delta, True)

    w_opt = fd.unconstrained_optimal_portfolio(sigma_hat, pi_hat, delta)
    print("Revised Optimal Portfolio")
    print(w_opt)
    print("Change from Original")
    print(w_opt - (w_hel/ (1 + tau_hel)))
    lam, dlam_dq = fd.he_litterman_lambda(w_hel, cov_hel, P, q, Omega, tau_hel, delta, True)
    print('Lambda')
    print(lam)
    print('dlamdq')
    print(dlam_dq) 
    print('Fusai-Meucci Consistency')
    print(fd.fusai_meucci_consistency(pi_hat, pi, tau_hel*cov_hel, P, Omega))
    print('Theils View Compatibility')
    print(fd.theils_view_compatibility(q, pi, tau_hel*cov_hel, P, Omega))
    te, dtedq = fd.braga_natale_measure(w_hel, cov_hel, P, q, Omega, tau_hel, delta)
    print('Braga Natale measure = tracking error = {0:5.2f}%'.format(te))
    print(dtedq)   
    ans_opt = np.array([0.0152381, 0.41893107, -0.03471219, 0.33792671, 0.11047619, -0.08321452, 0.1877356])

    np.testing.assert_allclose(w_opt, ans_opt, rtol=1e-6)

def test_expected_shortfall():
    """
    Test the expected shortfall statistics
    """
    df_ret = fd.get_test_data()    
    sp_ret = df_ret[df_ret.Ticker == "^GSPC"]["LogReturn"].values
    (es, var) = fd.sample_expected_shortfall(sp_ret, 0.95)
    np.testing.assert_approx_equal(es, -3.0182639295131275, 3)
    gs = fd.GeneralisedSkewT.fitclass(sp_ret, display_progress=False)
    (es_dist, var_dist) = fd.expected_shortfall(gs, 0.95 )
    np.testing.assert_approx_equal(es_dist, -2.9305505707305555, 3)
    sp_tail = fd.fit_tail_model(sp_ret, 0.95)
    np.testing.assert_approx_equal(sp_tail[0], 0.20524897090856728, 3)
    (es_tail, var_tail) = fd.expected_shortfall_tail_model(0.95, sp_tail[0], sp_tail[1], sp_tail[2], 0.99)
    np.testing.assert_approx_equal(es_tail, -5.344663974593491, 3)


def test_fit_johnson():

    def pdf_su_shape(x, gamma, delta):
       return fd.JohnsonSU(gamma, delta, 0, 1.0).pdf(x)

    lst_dist = {
        "gamma = -1.1, delta = 1.5": [lambda x : pdf_su_shape(x, -1.1, 1.5), "r--"],
        "gamma = -1.1, delta = 0.8": [lambda x : pdf_su_shape(x, -1.1, 0.8), "b--"],
        "gamma = 0.5, delta = 0.8": [lambda x : pdf_su_shape(x, 0.5, 0.8), "g--"],
        "gamma = 0.5, delta = 0.05": [lambda x : pdf_su_shape(x, 0.5, 0.05), "k--"],
        "gamma = 0.0, delta = 0.1": [lambda x : pdf_su_shape(x, 0.0, 0.1), "y--"],        
        "Normal": [lambda x : norm.pdf(x), "k-"],
    }
    fd.plot_multi_function(lst_dist, y_label='Probability Density',
                   x_lim=[-10.0, 10.0], y_log_scale=False, title='Johnson S_U Distribution')
    df_ret = fd.get_test_data()
    sp_ret = df_ret[df_ret.Ticker == "^GSPC"]["LogReturn"].values
    jsu_fit = fd.JohnsonSU.fitclass(sp_ret)
    print(jsu_fit.std())
    (x_vals, x_wts) = fd.gauss_legendre_sample(1000)
    mean = jsu_fit.mean()
    sd = jsu_fit.std()
    mean_est = np.sum(x_wts * jsu_fit.pdf(x_vals) * x_vals) # Use numerical integration to test the mean of the distribution
    sd_est = np.sqrt(np.sum(x_wts*jsu_fit.pdf(x_vals)*x_vals**2) - mean_est*mean_est)
    np.testing.assert_approx_equal(mean, mean_est, 1e-6)
    np.testing.assert_approx_equal(sd, sd_est, 1e-6)
    norm_mod = norm.fit(sp_ret)
    norm_fit = norm(norm_mod[0], norm_mod[1])
    dict_pdf = {
        "Normal Distribution":   [norm_fit.pdf, "b-"],
        "Johnson SU Distribution": [jsu_fit.pdf, "y-"],
    }
    fd.plot_hist_fit(sp_ret, "SP 500", dict_pdf, 50, log_scale=True)





if __name__ == "__main__":
    test_dists()

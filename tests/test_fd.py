"""
Testing module for the FinanceDistributions package
"""

import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
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
    _ = fd.PRIIPS_stats_df(df_sample_5y, index_field="Ticker")

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

    _ = fd.PRIIPS_stats_bootstrap(df_sample_5y, index_field="Ticker")
    print("Finished")
    return


def calc_correl(df):
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
    """
    Test the Newey-West adjusted correlation function
    """
    pivot_df = df.pivot(index="Date", columns="Name", values="LogReturn")
    pivot_df = pivot_df.dropna()
    lst_names = df["Name"].unique()
    x_vals = pivot_df[lst_names].values
    corr_mat = fd.newey_adj_corr(x_vals, lag)
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
    df_prices = fd.get_test_data()

    df_month = (
        df_prices.groupby(["Name", "EndOfMonth"])
        .agg({"LogReturn": "sum"})
        .reset_index()
        .rename({"EndOfMonth": "Date"})
    )
    df_month.columns = ["Name", "Date", "LogReturn"]
    _ = fd.rolling_backtest(df_month, calc_adj_corr, WINDOW_YEARS, BACKTEST_YEARS)


def test_new_priips_aex():
    """
    Test the new PRIIPS functionality on the AEX index
    """
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
    Test robust correlation using EM Algorithm to fit Multivariate T-Distribution
    """
    df_prices = fd.get_test_data(5)
    pivot_df = df_prices.pivot(index="Date", columns="Name", values="LogReturn")
    pivot_df = pivot_df.dropna()
    (samp_ave, samp_covar, nu, _) = fd.TDist.em_fit(pivot_df.values, dof=-8.0)
    np.testing.assert_approx_equal(nu, 4.451310494728308, 4)
    td = fd.TDist(samp_ave, samp_covar, nu)
    x = td.simulate(1000)
    print(np.mean(x, axis=0))
    _, _ = td.mahal_dist(pivot_df.values)
    _ = td.cumdf(pivot_df.values)
    _ = fd.corr_conv(samp_covar)
    (_, norm_cov, _, _) = fd.TDist.em_fit(pivot_df.values, dof=1000)
    norm_corr = fd.corr_conv(norm_cov)
    act_corr = pivot_df.corr()
    corr_diff = norm_corr - act_corr
    print(corr_diff)
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
    print(gsd_skew.pdf(-1.0))
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
    """
    Cached read checks whether file has been downloaded
    before reading in a cSV file
    """
    df_prices = fd.read_cached_excel(
        "https://raw.githubusercontent.com/TimWilding/FinanceDistributions/refs/heads/master/data/HF_Index.xlsx",
        header=3,
    )
    df_prices["LogReturn"] = np.log((100 + df_prices["PercentReturn"]) / 100)
    df_prices = df_prices.dropna()
    print(fd.probabilistic_sharpe_ratio(df_prices["LogReturn"].values))
    print("Finished testing cached Excel read")


def test_regress():
    """
    Tests the sample regression function
    """
    df_prices = fd.get_test_data(5)
    pivot_df = df_prices.pivot(index="Date", columns="Name", values="LogReturn")
    pivot_df = pivot_df.dropna()
    _ = fd.sample_regress(pivot_df, "SP 500", True, True)
    print("Finished testing regression")


def test_risk_parity():
    """This is a test of risk parity optimisation
    on a known problem"""
    sigma = np.vstack(
        [
            np.array((1.0000, 0.0015, -0.0119)),
            np.array((0.0015, 1.0000, -0.0308)),
            np.array((-0.0119, -0.0308, 1.0000)),
        ]
    )
    risk_budget = np.array((0.1594, 0.0126, 0.8280))
    ans = np.array([0.2798628, 0.08774909, 0.63238811])
    rpp = fd.get_risk_parity_pf(sigma, risk_budgets=risk_budget)
    np.testing.assert_allclose(rpp, ans, rtol=1e-4)
    # assert rpp.risk_concentration.evaluate() < 1e-9


def test_black_litterman():
    """
    Test Black-Litterman functionality
    """
    # Data given in He & Litterman 1999 for illustrative calculations
    # lst_countries = ["Australia", "Canada", "France", "Germany", "Japan", "UK", "USA"]

    # Table 1 - Correlations
    correl_hel = np.array(
        [
            [1, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
            [0.488, 1, 0.664, 0.655, 0.31, 0.608, 0.779],
            [0.478, 0.664, 1, 0.861, 0.355, 0.783, 0.668],
            [0.515, 0.655, 0.861, 1, 0.354, 0.777, 0.653],
            [0.439, 0.31, 0.355, 0.354, 1, 0.405, 0.306],
            [0.512, 0.608, 0.783, 0.777, 0.405, 1, 0.652],
            [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1],
        ]
    )

    # Table 2 - volatilities and market cap weights
    sigma_hel = np.array([16, 20.3, 24.8, 27.1, 21, 20, 18.7]) / 100
    w_hel = np.array([1.6, 2.2, 5.2, 5.5, 11.6, 12.4, 61.5]) / 100

    # View Portfolios
    pfs = (
        np.array(
            [
                [0.0, 0.0, -29.5, 100.0, 0.0, -70.5, 0.0],
                [0.0, 100.0, 0.0, 0.0, 0.0, 0.0, -100.0],
            ]
        )
        / 100
    )

    # Calculate the covariance matrix from the data
    cov_hel = fd.cov_from_correl(correl_hel, sigma_hel)

    # Calculate the equilibrium returns from the equilibrium weights and the covariance
    # matrix
    delta = 2.5
    tau_hel = 0.05
    q = np.array([5, 3]) / 100
    omega = np.array([[0.021, 0.0], [0.0, 0.017]]) * tau_hel
    pi = fd.reverse_optimise(w_hel, cov_hel, delta)  # calculate equilibrium returns
    print(pi)
    pi_hat, sigma_hat = fd.black_litterman_stats(
        w_hel, cov_hel, pfs, q, omega, tau_hel, delta, True
    )

    w_opt = fd.unconstrained_optimal_portfolio(sigma_hat, pi_hat, delta)
    print("Revised Optimal Portfolio")
    print(w_opt)
    print("Change from Original")
    print(w_opt - (w_hel / (1 + tau_hel)))
    lam, dlam_dq = fd.he_litterman_lambda(
        w_hel, cov_hel, pfs, q, omega, tau_hel, delta, True
    )
    print("Lambda")
    print(lam)
    print("dlamdq")
    print(dlam_dq)
    print("Fusai-Meucci Consistency")
    print(fd.fusai_meucci_consistency(pi_hat, pi, tau_hel * cov_hel, pfs, omega))
    print("Theils View Compatibility")
    print(fd.theils_view_compatibility(q, pi, tau_hel * cov_hel, pfs, omega))
    te, dtedq = fd.braga_natale_measure(w_hel, cov_hel, pfs, q, omega, tau_hel, delta)
    print(f"Braga Natale measure = tracking error = {te:5.2f}%")
    print(dtedq)
    ans_opt = np.array(
        [
            0.0152381,
            0.41893107,
            -0.03471219,
            0.33792671,
            0.11047619,
            -0.08321452,
            0.1877356,
        ]
    )

    np.testing.assert_allclose(w_opt, ans_opt, rtol=1e-6)


def test_expected_shortfall():
    """
    Test the expected shortfall statistics
    """
    df_ret = fd.get_test_data()
    sp_ret = df_ret[df_ret.Ticker == "^GSPC"]["LogReturn"].values
    (es, _) = fd.sample_expected_shortfall(sp_ret, 0.95)
    np.testing.assert_approx_equal(es, -3.0182639295131275, 3)
    gs = fd.GeneralisedSkewT.fitclass(sp_ret, display_progress=False)
    (es_dist, _) = fd.expected_shortfall(gs, 0.95)
    np.testing.assert_approx_equal(es_dist, -2.9305505707305555, 3)
    sp_tail = fd.fit_tail_model(sp_ret, 0.95)
    np.testing.assert_approx_equal(sp_tail[0], 0.20524897090856728, 3)
    (es_tail, _) = fd.expected_shortfall_tail_model(
        0.95, sp_tail[0], sp_tail[1], sp_tail[2], 0.99
    )
    np.testing.assert_approx_equal(es_tail, -5.344663974593491, 3)


def approx_stats_tests(dist):
    """
    use Gauss-Legendre quadrature to estimate
    mean and standard deviation of a particular
    distribution
    """

    mean, var, skew, kurt = dist.stats(moments="mvsk")
    sd = np.sqrt(var)
    mean_est = quad(lambda x: x * dist.pdf(x), -np.inf, np.inf)[0]
    m2_est = quad(lambda x: x * x * dist.pdf(x), -np.inf, np.inf)[0]
    sd_est = np.sqrt(m2_est - mean_est * mean_est)
    skew_est = quad(
        lambda x: dist.pdf(x) * ((x - mean_est) / sd_est) ** 3, -np.inf, np.inf
    )[0]
    kurt_est = quad(
        lambda x: dist.pdf(x) * ((x - mean_est) / sd_est) ** 4, -np.inf, np.inf
    )[0]
    np.testing.assert_approx_equal(mean, mean_est, 1e-6)
    np.testing.assert_approx_equal(sd, sd_est, 1e-6)
    np.testing.assert_approx_equal(skew, skew_est, 1e-6)
    np.testing.assert_approx_equal(kurt, kurt_est, 1e-6)
    return


def test_fit_johnson():
    """
    Test the Johnson SU Distribution - fitting and statistics
    """

    def pdf_su_shape(x, gamma, delta):
        return fd.JohnsonSU(gamma, delta, 0, 1.0).pdf(x)

    lst_dist = {
        "gamma = -1.1, delta = 1.5": [lambda x: pdf_su_shape(x, -1.1, 1.5), "r--"],
        "gamma = -1.1, delta = 0.8": [lambda x: pdf_su_shape(x, -1.1, 0.8), "b--"],
        "gamma = 0.5, delta = 0.8": [lambda x: pdf_su_shape(x, 0.5, 0.8), "g--"],
        "gamma = 0.5, delta = 0.05": [lambda x: pdf_su_shape(x, 0.5, 0.05), "k--"],
        "gamma = 0.0, delta = 0.1": [lambda x: pdf_su_shape(x, 0.0, 0.1), "y--"],
        "Normal": [norm.pdf, "k-"],
    }
    fd.plot_multi_function(
        lst_dist,
        y_label="Probability Density",
        x_lim=[-10.0, 10.0],
        y_log_scale=False,
        title="Johnson S_U Distribution",
    )

    def cdf_su_shape(x, gamma, delta):
        return fd.JohnsonSU(gamma, delta, 0, 1.0).cdf(x)

    lst_dist = {
        "gamma = -1.1, delta = 1.5": [lambda x: cdf_su_shape(x, -1.1, 1.5), "r--"],
        "gamma = -1.1, delta = 0.8": [lambda x: cdf_su_shape(x, -1.1, 0.8), "b--"],
        "gamma = 0.5, delta = 0.8": [lambda x: cdf_su_shape(x, 0.5, 0.8), "g--"],
        "gamma = 0.5, delta = 0.05": [lambda x: cdf_su_shape(x, 0.5, 0.05), "k--"],
        "gamma = 0.0, delta = 0.1": [lambda x: cdf_su_shape(x, 0.0, 0.1), "y--"],
        "Normal": [norm.cdf, "k-"],
    }
    fd.plot_multi_function(
        lst_dist,
        y_label="Cumulative Density",
        x_lim=[-10.0, 10.0],
        y_log_scale=False,
        title="Johnson S_U Distribution",
    )

    jsu = fd.JohnsonSU(0.5, 0.05, 0, 1.0)
    test_val = 3
    cdf = jsu.cdf(test_val)
    cdf_approx = quad(jsu.pdf, -np.inf, test_val)[0]
    np.testing.assert_approx_equal(cdf, cdf_approx, 1e-6)

    jsu = fd.JohnsonSU(-1.1, 0.8, 0, 1.0)
    test_val = 0.5
    cdf = jsu.cdf(test_val)
    cdf_approx = quad(jsu.pdf, -np.inf, test_val)[0]
    np.testing.assert_approx_equal(cdf, cdf_approx, 1e-6)

    df_ret = fd.get_test_data()
    sp_ret = df_ret[df_ret.Ticker == "^GSPC"]["LogReturn"].values
    jsu_fit = fd.JohnsonSU.fitclass(sp_ret)

    jsu_rvs = jsu_fit.rvs(100000)
    u = np.random.uniform(0, 1.0, 10000)
    x = jsu_fit.ppf(u)

    fd.plot_hist_fit(jsu_rvs, "TEST DIST", {"Johnson SU": [jsu_fit.pdf, "r--"]}, 300)

    approx_stats_tests(jsu_fit)

    norm_mod = norm.fit(sp_ret)
    norm_fit = norm(norm_mod[0], norm_mod[1])
    dict_pdf = {
        "Normal Distribution": [norm_fit.pdf, "b-"],
        "Johnson SU Distribution": [jsu_fit.pdf, "y-"],
    }
    fd.plot_hist_fit(sp_ret, "SP 500", dict_pdf, 50, log_scale=True)
    s = fd.edf_stats(sp_ret, jsu_fit)
    print(s)
    np.testing.assert_approx_equal(s[0], 1.1395746330408656, 1e-6)

    x = np.linspace(-10, 10, 1000)
    cdf_vals = jsu_fit.cdf(x)
    plt.plot(x, cdf_vals)

    p = np.linspace(0.01, 0.99, 100)
    x_vals = jsu_fit.ppf(p)
    plt.plot(x_vals, p, color="r")
    plt.show()

    df = pd.DataFrame({"p": p, "x": x_vals})
    df_csv = pd.DataFrame({"x": x, "cdf": cdf_vals})
    df_csv.to_csv("jsu_cdf.csv")
    df.to_csv("jsu_prob.csv")


def test_fit_meixner():
    """
    Test the Miexner Distribution - fitting and statistics
    """

    def pdf_su_shape(x, beta, delta):
        return fd.Meixner(beta, delta, 0, 2.0).pdf(x)

    lst_dist = {
        "beta = -1.3, delta = 1.0": [lambda x: pdf_su_shape(x, -1.3, 1.0), "r--"],
        "beta = 0.0, delta = 1.0": [lambda x: pdf_su_shape(x, 0.0, 1.0), "b--"],
        "beta = 1.5, delta = 1.0": [lambda x: pdf_su_shape(x, 1.5, 1.0), "g--"],
        "Normal": [norm.pdf, "k-"],
    }
    fd.plot_multi_function(
        lst_dist,
        y_label="Probability Density",
        x_lim=[-10.0, 10.0],
        y_log_scale=False,
        title="Meixner Distribution",
    )
    meix = fd.Meixner(
        -1.3, 1.0, 0, 2.0
    )  # Excessive skew makes this a pathological case!
    jsu_dist = fd.JohnsonSU.moment_match(*meix._stats())

    def pq(x):
        return meix.pdf(x) / jsu_dist.pdf(x)
    
    fd.plot_function(pq, [-20, 20], y_lim=[0, 10], title="PDF ratio")


    meix_rv = meix.rvs(size=10000)

    fd.plot_hist_fit(meix_rv, "TEST DIST", lst_dist, ylim=[1e-5,
                    0.8], nbins=50, log_scale=True)

    # https://demonstrations.wolfram.com/NonuniquenessOfOptionPricingUnderTheMeixnerModel/
    def cdf_su_shape(x, beta, delta):
        return fd.Meixner(beta, delta, 0, 2.0).cdf(x)

    lst_dist = {
        "beta = -2.0, delta = 1.0": [lambda x: cdf_su_shape(x, -2.0, 1.0), "r--"],
        "beta = 0.0, delta = 1.0": [lambda x: cdf_su_shape(x, 0.0, 1.0), "b--"],
        "beta = 1.5, delta = 1.0": [lambda x: cdf_su_shape(x, 1.5, 1.0), "g--"],
        "Normal": [norm.cdf, "k-"],
    }
    fd.plot_multi_function(
        lst_dist,
        y_label="Cumulative Density",
        x_lim=[-10.0, 10.0],
        y_lim=[0.0, 1.0],
        y_log_scale=False,
        title="Meixner Distribution",
    )

    #    jsu = fd.JohnsonSU(0.5, 0.05, 0, 1.0)
    #    test_val = 3
    #    cdf = jsu.cdf(test_val)
    #    cdf_approx = quad(jsu.pdf, -np.inf, test_val)[0]
    #    np.testing.assert_approx_equal(cdf, cdf_approx, 1e-6)

    #   jsu = fd.JohnsonSU(-1.1, 0.8, 0, 1.0)
    #   test_val = 0.5
    #    cdf = jsu.cdf(test_val)
    #    cdf_approx = quad(jsu.pdf, -np.inf, test_val)[0]
    #    np.testing.assert_approx_equal(cdf, cdf_approx, 1e-6)

    df_ret = fd.get_test_data()
    sp_ret = df_ret[df_ret.Ticker == "^GSPC"]["LogReturn"].values
    meix_fit = fd.Meixner.fitclass(sp_ret)

    approx_stats_tests(meix_fit)

    norm_mod = norm.fit(sp_ret)
    norm_fit = norm(norm_mod[0], norm_mod[1])
    dict_pdf = {
        "Normal Distribution": [norm_fit.pdf, "b-"],
        "Meixner Distribution": [meix_fit.pdf, "y-"],
    }
    fd.plot_hist_fit(sp_ret, "SP 500", dict_pdf, 50, log_scale=True)
    s = fd.edf_stats(sp_ret, meix_fit)
    print(s)
    np.testing.assert_approx_equal(s[0], 1.1395746330408656, 1e-6)

    x = np.linspace(-10, 10, 1000)
    cdf_vals = meix_fit.cdf(x)
    plt.plot(x, cdf_vals)

    p = np.linspace(0.01, 0.99, 100)
    x_vals = meix_fit.ppf(p)
    plt.plot(x_vals, p, color="r")
    plt.show()


def cdf_testing(dist, test_val):
    """
    Test the cdf of a distribution
    """
    cdf = dist.cdf(test_val)
    cdf_approx = quad(dist.pdf, -np.inf, test_val)[0]
    np.testing.assert_approx_equal(cdf, cdf_approx, 1e-6)
    return


def test_generalised_skewt():
    """
    Short tests of the Generalised Skew-T distribution
    """
    gsd = fd.GeneralisedSkewT(
        0.0, 1.0, 6, 0.0, 1.0
    )  # location and scale parameters at end = 0 and 1

    gsd_skew = fd.GeneralisedSkewT(
        0.5, 1.0, 6, 0.0, 1.0
    )  # location and scale parameters at end = 0 and 1

    approx_stats_tests(gsd_skew)

    approx_stats_tests(gsd)

    lst_dist = {
        "GSD": [gsd.pdf, "r--"],
        "GSD_Skew": [gsd_skew.pdf, "b--"],
        "Normal": [norm.pdf, "k-"],
    }
    fd.plot_multi_function(
        lst_dist,
        y_label="Probability Density",
        x_lim=[-10.0, 10.0],
        y_log_scale=False,
        title="Generalised Skew-T",
    )
    df_ret = fd.get_test_data()
    sp_ret = df_ret[df_ret.Ticker == "^GSPC"]["LogReturn"].values

    gsd_fit = fd.GeneralisedSkewT.fitclass(sp_ret)
    rvs = gsd_fit.rvs(size=100)
    fd.plot_ks(rvs, gsd_fit, "GST", "SP 500")
    approx_stats_tests(gsd_fit)

    cdf_testing(gsd_fit, 0.9)
    cdf_testing(gsd_fit, -1.0)
    cdf_testing(gsd_fit, 1.0)

    cdf_testing(gsd_skew, 3.5)
    cdf_testing(gsd_skew, -3.5)
    cdf_testing(gsd_skew, 0.0)

    x = np.linspace(-10, 10, 1000)
    cdf_vals = gsd_fit.cdf(x)
    plt.plot(x, cdf_vals)

    p = np.linspace(0.01, 0.99, 100)
    x = gsd_fit.ppf(p)
    plt.plot(x, p, color="r")
    plt.show()


#    df = pd.DataFrame({'p': p, 'x': x})
#    df_csv = pd.DataFrame({'x': x, 'cdf': cdf_vals})
#    df_csv.to_csv("gsd_cdf.csv")
#    df.to_csv("gsd_prob.csv")


if __name__ == "__main__":
    test_dists()

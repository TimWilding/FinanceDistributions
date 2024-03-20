Utilities
=========

FastDistributions contains a number of utilities to help with modelling financial distributions:

Statistical Functions
---------------------

* rolling_backtest - a simple utility that uses a rolling window of data to calculate some statistics and build up a dataframe
* parallel_bootstrap - a simple utility that bootstraps a data sample. This is a simpler version of scipy.stats.bootstrap that is easy to use.
* download_yahoo_returns - uses yfinance to download prices and calculate returns in a single line


Plot Functions
--------------
There are several plot functions that use matplotlib to help with the visualisation of different
distributions.

* plot_hist_fit - plots histogram of results with a dictionary of functions for comparison
* plot_mahal_cdf - plots empirical percentiles of Mahalanobis distance vs theoretical percentiles
* plot_mahal_dist - plots time series of Mahalanobis distance
* plot_function - plots a function on a chart
* plot_qq - q-q plots
* plot_indexed_prices - plots time series of prices on a chart indexed to 100 on the first dataframe
* plot_log_cdf - plot pr(X>x) vs x on a log scale to help with looking at tails of the distribution


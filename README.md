Finance Distributions is a set of useful calculations and plots for statistical distribution used in financial models. There are a set of univariate 
distributions that include skewness and kurtosis parameters. There are some routines for calculating robust correlations, and some simple allocation 
recipes for using those routines.

The code can be installed by running python setup.py install. The following distributions are added:
- LevyStableInterp - an interpolated version of the Levy Stable distribution for fast fitting
- GeneralisedSkewT - an analytical distribution with skewness and fat tails that can be used for models
- EntropyDistributions - a generalisation of the Normal Distribution to include skewness and kurtosis based on the principles of maximum entropy
- MeixnerDistribution - a Generalised HyperSecant Distribution that includes skewness and kurtosis parameters
- Johnson SU - the unbounded version of the Johnson distribution

- fit_tail_model - fits a Generalised Pareto Distribution to the tail of a particular sample (Extreme Value Theory)



There are also routines for calculation of useful multivariate statistics relating to correlation
- Multivariate T-Distribution - multivariate distribution including fat tails
- correlations - various adjustments for serial correlation in financial returns
- PRIIPSCalcs - calculations for PRIIPS statistics

And a few useful plotting routines
- plot_qq - multiple qq plots on a single figure
- plot_hist - multiple distribution comparisons on a single histogram
- plot_log_cdf - plots the Cumulative Density Function tails and compares to the sample

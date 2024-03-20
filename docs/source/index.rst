.. FastDistributions documentation master file, created by
   sphinx-quickstart on Mon Mar 18 09:35:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FastDistributions's documentation!
=============================================

FastDistributions is a python package that contains routines used to efficiently model returns of financial instruments
while accounting for outliers. 

For univariate series, the package contains two univariate probability distribution functions that incorporate skewness, and kurtosis measures alongside the
regular location and scale measures. It also includes the Cornish-Fisher distribution models used 
by the EU's PRIIPs regulations.

For multivariate series, the package also contains two robust correlation estimates. 
The first is a robust estimate of the correlation matrix adjusted for autocorrelation, and the
second is a robust, multivariate model of the covariance matrix of financial returns.
Finally, there is a set of utilities for modelling and plotting these different results.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   install
   support
   gen_skew_t
   levy_stable
   priips
   multivariate_t
   correlation
   utilities



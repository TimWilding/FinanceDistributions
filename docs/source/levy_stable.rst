The Levy Stable Distribution
============================

The Levy Stable family of distributions is a generalisation of the Normal distribution that has parameters
that control for skewness and kurtosis. The Levy Stable is an interesting distribution because of its stability property. 
Linear combinations of a stable distribution have the same shape parameter. 
This is appealing because it allows us to present returns as having the same distribution over different time scales whilst having fat tails and skewness. 
Mandelbrot (1963, https://econpapers.repec.org/article/ucpjnlbus/v_3a36_3ay_3a1963_3ap_3a394.htm) popularised the distributions in financial markets 
with an examination of historic cotton prices. He concluded that the stability parameter (α) was around 1.7 by examining the behaviour of the prices in the tail.

This implementation is designed to allow quick fitting of the distribution to a data sample using a cubic interpolated version of the probability distribution function.

The set of grid points used for the interpolation can be built using::
    python build_levy_interpolator.python

This will create a set of 5 files in the directory that can be used as the grid.
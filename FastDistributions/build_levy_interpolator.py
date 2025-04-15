"""
Use https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html to build a
grid of values for the shape of the Levy Stable distribution characterised by log of the probability
distribution function. This grid can then be used for interpolation using the Scipy
RegularGridInterpolator. Levy Stable distributions are heavy-tailed distributions that are often
used to represent equity market behaviour.

https://en.wikipedia.org/wiki/Stable_distribution contains an extensive discussion of the
distribution. The Levy Stable distribution is characterised by two parameters - alpha and beta
- 0 <= alpha <= 2.0 - stability parameter
- -1.0 <= beta <=1.0 - skewness parameter

alpha = 2.0 is equivalent to the Normal distribution

Use Python build_levy_interpolator.py to build the interpolator and save it to a file. The
file is split into several files so that they can be uploaded to GitHub. The files are saved in the
interp_data directory.

The file can then be used in the LevyStable class to get the PDF value for given alpha, beta and x.
"""
import math
import pickle
import time
import numpy as np
from scipy.stats import levy_stable
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
from .splich import file_split

# Here is the grid - what is a good grid?
N_ALPHA = 150
N_BETA = 151
N_PTS = 1000


# Define the grid
alpha_vals = np.linspace(0.4, 2.0, N_ALPHA)
beta_vals = np.linspace(-1.0, 1.0, N_BETA)
x_vals = np.linspace(-0.999 * math.pi / 2, 0.999 * math.pi / 2, N_PTS)
x_tan_vals = np.tan(x_vals)  # Use tan transformation to stretch from -inf to +inf


start = time.time()
parallel_pdf_vals = np.zeros((N_ALPHA, N_BETA, N_PTS))


def process(i):
    """
    Produce grid values for a specific Alpha Value by cycling through beta values
    """
    for j in range(N_BETA):
        parallel_pdf_vals[i, j, :] = np.log(
            levy_stable(alpha_vals[i], beta_vals[j]).pdf(x_tan_vals)
        )


Parallel(n_jobs=32, require="sharedmem")(delayed(process)(i) for i in range(N_ALPHA))
print(f"Parallel Grid calc time = {time.time() - start:.2f} s")
# Define a RegularGridInterpolator - note can specify default interpolation method
# 'linear' is the default, but 'cubic' or 'spline' can be used.
min_pdf_val = np.min(parallel_pdf_vals[np.isfinite(parallel_pdf_vals)])
# We need to do this to make the interpolation work, but it should be done on the output
# so that we can preserve the 0 values of the PDF function
# parallel_pdf_vals[~np.isfinite(parallel_pdf_vals)] = min_pdf_val - 100.0
interp = RegularGridInterpolator(
    (alpha_vals, beta_vals, x_vals), parallel_pdf_vals, method="linear"
)


with open(r'interp_data\ll_levy_stable_interp.pickle', "wb") as handle:
    pickle.dump(interp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Split the file into 5 so Github can handle it
file_split(r'interp_data\ll_levy_stable_interp.pickle', 5)

# We should also run some statistics on the rate of change in the function values
# across the grid
# look at abs(x_ij(k-1) + x_ij(k+1) -2*x_ijk) look at average and min/max across all three axes
# Ideally, we should pick N_ALPHA x N_BETA x N_PTS so that these are roughly even given a set number
# of final grid points
# I guess, theoretically, we could use the Hessian to try and get that roughly right
# An individual position should have a value of ~d2f/dx^2 *x^2

test_vals = np.abs(
    parallel_pdf_vals[:, :, 2:]
    + parallel_pdf_vals[:, :, 0:(N_PTS - 3)]
    - parallel_pdf_vals[:, :, 1:(N_PTS - 2)]
)
# Test out the function
ALPHA_TEST = 1.75
BETA_TEST = 0.52
X_TEST = 3.17
ls = interp([ALPHA_TEST, BETA_TEST, np.arctan(X_TEST)])
print(f"Interpolated value = {ls}")
av = np.log(levy_stable(ALPHA_TEST, BETA_TEST).pdf(X_TEST))
print(f"Actual value = {av}")

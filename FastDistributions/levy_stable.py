"""
Fast fitting of Levy-Stable distributions using interpolation of prebuilt functions
"""
import math
import pickle
import urllib.request
import numpy as np
import pybobyqa
from scipy.stats import levy_stable, FitError
from scipy.ndimage import map_coordinates

LOC_VAR = 0
SCALE_VAR = 1
ALPHA_VAR = 3 # -1 < skew_var< -1
BETA_VAR = 2 # K>0

INTERP_FILE_ID = 'https://www.dropbox.com/scl/fi/ep42m99qba2kekr7w5qz8/ll_levy_stable_interp_tan.pickle?rlkey=l47vzadf8ch7zabu5pakjtf63&dl=0'
DESTINATION = 'model.pkl'

class CartesianGridInterpolator:
    """
    This is an interpolator for a Cartesian Grid - where the grid values
    are evenly spaced
    This code is taken from the page about regular grid interpolation
    It is much faster than RGI because the grid is evenly spaced
    https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_regular_grid.html
    """
    def __init__(self, points, values, method='linear'):
        self.limits = np.array([[min(x), max(x)] for x in points])
        self.values = np.asarray(values, dtype=float)
        self.order = {'linear': 1, 'cubic': 3, 'quintic': 5}[method]

    def __call__(self, xi):
        """
        `xi` here is an array-like (an array or a list) of points.

        Each "point" is an ndim-dimensional array_like, representing
        the coordinates of a point in ndim-dimensional space.
        """
        xi = np.asarray(xi)

        # convert from data coordinates to pixel coordinates
        ns = self.values.shape
        coords = [(n-1)*(val - lo) / (hi - lo)
                  for val, n, (lo, hi) in zip(xi, ns, self.limits)]

        # interpolate
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        return map_coordinates(self.values, coords,
                               order=self.order,
                               cval=np.nan)  # fill_value
    
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent',  'Wget/1.16 (linux-gnu)')]
urllib.request.install_opener(opener)
urllib.request.urlretrieve(INTERP_FILE_ID, DESTINATION)

with open(DESTINATION, 'rb') as handle:
    GRID_INT = pickle.load(handle)
    min_pdf_val = np.min(GRID_INT.values[np.isfinite(GRID_INT.values)])
    GRID_INT.values[~np.isfinite(GRID_INT.values)] = min_pdf_val - 100
    GRID_INT.bounds_error = False
    
# Convert the grid interpolator to a Cartesian Grid Interpolator because it is much faster
GRID_INT = CartesianGridInterpolator(GRID_INT.grid, GRID_INT.values, method='cubic')

class LevyStableInterp():
    @staticmethod
    def ll(x, alpha, beta, loc=0, scale=1):
        """
        Use linear interpolation to estimate the
        log-likelihood value for the following:
         - alpha = levy-stable alpha parameter
         - beta = levy-stable beta parameter
         - loc = location parameter
         - scale = scale parameter
         - x = array of observed values
        returns an array of log-likelihoods for
        each x
        """
        n = x.shape[0]
        alpha_use = alpha
        beta_use = beta
        if n>1:
            alpha_use = alpha*np.ones(n)
            beta_use = beta*np.ones(n)

        norm_x = (x - loc) / scale
        at_vals = np.arctan(norm_x)
        MAX_AT_VAL = 0.99
        at_vals[at_vals>MAX_AT_VAL*math.pi/2] = MAX_AT_VAL*math.pi/2
        at_vals[at_vals<-MAX_AT_VAL*math.pi/2] = -MAX_AT_VAL*math.pi/2
#          print('NORM:   Maximum = {0}, Minimum = {1}'.format(np.max(norm_x), np.min(norm_x)))
#          print('ARCTAN: Maximum = {0}, Minimum = {1}'.format(np.max(at_vals), np.min(at_vals)))
        log_like = GRID_INT((alpha_use, beta_use, at_vals)) - np.log(scale)
        return log_like

    @staticmethod
    def fit(x, prob=None, display_progress=True):
        """
        Uses MLE to return Generalised Skew-T parameters this is
        set to be similar to fit in the scipy.stats package so
        it should raises a TypeError or ValueError if the input
        is invalid and a scipy.stats.FitError if fitting fails
        or the fit produced would be invalid
        Returns
        -------
        parameter_tuple : tuple of floats
            Estimates for any shape parameters (if applicable), followed by
            those for location and scale. For most random variables, shape
            statistics will be returned, but there are exceptions (e.g.
            ``norm``).
        """
        sol = _gen_levy_fit(x, prob, display_progress)
        return (sol.x[ALPHA_VAR], sol.x[BETA_VAR], sol.x[LOC_VAR], sol.x[SCALE_VAR])

    @staticmethod
    def fitclass(x, prob=None, display_progress=True):
        """
        Uses MLE to return Generalised Skew-T parameters this is
        set to be similar to fit in the scipy.stats package so
        it should raises a TypeError or ValueError if the input
        is invalid and a scipy.stats.FitError if fitting fails
        or the fit produced would be invalid
        Returns
        -------
        parameter_tuple : tuple of floats
            Estimates for any shape parameters (if applicable), followed by
            those for location and scale. For most random variables, shape
            statistics will be returned, but there are exceptions (e.g.
            ``norm``).
        """
        sol = _gen_levy_fit(x, prob, display_progress)
        return levy_stable(sol.x[ALPHA_VAR], sol.x[BETA_VAR], sol.x[LOC_VAR], sol.x[SCALE_VAR])



def ll(returns_data, x, display_progress):
    if display_progress:
        print('Parameters')
        print('==========')
        print(f'Alpha = {x[ALPHA_VAR]}')
        print(f'Beta = {x[BETA_VAR]}')
        print(f'Loc = {x[LOC_VAR]}')
        print(f'Scale = {x[SCALE_VAR]}')
    log_like = np.sum(LevyStableInterp.ll(returns_data, x[ALPHA_VAR],
                                          x[BETA_VAR], x[LOC_VAR],
                                          x[SCALE_VAR]))
    if display_progress:
        print(f'Likelihood = {log_like}')
    return log_like

def _gen_levy_fit(returns_data, prob=None, display_progress=True):
    """
	Routine to fit a generalised skew-t distribution to a set of data sets with
	weights using the BOBYQA algorithm of Powell (https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf)
	This routine does not require derivatives

    Note that this seems to alight on gridd points of the interpolated values and 
    may need looking at to see if we can refine the solution
    """

    n = returns_data.shape[0]
    if prob is None:
        wt_prob = np.ones(n) / n
    else:
        wt_prob = prob

    mean_dist = np.average(returns_data, weights=wt_prob)
    sd_dist= np.sqrt(np.cov(returns_data, aweights=wt_prob))

# Initialise NLopt engine to use BOBYQA algorithm

# Set up starting parameters & upper & lower bounds
    init_x = np.array([mean_dist, sd_dist, 0.0, 1.5])
    lower = np.array([-10*sd_dist, 0.01*sd_dist, -0.99, 0.5])
    upper = np.array([10*sd_dist, 5*sd_dist, 0.99, 2.0])


# NLopt function must return a gradient even if algorithm is derivative-free
# - this function will return an empty gradient

    ll_func = lambda x : -ll(returns_data, x, display_progress)
    if display_progress:
        print("Fitting Levy Stable Distribution")
        print("=======================================")
        print("Initial Log Likelihood")
        print(ll_func(init_x))
    soln = pybobyqa.solve(ll_func, init_x, bounds=(lower, upper), scaling_within_bounds=True)
  #  https://nlopt.readthedocs.io/en/latest/NLopt_Reference/ says that ROUNDOFF_LIMITED results are usually useful so I'm going to assume they are
  #  print("")
  #  print("Solution xmin = %s" % str(soln.x))
  #  print("Objective value f(xmin) = %.10g" % (soln.fun))
  #  print("Needed %g objective evaluations" % soln.nfev)
  #  print("Exit flag = %g" % soln.status)
  #  print(soln.message)
    if display_progress:
        print(soln)
    if soln.flag == soln.EXIT_INPUT_ERROR:
        raise ValueError('input value problem in fitting routine')
    if (soln.flag < soln.EXIT_SUCCESS) & (soln.flag!=soln.EXIT_LINALG_ERROR):
        print(soln)
        raise FitError("Fitting error in Fast Levy Stable fitting")
    return soln

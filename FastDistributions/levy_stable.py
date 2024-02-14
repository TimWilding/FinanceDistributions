"""
Fast fitting of Levy-Stable distributions using interpolation of prebuilt functions
"""
import math
import numpy as np
import pickle
import urllib.request
import pybobyqa
from scipy.stats import levy_stable

MEAN_VAR = 0
VAR_VAR = 1
ALPHA_VAR = 3 # -1 < skew_var< -1
BETA_VAR = 2 # K>0

INTERP_FILE_ID = 'https://www.dropbox.com/scl/fi/ep42m99qba2kekr7w5qz8/ll_levy_stable_interp_tan.pickle?rlkey=l47vzadf8ch7zabu5pakjtf63&dl=0'
DESTINATION = 'model.pkl'


opener = urllib.request.build_opener()
opener.addheaders = [('User-agent',  'Wget/1.16 (linux-gnu)')]
urllib.request.install_opener(opener)
urllib.request.urlretrieve(INTERP_FILE_ID, DESTINATION)
with open(DESTINATION, 'rb') as handle:
    GRID_INT = pickle.load(handle)
    min_pdf_val = np.min(GRID_INT.values[np.isfinite(GRID_INT.values)])
    GRID_INT.values[~np.isfinite(GRID_INT.values)] = min_pdf_val - 100


class LevyStableInterp():
  #  def __init__(self):
  #      super().__init__(self)
    @staticmethod
    def ll(x, alpha, beta, loc=0, scale=1):
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
        ll = GRID_INT((alpha_use, beta_use, at_vals), method='linear') - np.log(scale)
        return ll

    @staticmethod
    def fit(x, prob=None, display_progress=True):
        sol = gen_levy_fit(x, prob, display_progress)
        return levy_stable(sol.x[ALPHA_VAR], sol.x[BETA_VAR], sol.x[MEAN_VAR], sol.x[VAR_VAR])



def ll(returns_data, x, display_progress):
    if display_progress:
        print('Parameters')
        print('==========')
        print('Alpha = {0}'.format(x[ALPHA_VAR]))
        print('Beta = {0}'.format(x[BETA_VAR]))
        print('Loc = {0}'.format(x[MEAN_VAR]))
        print('Scale = {0}'.format(x[VAR_VAR]))
    ll = np.sum(LevyStableInterp.ll(returns_data, x[ALPHA_VAR], x[BETA_VAR], x[MEAN_VAR], x[VAR_VAR]))
    if display_progress:
        print('Likelihood = {0}'.format(ll))
    return ll

def gen_levy_fit(returns_data, prob=None, display_progress=True):
    """
	Routine to fit a generalised skew-t distribution to a set of data sets with
	weights using the BOBYQA algorithm of Powell (https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf)
	This routine does not require derivatives
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
    lower = np.array([-10*sd_dist, 1e-8, -0.99, 0.5])
    upper = np.array([10*sd_dist, 5*sd_dist, 0.99, 2.0])


# NLopt function must return a gradient even if algorithm is derivative-free
# - this function will return an empty gradient

    ll_func = lambda x : -ll(returns_data, x, display_progress)
    if display_progress:
        print("Fitting Levy Stable Distribution")
        print("=======================================")
        print("Initial Log Likelihood")
        print(ll_func(init_x))
    soln = pybobyqa.solve(ll_func, init_x, bounds=(lower, upper))
  #  https://nlopt.readthedocs.io/en/latest/NLopt_Reference/ says that ROUNDOFF_LIMITED results are usually useful so I'm going to assume they are
  #  print("")
  #  print("Solution xmin = %s" % str(soln.x))
  #  print("Objective value f(xmin) = %.10g" % (soln.fun))
  #  print("Needed %g objective evaluations" % soln.nfev)
  #  print("Exit flag = %g" % soln.status)
  #  print(soln.message)
    if display_progress:
         print(soln)
    return soln
import numpy as np
import pybobyqa
from scipy.stats import rv_continuous, FitError
from dist_plots import plot_multi_function
from .stat_functions import _basestats


def log_pdf_johnson_sb(x, gamma, delta, xi, lambd):
    """
    Computes the log of the PDF for the Johnson SB distribution.

    Parameters:
        x (float or np.ndarray): Value(s) where the PDF is evaluated.
        gamma (float): Shape parameter.
        delta (float): Shape parameter.
        xi (float): Location parameter.
        lambd (float): Scale parameter.

    Returns:
        float or np.ndarray: Log of the PDF at x.
    """
    if np.any(x <= xi) or np.any(x >= xi + lambd):
        raise ValueError("x must be in the range (xi, xi + lambd)")

    z = (x - xi) / (lambd - x + xi)
    transformed = gamma + delta * np.log(z)

    log_pdf = -0.5 * transformed**2 - np.log(lambd) - np.log(x - xi) - np.log(lambd - x + xi) - 0.5 * np.log(2 * np.pi)
    return log_pdf

def log_pdf_johnson_su(x, gamma, delta, xi, lambd):
    """
    Computes the log of the PDF for the Johnson SU distribution.

    Parameters:
        x (float or np.ndarray): Value(s) where the PDF is evaluated.
        gamma (float): Shape parameter.
        delta (float): Shape parameter.
        xi (float): Location parameter.
        lambd (float): Scale parameter.

    Returns:
        float or np.ndarray: Log of the PDF at x.
    """
    if lambd <= 0:
        raise ValueError("lambd must be positive.")
    if delta <=0:
        raise ValueError("delta must be positive.")
    
    z = (x - xi) / lambd
    sinh_inv = np.arcsinh(z)
    
    transformed = gamma + delta * sinh_inv

    log_pdf = np.log(delta) -0.5 * transformed**2 - np.log(lambd) - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(1 + z**2)
    return log_pdf

# Example usage:
# Parameters for the distribution
#gamma = 0.5
#delta = 1.2
#xi = 1.0
#lambd = 2.0

# Evaluate log-PDF for specific values
#x_values = np.array([1.5, 1.8, 2.1])
#print("Log-PDF for Johnson SB:", log_pdf_johnson_sb(x_values, gamma, delta, xi, lambd))
#print("Log-PDF for Johnson SU:", log_pdf_johnson_su(x_values, gamma, delta, xi, lambd))




class JohnsonSU(rv_continuous):
    """
    Class to represent the JohnsonSU distribution using the same form as the
    scipy.stats.distribution objects
    """

    def __init__(self, gamma, delta, loc: float = 0.0, scale: float = 1.0):
#        gamma (float): Shape parameter.
#        delta (float): Shape parameter.
#        xi (float): Location parameter.
#        lambd (float): Scale parameter.
        super().__init__(self)
        self.gamma = gamma
        self.delta = delta
        self.lambd = scale
        self.xi = loc

    def _pdf(self, x):
        pd = np.exp(self._logpdf(x))
        if isinstance(pd, (float, np.float64)):
        # Handle single float
            return 0.0 if np.isnan(pd) else pd
        elif isinstance(pd, np.ndarray):
        # Handle numpy array
            return np.nan_to_num(pd, nan=0.0)
        else:
            raise TypeError('Must be a float or numpy array')

    def _logpdf(self, x):
        return log_pdf_johnson_su(x, self.gamma, self.delta, self.xi, self.lambd)
    
    def _var(self):
        m = np.exp(1.0/self.delta**2)
        var = (self.lambd**2 / 2) * (m-1)* (m*np.cosh(2 * self.gamma / self.delta) + 1) 
        return var

    def std(self):
        return np.sqrt(self.var())

    def _mean(self):
        m = np.exp(0.5/self.delta**2)
        mean = self.xi - self.lambd * m * np.sinh(self.gamma / self.delta)
        return mean
    
    def _stats(self):
        return _basestats(self)

    
    @staticmethod
    def fit(returns_data, prob=None, display_progress=True):
        return _johnson_su_fit(returns_data, prob, display_progress)
    
    @staticmethod
    def fitclass(returns_data, prob=None, display_progress=True):
        sol =  _johnson_su_fit(returns_data, prob, display_progress)
        return JohnsonSU(
            sol.x[0], sol.x[1], sol.x[2], sol.x[3]
        )


def _johnson_su_fit(returns_data, prob=None, display_progress=True):
    """
        Routine to fit a generalised skew-t distribution to a set of data sets with
        weights using the BOBYQA algorithm of Powell
    (https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf)
        This routine does not require derivatives
    """

    n = returns_data.shape[0]
    if prob is None:
        wt_prob = np.ones(n) / n
    else:
        wt_prob = prob
    mean_dist = np.average(returns_data, weights=wt_prob)
    sd_dist = np.sqrt(np.cov(returns_data, aweights=wt_prob))

    # Initialise NLopt engine to use BOBYQA algorithm

    # Set up starting parameters & upper & lower bounds
    # Note that this assumes that
    init_x = np.array(
        [ 
            0.0, 
            1.0, 
            mean_dist,
            sd_dist, 
        ]
    )
    EPS = 1e-8
    lower = np.array(
        [
            -10,
            EPS,
            -100 * sd_dist,
            1e-8 * sd_dist,
        ]
    )
    upper = np.array(
        [ 
            10, 
            10,
            100 * sd_dist,
            10 * sd_dist,                       
        ]
    )

    # NLopt function must return a gradient even if algorithm is derivative-free
    # - this function will return an empty gradient
    ll_func = lambda x: -np.sum(
        wt_prob
        * log_pdf_johnson_su(
            returns_data,
            x[0],
            x[1],
            x[2],
            x[3],
        )
    )
    if display_progress:
        print("Fitting Generalised Skew-T Distribution")
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
    if soln.flag == soln.EXIT_INPUT_ERROR:
        raise ValueError("input value problem in fitting routine")
    if (soln.flag < soln.EXIT_SUCCESS) & (soln.flag != soln.EXIT_LINALG_ERROR):
        print(soln)
        raise FitError("Fitting error in Generalised Skew-T fitting")
    return soln

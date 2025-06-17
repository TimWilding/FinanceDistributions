"""
Simple regression calculations using statsmodels and numpy used in FastDistributions.
"""
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pandas as pd
import numpy as np


def sample_regress(
    df_sample, regress_col="Bitcoin", print_summary=False, add_intercept=True
):
    """
    Regress the contents of the regress_col on to the contents
    of the other columns. All columns are standardised before
    regression so that we return the standardised coefficients
    which makes them comparable
    """
    try:
        del df_sample["Date"]
    except KeyError:
        print("No date column found")
    lst_x_cols = list(df_sample.columns)
    scaler = StandardScaler()
    df_scale = scaler.fit_transform(df_sample)
    df_scale = pd.DataFrame(df_scale, columns=df_sample.columns)
    lst_x_cols.remove(regress_col)
    X = df_scale[lst_x_cols]
    y = df_scale[regress_col]
    if add_intercept:
        X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    if print_summary:
        print(model.summary())
    p = pd.DataFrame(pd.DataFrame({"coeff": model.params, "StdErr": model.bse}))

    return p.reset_index()


def wls_regress(y, X, w):
    """
    Weighted Least-Squares using the weights given in the
    vector w.
    y = X * b_hat + e
    b_hat = (X'W X)^{-1} X' W y

    Parameters
    y: dependent variable array-like, shape (n_samples,)
    X: independent variables array-like, shape (n_samples, n_features)
    w: weights array-like, shape (n_samples,) - weights for each sample

    Returns
    b_hat: array-like, shape (n_features,) - estimated coefficients
    s: float - standard error of the regression

    """
    n = X.shape[0]
    sw = np.sqrt(w)
    if X.ndim == 1:
        wX = X * sw
        wX = wX[:, np.newaxis]
    else:
        wX = X * sw[:, np.newaxis]

    wy = y * sw
    b_hat = np.linalg.lstsq(wX, wy)[0]
    e = wy - wX @ b_hat
    s = np.sqrt(np.sum(e * e) / np.sum(w))
    return b_hat, s

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pandas as pd


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
    lst_X_cols = list(df_sample.columns)
    scaler = StandardScaler()
    df_scale = scaler.fit_transform(df_sample)
    df_scale = pd.DataFrame(df_scale, columns=df_sample.columns)
    lst_X_cols.remove(regress_col)
    X = df_scale[lst_X_cols]
    y = df_scale[regress_col]
    if add_intercept:
        X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    if print_summary:
        print(model.summary())
    p = pd.DataFrame(pd.DataFrame({"coeff": model.params, "StdErr": model.bse}))

    return p.reset_index()

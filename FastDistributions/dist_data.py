"""
Utility functions for downloading data from e.g. yfinance
"""

import os
import os.path
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pandas import read_csv


def _download_single_asset(asset, download_period, start, end):
    """
    download a single asset using the yfinance api and append
    a ticker and name column
    https://github.com/ranaroussi/yfinance/blob/main/README.md
    contains suggestions which might help robustness
    """
    ticker = asset[0]

    if ticker is None:
        return None
    if start is None:
        df_history = yf.download([ticker], threads=False, period=download_period)
    else:
        if end is None:
            df_history = yf.download([ticker], threads=False, start=start)
        else:
            df_history = yf.download([ticker], threads=False, start=start, end=end)
    df_history["Ticker"] = ticker
    df_history["Name"] = asset[1]
    df_history = df_history.reset_index()
    return df_history


def download_yahoo_returns(
    assets,
    download_period="10y",
    endweekday: int = 2,
    start=None,
    end=None,
    threads: int = 4,
    price_field="Close",
) -> pd.DataFrame:
    """
    Takes the list of assets and downloads them from Yahoo Finance
    using the tickers - calculates daily returns, and adds a marker
    for weekly and monthly returns if you need to calculate those as well
    Assets is a list of tuples like
        [('^GSPC', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')]
    The first entryof each tuple is the Yahoo ticker,
    and the second entry is a long name

    Available download period parameters are:
        period =  1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    or you can specify start and end date:
        start & end using 'YYYY-MM-DD' string

    Note that though yf can download multiple tickers
    the resultant dataframe is not arranged in a way that makes calculation
    of returns easy with a column for each ticker.
    """

    # multi thread the code
    # using https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread
    pool = ThreadPool(processes=threads)
    lst_results = []
    for dl_asset in assets:
        print(f"Downloading {dl_asset[1]}")
        lst_results.append(
            pool.apply_async(
                _download_single_asset, (dl_asset, download_period, start, end)
            )
        )

    df_out = None
    for result in lst_results:
        df_history = result.get()
        if df_history is not None:
            if df_out is None:
                df_out = df_history
            else:
                df_out = pd.concat([df_out, df_history])

    print("Completed Download")
    # Convert to datetime and drop timezone info
    df_out["Date"] = pd.to_datetime(df_out["Date"], utc=True).dt.normalize()

    return calculate_returns(
        df_out, "Ticker", "Date", price_field=price_field, endweekday=endweekday
    )


def calculate_returns(
    df,
    identifier_field="Ticker",
    date_field="Date",
    price_field="Close",
    endweekday: int = 2,
) -> pd.DataFrame:
    """
    Calculate returns for a dataframe with three columns - id, date,
    and price
    """
    # Calculate the returns
    df_out = df
    df_out.loc[:, "LogPrice"] = np.log(df_out[price_field])
    df_out.sort_values([identifier_field, date_field], inplace=True)
    df_out["LogReturn"] = 100 * df_out.groupby(identifier_field)["LogPrice"].diff()
    # Calculate end of month
    df_out.loc[:, "EndOfMonth"] = df_out[date_field] + pd.offsets.MonthEnd(0)
    # Calculate end of the week (assuming the week ends on Wednesday)
    df_out.loc[:, "EndOfWeek"] = (
        df_out[date_field] + pd.offsets.Day(-1)
    ) + pd.offsets.Week(weekday=endweekday)
    df_out = df_out.dropna()  # get rid of the log returns that are rubbish
    print("Calculated Returns")
    return df_out


def get_test_data(download_years=-1):
    """
    Returns a test dataset downloaded from Yahoo earlier
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    df = read_csv(os.path.join(data_path, "asset_returns_history.csv"), parse_dates=[0])
    if download_years > 0:
        latest_date = df["Date"].max()
        start_date = latest_date - relativedelta(years=download_years)
        df = df[(df["Date"] >= start_date) & (df["Date"] <= latest_date)]

    return calculate_returns(df)

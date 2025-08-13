"""
Utility functions for downloading data from e.g. yfinance
"""

import os
import os.path
import hashlib

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pandas import read_csv
import requests
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type, wait_random_exponential

@retry(wait=wait_fixed(1), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
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
        df_history = yf.download(
            [ticker], threads=False, period=download_period, multi_level_index=True
        )
    else:
        if end is None:
            df_history = yf.download(
                [ticker], threads=False, start=start, multi_level_index=True
            )
        else:
            df_history = yf.download(
                [ticker], threads=False, start=start, end=end, multi_level_index=True
            )
    df_history.columns = df_history.columns.get_level_values(
        0
    )  # Remove the second level of the multiindex
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
    price_field="Close",
) -> pd.DataFrame:
    """
    Takes the list of assets and downloads them from Yahoo Finance
    using the tickers - calculates daily returns, and adds a marker
    for weekly and monthly returns if you need to calculate those as well

    Assets - a list of tuples like
        [('^GSPC', 'SP 500'), ('^FTSE', 'FTSE 100'), ('^N225', 'Nikkei 225')]
        The first entryof each tuple is the Yahoo ticker,
        and the second entry is a long name

    download_period - Available download period parameters are:
        period =  1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        or you can specify start and end date:
        start & end using 'YYYY-MM-DD' string
    endweekday - the day of the week that you want to use as the end of the week
                0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday,
                5 = Saturday, 6 = Sunday, default is Wednesday
                used in calculating weekly returns
    price_field - 'Close' is the default, but you can use 'Adj Close' or 'Open' etc

    Note that though yf can download multiple tickers
    the resultant dataframe is not arranged in a way that makes calculation
    of returns easy with a column for each ticker.
    """

    # multi thread the code
    # using https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread
    #  pool = ThreadPool(processes=threads)
    lst_results = []
    for dl_asset in assets:
        print(f"Downloading {dl_asset[1]}")
        lst_results.append(
            _download_single_asset(dl_asset, download_period, start, end)
        )

    df_out = None
    for result in lst_results:
        df_history = result  # .get()
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


def read_cached_excel(url, cache_dir="cache", **read_excel_kwargs) -> pd.DataFrame:
    """
    Reads an Excel file into a DataFrame, caching it locally.

    Parameters:
        url (str): The URL of the Excel file.
        cache_dir (str): The directory where cached files are stored. Default is "cache".
        **read_excel_kwargs: Additional arguments passed to `pandas.read_excel`.

    Returns:
        pd.DataFrame: The DataFrame read from the Excel file.
    """
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a unique filename for the cached file based on the URL hash
    file_hash = hashlib.md5(url.encode()).hexdigest()
    local_filename = os.path.join(cache_dir, f"{file_hash}.xlsx")

    # Check if the file is already cached
    if not os.path.exists(local_filename):
        # Download the file if it is not cached
        try:
            response = requests.get(url, timeout=100)
            response.raise_for_status()  # Raise an error for bad responses
            with open(local_filename, "wb") as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download the file from {url}. Error: {e}") from e

    # Read the Excel file into a DataFrame and return it
    return pd.read_excel(local_filename, **read_excel_kwargs)


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

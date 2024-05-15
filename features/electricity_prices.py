import requests
from datetime import datetime, date, timedelta
import pandas as pd


def electricity_prices(historical: bool = False, area: list = None, start: str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), end: str = (date.today()).strftime("%Y-%m-%d")) -> pd.DataFrame:
    """
    Fetches electricity prices from Energinet (Dataservice API).

    Parameters:
    - historical (bool): If True, fetches historical data from start date to end date. If False, fetches data for the current day. Default is False.
    - start (str): Define a start date for the API call. Default is 'Yesterday'.
    - end (str): Define a end date for the API call. Default is 'Today'.
    
    Returns:
    - pd.DataFrame: DataFrame with electricity prices for different areas in Denmark (DK1, DK2).
    """

    # Define the API URL for electricity prices data and make a request to the API
    API_URL = 'https://api.energidataservice.dk/dataset/Elspotprices'
    r = requests.get(API_URL , params={
                'offset': 0,
                'start': start+'T00:00',
                'end': end+'T23:59',
                'filter': '{"PriceArea":["DK1", "DK2"]}',
                'sort': 'HourUTC DESC'
            })

    # Extract JSON data from the response and make a DataFrame
    data = r.json()['records']
    df = pd.DataFrame(data)

    # Format date and time
    df["date"] = df["HourDK"].map(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime("%Y-%m-%d"))
    df['datetime'] = pd.to_datetime(df['HourDK'])
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    # Divide the price to KWH
    df['SpotPriceDKK_KWH'] = df['SpotPriceDKK'] / 1000

    # Drop unnecessary columns
    df.drop('SpotPriceDKK', axis=1, inplace=True)

    # Filter the df based on the area
    if area is None:
        filtered_df = df
    else:
        filtered_df = df[df['PriceArea'].isin(area)]

    # Filter the df based on the historical parameter
    today = (date.today()).strftime("%Y-%m-%d")
    if historical:
        filtered_df = filtered_df[filtered_df.date != today]
    else:
        filtered_df = filtered_df[filtered_df.date == today]

    # Convert datetime to timestamp in milliseconds and add it as a new column
    filtered_df["timestamp"] = filtered_df["datetime"].apply(lambda x: int(x.timestamp() * 1000))

    # Reset the index to avoid duplicate entries
    filtered_df.reset_index(drop=True, inplace=True)

    # Select relevant columns for electricity prices data and reorder them
    reordered_df = filtered_df[['timestamp', 'datetime', 'date', 'hour', 'PriceArea', 'SpotPriceDKK_KWH']]

    # Unpivot DataFrame
    reordered_df = reordered_df.melt(id_vars=['timestamp', 'datetime', 'date', 'hour', "PriceArea"], var_name="attribute", value_name="value")

    # Combine columns into a single "heading" column
    reordered_df["heading"] = reordered_df["PriceArea"] + "_" + reordered_df["attribute"]

    # Drop the columns that are no longer needed
    reordered_df.drop(columns=["PriceArea"], inplace=True)
    reordered_df.drop(columns=["attribute"], inplace=True)

    # Pivot DataFrame
    electricity_prices = reordered_df.pivot_table(index=['timestamp', 'datetime', 'date', 'hour'], columns="heading", values="value").reset_index()

    # Converting column names to lowercase for consistency
    electricity_prices.columns = list(map(str.lower, electricity_prices.columns))

    # Replace spaces in column names with underscores
    electricity_prices.columns = electricity_prices.columns.str.replace(' ', '_')

    # Return the DataFrame with electricity prices data
    return electricity_prices

def forecast_renewable_energy(historical: bool = False, area: str = None, start: str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), end: str = (date.today()).strftime("%Y-%m-%d")) -> pd.DataFrame:
    """
    Fetches electricity prices from Energinet (Dataservice API).

    Parameters:
    - historical (bool): If True, fetches historical data from start date to end date. If False, fetches data for the current day. Default is False.
    - start (str): Define a start date for the API call. Defaul is 'Yesterday'.
    - end (str): Define a end date for the API call. Default is 'Today'.
    
    Returns:
    - pd.DataFrame: DataFrame with electricity prices for different areas in Denmark (DK1, DK2).
    """

    # Define the API URL for forecasted renewable energy data and make a request to the API
    API_URL = 'https://api.energidataservice.dk/dataset/Forecasts_Hour'
    r = requests.get(API_URL , params={
                'offset': 0,
                'start': start+'T00:00',
                'end': end+'T23:59',
            })

    # Extract JSON data from the response and make a DataFrame
    data = r.json()['records']
    df = pd.DataFrame(data)

    # Format date and time
    df["date"] = df["HourDK"].map(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime("%Y-%m-%d"))
    df['datetime'] = pd.to_datetime(df['HourDK'])
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    # Drop unnecessary columns
    df.drop('Forecast5Hour', axis=1, inplace=True)
    df.drop('Forecast1Hour', axis=1, inplace=True)
    df.drop('ForecastDayAhead', axis=1, inplace=True)
    df.drop('ForecastCurrent', axis=1, inplace=True)
    df.drop('HourUTC', axis=1, inplace=True)
    df.drop('HourDK', axis=1, inplace=True)
    df.drop('TimestampDK', axis=1, inplace=True)
    df.drop('TimestampUTC', axis=1, inplace=True)

    # Filter the df based on the area
    if area is None:
        filtered_df = df
    else:
        filtered_df = df[df['PriceArea'].isin(area)]

    # Filter the df based on the historical parameter
    today = (date.today()).strftime("%Y-%m-%d")
    if historical:
        filtered_df = filtered_df[df.date != today]
    else:
        filtered_df = filtered_df[df.date == today]

    # Convert datetime to timestamp in milliseconds and add it as a new column
    filtered_df["timestamp"] = filtered_df["datetime"].apply(lambda x: int(x.timestamp() * 1000))

    # Multiply specified columns by 1000
    filtered_df["ForecastIntraday_KWH"] = filtered_df["ForecastIntraday"] * 1000

    # Drop unnecessary columns
    df.drop('ForecastIntraday', axis=1, inplace=True)

    # Reset the index to avoid duplicate entries
    filtered_df.reset_index(drop=True, inplace=True)

    # Select relevant columns for forecasted renewable energy data and reorder them
    reordered_df = filtered_df[['timestamp', 'datetime', 'date', 'hour', 'PriceArea', 'ForecastType', 'ForecastIntraday_KWH']]

    # Unpivot DataFrame
    reordered_df = reordered_df.melt(id_vars=["timestamp", 'datetime', "date", "hour", "PriceArea", "ForecastType"], var_name="attribute", value_name="value")

    # Combine columns into a single "heading" column
    reordered_df["heading"] = reordered_df["PriceArea"] + "_" + reordered_df["ForecastType"] + "_" + reordered_df["attribute"]
    
    # Drop the columns that are no longer needed
    reordered_df.drop(columns=["PriceArea"], inplace=True)
    reordered_df.drop(columns=["ForecastType"], inplace=True)
    reordered_df.drop(columns=["attribute"], inplace=True)

    # Pivot DataFrame
    forecast_renewable_energy = reordered_df.pivot_table(index=["timestamp", "datetime", "date", "hour"], columns="heading", values="value").reset_index()

    # Converting column names to lowercase for consistency
    forecast_renewable_energy.columns = list(map(str.lower, forecast_renewable_energy.columns))

    # Replace spaces in column names with underscores
    forecast_renewable_energy.columns = forecast_renewable_energy.columns.str.replace(' ', '_')

    # Return the DataFrame with forecast renewable energy data
    return forecast_renewable_energy
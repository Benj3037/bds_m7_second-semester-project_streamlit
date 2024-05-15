import requests
from datetime import datetime, date, timedelta
import pandas as pd


def historical_weather_measures(historical: bool = False, lat: float = 57.048, lon: float = 9.9187, start: str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), end: str = (date.today()).strftime("%Y-%m-%d")) -> pd.DataFrame:
    """
    Fetches weather measures from Open Meteo API.

    Parameters:
    - historical (bool): If True, fetches historical data from start date to end date. If False, fetches data for the current day. Default is False.
    - latitude and longitude: Default to coordinates to Aalborg.
    - start (str): Define a start date for the API call. Defaul is 'Yesterday'.
    - end (str): Define a end date for the API call. Default is 'Today'.

    Returns:
    - pd.DataFrame: DataFrame with weather data for defined area.
    """

    # Define the API URL for historical weather data and make a request to the API
    API_URL = 'https://archive-api.open-meteo.com/v1/archive'
    r = requests.get(API_URL , params={
                'latitude': lat,
                'longitude': lon,
                'start_date': start,
                'end_date': end,
                'hourly': 'temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,weather_code,cloud_cover,wind_speed_10m,wind_gusts_10m'
            })

    # Extract JSON data from the response and make a DataFrame
    data = r.json()['hourly']
    df = pd.DataFrame(data)

    # Extract date from the 'time' column and convert it to datetime format
    df["date"] = df['time'].str[:10]
    df['datetime'] = pd.to_datetime(df['time'])
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    # Filter the DataFrame based on whether historical data is requested or not
    today = (date.today()).strftime("%Y-%m-%d")
    if historical:
        df = df[df.date != today]
    else:
        df = df[df.date == today]

    # Convert datetime to timestamp in milliseconds and add it as a new column
    df["timestamp"] = df["datetime"].apply(lambda x: int(x.timestamp() * 1000))

    # Select relevant columns for weather data and reorder them
    weather = df[['timestamp', 'datetime', 'date', 'hour', 'temperature_2m', 'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'weather_code', 'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m']]

    # Deleting rows with missing values
    weather = weather.dropna()

    # Return the DataFrame with weather data
    return weather

def forecast_weather_measures(lat: float = 57.048, lon: float = 9.9187, forecast_length : int = 1) -> pd.DataFrame:
    """
    Fetches weather forecast from Open Meteo API.

    Parameters:
    - latitude and longitude: Default to coordinates to Aalborg.
    - forecast_length: Defining the length of the weather forecast. Default is 1 day.

    Returns:
    - pd.DataFrame: DataFrame with weather forecast for defined area.
    """

    API_URL = 'https://api.open-meteo.com/v1/forecast'
    r = requests.get(API_URL , params={
                'latitude': lat,
                'longitude': lon,
                'hourly': 'temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,weather_code,cloud_cover,wind_speed_10m,wind_gusts_10m',
                "forecast_days": forecast_length
            })

    # Extract JSON data from the response and make a DataFrame
    data = r.json()['hourly']
    df = pd.DataFrame(data)

    # Extract date from the 'time' column and convert it to datetime format
    df["date"] = df['time'].str[:10]
    df['datetime'] = pd.to_datetime(df['time'])
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    # Convert datetime to timestamp in milliseconds and add it as a new column
    df["timestamp"] = df["datetime"].apply(lambda x: int(x.timestamp() * 1000))

    # Select relevant columns for forecast weather data and reorder them
    forecast_weather = df[['timestamp', 'datetime', 'date', 'hour', 'temperature_2m', 'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'weather_code', 'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m']]

    # Convert columns to float
    forecast_weather['relative_humidity_2m'] = forecast_weather['relative_humidity_2m'].astype(float)
    forecast_weather['weather_code'] = forecast_weather['weather_code'].astype(float)
    forecast_weather['cloud_cover'] = forecast_weather['cloud_cover'].astype(float)

    # Deleting rows with missing values
    forecast_weather = forecast_weather.dropna()

    # Return the DataFrame with forecast weather data
    return forecast_weather
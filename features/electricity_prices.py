import requests
from datetime import datetime, date, timedelta
import pandas as pd
from features import calendar

def electricity_prices(historical: bool = False, area: list = None, start: str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), end: str = (date.today()).strftime("%Y-%m-%d")) -> pd.DataFrame:
    """
    Fetches electricity prices from Energinet (Dataservice API).

    Parameters:
    - historical (bool): If True, fetches historical data from start date to end date. If False, fetches data for the current day. Default is False.
    - area (list): Define the area for the API call. Default is None.
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
    df['date'] = pd.to_datetime(df['date'])

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

def electricity_prices_window(historical: bool = False, area: list = None, start: str = ('2020-01-01'), end: str = datetime.now().date() + timedelta(days=(7*1))) -> pd.DataFrame:
    """
    Fetches electricity prices and make rolling windows.

    Parameters:
    - historical (bool): If True, fetches historical data from start date to end date. If False, fetches data for the current day. Default is False.
    - area (list): List of areas to fetch data for. Default is None.
    - start (str): Define a start date for the API call. Default is '2020-01-01'.
    - end (str): Define a end date for the API call. Default is one week from today.
    
    Returns:
    - pd.DataFrame: DataFrame with electricity prices window.
    """
    
    electricity_df = electricity_prices(
        historical=True, 
        area=area, 
        start=start
    )

    calendar_hours_df = calendar.calendar_denmark(
        freq='H',
        start=start, 
        end=end
    )

    # Merging the electricity and calendar dataframes
    merge_data = pd.merge(electricity_df, calendar_hours_df, how='right', left_on='timestamp', right_on='timestamp')

    # Drop and rename columns
    merge_data = merge_data.drop(columns=['date_x', 'datetime_x', 'hour_x'])
    merge_data = merge_data.rename(columns={'date_y': 'date', 
                                            'datetime_y': 'datetime', 
                                            'hour_y': 'hour'})

    merge_data_for_rolling = merge_data

    # Get today's date (only the date part, without time)
    today = datetime.today().date()

    # Create a boolean mask where the 'datetime' column's date part is before today
    mask = merge_data_for_rolling['datetime'].dt.date < today

    # Fill NaN values in the 'dk1_spotpricedkk_kwh' column with the previous row's value
    # Only fill NaNs for rows where the date is before today (using the mask)
    merge_data_for_rolling.loc[mask, 'dk1_spotpricedkk_kwh'] = merge_data_for_rolling['dk1_spotpricedkk_kwh'].ffill()

    # Defining a copy of the combined data to avoid modifying the original dataframe
    electricity_window_df = merge_data_for_rolling

    # Adding a column with the mean for the previous 1 week
    electricity_window_df['prev_1w_mean'] = electricity_window_df['dk1_spotpricedkk_kwh'].rolling(window=24*(1*7), min_periods=1).mean()

    # Adding a column with the mean for the previous 2 weeks
    electricity_window_df['prev_2w_mean'] = electricity_window_df['dk1_spotpricedkk_kwh'].rolling(window=24*(2*7), min_periods=1).mean()

    # Adding a column with the mean for the previous 4 weeks
    electricity_window_df['prev_4w_mean'] = electricity_window_df['dk1_spotpricedkk_kwh'].rolling(window=24*(4*7), min_periods=1).mean()

    # Adding a column with the mean for the previous 6 weeks
    electricity_window_df['prev_6w_mean'] = electricity_window_df['dk1_spotpricedkk_kwh'].rolling(window=24*(6*7), min_periods=1).mean()

    # Adding a column with the mean for the previous 8 weeks
    electricity_window_df['prev_8w_mean'] = electricity_window_df['dk1_spotpricedkk_kwh'].rolling(window=24*(8*7), min_periods=1).mean()

    # Adding a column with the mean for the previous 12 weeks
    electricity_window_df['prev_12w_mean'] = electricity_window_df['dk1_spotpricedkk_kwh'].rolling(window=24*(12*7), min_periods=1).mean()

    # Filter the df based on the historical parameter
    today = (date.today()).strftime("%Y-%m-%d")
    if historical:
        electricity_window_df_filtered = electricity_window_df[electricity_window_df.date < today]
    else:
        electricity_window_df_filtered = electricity_window_df[electricity_window_df.date >= today]

    # Only keep the columns 'timestamp', 'prev_1w_mean', 'prev_2w_mean', and 'prev_4w_mean'
    electricity_window_df_filtered=electricity_window_df_filtered[['timestamp', 'datetime', 'prev_1w_mean', 'prev_2w_mean', 'prev_4w_mean', 'prev_6w_mean', 'prev_8w_mean', 'prev_12w_mean']]

    # Return the DataFrame with electricity prices data
    return electricity_window_df_filtered
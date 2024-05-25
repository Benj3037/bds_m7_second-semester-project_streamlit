from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import holidays

def calendar_denmark(freq: str = 'D', start: str = (pd.Timestamp.now().year - 2), end: str = (pd.Timestamp.now().year + 1)) -> pd.DataFrame:
    """
    Fetches calendar for Denmark.

    Parameters:
    - freq (str): Define the frequency of the calendar. Default 'D' (daily).
    - start (str): Define a start year. Default current year minus 2 years.
    - end (str): Define a end year. Default current year and 1 year ahead.

    Returns:
    - pd.DataFrame: DataFrame with calendar for Denmark.
    """

    # Define the start and end dates
    start_date = str(start) + ' 00:00:00'
    end_date = str(end) + ' 23:00:00'

    # Create the datetime range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Convert to DataFrame
    calendar_df = pd.DataFrame(date_range, columns=['datetime'])

    # Extract date-related features
    calendar_df['date'] = calendar_df['datetime'].dt.date
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    calendar_df['hour'] = calendar_df['datetime'].dt.hour
    calendar_df['dayofweek'] = calendar_df['datetime'].dt.dayofweek
    calendar_df['day'] = calendar_df['datetime'].dt.day
    calendar_df['month'] = calendar_df['datetime'].dt.month
    calendar_df['year'] = calendar_df['datetime'].dt.year
    calendar_df['timestamp'] = calendar_df['datetime'].apply(lambda x: int(x.timestamp() * 1000))

    # Select country
    dk_holidays = holidays.Denmark()

    # Initialize a list to store workday data
    workday_data = []

    # Iterate over each date in the date range
    for date in calendar_df['date'].unique():
        # Check if it's a holiday
        if pd.Timestamp(date) in dk_holidays:
            is_workday = 0  # Not a workday
        # Check if it's a weekend (Saturday or Sunday)
        elif pd.Timestamp(date).dayofweek >= 5:
            is_workday = 0  # Not a workday
        else:
            is_workday = 1  # Workday

        # Append data to the list
        workday_data.append({'date': pd.Timestamp(date), 'workday': is_workday})

    # Create a DataFrame for workday information
    workday_df = pd.DataFrame(workday_data)

    # Merge workday information into the calendar DataFrame
    calendar_df = pd.merge(calendar_df, workday_df, on='date', how='right')

    return calendar_df
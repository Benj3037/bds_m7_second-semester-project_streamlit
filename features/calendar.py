from datetime import datetime, date
import numpy as np
import pandas as pd
import holidays


def dk_calendar() -> pd.DataFrame:
    """
    Fetches calendar for Denmark.

    Returns:
    - pd.DataFrame: DataFrame with danish calendar.
    """

    df = pd.read_csv('https://raw.githubusercontent.com/Camillahannesbo/MLOPs-Assignment-/main/data/calendar_incl_holiday.csv', delimiter=';', usecols=['date', 'type'])

    # Formatting the date column to 'YYYY-MM-DD' dateformat
    df["date"] = df["date"].map(lambda x: datetime.strptime(x, '%d/%m/%Y').strftime("%Y-%m-%d"))

    # Add features to the calendar dataframe
    df['date_'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date_'].dt.dayofweek
    df['day'] = df['date_'].dt.day
    df['month'] = df['date_'].dt.month
    df['year'] = df['date_'].dt.year
    df['workday'] = np.where(df['type'] == 'Not a Workday', 0, 1)

    # Drop the columns 'type' and 'date_' to finalize the calendar dataframe
    calendar = df.drop(['type','date_'], axis=1)

    # Return the DataFrame with calendar data
    return calendar

def dk_calendar_auto() -> pd.DataFrame:
    """
    Fetches calendar for Denmark.

    Returns:
    - pd.DataFrame: DataFrame with danish calendar.
    """

    # Define the start and end dates
    start_date = pd.Timestamp.now().year - 2
    end_date = pd.Timestamp.now().year + 1

    # Generate the date range
    date_range = pd.date_range(start=str(start_date), end=str(end_date), freq='D')

    # Convert the date range to a DataFrame
    calendar_df = pd.DataFrame({'date': date_range})

    # Add features to the calendar dataframe
    calendar_df['dayofweek'] = calendar_df['date'].dt.dayofweek
    calendar_df['day'] = calendar_df['date'].dt.day
    calendar_df['month'] = calendar_df['date'].dt.month
    calendar_df['year'] = calendar_df['date'].dt.year

    # Select country
    dk_holidays = holidays.Denmark()

    # Initialize lists to store data
    data = []

    # Iterate over each date in the date range
    for date in calendar_df['date']:
        # Check if it's a holiday
        if date in dk_holidays:
            is_workday = 0  # Not a workday
        # Check if it's a weekend (Saturday or Sunday)
        elif date.dayofweek >= 5:
            is_workday = 0  # Not a workday
        else:
            is_workday = 1  # Workday

        # Append data to list
        data.append({'date': date, 'workday': is_workday})

    # Create DataFrame
    workday_df = pd.DataFrame(data)

    # Merge workday information into the calendar DataFrame
    calendar_df = pd.merge(calendar_df, workday_df, on='date', how='left')

    return calendar_df
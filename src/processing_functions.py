import pandas as pd
import numpy as np
from geopy.distance import geodesic

def convert_columns_to_integers(df):
    """
    Converts specified columns in the DataFrame to integer format, handling NaN values appropriately.

    Parameters:
    df (pd.DataFrame): The DataFrame with columns to convert.

    Returns:
    pd.DataFrame: The DataFrame with specified columns converted to integers.
    """
    
    # Convert 'yr_renovated'
    df['yr_renovated'].replace(np.nan, 0.000, inplace=True)  # Replace NaN entries with 0.000
    df['yr_renovated'] = (df['yr_renovated'] / 10).astype('int64')  # Correct for factor 10 and convert to int64
    df['yr_renovated'] = df['yr_renovated'].replace(0, pd.NA)  # Replace 0's with <NA>

    # Convert other columns to int64
    int_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15', 'sqft_lot15', 'price']
    for col in int_columns:
        df[col] = df[col].astype('int64')
    
    # Convert 'sqft_basement'
    df['sqft_basement'].replace(np.nan, -1.000, inplace=True)  # Replace NaN entries with -1.000 as a placeholder
    df['sqft_basement'] = df['sqft_basement'].astype('int64')  # Convert to int64
    df['sqft_basement'] = df['sqft_basement'].replace(-1, pd.NA)  # Replace -1's with <NA>

    return df

def convert_to_boolean(df, column_name, yes_value=1.000, no_value=0.000):
    """
    Converts a specified column in the DataFrame to 'yes'/'no' format, preserving NaN values.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to convert.
    column_name (str): The name of the column to convert to 'yes'/'no'.
    yes_value: The value in the column that should be mapped to 'yes' (default is 1.000).
    no_value: The value in the column that should be mapped to 'no' (default is 0.000).

    Returns:
    pd.DataFrame: The DataFrame with the specified column converted to 'yes'/'no' format.
    """
    # Map the values to 'yes' and 'no', preserving NaN
    df[column_name] = df[column_name].map({no_value: 'no', yes_value: 'yes', np.nan: np.nan})
    
    # The column will now contain strings 'yes', 'no', or NaN
    return df

def remove_older_duplicates(df, id_column='id', date_column='date'):
    """
    Removes duplicates from the DataFrame, keeping only the most recent entries based on the date column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    id_column (str): The column to check for duplicates (default is 'id').
    date_column (str): The column used to determine the most recent entry (default is 'date').

    Returns:
    pd.DataFrame: A DataFrame with only the most recent entries for each ID.
    """
    # Step 1: Sort the DataFrame by 'id' and 'date' in descending order
    df_sorted = df.sort_values(by=[id_column, date_column], ascending=[True, False])
    
    # Step 2: Drop duplicates based on 'id', keeping the first (most recent date) occurrence
    df_no_older_duplicates = df_sorted.drop_duplicates(subset=id_column, keep='first')
    
    # Step 3: Sort the DataFrame back to its original order if needed
    df = df_no_older_duplicates.sort_index()
    
    return df

def distance_to_center(row):
    """
    Calculates the distance in miles from a house to the center of Seattle.

    Parameters:
    row (pd.Series): A row from the DataFrame containing 'lat' and 'long' columns.

    Returns:
    float: The distance from the house to Seattle's center in miles.
    """
    seattle_center = (47.6062, -122.3321)
    house_location = (row['lat'], row['long'])

    return geodesic(seattle_center, house_location).miles

def classify_by_zip_code(zipcode):
    '''A function that classifies the data according to the ZIP codes of Seattle'''
    # List with the actual ZIP codes of Seattle
    city_zip_codes = [
        98101, 98102, 98103, 98104, 98105,
        98106, 98107, 98108, 98109, 98112,
        98115, 98116, 98117, 98118, 98119,
        98121, 98122, 98125, 98126, 98133,
        98134, 98136, 98144, 98146, 98154,
        98164, 98174, 98177, 98178, 98195,
        98199
    ]
    if zipcode in city_zip_codes:
        return 'city'
    else:
        return 'countryside'

import pandas as pd

def descriptive_label_mapping(df, column_name, category_mapping, category_order=None):
    """
    Converts a specified column in the DataFrame from numeric categories to string names, preserving NaN values.
    Optionally, converts the new column to a Categorical type with a specified order.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to convert.
    column_name (str): The name of the column to convert to string categories.
    category_mapping (dict): A dictionary mapping numeric values to string categories.
    category_order (list, optional): A list specifying the desired order of the categorical labels.

    Returns:
    pd.DataFrame: The DataFrame with the specified column converted to string categories and optionally
                  converted to an ordered Categorical type.
    """
    # Use .map() to apply the mapping, preserving NaN values
    df[f'{column_name}_cat'] = df[column_name].map(category_mapping)
    
    # If a category order is provided, convert the new column to an ordered Categorical type
    if category_order:
        df[f'{column_name}_cat'] = pd.Categorical(df[f'{column_name}_cat'], categories=category_order, ordered=True)

    return df

def month_to_season(month):
    """
    This function takes a numeric representation of a month (from 1 to 12) as input and returns 
    the corresponding season as a string ('Winter', 'Spring', 'Summer', or 'Autumn'). 
    The function is useful for mapping individual months to their respective meteorological 
    seasons, allowing for seasonal analysis of data.
    
    Parameters:
    - month (int): An integer value representing the month (1 for January, 2 for February, ..., 12 for December).
    
    Returns:
    - str: A string indicating the season that corresponds to the given month.
    
    Season Mapping:
    - Winter: December (12), January (1), February (2)
    - Spring: March (3), April (4), May (5)
    - Summer: June (6), July (7), August (8)
    - Autumn: September (9), October (10), November (11)
    """
    
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # 9, 10, 11
        return 'Autumn'


import pandas as pd
import numpy as np

def compare_dataframes(df1, df2):
    """
    Compares two DataFrames to check if they have the same shape and identical data.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame to compare.
    df2 (pd.DataFrame): The second DataFrame to compare.

    Returns:
    str: A message indicating whether the DataFrames are identical or not.
    """
    # Check if the DataFrames have the same shape
    if df1.shape != df2.shape:
        return "The DataFrames have different shapes, so they are not the same."
    else:
        # Check if all the elements in the DataFrames are the same
        if df1.equals(df2):
            return "The DataFrames are identical."
        else:
            return "The DataFrames are not identical. There are differences in the data."
        
def check_duplicates(df, id_column='id', date_column='date'):
    """
    Check for duplicate rows in the DataFrame based on the specified 'id' and 'date' columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to check for duplicates.
    id_column (str): The column name to check for duplicates (default is 'id').
    date_column (str): The column name to further check duplicates against (default is 'date').

    Returns:
    str: A message indicating whether duplicates exist and how many there are.
    """
    # Filter the DataFrame to include only rows with duplicated 'id'
    duplicates_df = df[df[id_column].duplicated(keep=False)]
    
    # Check for duplicate rows based on both 'id' and 'date'
    duplicates_count = duplicates_df.duplicated(subset=[id_column, date_column], keep=False).sum()
    
    if duplicates_count == 0:
        return f"There are no duplicates based on '{id_column}' and '{date_column}' together."
    else:
        return f"There are {duplicates_count} duplicate rows in the DataFrame based on '{id_column}' and '{date_column}' together."

def check_float_columns_for_int_conversion(df):
    """
    Check which float columns in a DataFrame can be converted to integer data types.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.

    Returns:
    dict: A dictionary where the keys are column names and the values are booleans indicating
          whether the float column can be safely converted to integers.
    """
    can_be_int = {}

    # Iterate through columns in the DataFrame
    for column in df.columns:
        # Check if the column data type is float
        if pd.api.types.is_float_dtype(df[column]):
            # Check if the float values only have a zero decimal part by comparing with their integer conversion
            # Use dropna() to ignore NaN values as NaN cannot be compared to integers
            can_be_int[column] = np.all((df[column].dropna() == df[column].dropna().astype(int)))

    return can_be_int

def check_potential_boolean_columns(df):
    """
    Identifies and returns the column names in a DataFrame that can be interpreted as boolean type,
    considering columns with NaNs or missing values.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.

    Returns:
    list: A list of column names that can be interpreted as boolean type.
    """
    boolean_columns = []

    for column in df.columns:
        # Drop NaNs and check if the remaining values are either 0/1 or True/False
        non_na_values = df[column].dropna().unique()
        if set(non_na_values).issubset({0, 1, True, False}):
            boolean_columns.append(column)

    return boolean_columns

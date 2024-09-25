import pandas as pd
import json

def apply_category_order_from_json(df, column_name, json_filepath):
    """
    Loads a category order from a JSON file and applies it to a specified categorical column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to which the category order will be applied.
    column_name (str): The name of the column to convert to an ordered Categorical type.
    json_filepath (str): The file path of the JSON file containing the category order.

    Returns:
    pd.DataFrame: The DataFrame with the specified column converted to an ordered Categorical type.
    """
    # Load the saved category order from the JSON file
    with open(json_filepath, 'r') as f:
        category_order = json.load(f)

    if not isinstance(category_order, list):
        raise TypeError("Category order must be a list.")

    if not category_order:
        raise ValueError("Category order cannot be empty.")

    # Reapply the order to the specified column as a Categorical feature
    df[column_name] = pd.Categorical(df[column_name], categories=category_order, ordered=True)

    return df
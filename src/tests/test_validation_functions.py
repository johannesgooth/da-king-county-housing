import unittest
import pandas as pd
import numpy as np
import sys
import os

# Adjust the import path to include the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, parent_dir)

from validation_functions import (
    compare_dataframes,
    check_duplicates,
    check_float_columns_for_int_conversion,
    check_potential_boolean_columns
)

class TestValidationFunctions(unittest.TestCase):
    
    def setUp(self):
        # Sample DataFrames for compare_dataframes
        self.df_identical_1 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        self.df_identical_2 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        self.df_different_shape = pd.DataFrame({
            'A': [1, 2],
            'B': ['x', 'y']
        })
        self.df_different_data = pd.DataFrame({
            'A': [1, 2, 4],
            'B': ['x', 'y', 'w']
        })
        
        # Sample DataFrames for check_duplicates
        self.df_no_duplicates = pd.DataFrame({
            'id': [1, 2, 3],
            'date': ['2021-01-01', '2021-02-01', '2021-03-01'],
            'value': [100, 200, 300]
        })
        self.df_with_duplicates = pd.DataFrame({
            'id': [1, 1, 2, 3, 3, 3],
            'date': ['2021-01-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-03-01', '2021-04-01'],
            'value': [100, 100, 200, 300, 300, 400]
        })
        self.df_all_duplicates = pd.DataFrame({
            'id': [1, 1, 1],
            'date': ['2021-01-01', '2021-01-01', '2021-01-01'],
            'value': [100, 100, 100]
        })
        self.df_empty = pd.DataFrame(columns=['id', 'date', 'value'])
        
        # Sample DataFrames for check_float_columns_for_int_conversion
        self.df_float_int_convertible = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0],
            'C': [7.1, 8.2, 9.3]  # Not convertible
        })
        self.df_float_with_nans = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, 5.0, np.nan],
            'C': [7.1, 8.2, 9.3]
        })
        self.df_no_float = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        self.df_negative_floats = pd.DataFrame({
            'A': [-1.0, -2.0, -3.0],
            'B': [-4.5, -5.0, -6.0]
        })
        
        # Sample DataFrames for check_potential_boolean_columns
        self.df_boolean_columns = pd.DataFrame({
            'is_active': [1, 0, 1, 0, np.nan],
            'is_new': [True, False, True, False, np.nan],
            'is_verified': [1, 1, 1, 1, 1]
        })
        self.df_non_boolean_columns = pd.DataFrame({
            'status': [1, 2, 3, 4, 5],
            'flag': [0, 1, 2, 3, 4],
            'description': ['a', 'b', 'c', 'd', 'e']
        })
        self.df_mixed_columns = pd.DataFrame({
            'is_available': [1, 0, 1, 0, np.nan],
            'count': [10, 20, 30, 40, 50],
            'verified': [True, False, True, False, True]
        })
        self.df_all_boolean = pd.DataFrame({
            'flag1': [0, 1, 0, 1],
            'flag2': [False, True, False, True],
            'flag3': [1, 1, 0, 0]
        })
        self.df_with_nans_boolean = pd.DataFrame({
            'flag1': [1, 0, 1, np.nan],
            'flag2': [True, False, True, np.nan],
            'flag3': [0, 1, 0, 1]
        })
        self.df_non_boolean_with_nans = pd.DataFrame({
            'status': [1, 2, np.nan, 4],
            'flag': [0, 1, np.nan, 2],
            'description': ['a', 'b', np.nan, 'd']
        })
    
    # Tests for compare_dataframes
    def test_compare_dataframes_identical(self):
        """
        Test that compare_dataframes returns "The DataFrames are identical." for identical DataFrames.
        """
        result = compare_dataframes(self.df_identical_1, self.df_identical_2)
        self.assertEqual(result, "The DataFrames are identical.")
    
    def test_compare_dataframes_different_shapes(self):
        """
        Test that compare_dataframes detects DataFrames with different shapes.
        """
        result = compare_dataframes(self.df_identical_1, self.df_different_shape)
        self.assertEqual(result, "The DataFrames have different shapes, so they are not the same.")
    
    def test_compare_dataframes_different_data(self):
        """
        Test that compare_dataframes detects DataFrames with the same shape but different data.
        """
        result = compare_dataframes(self.df_identical_1, self.df_different_data)
        self.assertEqual(result, "The DataFrames are not identical. There are differences in the data.")
    
    def test_compare_dataframes_different_columns(self):
        """
        Test that compare_dataframes detects DataFrames with different columns.
        """
        df_diff_cols_1 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        df_diff_cols_2 = pd.DataFrame({
            'A': [1, 2, 3],
            'C': ['x', 'y', 'z']
        })
        result = compare_dataframes(df_diff_cols_1, df_diff_cols_2)
        self.assertEqual(result, "The DataFrames are not identical. There are differences in the data.")
    
    def test_compare_dataframes_different_order(self):
        """
        Test that compare_dataframes detects DataFrames with the same data but different row orders.
        """
        df_order_1 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        df_order_2 = pd.DataFrame({
            'A': [3, 2, 1],
            'B': ['z', 'y', 'x']
        })
        result = compare_dataframes(df_order_1, df_order_2)
        self.assertEqual(result, "The DataFrames are not identical. There are differences in the data.")
    
    # Tests for check_duplicates
    def test_check_duplicates_no_duplicates(self):
        """
        Test that check_duplicates correctly identifies no duplicates.
        """
        result = check_duplicates(self.df_no_duplicates, id_column='id', date_column='date')
        expected_message = "There are no duplicates based on 'id' and 'date' together."
        self.assertEqual(result, expected_message)
    
    def test_check_duplicates_with_duplicates(self):
        """
        Test that check_duplicates correctly identifies duplicates.
        """
        result = check_duplicates(self.df_with_duplicates, id_column='id', date_column='date')
        expected_message = "There are 4 duplicate rows in the DataFrame based on 'id' and 'date' together."
        self.assertEqual(result, expected_message)
    
    def test_check_duplicates_all_duplicates(self):
        """
        Test that check_duplicates correctly identifies duplicates when all rows are duplicates.
        """
        result = check_duplicates(self.df_all_duplicates, id_column='id', date_column='date')
        expected_message = "There are 3 duplicate rows in the DataFrame based on 'id' and 'date' together."
        self.assertEqual(result, expected_message)
    
    def test_check_duplicates_empty_dataframe(self):
        """
        Test that check_duplicates correctly handles an empty DataFrame.
        """
        result = check_duplicates(self.df_empty, id_column='id', date_column='date')
        expected_message = "There are no duplicates based on 'id' and 'date' together."
        self.assertEqual(result, expected_message)
    
    def test_check_duplicates_single_row(self):
        """
        Test that check_duplicates correctly handles a DataFrame with a single row.
        """
        df_single = pd.DataFrame({
            'id': [1],
            'date': ['2021-01-01'],
            'value': [100]
        })
        result = check_duplicates(df_single, id_column='id', date_column='date')
        expected_message = "There are no duplicates based on 'id' and 'date' together."
        self.assertEqual(result, expected_message)
    
    def test_check_duplicates_custom_columns(self):
        """
        Test that check_duplicates works correctly with custom id and date columns.
        """
        df_custom = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'signup_date': ['2021-01-01', '2021-01-01', '2021-02-01', '2021-02-02', '2021-03-01'],
            'value': [100, 100, 200, 250, 300]
        })
        result = check_duplicates(df_custom, id_column='user_id', date_column='signup_date')
        expected_message = "There are 2 duplicate rows in the DataFrame based on 'user_id' and 'signup_date' together."
        self.assertEqual(result, expected_message)
    
    # Tests for check_float_columns_for_int_conversion
    def test_check_float_columns_can_be_converted(self):
        """
        Test that check_float_columns_for_int_conversion identifies float columns that can be converted to integers.
        """
        result = check_float_columns_for_int_conversion(self.df_float_int_convertible)
        expected = {
            'A': True,
            'B': True,
            'C': False
        }
        self.assertEqual(result, expected)
    
    def test_check_float_columns_cannot_be_converted(self):
        """
        Test that check_float_columns_for_int_conversion identifies float columns that cannot be converted to integers.
        """
        df_non_convertible = pd.DataFrame({
            'A': [1.1, 2.2, 3.3],
            'B': [4.0, 5.5, 6.0]
        })
        result = check_float_columns_for_int_conversion(df_non_convertible)
        expected = {
            'A': False,
            'B': False
        }
        self.assertEqual(result, expected)
    
    def test_check_float_columns_with_nans(self):
        """
        Test that check_float_columns_for_int_conversion handles NaN values correctly.
        """
        result = check_float_columns_for_int_conversion(self.df_float_with_nans)
        expected = {
            'A': True,  # 1.0, NaN, 3.0 can be converted (ignoring NaN)
            'B': True,  # 4.0, 5.0, NaN can be converted
            'C': False  # 7.1, 8.2, 9.3 cannot be converted
        }
        self.assertEqual(result, expected)
    
    def test_check_float_columns_no_float(self):
        """
        Test that check_float_columns_for_int_conversion handles DataFrames with no float columns.
        """
        result = check_float_columns_for_int_conversion(self.df_no_float)
        expected = {}
        self.assertEqual(result, expected)
    
    def test_check_float_columns_with_negative_values(self):
        """
        Test that check_float_columns_for_int_conversion handles negative float values correctly.
        """
        result = check_float_columns_for_int_conversion(self.df_negative_floats)
        expected = {
            'A': True,
            'B': False
        }
        self.assertEqual(result, expected)
    
    # Tests for check_potential_boolean_columns
    def test_check_potential_boolean_columns_yes(self):
        """
        Test that check_potential_boolean_columns identifies columns that can be interpreted as boolean.
        """
        result = check_potential_boolean_columns(self.df_boolean_columns)
        expected = ['is_active', 'is_new', 'is_verified']
        self.assertListEqual(sorted(result), sorted(expected))
    
    def test_check_potential_boolean_columns_no(self):
        """
        Test that check_potential_boolean_columns does not identify non-boolean columns.
        """
        result = check_potential_boolean_columns(self.df_non_boolean_columns)
        expected = []
        self.assertListEqual(result, expected)
    
    def test_check_potential_boolean_columns_mixed(self):
        """
        Test that check_potential_boolean_columns correctly identifies only the boolean columns.
        """
        result = check_potential_boolean_columns(self.df_mixed_columns)
        expected = ['is_available', 'verified']
        self.assertListEqual(sorted(result), sorted(expected))
    
    def test_check_potential_boolean_columns_all_boolean(self):
        """
        Test that check_potential_boolean_columns identifies all columns as boolean when appropriate.
        """
        result = check_potential_boolean_columns(self.df_all_boolean)
        expected = ['flag1', 'flag2', 'flag3']
        self.assertListEqual(sorted(result), sorted(expected))
    
    def test_check_potential_boolean_columns_with_nans(self):
        """
        Test that check_potential_boolean_columns correctly identifies boolean columns even with NaNs.
        """
        result = check_potential_boolean_columns(self.df_with_nans_boolean)
        expected = ['flag1', 'flag2', 'flag3']
        self.assertListEqual(sorted(result), sorted(expected))
    
    def test_check_potential_boolean_columns_all_non_boolean_with_nans(self):
        """
        Test that check_potential_boolean_columns does not incorrectly identify non-boolean columns even with NaNs.
        """
        result = check_potential_boolean_columns(self.df_non_boolean_with_nans)
        expected = []
        self.assertListEqual(result, expected)
    
    # Additional Tests for robustness
    
    def test_compare_dataframes_empty_dataframes(self):
        """
        Test that compare_dataframes correctly identifies two empty DataFrames as identical.
        """
        df_empty1 = pd.DataFrame()
        df_empty2 = pd.DataFrame()
        result = compare_dataframes(df_empty1, df_empty2)
        self.assertEqual(result, "The DataFrames are identical.")
    
    def test_compare_dataframes_one_empty_one_non_empty(self):
        """
        Test that compare_dataframes correctly identifies when one DataFrame is empty and the other is not.
        """
        df_empty = pd.DataFrame()
        df_non_empty = pd.DataFrame({
            'A': [1],
            'B': [2]
        })
        result = compare_dataframes(df_empty, df_non_empty)
        self.assertEqual(result, "The DataFrames have different shapes, so they are not the same.")
    
    def test_check_duplicates_non_default_columns(self):
        """
        Test that check_duplicates works correctly with non-default id and date columns.
        """
        df_custom = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'signup_date': ['2021-01-01', '2021-01-01', '2021-02-01', '2021-02-02', '2021-03-01'],
            'value': [100, 100, 200, 250, 300]
        })
        result = check_duplicates(df_custom, id_column='user_id', date_column='signup_date')
        expected_message = "There are 2 duplicate rows in the DataFrame based on 'user_id' and 'signup_date' together."
        self.assertEqual(result, expected_message)

# Run the tests
if __name__ == '__main__':
    unittest.main()
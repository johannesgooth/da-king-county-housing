import unittest
from unittest.mock import mock_open, patch
import pandas as pd
import json
import sys
import os

# Adjust the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from basic_functions import apply_category_order_from_json

class TestApplyCategoryOrderFromJson(unittest.TestCase):
    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'category': ['apple', 'banana', 'cherry', 'banana', 'apple']
        })

        # Expected category order
        self.category_order = ['banana', 'apple', 'cherry']

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps(['banana', 'apple', 'cherry']))
    def test_apply_category_order_success(self, mock_file):
        """
        Test that the function correctly applies the category order from JSON.
        """
        result_df = apply_category_order_from_json(self.df.copy(), 'category', 'dummy_path.json')

        # Check if the column has categorical dtype
        self.assertTrue(pd.api.types.is_categorical_dtype(result_df['category']))

        # Check if categories are ordered correctly
        self.assertEqual(list(result_df['category'].cat.categories), self.category_order)

        # Check if the categorical is ordered
        self.assertTrue(result_df['category'].cat.ordered)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps(['banana', 'apple']))
    def test_apply_category_order_with_missing_categories(self, mock_file):
        """
        Test how the function handles categories in the DataFrame that are not present in the JSON.
        """
        result_df = apply_category_order_from_json(self.df.copy(), 'category', 'dummy_path.json')

        # Categories in JSON are missing 'cherry'
        self.assertEqual(list(result_df['category'].cat.categories), ['banana', 'apple'])

        # 'cherry' should be set to NaN since it's not in the category order
        self.assertTrue(result_df['category'].isna().sum() == 1)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps(['banana', 'apple', 'cherry', 'date']))
    def test_apply_category_order_with_extra_categories(self, mock_file):
        """
        Test how the function handles additional categories in the JSON that are not present in the DataFrame.
        """
        result_df = apply_category_order_from_json(self.df.copy(), 'category', 'dummy_path.json')

        # JSON has an extra category 'date'
        self.assertEqual(list(result_df['category'].cat.categories), ['banana', 'apple', 'cherry', 'date'])

        # All original categories should be present and correctly ordered
        self.assertTrue(result_df['category'].cat.ordered)
        self.assertFalse(result_df['category'].isna().any())

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_apply_category_order_file_not_found(self, mock_file):
        """
        Test that the function raises a FileNotFoundError when the JSON file does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            apply_category_order_from_json(self.df.copy(), 'category', 'nonexistent.json')

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    def test_apply_category_order_invalid_json(self, mock_file):
        """
        Test that the function raises a JSONDecodeError when the JSON file contains invalid JSON.
        """
        with self.assertRaises(json.JSONDecodeError):
            apply_category_order_from_json(self.df.copy(), 'category', 'invalid.json')

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps(['banana', 'apple', 'cherry']))
    def test_apply_category_order_column_not_in_df(self, mock_file):
        """
        Test that the function raises a KeyError when the specified column does not exist in the DataFrame.
        """
        with self.assertRaises(KeyError):
            apply_category_order_from_json(self.df.copy(), 'nonexistent_column', 'dummy.json')

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps(['banana', 'apple', 'cherry']))
    def test_apply_category_order_empty_dataframe(self, mock_file):
        """
        Test that the function can handle an empty DataFrame.
        """
        empty_df = pd.DataFrame(columns=['category'])
        result_df = apply_category_order_from_json(empty_df, 'category', 'dummy.json')

        # Ensure the DataFrame remains empty and the category dtype is set correctly
        self.assertTrue(result_df.empty)
        self.assertIn('category', result_df.columns)
        self.assertTrue(pd.api.types.is_categorical_dtype(result_df['category']))
        self.assertEqual(list(result_df['category'].cat.categories), self.category_order)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps([]))
    def test_apply_category_order_empty_category_order(self, mock_file):
        """
        Test that the function handles an empty category order list.
        """
        with self.assertRaises(ValueError):
            apply_category_order_from_json(self.df.copy(), 'category', 'empty_order.json')

if __name__ == '__main__':
    unittest.main()
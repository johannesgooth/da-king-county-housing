import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import sys
import os

# Adjust the import path to include the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from processing_functions import (
    convert_columns_to_integers,
    convert_to_boolean,
    remove_older_duplicates,
    distance_to_center,
    classify_by_zip_code,
    descriptive_label_mapping,
    month_to_season
)

class TestPreprocessingFunctions(unittest.TestCase):
    
    def setUp(self):
        # Sample DataFrame for testing convert_columns_to_integers
        self.df_int = pd.DataFrame({
            'yr_renovated': [np.nan, 1990, 2000, np.nan],
            'bedrooms': [3.0, 4.0, 2.0, 3.0],
            'sqft_living': [1500.0, 2000.0, 1800.0, 1600.0],
            'sqft_lot': [5000.0, 6000.0, 5500.0, 5800.0],
            'sqft_above': [1200.0, 1800.0, 1600.0, 1400.0],
            'sqft_living15': [1400.0, 1900.0, 1700.0, 1500.0],
            'sqft_lot15': [4800.0, 5900.0, 5300.0, 5700.0],
            'price': [300000.0, 450000.0, 350000.0, 320000.0],
            'sqft_basement': [800.0, np.nan, 700.0, 750.0]
        })

        # Sample DataFrame for testing convert_to_boolean
        self.df_bool = pd.DataFrame({
            'is_featured': [1.0, 0.0, 1.0, np.nan, 0.0],
            'is_new': [0.0, 0.0, 1.0, 1.0, np.nan]
        })

        # Sample DataFrame for testing remove_older_duplicates
        self.df_dup = pd.DataFrame({
            'id': [1, 1, 2, 3, 3, 3],
            'date': ['2021-01-01', '2021-06-01', '2020-05-20', '2019-07-15', '2020-08-25', '2021-09-10'],
            'value': [100, 150, 200, 300, 350, 400]
        })

        # Sample DataFrame for testing distance_to_center
        self.df_distance = pd.DataFrame({
            'lat': [47.6097, 47.6205, 47.6038],
            'long': [-122.3331, -122.3493, -122.3301]
        })

        # Sample DataFrame for testing classify_by_zip_code
        self.df_zip = pd.DataFrame({
            'zipcode': [98101, 98052, 98115, 98075, 98109]
        })

        # Sample DataFrame for testing descriptive_label_mapping
        self.df_label = pd.DataFrame({
            'category_numeric': [1, 2, 3, 2, np.nan]
        })
        self.category_mapping = {1: 'Low', 2: 'Medium', 3: 'High'}
        self.category_order = ['Low', 'Medium', 'High']

        # Sample DataFrame for testing month_to_season
        self.df_season = pd.DataFrame({
            'month': [1, 4, 7, 10, 12]
        })

    # Tests for convert_columns_to_integers
    def test_convert_columns_to_integers_success(self):
        """
        Test that the function correctly converts specified columns to integers, handling NaN values.
        """
        result_df = convert_columns_to_integers(self.df_int.copy())

        # Check 'yr_renovated'
        expected_yr_renovated = pd.Series([pd.NA, 199, 200, pd.NA], name='yr_renovated', dtype='Int64')
        pd.testing.assert_series_equal(result_df['yr_renovated'], expected_yr_renovated)

        # Check integer columns
        int_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 
                       'sqft_living15', 'sqft_lot15', 'price']
        for col in int_columns:
            self.assertTrue(pd.api.types.is_integer_dtype(result_df[col]))
            self.assertEqual(result_df[col].dtype, 'Int64')  # Updated to 'Int64'

        # Check 'sqft_basement'
        expected_sqft_basement = pd.Series([800, pd.NA, 700, 750], name='sqft_basement', dtype='Int64')
        pd.testing.assert_series_equal(result_df['sqft_basement'], expected_sqft_basement)

    def test_convert_columns_to_integers_all_nan(self):
        """
        Test that the function correctly handles columns with all NaN values.
        """
        df_all_nan = pd.DataFrame({
            'yr_renovated': [np.nan, np.nan],
            'bedrooms': [np.nan, np.nan],
            'sqft_basement': [np.nan, np.nan]
        })
        result_df = convert_columns_to_integers(df_all_nan.copy())

        # Check 'yr_renovated'
        expected_yr_renovated = pd.Series([pd.NA, pd.NA], name='yr_renovated', dtype='Int64')
        pd.testing.assert_series_equal(result_df['yr_renovated'], expected_yr_renovated)

        # Check integer columns
        int_columns = ['bedrooms']
        for col in int_columns:
            expected_series = pd.Series([pd.NA, pd.NA], name=col, dtype='Int64')
            pd.testing.assert_series_equal(result_df[col], expected_series)

        # Check 'sqft_basement'
        expected_sqft_basement = pd.Series([pd.NA, pd.NA], name='sqft_basement', dtype='Int64')
        pd.testing.assert_series_equal(result_df['sqft_basement'], expected_sqft_basement)

    # Tests for convert_to_boolean
    def test_convert_to_boolean_success(self):
        """
        Test that the function correctly converts specified columns to 'yes'/'no', preserving NaN.
        """
        result_df = convert_to_boolean(self.df_bool.copy(), 'is_featured')
        expected_series = pd.Series(['yes', 'no', 'yes', np.nan, 'no'], name='is_featured')
        pd.testing.assert_series_equal(result_df['is_featured'], expected_series)

        result_df = convert_to_boolean(self.df_bool.copy(), 'is_new')
        expected_series_new = pd.Series(['no', 'no', 'yes', 'yes', np.nan], name='is_new')
        pd.testing.assert_series_equal(result_df['is_new'], expected_series_new)

    def test_convert_to_boolean_custom_values(self):
        """
        Test that the function correctly handles custom yes and no values.
        """
        df_custom = pd.DataFrame({
            'status': [2.0, 1.0, 2.0, np.nan, 1.0]
        })
        result_df = convert_to_boolean(df_custom.copy(), 'status', yes_value=2.0, no_value=1.0)
        expected_series = pd.Series(['yes', 'no', 'yes', np.nan, 'no'], name='status')
        pd.testing.assert_series_equal(result_df['status'], expected_series)

    # Tests for remove_older_duplicates
    def test_remove_older_duplicates_success(self):
        """
        Test that the function correctly removes older duplicates, keeping the most recent entry.
        """
        df_dup_sorted = self.df_dup.copy()
        df_dup_sorted['date'] = pd.to_datetime(df_dup_sorted['date'])
        result_df = remove_older_duplicates(df_dup_sorted, id_column='id', date_column='date')

        expected_df = pd.DataFrame({
            'id': [1, 2, 3],
            'date': [pd.Timestamp('2021-06-01'), pd.Timestamp('2020-05-20'), pd.Timestamp('2021-09-10')],
            'value': [150, 200, 400]
        }, index=[1, 2, 5])

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_remove_older_duplicates_no_duplicates(self):
        """
        Test that the function correctly handles DataFrames with no duplicates.
        """
        df_no_dup = pd.DataFrame({
            'id': [1, 2, 3],
            'date': ['2021-01-01', '2020-05-20', '2021-09-10'],
            'value': [100, 200, 400]
        })
        df_no_dup['date'] = pd.to_datetime(df_no_dup['date'])
        result_df = remove_older_duplicates(df_no_dup.copy(), id_column='id', date_column='date')

        pd.testing.assert_frame_equal(result_df, df_no_dup)

    def test_remove_older_duplicates_all_duplicates(self):
        """
        Test that the function correctly handles DataFrames where all rows are duplicates.
        """
        df_all_dup = pd.DataFrame({
            'id': [1, 1, 1],
            'date': ['2020-01-01', '2021-01-01', '2022-01-01'],
            'value': [100, 150, 200]
        })
        df_all_dup['date'] = pd.to_datetime(df_all_dup['date'])
        result_df = remove_older_duplicates(df_all_dup.copy(), id_column='id', date_column='date')

        expected_df = pd.DataFrame({
            'id': [1],
            'date': [pd.Timestamp('2022-01-01')],
            'value': [200]
        }, index=[2])

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_remove_older_duplicates_empty_dataframe(self):
        """
        Test that the function correctly handles an empty DataFrame.
        """
        df_empty = pd.DataFrame(columns=['id', 'date', 'value'])
        result_df = remove_older_duplicates(df_empty.copy(), id_column='id', date_column='date')
        pd.testing.assert_frame_equal(result_df, df_empty)

    # Tests for distance_to_center
    def test_distance_to_center_correctness(self):
        """
        Test that the function correctly calculates the distance to Seattle center.
        """
        # Manually calculate expected distances
        seattle_center = (47.6062, -122.3321)
        expected_distances = [
            geodesic(seattle_center, (47.6097, -122.3331)).miles,
            geodesic(seattle_center, (47.6205, -122.3493)).miles,
            geodesic(seattle_center, (47.6038, -122.3301)).miles
        ]

        result_distances = self.df_distance.apply(distance_to_center, axis=1)
        for result, expected in zip(result_distances, expected_distances):
            self.assertAlmostEqual(result, expected, places=2)

    def test_distance_to_center_same_location(self):
        """
        Test that the distance is zero when the house is at Seattle center.
        """
        df_same = pd.DataFrame({
            'lat': [47.6062],
            'long': [-122.3321]
        })
        result = df_same.apply(distance_to_center, axis=1)
        self.assertEqual(result.iloc[0], 0.0)

    def test_distance_to_center_invalid_coordinates(self):
        """
        Test that the function handles invalid coordinates gracefully.
        """
        df_invalid = pd.DataFrame({
            'lat': [None, 47.6062],
            'long': [-122.3321, 'invalid']
        })
        result = df_invalid.apply(distance_to_center, axis=1)
        expected = pd.Series([np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected)

    # Tests for classify_by_zip_code
    def test_classify_by_zip_code_city(self):
        """
        Test that the function classifies known Seattle ZIP codes as 'city'.
        """
        for zipcode in [98101, 98115, 98109]:
            classification = classify_by_zip_code(zipcode)
            self.assertEqual(classification, 'city')

    def test_classify_by_zip_code_countryside(self):
        """
        Test that the function classifies non-Seattle ZIP codes as 'countryside'.
        """
        for zipcode in [98052, 98075, 98200]:
            classification = classify_by_zip_code(zipcode)
            self.assertEqual(classification, 'countryside')

    def test_classify_by_zip_code_invalid_zip(self):
        """
        Test that the function handles invalid ZIP codes gracefully.
        """
        classification = classify_by_zip_code('invalid_zip')
        self.assertEqual(classification, 'countryside')  # Since 'invalid_zip' not in list

    # Tests for descriptive_label_mapping
    def test_descriptive_label_mapping_with_order(self):
        """
        Test that the function correctly maps numeric categories to strings and sets the categorical order.
        """
        result_df = descriptive_label_mapping(
            self.df_label.copy(), 
            'category_numeric', 
            self.category_mapping, 
            category_order=self.category_order
        )
        
        expected_dtype = pd.CategoricalDtype(categories=self.category_order, ordered=True)
        expected_series = pd.Series(['Low', 'Medium', 'High', 'Medium', pd.NA], name='category_numeric_cat', dtype=expected_dtype)
        
        pd.testing.assert_series_equal(result_df['category_numeric_cat'], expected_series)

    def test_descriptive_label_mapping_without_order(self):
        """
        Test that the function correctly maps numeric categories to strings without setting categorical order.
        """
        result_df = descriptive_label_mapping(
            self.df_label.copy(), 
            'category_numeric', 
            self.category_mapping
        )
        
        expected_series = pd.Series(['Low', 'Medium', 'High', 'Medium', pd.NA], name='category_numeric_cat', dtype='object')
        pd.testing.assert_series_equal(result_df['category_numeric_cat'], expected_series)

    def test_descriptive_label_mapping_unmapped_values(self):
        """
        Test that the function maps unmapped numeric values to NaN.
        """
        df_unmapped = pd.DataFrame({
            'category_numeric': [1, 2, 4, 2, np.nan]
        })
        result_df = descriptive_label_mapping(
            df_unmapped.copy(), 
            'category_numeric', 
            self.category_mapping, 
            category_order=self.category_order
        )
        
        expected_dtype = pd.CategoricalDtype(categories=self.category_order, ordered=True)
        expected_series = pd.Series(['Low', 'Medium', pd.NA, 'Medium', pd.NA], name='category_numeric_cat', dtype=expected_dtype)
        
        pd.testing.assert_series_equal(result_df['category_numeric_cat'], expected_series)

    # Tests for month_to_season
    def test_month_to_season_correctness(self):
        """
        Test that the function correctly maps each month to the appropriate season.
        """
        season_mapping = {
            1: 'Winter',
            4: 'Spring',
            7: 'Summer',
            10: 'Autumn',
            12: 'Winter'
        }
        for month, expected_season in season_mapping.items():
            with self.subTest(month=month):
                season = month_to_season(month)
                self.assertEqual(season, expected_season)

    def test_month_to_season_invalid_month(self):
        """
        Test that the function handles invalid month numbers gracefully.
        """
        with self.assertRaises(ValueError):
            month_to_season(13)  # Invalid month

        with self.assertRaises(ValueError):
            month_to_season(0)  # Invalid month

        with self.assertRaises(TypeError):
            month_to_season('January')  # Invalid type

# Run the tests
if __name__ == '__main__':
    unittest.main()
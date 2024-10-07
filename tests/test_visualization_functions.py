import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Adjust the import path to include the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px

from src.visualization_functions import (
    plot_box_plots,
    plot_violin_plots,
    plot_histograms,
    plot_correlation_matrix_heatmap,
    plot_map,
    plot_bar_plots_with_categories,
    plot_box_plot,
    plot_violin_plot,
    plot_bins,
    plot_scatter_plot,
    plot_bar_plot,
    plot_choropleth_map,
    colors  # Import colors
)

class TestVisualizationFunctions(unittest.TestCase):
    
    def setUp(self):
        # Sample DataFrame for plotting functions
        self.sample_df = pd.DataFrame({
            'bedrooms': [2, 3, 4, 3, 5],
            'bathrooms': [1, 2, 3, 2, 4],
            'sqft_living': [1500, 2500, 3500, 2500, 4500],
            'sqft_lot': [5000, 6000, 7000, 6000, 8000],
            'price': [300000, 450000, 600000, 450000, 750000],
            'age_at_sale': [10, 15, 20, 15, 25],
            'mile_dist_center': [5, 10, 15, 10, 20],
            'location_type': ['city', 'countryside', 'city', 'countryside', 'city'],
            'zipcode': [98101, 98052, 98115, 98075, 98109],
            'lat': [47.6062, 47.6097, 47.6189, 47.6205, 47.6050],
            'long': [-122.3321, -122.3331, -122.3410, -122.3493, -122.3352],
            'latitude': [47.6062, 47.6097, 47.6189, 47.6205, 47.6050],      # If needed
            'longitude': [-122.3321, -122.3331, -122.3410, -122.3493, -122.3352]  # If needed
        })
        
        # Ensure 'zipcode' is numeric (in case it's read as string)
        self.sample_df['zipcode'] = pd.to_numeric(self.sample_df['zipcode'], errors='coerce')
        
        # Compute the correlation matrix using only numeric columns
        self.corr_matrix = self.sample_df.select_dtypes(include=[np.number]).corr()
        
        # Sample data for bins
        self.bins = np.array([0, 5, 10, 15, 20, 25])
        
        # Sample data for bar plots with categories
        self.avg_price_data1 = pd.DataFrame({
            'x': ['0-5', '5-10', '10-15', '15-20'],
            'y': [300000, 450000, 600000, 750000]
        })
        self.avg_price_data2 = pd.DataFrame({
            'x': ['0-5', '5-10', '10-15', '15-20'],
            'y': [320000, 470000, 620000, 770000]
        })
    
    # Tests for plot_box_plots
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_box_plots_execution(self, mock_subplots, mock_show):
        """
        Test that plot_box_plots executes without errors and calls plt.show().
        """
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(5)]  # Assuming 5 box plots
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        try:
            plot_box_plots(self.sample_df)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_box_plots raised an exception {e}")
    
    # Tests for plot_violin_plots
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_violin_plots_execution(self, mock_subplots, mock_show):
        """
        Test that plot_violin_plots executes without errors and calls plt.show().
        """
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(5)]  # Assuming 5 violin plots
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        try:
            plot_violin_plots(self.sample_df)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_violin_plots raised an exception {e}")
    
    # Tests for plot_histograms
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_histograms_execution(self, mock_subplots, mock_show):
        """
        Test that plot_histograms executes without errors and calls plt.show().
        """
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(6)]  # Assuming 6 histograms
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        try:
            plot_histograms(self.sample_df)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_histograms raised an exception {e}")
    
    # Tests for plot_correlation_matrix_heatmap
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.show')
    def test_plot_correlation_matrix_heatmap_execution(self, mock_show, mock_heatmap):
        """
        Test that plot_correlation_matrix_heatmap executes without errors and calls sns.heatmap and plt.show().
        """
        try:
            plot_correlation_matrix_heatmap(self.corr_matrix)
            mock_heatmap.assert_called_once()
            mock_show.assert_called_once()
            
            # Optionally, verify specific parameters
            args, kwargs = mock_heatmap.call_args
            pd.testing.assert_frame_equal(args[0], self.corr_matrix)
            self.assertTrue(kwargs.get('annot'))
            self.assertEqual(kwargs.get('fmt'), ".2f")
        except Exception as e:
            self.fail(f"plot_correlation_matrix_heatmap raised an exception {e}")
    
    # Tests for plot_map
    @patch('plotly.graph_objects.Figure.show')
    @patch('plotly.express.scatter_mapbox')
    def test_plot_map_execution_single_color(self, mock_scatter_mapbox, mock_fig_show):
        """
        Test that plot_map executes without errors for single color plots and calls the appropriate Plotly functions.
        """
        # Mock the scatter_mapbox to return a mock figure
        mock_scatter_mapbox.return_value = MagicMock(show=mock_fig_show)
        
        try:
            plot_map(self.sample_df)
            mock_scatter_mapbox.assert_called_once_with(
                self.sample_df,
                lat='lat',
                lon='long',  # Changed to 'long'
                hover_name='id',
                color_discrete_sequence=[colors[2]],
                size_max=5,
                zoom=8.55,
                center={"lat": 47.45, "lon": -122.10},
                mapbox_style="carto-positron",
            )
            mock_fig_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_map (single color) raised an exception {e}")
    
    @patch('plotly.graph_objects.Figure.show')
    @patch('plotly.express.scatter_mapbox')
    def test_plot_map_execution_categorized(self, mock_scatter_mapbox, mock_fig_show):
        """
        Test that plot_map executes without errors for categorized plots and calls the appropriate Plotly functions.
        """
        # Mock the scatter_mapbox to return a mock figure
        mock_scatter_mapbox.return_value = MagicMock(show=mock_fig_show)
        
        try:
            plot_map(self.sample_df, location_type_col='location_type')
            mock_scatter_mapbox.assert_called_once_with(
                self.sample_df,
                lat='lat',
                lon='long',  # Changed to 'long'
                hover_name='id',
                color='location_type',
                color_discrete_map={'city': '#84a8cb', 'countryside': '#bd8585'},
                category_orders={'location_type': ['city', 'countryside']},
                labels={'location_type': 'Area Type'},
                size_max=5,
                zoom=8.55,
                center={"lat": 47.45, "lon": -122.10},
                mapbox_style="carto-positron",
            )
            mock_fig_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_map (categorized) raised an exception {e}")
    
    # Tests for plot_bar_plots_with_categories
    @patch('plotly.graph_objects.Figure.show')
    def test_plot_bar_plots_with_categories_execution(self, mock_show):
        """
        Test that plot_bar_plots_with_categories executes without errors and calls fig.show().
        """
        try:
            plot_bar_plots_with_categories(
                self.sample_df,
                category_col='location_type',
                price_col='price',
                sqft_col='sqft_living'
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_bar_plots_with_categories raised an exception {e}")
    
    # Tests for plot_box_plot
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_box_plot_execution_with_category(self, mock_subplots, mock_show):
        """
        Test that plot_box_plot executes without errors with a category column and calls plt.show().
        """
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        try:
            plot_box_plot(
                self.sample_df, 
                value_col='price', 
                category_col='location_type',
                custom_legend_names={'city': 'Urban', 'countryside': 'Rural'},
                color_map={'Urban': '#84a8cb', 'Rural': '#bd8585'},
                plot_title='Price Distribution by Location',
                y_label='Price ($)'
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_box_plot with category raised an exception {e}")
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_box_plot_execution_without_category(self, mock_subplots, mock_show):
        """
        Test that plot_box_plot executes without errors without a category column and calls plt.show().
        """
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        try:
            plot_box_plot(
                self.sample_df, 
                value_col='price'
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_box_plot without category raised an exception {e}")
    
    # Tests for plot_violin_plot
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_violin_plot_execution_with_category(self, mock_subplots, mock_show):
        """
        Test that plot_violin_plot executes without errors with a category column and calls plt.show().
        """
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        try:
            plot_violin_plot(
                self.sample_df, 
                value_col='price', 
                category_col='location_type',
                custom_legend_names={'city': 'Urban', 'countryside': 'Rural'},
                color_map={'Urban': '#84a8cb', 'Rural': '#bd8585'},
                plot_title='Price Distribution by Location',
                y_label='Price ($)'
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_violin_plot with category raised an exception {e}")
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_violin_plot_execution_without_category(self, mock_subplots, mock_show):
        """
        Test that plot_violin_plot executes without errors without a category column and calls plt.show().
        """
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        try:
            plot_violin_plot(
                self.sample_df, 
                value_col='price'
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_violin_plot without category raised an exception {e}")
    
    # Tests for plot_bins
    @patch('plotly.graph_objects.Figure.show')
    @patch('plotly.express.bar')
    def test_plot_bins_execution(self, mock_bar, mock_fig_show):
        """
        Test that plot_bins executes without errors and calls the appropriate Plotly functions.
        """
        # Mock the bar to return a mock figure
        mock_bar.return_value = MagicMock(show=mock_fig_show)
        
        try:
            plot_bins(
                self.sample_df, 
                column_name='price', 
                y_label='Average Price ($)', 
                bins=self.bins, 
                y_tick_dist=50000, 
                error_bars=False,
                legend_labels={'city': 'Urban', 'countryside': 'Rural'}
            )
            mock_bar.assert_called_once()
            mock_fig_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_bins raised an exception {e}")
    
    # Tests for plot_scatter_plot
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    @patch('seaborn.scatterplot')
    def test_plot_scatter_plot_execution(self, mock_scatterplot, mock_subplots, mock_show):
        """
        Test that plot_scatter_plot executes without errors and calls plt.show().
        """
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        try:
            plot_scatter_plot(self.sample_df)
            mock_show.assert_called_once()
            
            # Assert that scatterplot was called four times with correct colors
            expected_calls = [
                unittest.mock.call(x=self.sample_df['bedrooms'], y=self.sample_df['price'], ax=mock_axes[0], color=colors[2], zorder=3),
                unittest.mock.call(x=self.sample_df['bathrooms'], y=self.sample_df['price'], ax=mock_axes[1], color=colors[2], zorder=3),
                unittest.mock.call(x=self.sample_df['sqft_living'], y=self.sample_df['price'], ax=mock_axes[2], color=colors[2], zorder=3),
                unittest.mock.call(x=self.sample_df['sqft_lot'], y=self.sample_df['price'], ax=mock_axes[3], color=colors[2], zorder=3)
            ]
            mock_scatterplot.assert_has_calls(expected_calls, any_order=False)
            self.assertEqual(mock_scatterplot.call_count, 4)
        except Exception as e:
            self.fail(f"plot_scatter_plot raised an exception {e}")
    
    # Tests for plot_bar_plot
    @patch('plotly.graph_objects.Figure.show')
    def test_plot_bar_plot_execution_single_plot(self, mock_show):
        """
        Test that plot_bar_plot executes without errors for a single plot and calls fig.show().
        """
        try:
            plot_bar_plot(
                avg_price_data1=self.avg_price_data1,
                y_label1="Average Price ($)"
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_bar_plot (single plot) raised an exception {e}")
    
    @patch('plotly.graph_objects.Figure.show')
    def test_plot_bar_plot_execution_two_plots(self, mock_show):
        """
        Test that plot_bar_plot executes without errors for two plots and calls fig.show().
        """
        try:
            plot_bar_plot(
                avg_price_data1=self.avg_price_data1,
                avg_price_data2=self.avg_price_data2,
                y_label1="Average Price ($)",
                y_label2="Average Price ($)"
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_bar_plot (two plots) raised an exception {e}")
    
    # Tests for plot_choropleth_map
    @patch('plotly.graph_objects.Figure.show')
    @patch('plotly.express.choropleth_mapbox')
    @patch('urllib.request.urlopen')
    def test_plot_choropleth_map_execution(self, mock_urlopen, mock_choropleth_mapbox, mock_fig_show):
        """
        Test that plot_choropleth_map executes without errors and calls the appropriate Plotly functions.
        """
        # Mock the urlopen to return a mock response
        mock_response = MagicMock()
        mock_response.__enter__.return_value.read.return_value = b'{}'  # Mocked GeoJSON data
        mock_urlopen.return_value = mock_response
        
        # Mock the choropleth_mapbox to return a mock figure
        mock_choropleth_mapbox.return_value = MagicMock(show=mock_fig_show)
        
        try:
            plot_choropleth_map(
                self.sample_df,
                price_col='price',
                zipcode_col='zipcode'
            )
            mock_choropleth_mapbox.assert_called_once()
            mock_fig_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_choropleth_map raised an exception {e}")

# Run the tests
if __name__ == '__main__':
    unittest.main()
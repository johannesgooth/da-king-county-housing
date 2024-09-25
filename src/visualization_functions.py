import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.collections import PolyCollection
import json
from urllib.request import urlopen

# Define custom color palette
colors = ['#84a8cb', '#bd8585', '#a4bdc3', '#67cff5', '#fb9090', '#72acae', '#bcb9ba', '#e4e5e6']  # blue, red, green, blue_2, red_2, green_2, grey, light_grey

def plot_box_plots(df, label_font_size=11, y_tick_font_size=11, y_tick_intervals=None, y_labels=None):
    """
    Creates vertical box plots to visualize the distribution of numeric columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the numeric columns to plot.
    label_font_size (int): Font size for the y-axis labels. Default is 12.
    y_tick_font_size (int): Font size for the y-axis tick labels. Default is 10.
    y_tick_intervals (list of tuples): A list of tuples for custom tick intervals for each plot. 
                                       Each tuple defines the tick interval (start, end, step) for the y-axis of the respective plot. If None, default intervals will be used.
    y_labels (list of str): A list of custom labels for the y-axis. Each label corresponds to a column. If None, the column names will be used as default labels.

    Returns:
    None: The function creates and shows the box plots.
    """
    # Set default y-axis tick intervals if not provided
    if y_tick_intervals is None:
        y_tick_intervals = [
            (0, 12, 2),       # Default tick interval for bedrooms
            (0, 10, 2),       # Default tick interval for bathrooms
            (0, 15000, 5000), # Default tick interval for sqft_living
            (0, 60000, 20000), # Default tick interval for sqft_lot
            (0, 2000000, 500000)  # Default tick interval for price
        ]
    
    # Set default y-axis labels if not provided
    if y_labels is None:
        y_labels = ['Number of Bedrooms', 'Number of Bathrooms', 'Living Space (sqft)', 'Lot Size (sqft)', 'Price ($)']
    
    # Define the columns to plot
    columns_to_plot = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'price']

    # Create subplots: 1 row, 5 columns (to display in one row)
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))  # Adjust the width and height for one row
    plt.subplots_adjust(wspace=0.5)

    # Loop over each column and plot the vertical boxplot
    for i, col in enumerate(columns_to_plot):
        sns.boxplot(
            y=df[col],
            ax=axes[i],
            boxprops=dict(color='black', facecolor=colors[-1]),  # Light grey fill, black borders
            linewidth=1,        # Line thickness
            flierprops=dict(marker='.', color='black', markersize=6),  # Small black outliers
            whiskerprops=dict(color='black', linewidth=1),  # Black whiskers
            medianprops=dict(color='black', linewidth=2),  # Bold black median line
            showcaps=False,  # Remove caps
            width=0.2  # Narrower boxes
        )

        # Customize spines to remove the right and top axes
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)

        # Set the linewidth of the left and bottom axes
        axes[i].spines['left'].set_linewidth(1)
        axes[i].spines['bottom'].set_linewidth(1)

        # Set the y-axis labels
        axes[i].set_ylabel(y_labels[i], fontsize=label_font_size)

        # Set custom y-axis tick intervals if provided
        axes[i].set_ylim(y_tick_intervals[i][0], y_tick_intervals[i][1])
        axes[i].set_yticks(range(y_tick_intervals[i][0], y_tick_intervals[i][1] + 1, y_tick_intervals[i][2]))

        # Customize the y-axis tick label font size
        axes[i].tick_params(axis='y', labelsize=y_tick_font_size)

        # Remove x-axis labels and titles
        axes[i].set_xticklabels([])
        axes[i].set_xlabel('')  # Removing the x-axis title

    # Show the plots
    plt.show()

def plot_violin_plots(df, label_font_size=11, y_tick_font_size=11, y_tick_intervals=None, y_labels=None):
    """
    Creates vertical violin plots to visualize the distribution of numeric columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the numeric columns to plot.
    label_font_size (int): Font size for the y-axis labels. Default is 12.
    y_tick_font_size (int): Font size for the y-axis tick labels. Default is 10.
    y_tick_intervals (list of tuples): A list of tuples for custom tick intervals for each plot. 
                                       Each tuple defines the tick interval (start, end, step) for the y-axis of the respective plot. If None, default intervals will be used.
    y_labels (list of str): A list of custom labels for the y-axis. Each label corresponds to a column. If None, the column names will be used as default labels.

    Returns:
    None: The function creates and shows the violin plots.
    """
    # Set default y-axis tick intervals if not provided
    if y_tick_intervals is None:
        y_tick_intervals = [
            (0, 12, 2),       # Default tick interval for bedrooms
            (0, 10, 2),       # Default tick interval for bathrooms
            (0, 15000, 5000), # Default tick interval for sqft_living
            (0, 60000, 20000), # Default tick interval for sqft_lot
            (0, 2000000, 500000)  # Default tick interval for price
        ]
    
    # Set default y-axis labels if not provided
    if y_labels is None:
        y_labels = ['Number of Bedrooms', 'Number of Bathrooms', 'Living Space (sqft)', 'Lot Size (sqft)', 'Price ($)']
    
    # Define the columns to plot
    columns_to_plot = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'price']

    # Create subplots: 1 row, 5 columns (to display in one row)
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))  # Adjust the width and height for one row
    plt.subplots_adjust(wspace=0.5)

    # Loop over each column and plot the vertical violin plot
    for i, col in enumerate(columns_to_plot):
        violin = sns.violinplot(
            y=df[col],
            ax=axes[i],
            inner=None,  # Removes mini box plot inside the violin
            linewidth=1,  # Line thickness for the plot outline
            color=colors[-1]  # Filling color for the violins
        )

        # Customize spines to remove the right and top axes
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)

        # Set the linewidth of the left and bottom axes to 1
        axes[i].spines['left'].set_linewidth(1)
        axes[i].spines['bottom'].set_linewidth(1)

        # Set the y-axis labels
        axes[i].set_ylabel(y_labels[i], fontsize=label_font_size)

        # Set custom y-axis tick intervals if provided
        axes[i].set_ylim(y_tick_intervals[i][0], y_tick_intervals[i][1])
        axes[i].set_yticks(range(y_tick_intervals[i][0], y_tick_intervals[i][1] + 1, y_tick_intervals[i][2]))

        # Customize the y-axis tick label font size
        axes[i].tick_params(axis='y', labelsize=y_tick_font_size)

        # Remove x-axis labels and titles
        axes[i].set_xticklabels([])
        axes[i].set_xlabel('')  # Removing the x-axis title

        # Modify the violin plot's outline color to black
        for part in violin.collections:
            part.set_edgecolor('black')
            part.set_facecolor(colors[-1])  # Ensuring fill stays light grey

    # Show the plots
    plt.show()

def plot_histograms(df, x_tick_intervals=None, y_tick_intervals=None, x_labels=None, y_labels=None):
    """
    Creates histograms to visualize the distribution of numeric columns in the DataFrame using Matplotlib.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the numeric columns to plot.
    x_tick_intervals (list of tuples): A list of tuples for custom tick intervals for each plot. Each tuple defines the tick interval
                                       (start, end, step) for the respective plot. If None, default intervals will be used.
    y_tick_intervals (list of tuples): A list of tuples for custom tick intervals for the y-axis of each plot. Each tuple defines the tick interval
                                       (start, end, step) for the respective plot. If None, default intervals will be used.
    x_labels (list of str): A list of custom x-axis labels for each plot. If None, no labels will be added.
    y_labels (list of str): A list of custom y-axis labels for each plot. If None, no labels will be added.

    Returns:
    None: The function creates and shows the histograms.
    """
    # Define the columns to plot
    columns_to_plot = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'price', 'age_at_sale']
    
    # Check if all columns exist in the DataFrame
    missing_columns = [col for col in columns_to_plot if col not in df.columns]
    if (missing_columns):
        raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_columns)}")
    
    # Set default tick intervals if not provided
    if x_tick_intervals is None:
        x_tick_intervals = [
            (0, 9, 2),       # Default tick interval for bedrooms (start, end, step)
            (0, 7, 2),       # Default tick interval for bathrooms
            (0, 7000, 2000), # Default tick interval for sqft_living
            (0, 90000, 25000), # Default tick interval for sqft_lot
            (0, 3500000, 1000000), # Default tick interval for price
            (0, 120, 40)      # Default tick interval for age_at_sale
        ]

    # Set default y-axis tick intervals if not provided
    if y_tick_intervals is None:
        y_tick_intervals = [
            (0, 10000, 3000),   # Default tick interval for bedrooms
            (0, 11000, 3000),   # Default tick interval for bathrooms
            (0, 6000, 2000),  # Default tick interval for sqft_living
            (0, 7000, 2000),  # Default tick interval for sqft_lot
            (0, 3500, 1000),   # Default tick interval for price
            (0, 1200, 400)    # Default tick interval for age_at_sale
        ]

    # Set default labels if not provided
    if x_labels is None:
        x_labels = ['Number of Bedrooms', 'Number of Bathrooms', 'Living Space (sqft)', 'Lot Size (sqft)', 'Price ($)', 'Age at Sale (years)']
    
    if y_labels is None:
        y_labels = ['Count', 'Count', 'Count', 'Count', 'Count', 'Count']

    # Create subplots: 2 rows, 3 columns
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    
    # Adjust subplot layout
    plt.subplots_adjust(hspace=0.5, wspace=0.2, top=0.9)
    
    # Apply Y-axis grid lines only and set X-axis to match grid style with outer ticks
    for i in range(2):
        for j in range(3):
            idx = i * 3 + j  # Calculate the index of the plot (0 to 5)

            # Modify X-axis spine to match the grid style
            ax[i][j].spines['bottom'].set_color(colors[-1])  # Set X-axis color to light grey
            ax[i][j].spines['bottom'].set_linewidth(1)      # Match the grid line width

            # Set ticks on the bottom, with the same style as gridlines
            ax[i][j].tick_params(axis='x', which='both', bottom=True, top=False, 
                                 direction='out', length=6, width=1, color=colors[-1])  # Match tick style to grid lines

            # Set ticks on the y-axis, styled the same way as the x-axis
            ax[i][j].tick_params(axis='y', which='both', left=True, right=False, 
                                 direction='out', length=6, width=1, color=colors[-1])  # Match tick style to grid lines on y-axis

            # Remove other spines (top, left, right)
            ax[i][j].spines['top'].set_visible(False)
            ax[i][j].spines['right'].set_visible(False)
            ax[i][j].spines['left'].set_visible(False)

            # Ensure the gridlines are behind the data
            ax[i][j].grid(axis='y', color=colors[-1], linestyle='-', linewidth=1)  # Enable horizontal grid lines only
            ax[i][j].grid(axis='x', visible=False)  # Disable vertical grid lines
            ax[i][j].set_axisbelow(True)  # Ensure axis is below the plot

            # Add custom x and y labels
            ax[i][j].set_xlabel(x_labels[idx], fontsize=10)  # Custom x-axis label
            ax[i][j].set_ylabel(y_labels[idx], fontsize=10)  # Custom y-axis label

    # Plot histograms for each numeric column with space between bars
    ax[0][0].hist(df['bedrooms'], bins=range(0, 13, 1), color=colors[2], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.97, linewidth=1, zorder=3)
    ax[0][0].set_xlim(x_tick_intervals[0][0], x_tick_intervals[0][1])
    ax[0][0].set_xticks(range(x_tick_intervals[0][0], x_tick_intervals[0][1] + 1, x_tick_intervals[0][2]))
    ax[0][0].set_ylim(y_tick_intervals[0][0], y_tick_intervals[0][1])
    ax[0][0].set_yticks(range(y_tick_intervals[0][0], y_tick_intervals[0][1] + 1, y_tick_intervals[0][2]))

    ax[0][1].hist(df['bathrooms'], bins=range(0, 11, 1), color=colors[2],
                  edgecolor=None,
                  alpha=1.0, rwidth=0.97, linewidth=1, zorder=3)
    ax[0][1].set_xlim(x_tick_intervals[1][0], x_tick_intervals[1][1])
    ax[0][1].set_xticks(range(x_tick_intervals[1][0], x_tick_intervals[1][1] + 1, x_tick_intervals[1][2]))
    ax[0][1].set_ylim(y_tick_intervals[1][0], y_tick_intervals[1][1])
    ax[0][1].set_yticks(range(y_tick_intervals[1][0], y_tick_intervals[1][1] + 1, y_tick_intervals[1][2]))

    ax[0][2].hist(df['sqft_living'], bins=range(0, 15001, 500), color=colors[2], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.95, linewidth=1.5, zorder=3)
    ax[0][2].set_xlim(x_tick_intervals[2][0], x_tick_intervals[2][1])
    ax[0][2].set_xticks(range(x_tick_intervals[2][0], x_tick_intervals[2][1] + 1, x_tick_intervals[2][2]))
    ax[0][2].set_ylim(y_tick_intervals[2][0], y_tick_intervals[2][1])
    ax[0][2].set_yticks(range(y_tick_intervals[2][0], y_tick_intervals[2][1] + 1, y_tick_intervals[2][2]))

    ax[1][0].hist(df['sqft_lot'], bins=500, color=colors[2], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.945, linewidth=1, zorder=3)
    ax[1][0].set_xlim(x_tick_intervals[3][0], x_tick_intervals[3][1])
    ax[1][0].set_xticks(range(x_tick_intervals[3][0], x_tick_intervals[3][1] + 1, x_tick_intervals[3][2]))
    ax[1][0].set_ylim(y_tick_intervals[3][0], y_tick_intervals[3][1])
    ax[1][0].set_yticks(range(y_tick_intervals[3][0], y_tick_intervals[3][1] + 1, y_tick_intervals[3][2]))

    ax[1][1].hist(df['price'], bins=100, color=colors[2], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.88, linewidth=1, zorder=3)
    ax[1][1].set_xlim(x_tick_intervals[4][0], x_tick_intervals[4][1])
    ax[1][1].set_xticks(range(x_tick_intervals[4][0], x_tick_intervals[4][1] + 1, x_tick_intervals[4][2]))
    ax[1][1].set_ylim(y_tick_intervals[4][0], y_tick_intervals[4][1])
    ax[1][1].set_yticks(range(y_tick_intervals[4][0], y_tick_intervals[4][1] + 1, y_tick_intervals[4][2]))

    ax[1][2].hist(df['age_at_sale'], bins=50, color=colors[2], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.88, linewidth=1, zorder=3)
    ax[1][2].set_xlim(x_tick_intervals[5][0], x_tick_intervals[5][1])
    ax[1][2].set_xticks(range(x_tick_intervals[5][0], x_tick_intervals[5][1] + 1, x_tick_intervals[5][2]))
    ax[1][2].set_ylim(y_tick_intervals[5][0], y_tick_intervals[5][1])
    ax[1][2].set_yticks(range(y_tick_intervals[5][0], y_tick_intervals[5][1] + 1, y_tick_intervals[5][2]))

    # Show the plots
    plt.show()

def plot_correlation_matrix_heatmap(corr_mtrx, figsize=(14, 12), annot=True, linewidths=0.5):
    """
    Plots a non-redundant correlation matrix heatmap (lower left triangle only).
    
    Parameters:
    - corr_mtrx (pd.DataFrame): The correlation matrix to plot.
    - figsize (tuple): The size of the figure. Default is (14, 12).
    - annot (bool): Whether to annotate the heatmap with correlation values. Default is True.
    - linewidths (float): Width of the lines that will divide each cell in the heatmap. Default is 0.5.
    
    Returns:
    None: The function will display the heatmap.
    """

    # Define the custom color map for the heatmap
    colors_map = [colors[0], colors[-1], colors[1]]
    custom_diverging_palette = LinearSegmentedColormap.from_list("custom_palette", colors_map)
    
    # Create a mask for the upper triangle (we only want to show the lower left triangle)
    mask = np.triu(np.ones_like(corr_mtrx, dtype=bool))
    
    # Set up the figure and axis
    plt.figure(figsize=figsize)
    
    # Plot the heatmap with the mask
    ax = sns.heatmap(
        corr_mtrx,
        mask=mask,  # Apply the mask
        cmap=custom_diverging_palette,  # Custom color palette
        annot=annot,  # Annotate with correlation values
        fmt=".2f",     # Format annotations
        linewidths=linewidths,  # Line width for cell borders
        cbar_kws={"shrink": .8}  # Shrink colorbar slightly
    )
    
    # Remove the 'price' label from the y-axis and 'price_per_sqft' from the x-axis
    x_labels = ax.get_xticklabels()  # Get current x-axis labels
    y_labels = ax.get_yticklabels()  # Get current y-axis labels
    
    # Identify the indices of 'price_per_sqft' and 'price'
    x_tick_pos = [i for i, label in enumerate(x_labels) if label.get_text() == 'price_per_sqft']
    y_tick_pos = [i for i, label in enumerate(y_labels) if label.get_text() == 'price']

    # Set new x-axis labels and remove the corresponding tick for 'price_per_sqft'
    ax.set_xticklabels([label.get_text() if label.get_text() != 'price_per_sqft' else '' for label in x_labels])
    ax.set_yticklabels([label.get_text() if label.get_text() != 'price' else '' for label in y_labels])
    
    # Remove the ticks for 'price_per_sqft' on the x-axis and 'price' on the y-axis
    if x_tick_pos:
        ax.set_xticks([tick for i, tick in enumerate(ax.get_xticks()) if i not in x_tick_pos])
    if y_tick_pos:
        ax.set_yticks([tick for i, tick in enumerate(ax.get_yticks()) if i not in y_tick_pos])
    
    # Show the plot
    plt.show()

def plot_map(df, lat_col='lat', lon_col='long', hover_name_col='id', 
             location_type_col=None, custom_legend_names=None, 
             color_map=None, category_order=None, labels=None, 
             single_color=colors[2], house_size=5, zoom=8.55, 
             map_center={"lat": 47.45, "lon": -122.10}, map_style="carto-positron", 
             seattle_center={"lat": 47.6062, "lon": -122.3321}, 
             add_seattle_center=True):
    """
    Creates a unified map for both single color or categorized (e.g., by location type) cases,
    with options for custom legend names and adding Seattle's center marker.

    Parameters:
    - df (DataFrame): The input DataFrame with house data.
    - lat_col (str): The name of the latitude column in the DataFrame. Default is 'lat'.
    - lon_col (str): The name of the longitude column in the DataFrame. Default is 'long'.
    - hover_name_col (str): The column to display when hovering over points. Default is 'id'.
    - location_type_col (str): The column used for coloring the points (for categorized plots). Default is None.
    - custom_legend_names (dict): Dictionary for renaming legend categories. Default is None.
    - color_map (dict): Custom colors for the different categories. Default is None.
    - category_order (dict): Dictionary specifying the order of categories. Default is None.
    - labels (dict): Dictionary for custom labels. Default is None.
    - single_color (str): The color to use for points if `location_type_col` is not provided. Default is '#72acae'.
    - house_size (int): The size of the house markers. Default is 5.
    - zoom (float): Zoom level for the map. Default is 8.55.
    - map_center (dict): The coordinates to center the map. Default is {"lat": 47.45, "lon": -122.10}.
    - map_style (str): The style of the map. Default is "carto-positron".
    - seattle_center (dict): The coordinates for the center of Seattle. Default is {"lat": 47.6062, "lon": -122.3321}.
    - add_seattle_center (bool): Whether to add a marker for the center of Seattle. Default is True.
    
    Returns:
    - fig: A Plotly Figure object with the map.
    """
    
    # If location_type_col is provided, we plot based on categories, otherwise a single color
    if location_type_col:
        if custom_legend_names is None:
            custom_legend_names = {'city': 'City Houses', 'countryside': 'Countryside Houses'}
        
        if color_map is None:
            color_map = {'city': '#84a8cb', 'countryside': '#bd8585'}
        
        if category_order is None:
            category_order = {'location_type': ['city', 'countryside']}
        
        if labels is None:
            labels = {'location_type': 'Area Type'}
        
        # Create map with location_type-based categories
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            hover_name=hover_name_col,
            color=location_type_col,
            color_discrete_map=color_map,
            category_orders=category_order,
            labels=labels,
            size_max=house_size,
            zoom=zoom,
            center=map_center,
            mapbox_style=map_style,
        )
        
        # Rename the legend entries
        for trace in fig.data:
            if trace.name in custom_legend_names:
                trace.name = custom_legend_names[trace.name]
    
    else:
        # Create map with a single color for points
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            hover_name=hover_name_col,
            color_discrete_sequence=[single_color],
            size_max=house_size,
            zoom=zoom,
            center=map_center,
            mapbox_style=map_style,
        )
        
        # Add a custom trace for houses with the given single color
        house_trace = go.Scattermapbox(
            lat=df[lat_col], 
            lon=df[lon_col],
            mode='markers',
            marker=dict(
                size=house_size,
                color=single_color,
            ),
            name='Houses in the King County Dataset',
            showlegend=True,
        )
        fig.add_trace(house_trace)
    
    # Optionally add a point for the center of Seattle
    if add_seattle_center:
        fig.add_trace(go.Scattermapbox(
            lat=[seattle_center['lat']],
            lon=[seattle_center['lon']],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=5,
                color='black',
                opacity=1
            ),
            name='Center of Seattle',
            showlegend=True,
            text=['Center of Seattle'],
        ))

    # Customize the legend
    fig.update_layout(
        width=1000, 
        height=700,
        legend=dict(
            title=None,
            font=dict(size=11, color='black'),
            bgcolor="rgba(255, 255, 255, 0.0)",
            borderwidth=0,
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0,
        )
    )
    # Plot the map
    fig.show()

def plot_bar_plots_with_categories(df, category_col, price_col, sqft_col,
                                   x_labels=None, y_label1="Average Price ($)", 
                                   y_label2="Average Sqft of Living (sqft)",
                                   y_tick_spacing1=50000, y_tick_spacing2=500, 
                                   x_range1=None, y_range1=None, 
                                   x_range2=None, y_range2=None,
                                   grid_color='lightgray', grid_linewidth=1,
                                   bar_colors=None):
    """
    Plots two comparative bar plots (one for average price and one for average living space) 
    with categories, and calculates standard deviation for each category.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the category, price, and living space data.
    - category_col (str): Column name for the categories (e.g., city or countryside).
    - price_col (str): Column name for the price data.
    - sqft_col (str): Column name for the living space data.
    - x_labels (list, optional): Custom labels for the x-axis. Default is None.
    - y_label1 (str, optional): Label for the y-axis of the first plot (price). Default is "Average Price ($)".
    - y_label2 (str, optional): Label for the y-axis of the second plot (living space). Default is "Average Sqft of Living (sqft)".
    - y_tick_spacing1 (int, optional): Spacing for y-ticks of the first plot. Default is 50000.
    - y_tick_spacing2 (int, optional): Spacing for y-ticks of the second plot. Default is 500.
    - x_range1 (tuple, optional): Range for the x-axis of the first plot. Default is None.
    - y_range1 (tuple, optional): Range for the y-axis of the first plot. Default is None.
    - x_range2 (tuple, optional): Range for the x-axis of the second plot. Default is None.
    - y_range2 (tuple, optional): Range for the y-axis of the second plot. Default is None.
    - grid_color (str, optional): Color of the grid lines and ticks. Default is 'lightgray'.
    - grid_linewidth (int, optional): Line width of the grid lines and error bars. Default is 1.
    - bar_colors (list, optional): List of colors for the bars. Default is None.

    Returns:
    - None: Displays the comparative bar plots.
    """

    # Calculate the mean and standard deviation for both price and living space by category
    summary_df = df.groupby(category_col).agg(
        avg_price=(price_col, 'mean'),
        std_price=(price_col, 'std'),
        avg_sqft=(sqft_col, 'mean'),
        std_sqft=(sqft_col, 'std')
    ).reset_index()

    # Create subplots: one for price and one for living space
    fig = make_subplots(rows=1, cols=2, subplot_titles=(None, None))

    # Determine colors for the bars
    num_categories = len(summary_df)
    if bar_colors is None:
        # Default colors if not provided
        bar_colors = [f'rgba({50 + idx * 50}, 100, 200, 0.8)' for idx in range(num_categories)]

    # Adding bar plot for the average price (first plot)
    for idx, row in summary_df.iterrows():
        # Price plot with error bars
        fig.add_trace(
            go.Bar(
                x=[x_labels[idx]] if x_labels else [row[category_col]], 
                y=[row['avg_price']], 
                marker_color=bar_colors[idx],
                error_y=dict(
                    type='data',  
                    array=[row['std_price']],  
                    visible=True,
                    thickness=grid_linewidth,  # Set the line width of error bars
                    color=grid_color,  # Set the color of error bars
                    width=0  # Remove the caps
                )
            ),
            row=1, col=1
        )

    # Adding bar plot for the average living space (second plot)
    for idx, row in summary_df.iterrows():
        # Sqft plot with error bars
        fig.add_trace(
            go.Bar(
                x=[x_labels[idx]] if x_labels else [row[category_col]], 
                y=[row['avg_sqft']], 
                marker_color=bar_colors[idx],
                error_y=dict(
                    type='data',  
                    array=[row['std_sqft']],  
                    visible=True,
                    thickness=grid_linewidth,  # Set the line width of error bars
                    color=grid_color,  # Set the color of error bars
                    width=0  # Remove the caps
                )
            ),
            row=1, col=2
        )

    # Update layout for both plots
    fig.update_layout(
        height=500, 
        width=1200,  
        showlegend=False,  # Remove legend
        plot_bgcolor='white',  # White background
        font=dict(color='black'),  # Axis labels font color
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins
    )

    # Update y-axes for the first plot (Price)
    fig.update_yaxes(
        title_text=y_label1, 
        row=1, col=1, 
        showline=False,  # Hide the Y-axis line
        ticks='outside', 
        tickwidth=grid_linewidth, 
        ticklen=6,
        tickcolor=grid_color,  # Set tick color same as grid color
        showgrid=True,  
        gridcolor=grid_color,
        gridwidth=grid_linewidth,
        dtick=y_tick_spacing1,  
        range=y_range1  
    )

    # Update y-axes for the second plot (Living Space)
    fig.update_yaxes(
        title_text=y_label2, 
        row=1, col=2, 
        showline=False,  
        ticks='outside', 
        tickwidth=grid_linewidth, 
        ticklen=6,
        tickcolor=grid_color,  # Set tick color same as grid color
        showgrid=True,  
        gridcolor=grid_color,
        gridwidth=grid_linewidth,
        dtick=y_tick_spacing2,  
        range=y_range2  
    )

    # Update x-axes for both plots (remove x-axis title)
    fig.update_xaxes(
        title_text="",  # Remove x-axis title
        showline=True, 
        linewidth=grid_linewidth, 
        linecolor=grid_color,  
        tickwidth=grid_linewidth, 
        ticklen=6,
        tickcolor=grid_color  # Set tick color same as grid color
    )

    # Show the figure
    fig.show()

def plot_box_plot(df, value_col, category_col=None, 
                  custom_legend_names=None, color_map=None,
                  label_font_size=11, y_tick_font_size=11, 
                  y_tick_intervals=(0, 900, 100),
                  plot_title=None, y_label=None):
    """
    Plots comparative box plots for different categories in one panel.
    It supports value distribution for multiple categories like city/countryside or renovated/unrenovated.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame with value data (e.g., price).
    - value_col (str): Column name for the value to plot (e.g., price).
    - category_col (str): Column name for the category (e.g., city/countryside or renovation status). Default is None.
    - custom_legend_names (dict): Custom names for the categories. Default is None.
    - color_map (dict): Custom colors for the different categories. Default is None.
    - label_font_size (int): Font size for the y-axis labels. Default is 11.
    - y_tick_font_size (int): Font size for the y-axis tick labels. Default is 11.
    - y_tick_intervals (tuple): Tick interval (start, end, step) for the y-axis.
    - plot_title (str): Title for the plot.
    - y_label (str): Custom label for the y-axis. Default is None.
    
    Returns:
    - None: The function creates and shows the box plots.
    """

    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Ensure that df contains no missing values in the relevant columns
    df_copy = df_copy.dropna(subset=[value_col, category_col] if category_col else [value_col])

    # If a category column is provided, plot based on categories
    if category_col:
        # Replace category names with custom legend names if provided
        if custom_legend_names:
            df_copy[category_col] = df_copy[category_col].map(custom_legend_names)
        
        # Get unique categories for plotting
        unique_categories = df_copy[category_col].unique()

        # Handle the color map. If none is provided, generate a color palette based on the number of categories
        if color_map is None:
            color_palette = sns.color_palette("husl", len(unique_categories))
            color_map = {category: color for category, color in zip(unique_categories, color_palette)}
        else:
            # Ensure all categories have a defined color
            if custom_legend_names:
                # Update the color map to match the custom legend names
                color_map = {custom_legend_names.get(key, key): color for key, color in color_map.items()}

            if not all(cat in color_map for cat in unique_categories):
                raise ValueError(f"color_map must include colors for all categories: {unique_categories}")

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the boxplot for each category
        sns.boxplot(
            x=category_col,
            y=value_col,
            data=df_copy,
            palette=[color_map[cat] for cat in unique_categories],  # Apply the color palette
            ax=ax,
            linewidth=1,
            flierprops=dict(
                marker='.', 
                markerfacecolor='black', 
                markeredgecolor='black', 
                markersize=6,
                linestyle='none'
            ),
            whiskerprops=dict(color='black', linewidth=1),
            medianprops=dict(color='black', linewidth=2),
            showcaps=False,
            width=0.2
        )
        
        # Remove x-axis label
        ax.set_xlabel('')  # This removes the x-axis label
        
        # Set the y-axis limits and ticks
        ax.set_ylim(y_tick_intervals[0], y_tick_intervals[1])
        ax.set_yticks(range(y_tick_intervals[0], y_tick_intervals[1] + 1, y_tick_intervals[2]))
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

        # Set y-axis label
        if y_label:
            ax.set_ylabel(y_label, fontsize=label_font_size)
        else:
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=label_font_size)

        # Set the plot title if provided
        if plot_title:
            ax.set_title(plot_title, fontsize=label_font_size + 2)

        # Remove the top and right axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set the linewidth of the left and bottom axes
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    else:
        # If no category column is provided, just plot a single box plot for the entire dataset
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the boxplot
        sns.boxplot(
            y=value_col,
            data=df_copy,
            color='#e4e5e6',  # Default color
            ax=ax,
            linewidth=1,
            flierprops=dict(
                marker='.', 
                markerfacecolor='black', 
                markeredgecolor='black', 
                markersize=6,
                linestyle='none'
            ),
            whiskerprops=dict(color='black', linewidth=1),
            medianprops=dict(color='black', linewidth=2),
            showcaps=False,
            width=0.2
        )
        
        # Set the y-axis limits and ticks
        ax.set_ylim(y_tick_intervals[0], y_tick_intervals[1])
        ax.set_yticks(range(y_tick_intervals[0], y_tick_intervals[1] + 1, y_tick_intervals[2]))
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

        # Set y-axis label
        if y_label:
            ax.set_ylabel(y_label, fontsize=label_font_size)
        else:
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=label_font_size)

        # Set the plot title if provided
        if plot_title:
            ax.set_title(plot_title, fontsize=label_font_size + 2)

        # Remove the top and right axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set the linewidth of the left and bottom axes
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_violin_plot(df, value_col, category_col=None, 
                     custom_legend_names=None, color_map=None,
                     label_font_size=11, y_tick_font_size=11, 
                     y_tick_intervals=(0, 900, 100),
                     plot_title=None, scale='width', split=False,
                     violin_width=0.3, y_label=None):
    """
    Plots comparative violin plots for different categories in one panel.
    It supports value distribution for multiple categories like city/countryside or renovated/unrenovated.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with value data (e.g., price).
    - value_col (str): Column name for the value to plot (e.g., price).
    - category_col (str): Column name for the category (e.g., city/countryside or renovation status). Default is None.
    - custom_legend_names (dict): Custom names for the categories. Default is None.
    - color_map (dict): Custom colors for the different categories. Default is None.
    - label_font_size (int): Font size for the y-axis labels. Default is 11.
    - y_tick_font_size (int): Font size for the y-axis tick labels. Default is 11.
    - y_tick_intervals (tuple): Tick interval (start, end, step) for the y-axis.
    - plot_title (str): Title for the plot.
    - scale (str): Determines the method for the width of the violins. Default is 'width'.
                   Other options are 'area', 'count'.
    - split (bool): If True, it splits the violins when the hue is used.
    - violin_width (float): Controls the width of the violins. Default is 0.3.
    - y_label (str): Custom label for the y-axis. Default is None.

    Returns:
    - None: The function creates and shows the violin plots.
    """

    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Ensure that df contains no missing values in the relevant columns
    df_copy = df_copy.dropna(subset=[value_col, category_col] if category_col else [value_col])

    # If a category column is provided, plot based on categories
    if category_col:
        # Replace category names with custom legend names if provided
        if custom_legend_names:
            df_copy[category_col] = df_copy[category_col].map(custom_legend_names)
        
        # Get unique categories for plotting
        unique_categories = df_copy[category_col].unique()

        # Handle the color map. If none is provided, generate a color palette based on the number of categories
        if color_map is None:
            color_palette = sns.color_palette("husl", len(unique_categories))
            color_map = {category: color for category, color in zip(unique_categories, color_palette)}
        else:
            # Ensure all categories have a defined color
            if custom_legend_names:
                # Update the color map to match the custom legend names
                color_map = {custom_legend_names.get(key, key): color for key, color in color_map.items()}

            if not all(cat in color_map for cat in unique_categories):
                raise ValueError(f"color_map must include colors for all categories: {unique_categories}")

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the violin plot for each category
        sns.violinplot(
            x=category_col,
            y=value_col,
            data=df_copy,
            palette=[color_map[cat] for cat in unique_categories],  # Apply the color palette
            ax=ax,
            scale=scale,
            split=split,
            linewidth=1,
            inner=None,  # No inner quartiles or medians
            width=violin_width  # Set the width of the violins
        )

        # Ensure edges of violins are black
        for violin in ax.findobj(PolyCollection):
            violin.set_edgecolor('black')
            violin.set_linewidth(1)

        # Remove x-axis label
        ax.set_xlabel('')  # This removes the x-axis label

        # Set the y-axis limits and ticks
        ax.set_ylim(y_tick_intervals[0], y_tick_intervals[1])
        ax.set_yticks(range(y_tick_intervals[0], y_tick_intervals[1] + 1, y_tick_intervals[2]))
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

        # Set y-axis label
        if y_label:
            ax.set_ylabel(y_label, fontsize=label_font_size)
        else:
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=label_font_size)

        # Set the plot title if provided
        if plot_title:
            ax.set_title(plot_title, fontsize=label_font_size + 2)

        # Remove the top and right axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set the linewidth of the left and bottom axes
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    else:
        # If no category column is provided, just plot a single violin plot for the entire dataset
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the violin plot
        sns.violinplot(
            y=value_col,
            data=df_copy,
            color='#e4e5e6',  # Default color
            ax=ax,
            scale=scale,
            linewidth=1,
            inner=None,  # No inner quartiles or medians
            width=violin_width  # Set the width of the violins
        )

        # Ensure edges of violins are black
        for violin in ax.findobj(PolyCollection):
            violin.set_edgecolor('black')
            violin.set_linewidth(1)

        # Set the y-axis limits and ticks
        ax.set_ylim(y_tick_intervals[0], y_tick_intervals[1])
        ax.set_yticks(range(y_tick_intervals[0], y_tick_intervals[1] + 1, y_tick_intervals[2]))
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

        # Set y-axis label
        if y_label:
            ax.set_ylabel(y_label, fontsize=label_font_size)
        else:
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=label_font_size)

        # Set the plot title if provided
        if plot_title:
            ax.set_title(plot_title, fontsize=label_font_size + 2)

        # Remove the top and right axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set the linewidth of the left and bottom axes
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_bins(df, column_name, y_label, bins, y_tick_dist,error_bars=False, legend_labels={'city': 'City Houses', 'countryside': 'Countryside Houses'}):
    """
    Plots a bar chart for the binned average values (e.g., price, price per sqft) based on the distance from the city center.
    Optionally adds error bars showing the standard deviation.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The column name in the DataFrame to plot on the y-axis (e.g., 'price' or 'price_per_sqft').
    y_label (str): The label to display on the y-axis (e.g., 'Average Price ($)' or 'Average Price per Sqft ($)').
    bins (np.ndarray): The bin edges for categorizing distances.
    error_bars (bool): Whether to add error bars for standard deviation. Only applicable if column_name represents continuous data like 'price'.
    legend_labels (dict): A dictionary to rename legend labels (e.g., {'city': 'Urban', 'countryside': 'Rural'}).

    Returns:
    fig: A Plotly bar chart figure.
    """
    
    # Create bins for distances
    df['distance_bin'] = pd.cut(df['mile_dist_center'], bins, labels=bins[:-1].astype(str))

    # Calculate the mean and standard deviation for each bin and location type
    if error_bars:
        # Calculate the mean and standard deviation for the given column
        binned_data = df.groupby(['distance_bin', 'location_type'])[column_name].agg(['mean', 'std']).reset_index()
    else:
        # Calculate only the mean without error bars
        binned_data = df.groupby(['distance_bin', 'location_type'])[column_name].mean().reset_index()
        binned_data.rename(columns={column_name: 'mean'}, inplace=True)

    # Set default color mapping
    color_discrete_map = {'city': '#84a8cb', 'countryside': '#bd8585'}

    # Apply custom legend labels if provided
    if legend_labels is not None:
        # Modify color_discrete_map keys to reflect the new labels
        color_discrete_map = {legend_labels.get(k, k): v for k, v in color_discrete_map.items()}
        # Apply the legend renaming to the data
        binned_data['location_type'] = binned_data['location_type'].map(legend_labels)

    # Plotting with customized style
    fig = px.bar(
        binned_data,
        x='distance_bin',
        y='mean',
        color='location_type',  # Differentiate bars by location type
        barmode='group',
        labels={'distance_bin': 'Distance from City Center (miles)', 'mean': y_label},
        color_discrete_map=color_discrete_map  # Custom colors
    )

    # # Add error bars if applicable
    # if error_bars:
    #     fig.update_traces(
    #         error_y=dict(
    #             array=binned_data['std'],  # Use the standard deviation for error bars
    #             thickness=1,  # Match the grid line thickness
    #             width=0,      # Set width to 0 to remove caps
    #             color=colors[-1]  # Match the color to the grid lines
    #         )
    #     )

    # Adjust the layout to match the style of your histograms and remove the legend title
    fig.update_layout(
        xaxis_title='Distance from City Center (miles)',
        yaxis_title=y_label,
        xaxis=dict(
            categoryorder='array',  # Control the X-axis ordering
            categoryarray=bins[:-1].astype(str),
            showgrid=False,  # Disable vertical grid lines
            tickmode='array',
            tickvals=bins[:-1],  # Ensure the ticks on X-axis align with bins
            ticktext=[f'{int(val)}' for val in bins[:-1]],  # Format tick text as integers
            showline=True,  # Show the X-axis line
            linewidth=1,  # Thickness of the X-axis
            linecolor=colors[-1],  # Match the grid line color
            tickwidth=1,  # Match the tick style to grid lines
            ticklen=6,  # Length of the ticks (similar to matplotlib)
            tickcolor=colors[-1],  # Set tick color to match grid lines
            ticks='outside',  # Match the style where ticks are outside
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=y_tick_dist,  # Adjust tick interval based on column
            gridcolor=colors[-1],  # Add grid lines for y-axis
            showline=False,  # No Y-axis line
            linewidth=1,
            tickwidth=1,
            ticklen=6,
            tickcolor=colors[-1],
            ticks='outside',  # Match the style where ticks are outside
        ),
        plot_bgcolor='white',  # Match background to matplotlib's default white
        bargap=0.1,  # Add some spacing between bars, similar to rwidth in matplotlib
        bargroupgap=0.1,  # Group spacing between city and countryside bars
        showlegend=True,  # Keep the legend to differentiate city and countryside
        legend=dict(
            x=0.85,  # Move the legend to the left (0.85 makes it end at the end of the x-axis)
            y=1,  # Keep it at the top
            title=None,  # Remove the legend title
            xanchor='left'  # Align the legend's left edge at x=0.85
        ),
        title=None  # Remove the plot title
    )

    # Hide the top and right axes (spines)
    fig.update_xaxes(showline=True, linewidth=1, linecolor=colors[-1])
    fig.update_yaxes(showline=False)

    # Display the plot
    fig.show()

def plot_scatter_plot(df, figsize=(15, 5)):
    """
    Function to create 4 customized scatter plots with Seaborn and Matplotlib.

    Parameters:
    df : DataFrame
        The DataFrame containing the data to be plotted.
    figsize : tuple
        Size of the figure (width, height).

    Returns:
    None
    """
    
    # Creating subplots: 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Function to set custom tick intervals and plot range
    def set_custom_ticks_and_limits(ax, x_interval=None, y_interval=None, x_lim=None, y_lim=None):
        if x_interval:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
        if y_interval:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_interval))
        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)

    # Plot 1: Number of Bedrooms vs. Price
    sns.scatterplot(x=df['bedrooms'], y=df['price'], ax=axes[0], color=colors[2], zorder=3)
    axes[0].set_xlabel('Number of Bedrooms', color='black')
    axes[0].set_ylabel('Price ($)', color='black')
    set_custom_ticks_and_limits(axes[0], x_interval=3, y_interval=2000000, x_lim=(0, 12.5), y_lim=(0, 8000000))

    # Plot 2: Number of Bathrooms vs. Price
    sns.scatterplot(x=df['bathrooms'], y=df['price'], ax=axes[1], color=colors[2], zorder=3)
    axes[1].set_xlabel('Number of Bathrooms', color='black')
    axes[1].set_ylabel('Price ($)', color='black')
    set_custom_ticks_and_limits(axes[1], x_interval=2, y_interval=2000000, x_lim=(0, 8.4), y_lim=(0, 8000000))

    # Plot 3: Square Footage vs. Price
    sns.scatterplot(x=df['sqft_living'], y=df['price'], ax=axes[2], color=colors[2], zorder=3)
    axes[2].set_xlabel('Living Space (Sqft)', color='black')
    axes[2].set_ylabel('Price ($)', color='black')
    set_custom_ticks_and_limits(axes[2], x_interval=4000, y_interval=2000000, x_lim=(0, 12500), y_lim=(0, 8000000))

    # Plot 4: Lot Size vs. Price
    sns.scatterplot(x=df['sqft_lot'], y=df['price'], ax=axes[3], color=colors[2], zorder=3)
    axes[3].set_xlabel('Lot Size (Sqft)', color='black')
    axes[3].set_ylabel('Price ($)', color='black')
    set_custom_ticks_and_limits(axes[3], x_interval=400000, y_interval=2000000, x_lim=(0, 1250000), y_lim=(0, 8000000))

    # Customizing the axes and gridlines for each plot
    for ax in axes:
        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set color and thickness for the bottom and left spines
        ax.spines['bottom'].set_color(colors[-1])   # Change to desired color
        ax.spines['left'].set_color(colors[-1])     # Change to desired color
        ax.spines['bottom'].set_linewidth(1)     # Change to desired thickness
        ax.spines['left'].set_linewidth(1)       # Change to desired thickness

        # Set tick parameters: color light grey for ticks, black for tick labels
        ax.tick_params(color=colors[-1], labelcolor='black', width=1, length=8)

        # Add gridlines (x and y) in light grey, with linewidth of 1, behind the datapoints
        ax.grid(True, which='both', axis='both', color=colors[-1], linewidth=1, zorder=1)

    # Adjust the layout to make sure labels and spacing don't overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_bar_plot(avg_price_data1, avg_price_data2=None, 
                       x_label1="X-Axis Label 1", y_label1="Y-Axis Label 1", 
                       x_label2=None, y_label2=None,
                       x_tick_spacing1=1, y_tick_spacing1=50000, 
                       x_tick_spacing2=1, y_tick_spacing2=50000, 
                       x_range1=None, y_range1=None, 
                       x_range2=None, y_range2=None):
    '''
    This function generates one or two side-by-side bar plots, depending on whether one or two datasets are provided. 
    The bar plots are customizable in terms of axis labels, tick spacing, axis ranges, and plot dimensions. 
    The function uses Plotly to create interactive bar charts, and it adjusts layout properties such as background color, margins, and tick styles.
    '''
    
    # Check if one or two plots are needed
    if avg_price_data2 is None:
        # Single plot
        fig = make_subplots(rows=1, cols=1, subplot_titles=(None,))
    else:
        # Two plots
        fig = make_subplots(rows=1, cols=2, subplot_titles=(None, None))

    # Adding bar plot for the first dataset (either single or left plot)
    fig.add_trace(
        go.Bar(x=avg_price_data1['x'], y=avg_price_data1['y'], name=x_label1, marker_color=colors[2]),
        row=1, col=1
    )

    # If a second dataset is provided, add a second bar plot
    if avg_price_data2 is not None:
        fig.add_trace(
            go.Bar(x=avg_price_data2['x'], y=avg_price_data2['y'], name=x_label2, marker_color=colors[2]),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        height=500, 
        width=900 if avg_price_data2 is None else 1200,  # Adjust width for single or two plots
        title_text=None,  # Remove figure title
        showlegend=False,  # Hide legend
        plot_bgcolor='white',  # White background
        font=dict(color='black'),  # Axis labels font color
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins
    )

    # Update x-axes for the first dataset
    fig.update_xaxes(
        title_text=x_label1, 
        row=1, col=1,
        showline=True, 
        linewidth=1, 
        linecolor=colors[-1],  # X-axis line color
        tickcolor=colors[-1],  # Tick mark color
        ticks='outside',  # Ticks outside
        tickwidth=1, 
        ticklen=6, 
        showgrid=False,  # Disable vertical gridlines
        dtick=x_tick_spacing1,  # Custom x-axis tick spacing
        range=x_range1  # Custom x-axis range
    )

    # Update y-axes for the first dataset (Remove Y-axis line but keep ticks and labels)
    fig.update_yaxes(
        title_text=y_label1, 
        row=1, col=1, 
        showline=False,  # Hide the Y-axis line
        tickcolor=colors[-1],  # Keep tick marks
        ticks='outside', 
        tickwidth=1, 
        ticklen=6, 
        showgrid=True,  # Keep horizontal gridlines
        gridcolor=colors[-1],
        dtick=y_tick_spacing1,  # Custom y-axis tick spacing
        range=y_range1  # Custom y-axis range
    )

    # If a second dataset is provided, update the second x- and y-axes
    if avg_price_data2 is not None:
        # Update x-axis for the second plot
        fig.update_xaxes(
            title_text=x_label2, 
            row=1, col=2, 
            showline=True, 
            linewidth=1, 
            linecolor=colors[-1], 
            tickcolor=colors[-1], 
            ticks='outside', 
            tickwidth=1, 
            ticklen=6, 
            showgrid=False,  # Disable vertical gridlines for the second plot
            dtick=x_tick_spacing2,  # Custom x-axis tick spacing
            range=x_range2  # Custom x-axis range
        )

        # Update y-axis for the second plot (Remove Y-axis line but keep ticks and labels)
        fig.update_yaxes(
            title_text=y_label2, 
            row=1, col=2, 
            showline=False,  # Hide the Y-axis line
            tickcolor=colors[-1],  # Keep tick marks
            ticks='outside', 
            tickwidth=1, 
            ticklen=6, 
            showgrid=True,  # Keep horizontal gridlines for the second plot
            gridcolor=colors[-1],
            dtick=y_tick_spacing2,  # Custom y-axis tick spacing
            range=y_range2  # Custom y-axis range
        )

    # Show the figure
    fig.show()

def plot_choropleth_map(df, price_col='price', zipcode_col='zipcode',
                        center_lat=47.407, center_lon=-121.9,
                        zoom=8, height=700, width=700,
                        geojson_url='https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/wa_washington_zip_codes_geo.min.json'):
    """
    Plots a choropleth map with a custom color scale.

    Parameters:
    - df (pd.DataFrame): Data frame containing the data.
    - price_col (str): Column name for the data to be represented (default 'price').
    - zipcode_col (str): Column name for the zip codes (default 'zipcode').
    - colors (list): List of three color codes or names for the color scale (default None).
    - center_lat (float): Latitude for the map center (default 47.407).
    - center_lon (float): Longitude for the map center (default -121.9).
    - zoom (int): Zoom level for the map (default 8).
    - height (int): Height of the map figure in pixels (default 700).
    - width (int): Width of the map figure in pixels (default 700).
    - geojson_url (str): URL to the GeoJSON file (default provided for Washington state).

    Returns:
    - None: Displays the choropleth map.
    """

    # Create the Plotly color scale
    plotly_color_scale = [
        (0.0, colors[0]),  # Minimum value
        (0.5, colors[-1]),  # Midpoint value
        (1.0, colors[1])   # Maximum value
    ]

    # Load the GeoJSON data
    with urlopen(geojson_url) as response:
        geojson_data = json.load(response)

    # Create the choropleth map
    fig = px.choropleth_mapbox(
        data_frame=df,
        geojson=geojson_data,
        featureidkey='properties.ZCTA5CE10',
        locations=zipcode_col,
        color=price_col,
        mapbox_style='open-street-map',
        center=dict(lat=center_lat, lon=center_lon),
        zoom=zoom,
        height=height,
        width=width,
        color_continuous_scale=plotly_color_scale
    )

    # Display the figure
    fig.show()
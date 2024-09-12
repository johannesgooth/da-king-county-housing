import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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

import matplotlib.pyplot as plt
import seaborn as sns

# Define custom color palette
light_grey = '#e4e5e6'  # Light grey for filling

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
            color=light_grey  # Filling color for the violins
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
            part.set_facecolor(light_grey)  # Ensuring fill stays light grey

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
    plt.subplots_adjust(hspace=.5, wspace=.2, top=.9)
    
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
    ax[0][0].hist(df['bedrooms'], bins=range(0, 13, 1), color=colors[0], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.97, linewidth=1, zorder=3)
    ax[0][0].set_xlim(x_tick_intervals[0][0], x_tick_intervals[0][1])
    ax[0][0].set_xticks(range(x_tick_intervals[0][0], x_tick_intervals[0][1] + 1, x_tick_intervals[0][2]))
    ax[0][0].set_ylim(y_tick_intervals[0][0], y_tick_intervals[0][1])
    ax[0][0].set_yticks(range(y_tick_intervals[0][0], y_tick_intervals[0][1] + 1, y_tick_intervals[0][2]))

    ax[0][1].hist(df['bathrooms'], bins=range(0, 11, 1), color=colors[0],
                  edgecolor=None,
                  alpha=1.0, rwidth=0.97, linewidth=1, zorder=3)
    ax[0][1].set_xlim(x_tick_intervals[1][0], x_tick_intervals[1][1])
    ax[0][1].set_xticks(range(x_tick_intervals[1][0], x_tick_intervals[1][1] + 1, x_tick_intervals[1][2]))
    ax[0][1].set_ylim(y_tick_intervals[1][0], y_tick_intervals[1][1])
    ax[0][1].set_yticks(range(y_tick_intervals[1][0], y_tick_intervals[1][1] + 1, y_tick_intervals[1][2]))

    ax[0][2].hist(df['sqft_living'], bins=range(0, 15001, 500), color=colors[0], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.95, linewidth=1.5, zorder=3)
    ax[0][2].set_xlim(x_tick_intervals[2][0], x_tick_intervals[2][1])
    ax[0][2].set_xticks(range(x_tick_intervals[2][0], x_tick_intervals[2][1] + 1, x_tick_intervals[2][2]))
    ax[0][2].set_ylim(y_tick_intervals[2][0], y_tick_intervals[2][1])
    ax[0][2].set_yticks(range(y_tick_intervals[2][0], y_tick_intervals[2][1] + 1, y_tick_intervals[2][2]))

    ax[1][0].hist(df['sqft_lot'], bins=500, color=colors[0], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.945, linewidth=1, zorder=3)
    ax[1][0].set_xlim(x_tick_intervals[3][0], x_tick_intervals[3][1])
    ax[1][0].set_xticks(range(x_tick_intervals[3][0], x_tick_intervals[3][1] + 1, x_tick_intervals[3][2]))
    ax[1][0].set_ylim(y_tick_intervals[3][0], y_tick_intervals[3][1])
    ax[1][0].set_yticks(range(y_tick_intervals[3][0], y_tick_intervals[3][1] + 1, y_tick_intervals[3][2]))

    ax[1][1].hist(df['price'], bins=100, color=colors[0], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.88, linewidth=1, zorder=3)
    ax[1][1].set_xlim(x_tick_intervals[4][0], x_tick_intervals[4][1])
    ax[1][1].set_xticks(range(x_tick_intervals[4][0], x_tick_intervals[4][1] + 1, x_tick_intervals[4][2]))
    ax[1][1].set_ylim(y_tick_intervals[4][0], y_tick_intervals[4][1])
    ax[1][1].set_yticks(range(y_tick_intervals[4][0], y_tick_intervals[4][1] + 1, y_tick_intervals[4][2]))

    ax[1][2].hist(df['age_at_sale'], bins=50, color=colors[0], 
                  edgecolor=None,
                  alpha=1.0, rwidth=0.88, linewidth=1, zorder=3)
    ax[1][2].set_xlim(x_tick_intervals[5][0], x_tick_intervals[5][1])
    ax[1][2].set_xticks(range(x_tick_intervals[5][0], x_tick_intervals[5][1] + 1, x_tick_intervals[5][2]))
    ax[1][2].set_ylim(y_tick_intervals[5][0], y_tick_intervals[5][1])
    ax[1][2].set_yticks(range(y_tick_intervals[5][0], y_tick_intervals[5][1] + 1, y_tick_intervals[5][2]))

    # Show the plots
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Define the function to plot the correlation matrix heatmap
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
    # Define custom color palette
    colors = ['#84a8cb', '#bd8585', '#a4bdc3', '#67cff5', 'fb9090', '72acae', '#bcb9ba', '#e4e5e6']  # blue, red, green, blue_2, red_2, green_2, grey, light_grey
    
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
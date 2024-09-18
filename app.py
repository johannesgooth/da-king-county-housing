import streamlit as st
import pandas as pd
import pydeck as pdk
from math import radians, cos, sin, sqrt, atan2

# Load the dataset
data = pd.read_csv('data/king_county_housing_data_cleaned_and_preprocessed.csv')

# Convert categorical columns to strings to ensure consistent data types
categorical_columns = ['condition_cat', 'grade_cat', 'season_cat', 'view_cat']
for col in categorical_columns:
    data[col] = data[col].astype(str)

# Calculate average prices of all city and countryside houses (without filters)
data_city = data[data['location_type'] == 'city']
data_countryside = data[data['location_type'] == 'countryside']

average_price_city_all = data_city['price'].mean()
average_price_countryside_all = data_countryside['price'].mean()

# Add an image at the top of the sidebar
st.sidebar.image('.streamlit/app_header.png', use_column_width=True)

# --- City Houses Filters ---
st.sidebar.header('City Houses Filters')

# Distance Filter
city_max_distance = st.sidebar.slider(
    'Maximum distance for city houses (miles)', 0.0, 30.0, 3.0, 0.5
)

# Number of Bedrooms
city_bedrooms_options = sorted(data['bedrooms'].unique())
city_default_bedrooms = [x for x in city_bedrooms_options if x <= 2]
city_bedrooms = st.sidebar.multiselect(
    'Number of Bedrooms (City)',
    city_bedrooms_options,
    default=city_default_bedrooms
)

# Number of Bathrooms
city_bathrooms_options = sorted(data['bathrooms'].unique())
city_default_bathrooms = [x for x in city_bathrooms_options if x <= 2]
city_bathrooms = st.sidebar.multiselect(
    'Number of Bathrooms (City)',
    city_bathrooms_options,
    default=city_default_bathrooms
)

# Condition of the House
city_condition_options = sorted(data['condition_cat'].unique())
city_default_conditions = [x for x in ['Good', 'Excellent'] if x in city_condition_options]
city_condition = st.sidebar.multiselect(
    'Condition of the House (City)',
    city_condition_options,
    default=city_default_conditions
)

# Grade Ranking
city_grade_options = sorted(data['grade_cat'].unique())
city_grade = st.sidebar.multiselect(
    'Grade Ranking (City)',
    city_grade_options,
    default=city_grade_options  # Default is all
)

# Purchasing Season
city_season_options = sorted(data['season_cat'].unique())
city_season = st.sidebar.multiselect(
    'Purchasing Season (City)',
    city_season_options,
    default=city_season_options  # Default is all
)

# View
city_view_options = sorted(data['view_cat'].unique())
city_view = st.sidebar.multiselect(
    'View (City)',
    city_view_options,
    default=city_view_options  # Default is all
)

# --- Countryside Houses Filters ---
st.sidebar.header('Countryside Houses Filters')

# Distance Filter
countryside_min_distance = st.sidebar.slider(
    'Minimum distance for countryside houses (miles)', 0.0, 50.0, 8.0, 0.5
)

# Number of Bedrooms
countryside_bedrooms_options = sorted(data['bedrooms'].unique())
countryside_default_bedrooms = [x for x in countryside_bedrooms_options if x <= 2]
countryside_bedrooms = st.sidebar.multiselect(
    'Number of Bedrooms (Countryside)',
    countryside_bedrooms_options,
    default=countryside_default_bedrooms
)

# Number of Bathrooms
countryside_bathrooms_options = sorted(data['bathrooms'].unique())
countryside_default_bathrooms = [x for x in countryside_bathrooms_options if x <= 2]
countryside_bathrooms = st.sidebar.multiselect(
    'Number of Bathrooms (Countryside)',
    countryside_bathrooms_options,
    default=countryside_default_bathrooms
)

# Condition of the House
countryside_condition_options = sorted(data['condition_cat'].unique())
countryside_default_conditions = [x for x in ['Poor', 'Fair'] if x in countryside_condition_options]
countryside_condition = st.sidebar.multiselect(
    'Condition of the House (Countryside)',
    countryside_condition_options,
    default=countryside_default_conditions
)

# Grade Ranking
countryside_grade_options = sorted(data['grade_cat'].unique())
countryside_grade = st.sidebar.multiselect(
    'Grade Ranking (Countryside)',
    countryside_grade_options,
    default=countryside_grade_options  # Default is all
)

# Purchasing Season
countryside_season_options = sorted(data['season_cat'].unique())
countryside_default_season = ['Winter'] if 'Winter' in countryside_season_options else []
countryside_season = st.sidebar.multiselect(
    'Purchasing Season (Countryside)',
    countryside_season_options,
    default=countryside_default_season
)

# View
countryside_view_options = sorted(data['view_cat'].unique())
countryside_view = st.sidebar.multiselect(
    'View (Countryside)',
    countryside_view_options,
    default=countryside_view_options  # Default is all
)

# Load or calculate 'mile_dist_center'
if 'mile_dist_center' not in data.columns:
    # Define Seattle city center coordinates
    city_center_lat = 47.6062
    city_center_lon = -122.3321

    # Function to calculate distance using Haversine formula
    def haversine_distance(row):
        lat1 = row['lat']
        lon1 = row['long']
        lat2 = city_center_lat
        lon2 = city_center_lon
        R = 3958.8  # Earth radius in miles

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        return distance

    # Calculate distance to city center for each house
    data['mile_dist_center'] = data.apply(haversine_distance, axis=1)

# --- Filtering Data ---

# Separate data into city and countryside based on 'location_type'
city_data = data[data['location_type'] == 'city'].copy()
countryside_data = data[data['location_type'] == 'countryside'].copy()

# Apply filters to city_data
city_data = city_data[city_data['mile_dist_center'] <= city_max_distance]

if city_bedrooms:
    city_data = city_data[city_data['bedrooms'].isin(city_bedrooms)]
if city_bathrooms:
    city_data = city_data[city_data['bathrooms'].isin(city_bathrooms)]
if city_condition:
    city_data = city_data[city_data['condition_cat'].isin(city_condition)]
if city_grade:
    city_data = city_data[city_data['grade_cat'].isin(city_grade)]
if city_season:
    city_data = city_data[city_data['season_cat'].isin(city_season)]
if city_view:
    city_data = city_data[city_data['view_cat'].isin(city_view)]

# Apply filters to countryside_data
countryside_data = countryside_data[countryside_data['mile_dist_center'] >= countryside_min_distance]

if countryside_bedrooms:
    countryside_data = countryside_data[countryside_data['bedrooms'].isin(countryside_bedrooms)]
if countryside_bathrooms:
    countryside_data = countryside_data[countryside_data['bathrooms'].isin(countryside_bathrooms)]
if countryside_condition:
    countryside_data = countryside_data[countryside_data['condition_cat'].isin(countryside_condition)]
if countryside_grade:
    countryside_data = countryside_data[countryside_data['grade_cat'].isin(countryside_grade)]
if countryside_season:
    countryside_data = countryside_data[countryside_data['season_cat'].isin(countryside_season)]
if countryside_view:
    countryside_data = countryside_data[countryside_data['view_cat'].isin(countryside_view)]

# Combine the filtered city and countryside data
filtered_data = pd.concat([city_data, countryside_data])

# Ensure 'location_type' column exists
if 'location_type' not in filtered_data.columns:
    st.error("The column 'location_type' does not exist in the dataset.")
    st.stop()

# --- Displaying Results ---

if not filtered_data.empty:
    # Calculate the average prices for city and countryside houses (filtered)
    average_price_city_filtered = city_data['price'].mean() if not city_data.empty else 0
    average_price_countryside_filtered = countryside_data['price'].mean() if not countryside_data.empty else 0

    # Calculate the relative differences for city houses
    if average_price_city_all != 0 and average_price_city_filtered != 0:
        price_difference_city = average_price_city_filtered - average_price_city_all
        price_percentage_city = (price_difference_city / average_price_city_all) * 100
        if price_difference_city < 0:
            arrow_city = '↓'
            savings_text_city = f"{abs(price_percentage_city):.2f}% savings"
        else:
            arrow_city = '↑'
            savings_text_city = f"{abs(price_percentage_city):.2f}% more"
    else:
        arrow_city = ''
        savings_text_city = 'No data'

    # Calculate the relative differences for countryside houses
    if average_price_countryside_all != 0 and average_price_countryside_filtered != 0:
        price_difference_countryside = average_price_countryside_filtered - average_price_countryside_all
        price_percentage_countryside = (price_difference_countryside / average_price_countryside_all) * 100
        if price_difference_countryside < 0:
            arrow_countryside = '↓'
            savings_text_countryside = f"{abs(price_percentage_countryside):.2f}% savings"
        else:
            arrow_countryside = '↑'
            savings_text_countryside = f"{abs(price_percentage_countryside):.2f}% more"
    else:
        arrow_countryside = ''
        savings_text_countryside = 'No data'

    # Calculate the sums and relative savings
    sum_average_price_all = average_price_city_all + average_price_countryside_all
    sum_average_price_filtered = average_price_city_filtered + average_price_countryside_filtered

    if sum_average_price_all != 0 and sum_average_price_filtered != 0:
        price_difference_sum = sum_average_price_filtered - sum_average_price_all
        price_percentage_sum = (price_difference_sum / sum_average_price_all) * 100
        if price_difference_sum < 0:
            arrow_sum = '↓'
            savings_text_sum = f"{abs(price_percentage_sum):.2f}% savings"
        else:
            arrow_sum = '↑'
            savings_text_sum = f"{abs(price_percentage_sum):.2f}% more"
    else:
        arrow_sum = ''
        savings_text_sum = 'No data'

    # --- Create a two-column layout ---
    col_left, col_right = st.columns([1, 3])  # Adjust the ratio as needed

    with col_left:
        # Display the average prices and savings in rows
        st.subheader("City House")
        st.write(f"${average_price_city_filtered:,.2f} {arrow_city}<br>{savings_text_city}", unsafe_allow_html=True)
        st.subheader("Countryside House")
        st.write(f"${average_price_countryside_filtered:,.2f} {arrow_countryside}<br>{savings_text_countryside}", unsafe_allow_html=True)
        st.subheader("Total")
        st.write(f"${sum_average_price_filtered:,.2f} {arrow_sum}<br>{savings_text_sum}", unsafe_allow_html=True)

    with col_right:
        # Define custom colors for 'city' and 'countryside' houses
        city_color = [103, 207, 245, 160]         # Custom color for city houses
        countryside_color = [251, 144, 144, 160]  # Custom color for countryside houses

        # Define custom datapoint sizes for 'city' and 'countryside' houses
        city_radius = 200         # Adjust this value to change city datapoint size
        countryside_radius = 200  # Adjust this value to change countryside datapoint size

        # Add a black point for Seattle's city center
        city_center_lat = 47.6062
        city_center_lon = -122.3321

        city_center_data = pd.DataFrame({
            'lat': [city_center_lat],
            'long': [city_center_lon]
        })

        city_center_layer = pdk.Layer(
            'ScatterplotLayer',
            data=city_center_data,
            get_position='[long, lat]',
            get_fill_color=[0, 0, 0, 255],  # Black color
            get_radius=200,
            pickable=False
        )

        # Define layers for city and countryside houses
        city_layer = pdk.Layer(
            'ScatterplotLayer',
            data=city_data,
            get_position='[long, lat]',
            get_fill_color=city_color,
            get_radius=city_radius,
            pickable=True
        )

        countryside_layer = pdk.Layer(
            'ScatterplotLayer',
            data=countryside_data,
            get_position='[long, lat]',
            get_fill_color=countryside_color,
            get_radius=countryside_radius,
            pickable=True
        )

        # Set up the view state
        view_state = pdk.ViewState(
            latitude=47.4847,
            longitude=-122,
            zoom=8.7,
            pitch=0
        )

        # Tooltip
        tooltip = {
            "html": "<b>Price:</b> ${price}<br/>"
                    "<b>Location Type:</b> {location_type}",
            "style": {"backgroundColor": "#a4bdc3", "color": "white"}
        }

        # Combine layers
        layers = [city_layer, countryside_layer, city_center_layer]

        # Create the deck.gl map
        r = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=layers,
            tooltip=tooltip
        )

        st.pydeck_chart(r)

else:
    st.write("No properties match the selected criteria.")
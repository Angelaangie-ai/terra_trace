import sys
import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from datetime import datetime, timezone
from shapely.geometry import Polygon, box, Point
from scipy.spatial import distance
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from geopy.distance import geodesic
from PIL import Image
from markupsafe import Markup
import openai
import ee

logging.basicConfig(level=logging.INFO)

# Configuration Constants
BASE_FOLDER_PATH = 'NDVI_DATASET'
CSV_FILEPATH = os.path.join(BASE_FOLDER_PATH, 'FILE_NAME')
MASTER_DF = pd.read_csv(CSV_FILEPATH)  
WILDFIRE_DATA = pd.read_csv('WILD_FIRE_DATA')

# OpenAI API Settings
openai.api_key = 'KEY'
openai.api_base = 'BASE'
openai.api_type = 'TYPE'
openai.api_version = 'VERSION'

# Initialize Earth Engine
ee.Initialize(project='soil-424920')

# Initialize Earth Engine
ee.Initialize(project='soil-424920')

app = Flask(__name__, static_url_path='/static', static_folder='static')

CATEGORIZATION_CODE_TO_LAND_COVER: Dict[int, str] = {
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    4: "Sorghum",
    5: "Soybeans",
    6: "Sunflower",
    10: "Peanuts",
    11: "Tobacco",
    12: "Sweet Corn",
    13: "Pop or Orn Corn",
    14: "Mint",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    30: "Speltz",
    31: "Canola",
    32: "Flaxseed",
    33: "Safflower",
    34: "Rape Seed",
    35: "Mustard",
    36: "Alfalfa",
    37: "Other Hay/Non Alfalfa",
    38: "Camelina",
    39: "Buckwheat",
    41: "Sugarbeets",
    42: "Dry Beans",
    43: "Potatoes",
    44: "Other Crops",
    45: "Sugarcane",
    46: "Sweet Potatoes",
    47: "Misc Vegs & Fruits",
    48: "Watermelons",
    49: "Onions",
    50: "Cucumbers",
    51: "Chick Peas",
    52: "Lentils",
    53: "Peas",
    54: "Tomatoes",
    55: "Caneberries",
    56: "Hops",
    57: "Herbs",
    58: "Clover/Wildflowers",
    59: "Sod/Grass Seed",
    60: "Switchgrass",
    61: "Fallow/Idle Cropland",
    62: "Pasture/Grass",
    63: "Forest",
    64: "Shrubland",
    65: "Barren",
    66: "Cherries",
    67: "Peaches",
    68: "Apples",
    69: "Grapes",
    70: "Christmas Trees",
    71: "Other Tree Crops",
    72: "Citrus",
    74: "Pecans",
    75: "Almonds",
    76: "Walnuts",
    77: "Pears",
    81: "Clouds/No Data",
    82: "Developed",
    83: "Water",
    87: "Wetlands",
    88: "Nonag/Undefined",
    92: "Aquaculture",
    111: "Open Water",
    112: "Perennial Ice/Snow",
    121: "Developed/Open Space",
    122: "Developed/Low Intensity",
    123: "Developed/Med Intensity",
    124: "Developed/High Intensity",
    131: "Barren",
    141: "Deciduous Forest",
    142: "Evergreen Forest",
    143: "Mixed Forest",
    152: "Shrubland",
    176: "Grassland/Pasture",
    190: "Woody Wetlands",
    195: "Herbaceous Wetlands",
    204: "Pistachios",
    205: "Triticale",
    206: "Carrots",
    207: "Asparagus",
    208: "Garlic",
    209: "Cantaloupes",
    210: "Prunes",
    211: "Olives",
    212: "Oranges",
    213: "Honeydew Melons",
    214: "Broccoli",
    215: "Avocados",
    216: "Peppers",
    217: "Pomegranates",
    218: "Nectarines",
    219: "Greens",
    220: "Plums",
    221: "Strawberries",
    222: "Squash",
    223: "Apricots",
    224: "Vetch",
    225: "Dbl Crop WinWht/Corn",
    226: "Dbl Crop Oats/Corn",
    227: "Lettuce",
    228: "Dbl Crop Triticale/Corn",
    229: "Pumpkins",
    230: "Dbl Crop Lettuce/Durum Wht",
    231: "Dbl Crop Lettuce/Cantaloupe",
    232: "Dbl Crop Lettuce/Cotton",
    233: "Dbl Crop Lettuce/Barley",
    234: "Dbl Crop Durum Wht/Sorghum",
    235: "Dbl Crop Barley/Sorghum",
    236: "Dbl Crop WinWht/Sorghum",
    237: "Dbl Crop Barley/Corn",
    238: "Dbl Crop WinWht/Cotton",
    239: "Dbl Crop Soybeans/Cotton",
    240: "Dbl Crop Soybeans/Oats",
    241: "Dbl Crop Corn/Soybeans",
    242: "Blueberries",
    243: "Cabbage",
    244: "Cauliflower",
    245: "Celery",
    246: "Radishes",
    247: "Turnips",
    248: "Eggplants",
    249: "Gourds",
    250: "Cranberries",
    254: "Dbl Crop Barley/Soybeans"
}

def find_closest_coordinate(target_coords, df):
    """
    Find the closest coordinate in df to target_coords using Euclidean distance.
    df must have 'lat' and 'lon' columns.
    """
    distances = np.sqrt((df['lat'] - target_coords[0])**2 + (df['lon'] - target_coords[1])**2)
    closest_idx = distances.idxmin()
    closest_point = df.loc[closest_idx]
    logging.info(f"Closest point found: {closest_point['lat']}, {closest_point['lon']}")
    return closest_point

def get_closest_ndvi_data(coord, df):
    """
    Given a coordinate (lat, lon) and a DataFrame df with NDVI data,
    find the closest point, melt the DataFrame, calculate yearly and percentage changes,
    and return a DataFrame with 'date' and 'NDVI' columns plus additional info.
    """
    lat, lon = coord
    closest_point = find_closest_coordinate((lat, lon), df)
    closest_lat, closest_lon = closest_point['lat'], closest_point['lon']

    df_filtered = df[(df['lat'] == closest_lat) & (df['lon'] == closest_lon)]
    if df_filtered.empty:
        logging.error(f"No data found for coordinates: {coord}")
        return pd.DataFrame()

    df_melted = df_filtered.melt(id_vars=['lat', 'lon'], var_name='date', value_name='NDVI')
    df_melted.dropna(subset=['NDVI'], inplace=True)
    df_melted['date'] = pd.to_datetime(df_melted['date'])
    df_melted.sort_values(by='date', inplace=True)

    df_melted['year'] = df_melted['date'].dt.year
    df_melted['month'] = df_melted['date'].dt.month
    df_melted['yearly_change'] = df_melted.groupby('year')['NDVI'].diff().fillna(0)
    df_melted['percentage_change'] = df_melted.groupby('year')['NDVI'].pct_change(fill_method=None).fillna(0)*100

    return df_melted

def plot_ndvi_timeseries(ndvi_df, save_path):
    """
    Plot NDVI timeseries data with a polynomial fit.
    """
    try:
        plt.figure(figsize=(10, 5))
        ndvi_df['Date'] = pd.to_datetime(ndvi_df['Date'])

        ndvi_df.set_index('Date', inplace=True)
        ndvi_resampled = ndvi_df.resample('M').mean()
        ndvi_df.reset_index(inplace=True)
        ndvi_resampled.reset_index(inplace=True)

        date_range = (ndvi_resampled['Date'] - ndvi_resampled['Date'].min()) / np.timedelta64(1, 'D')
        x = date_range.values
        y = ndvi_resampled['NDVI']

        # Interpolate for smooth curve
        x_new = np.linspace(x.min(), x.max(), 300)
        f = interp1d(x, y, kind='cubic')
        y_new = f(x_new)

        # Fit polynomial
        polynomial_coefficients = np.polyfit(x_new, y_new, 3)
        y_fit = np.polyval(polynomial_coefficients, x_new)

        plt.plot(ndvi_resampled['Date'], y, 'o', label='Resampled Data')
        plt.plot(ndvi_resampled['Date'].min() + np.timedelta64(1, 'D') * x_new, y_fit, '-', label='Polynomial Fit')
        plt.xlabel('Date')
        plt.ylabel('NDVI Value')
        plt.title('NDVI Timeseries Data with Polynomial Fit')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Plot saved at {save_path}")
    except Exception as e:
        logging.error(f"Error in plot_ndvi_timeseries: {e}")
        raise

def get_first_curve_data():
    """
    Return sample NDVI DataFrame for demonstration (random data).
    """
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    ndvi_values = np.random.rand(len(dates))
    data = {'Date': dates, 'NDVI': ndvi_values}
    return pd.DataFrame(data)

def generate_ndvi_analysis(ndvi_df):
    """
    Generate an NDVI analysis summary in HTML.
    """
    try:
        if 'NDVI' not in ndvi_df.columns:
            raise KeyError("'NDVI' column not found in the DataFrame")

        min_ndvi = ndvi_df['NDVI'].min()
        max_ndvi = ndvi_df['NDVI'].max()
        avg_ndvi = ndvi_df['NDVI'].mean()
        trend = np.polyfit(np.arange(len(ndvi_df['NDVI'])), ndvi_df['NDVI'], 1)[0]
        trend_description = "increasing" if trend > 0 else "decreasing"

        analysis = f"""
        <div class="ndvi-analysis">
            <h2 class="ndvi-title">NDVI Analysis Summary</h2>
            <div class="ndvi-section">
                <h3 class="ndvi-subtitle">📊 Key Metrics</h3>
                <ul class="ndvi-list">
                    <li>Minimum NDVI: <strong>{min_ndvi:.4f}</strong></li>
                    <li>Maximum NDVI: <strong>{max_ndvi:.4f}</strong></li>
                    <li>Average NDVI: <strong>{avg_ndvi:.4f}</strong></li>
                    <li>Trend: <strong>{trend_description}</strong></li>
                </ul>
            </div>
            <div class="ndvi-section">
                <h3 class="ndvi-subtitle">🌿 Vegetation Density</h3>
                <p>High NDVI ({max_ndvi:.4f}) indicates dense vegetation, while low NDVI ({min_ndvi:.4f}) suggests sparse vegetation.</p>
            </div>
            <div class="ndvi-section">
                <h3 class="ndvi-subtitle">📈 Temporal Analysis</h3>
                <p>The {trend_description} trend may indicate {trend_description} growth or seasonal changes.</p>
            </div>
            <p class="ndvi-note"><em>Note: Consider additional data and expert input for a complete analysis.</em></p>
        </div>
        """
        return analysis
    except Exception as e:
        logging.error(f"Error in generate_ndvi_analysis: {e}")
        return f"<div class='ndvi-analysis'><p>Error generating NDVI analysis: {e}</p></div>"

def get_satellite_image(coordinates, start_date, end_date):
    """
    Fetch a satellite image (S2) from Earth Engine for given coordinates and date range.
    Returns a thumbnail URL.
    """
    point = ee.Geometry.Point(coordinates)
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(point)
                  .filterDate(start_date, end_date)
                  .sort('CLOUDY_PIXEL_PERCENTAGE')
                  .first())

    vis_params = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2'],
        'gamma': 1.4
    }
    scale = 10
    region = point.buffer(1000).bounds().getInfo()['coordinates']
    url = collection.getThumbURL({
        'min': vis_params['min'],
        'max': vis_params['max'],
        'bands': vis_params['bands'],
        'gamma': vis_params['gamma'],
        'scale': scale,
        'region': region
    })
    return url

@app.route('/chat_ndvi', methods=['POST'])
def chat_ndvi():
    """
    Endpoint to handle NDVI chat requests.
    If 'analyze the first curve' requested:
      - Generate sample NDVI data, plot, analyze
    """
    data = request.get_json()
    message = data.get('message', '').lower()

    try:
        if 'analyze the first curve' in message:
            ndvi_df = get_first_curve_data()

            plot_filename = 'ndvi_first_curve.png'
            save_path = os.path.join('static', plot_filename)
            plot_ndvi_timeseries(ndvi_df, save_path)
            plot_url = url_for('static', filename=plot_filename)

            analysis_html = generate_ndvi_analysis(ndvi_df)
            if analysis_html is None:
                analysis_html = "<p>Error: Unable to generate NDVI analysis.</p>"
            analysis = Markup(analysis_html)

            return jsonify({'response': analysis, 'plot_url': plot_url})
        else:
            return jsonify({'response': "I'm sorry, I didn't understand that. Try asking about analyzing the first curve."})
    except Exception as e:
        logging.error(f"Error in chat_ndvi: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.get_json()
    coordinates = data.get('coordinates')
    coords = list(map(float, coordinates.split(',')))
    start_date = '2020-04-01'
    end_date = '2020-04-30'
    image_url = get_satellite_image(coords, start_date, end_date)
    return jsonify({'image_url': image_url})

def explain_ndvi(value):
    if pd.isna(value):
        return "No data"
    elif value < -0.1:
        return "indicating very sparse or non-vegetative area"
    elif -0.1 <= value < 0:
        return "indicating sparse vegetation"
    elif 0 <= value < 0.2:
        return "indicating some vegetation"
    else:
        return "indicating healthy vegetation"

def generate_explanation(df):
    explanation = ""
    for index, row in df.iterrows():
        month = row["month"]
        explanation += f"\n### {month}:\n"
        for year_col in df.columns[1:]:
            value = row[year_col]
            condition = explain_ndvi(value)
            explanation += f"  - {year_col}: {value} ({condition})\n"
    return explanation

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    deployment_id = 'gpt-35-turbo'

    try:
        response = openai.ChatCompletion.create(
            engine=deployment_id,
            messages=[
                {"role": "system", "content": "You are an expert in NDVI curve analysis."},
                {"role": "user", "content": question}
            ],
            max_tokens=2000
        )
        answer = response['choices'][0]['message']['content']
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/plot', methods=['POST'])
def plot_endpoint():
    """
    Endpoint to plot NDVI for given coordinates provided via form data.
    """
    coordinates_input = request.form['coordinates']
    coordinates = [tuple(map(float, coord.strip().split(','))) for coord in coordinates_input.split(';') if coord.strip()]

    plots = []
    for lat, lon in coordinates:
        ndvi_df = get_closest_ndvi_data((lat, lon), MASTER_DF)
        if ndvi_df.empty:
            continue
        # Convert to standard format (Date, NDVI)
        df_plot = ndvi_df[['date', 'NDVI']].rename(columns={'date': 'Date'})

        plot_filename = f'ndvi_plot_{lat}_{lon}.png'
        save_path = os.path.join('static', plot_filename)
        plot_ndvi_timeseries(df_plot, save_path)
        plots.append((lat, lon, save_path))

    return render_template('plot.html', plots=plots)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/tools/my_tool', methods=['POST'])
def handle_my_tool():
    data = request.get_json()
    arg1 = data.get('arg1')
    arg2 = data.get('arg2')
    result = plugin2.my_tool(arg1, arg2)
    return jsonify({'arg1': arg1, 'arg2': arg2, 'message': 'Tool executed successfully!'})

@app.route('/tools/simple_gpt_tool', methods=['POST'])
def handle_simple_gpt_tool():
    data = request.get_json()
    arg1 = data.get('arg1')
    arg2 = data.get('arg2')
    result = plugin2.simple_gpt_tool(arg1, arg2)
    return jsonify({'result': result})

@app.route('/tools/ndvi_explanation_tool', methods=['POST'])
def ndvi_explanation_tool():
    data = request.get_json()
    coordinates_str = data.get('coordinates')
    coord = tuple(map(float, coordinates_str.split(',')))
    
    ndvi_df = get_closest_ndvi_data(coord, MASTER_DF)
    if ndvi_df.empty:
        return jsonify({"error": "Data not found for given coordinates"}), 500

    # Pivot to wide format by year and month
    df_wide = ndvi_df.pivot(index='month', columns='year', values='NDVI').reset_index()
    df_wide.columns.name = None
    explanation = generate_explanation(df_wide)
    return jsonify({"explanation": explanation})

def process_coordinates(coordinates):
    """
    Use a categorical raster to find crop coverage for given polygon.
    """
    try:
        categorical_raster = CategoricalRaster(
            id="A2",
            time_range=(
                datetime.fromisoformat("2022-01-01").astimezone(timezone.utc),
                datetime.fromisoformat("2022-12-31").astimezone(timezone.utc)
            ),
            geometry=box(-127.8459, 24.3321, -67.0096, 49.3253).__geo_interface__,
            assets=[{"path_or_url": "/workspace/devkit23/plugins/plugin2/modules/2020_30m_cdls.tif"}],
            bands=[1],
            categories=CATEGORIZATION_CODE_TO_LAND_COVER
        )

        crops_finder = plugin2.CropsInAreaFinder(categorical_raster)
        polygon = Polygon(coordinates)
        crop_percentages = crops_finder.get_crop_area_percentage(polygon)
        active_classes = np.unique(crops_finder.read_raster_section(categorical_raster, polygon))
        crop_mask_length = len(crops_finder.get_crop_mask(categorical_raster))

        decoded_classes = {int(code): CATEGORIZATION_CODE_TO_LAND_COVER.get(int(code), "Unknown") for code in active_classes}

        return {
            'crop_percentages': crop_percentages,
            'active_classes': active_classes.tolist(),
            'crop_mask_length': crop_mask_length,
            'decoded_classes': decoded_classes,
        }
    except Exception as e:
        logging.error(f"Error in process_coordinates: {e}")
        return {
            'crop_percentages': {},
            'active_classes': [],
            'crop_mask_length': 0,
            'decoded_classes': {}
        }

@app.route('/generate_plots', methods=['POST'])
def generate_plots():
    coordinates_input = request.form['coordinates']
    try:
        coordinates = [tuple(map(float, coord.strip().split(','))) for coord in coordinates_input.split(';') if coord.strip()]
    except ValueError as e:
        logging.error(f"Error parsing coordinates: {e}")
        return jsonify({'error': 'Invalid coordinate format'}), 400
    
    plots = []
    image_url = None
    for lat, lon in coordinates:
        try:
            coords = [lon, lat]
            start_date = '2020-04-01'
            end_date = '2020-04-15'
            image_url = get_satellite_image(coords, start_date, end_date)

            ndvi_df = get_closest_ndvi_data((lat, lon), MASTER_DF)
            if ndvi_df.empty:
                continue

            df_plot = ndvi_df[['date', 'NDVI']].rename(columns={'date': 'Date'})
            plot_filename = f'ndvi_plot_{lat}_{lon}.png'
            save_path = os.path.join('static', plot_filename)
            plot_ndvi_timeseries(df_plot, save_path)
            plots.append((lat, lon, plot_filename))

        except Exception as e:
            logging.error(f"Error processing coordinates ({lat}, {lon}): {e}")
            return jsonify({'error': f"Error processing coordinates ({lat}, {lon}): {e}"}), 500
    return render_template('show_map.html', coordinates=coordinates_input, plots=plots, image_url=image_url)

def classify_ndvi_curve_function(x, y_smooth, coordinates=None):
    """
    Classify NDVI curve as vegetation, annual crop, perennial vegetation, etc.
    """
    logging.info(f"Classifying NDVI curve for {coordinates}")
    valid_indices = ~np.isnan(y_smooth)
    x = x[valid_indices]
    y_smooth = y_smooth[valid_indices]

    if len(y_smooth) < 5:
        return {
            'classification': 'Insufficient Data',
            'confidence': 0,
            'warning': 'Not enough valid data points'
        }

    y_shifted = y_smooth - np.min(y_smooth) + 0.1
    y_normalized = (y_shifted - np.min(y_shifted)) / (np.max(y_shifted) - np.min(y_shifted))

    max_ndvi = np.max(y_normalized)
    ndvi_range = np.ptp(y_normalized)
    peak_time = np.argmax(y_normalized) / len(y_normalized)
    median_ndvi = np.median(y_normalized)
    growth_rate = np.max(np.diff(y_normalized[:len(y_normalized)//2]))
    decline_rate = np.min(np.diff(y_normalized[len(y_normalized)//2:]))

    is_vegetation = max_ndvi > 0.3
    is_annual_crop = (
        is_vegetation and
        ndvi_range > 0.2 and
        0.2 < peak_time < 0.8 and
        growth_rate > 0.005 and
        decline_rate < -0.005
    )
    is_perennial = (
        is_vegetation and
        ndvi_range < 0.2 and
        median_ndvi > 0.4
    )

    if not is_vegetation:
        classification = 'Non-vegetation or Water'
    elif is_annual_crop:
        classification = 'Annual Crop (possibly Corn)'
    elif is_perennial:
        classification = 'Perennial Vegetation'
    else:
        classification = 'Mixed or Undefined Vegetation'

    confidence = min(1.0, max(0.0, ndvi_range * 2))

    return {
        'classification': classification,
        'confidence': float(confidence),
        'max_ndvi': float(np.max(y_smooth)),
        'min_ndvi': float(np.min(y_smooth)),
        'ndvi_range': float(ndvi_range),
        'peak_time': float(peak_time),
        'median_ndvi': float(median_ndvi),
        'growth_rate': float(growth_rate),
        'decline_rate': float(decline_rate),
        'is_vegetation': bool(is_vegetation),
        'is_annual_crop': bool(is_annual_crop),
        'is_perennial': bool(is_perennial),
        'warning': 'Negative NDVI values present' if (y_smooth < 0).any() else None
    }

@app.route('/show_map', methods=['GET'])
def show_map():
    coordinates_input = request.args.get('coordinates')
    if not coordinates_input:
        return jsonify({'error': 'No coordinates provided'}), 400

    try:
        first_coord = coordinates_input.split(';')[0].strip()
        lat, lon = map(float, first_coord.split(','))
        coordinates = (lat, lon)

        image_url = get_satellite_image([lon, lat], '2020-04-01', '2020-04-15')
        ndvi_df = get_closest_ndvi_data(coordinates, MASTER_DF)

        if not ndvi_df.empty:
            x = ndvi_df['date'].astype(int) / 10**9
            y_smooth = ndvi_df['NDVI'].values
            classification_result = classify_ndvi_curve_function(x, y_smooth, coordinates)
        else:
            classification_result = {
                'classification': 'No Data',
                'confidence': 0.0,
                'warning': f'No data found for {coordinates}'
            }

        categorical_raster = CategoricalRaster(
            id="A2",
            time_range=(
                datetime.fromisoformat("2022-01-01").astimezone(timezone.utc),
                datetime.fromisoformat("2022-12-31").astimezone(timezone.utc)
            ),
            geometry=box(-127.8459, 24.3321, -67.0096, 49.3253).__geo_interface__,
            assets=[{"path_or_url": "/workspace/devkit23/plugins/plugin2/modules/2020_30m_cdls.tif"}],
            bands=[1],
            categories=CATEGORIZATION_CODE_TO_LAND_COVER
        )

        polygon = Polygon([(lon, lat), (lon+0.001, lat), (lon+0.001, lat+0.001), (lon, lat+0.001)])
        crops_finder = plugin2.CropsInAreaFinder(categorical_raster)
        crop_percentages, active_classes = crops_finder.get_crop_area_percentage(polygon)

        return render_template('crop_results.html',
                               coordinates=coordinates,
                               crop_percentages=crop_percentages,
                               active_classes=active_classes,
                               image_url=image_url,
                               classification_result=classification_result)
    except Exception as e:
        logging.error(f"Error processing coordinates: {e}")
        return jsonify({'error': str(e)}), 500


def check_wildfire_proximity(latitude, longitude, radius_km=10):
    """
    Check if any wildfires occurred within radius_km of given location.
    """
    for index, row in WILDFIRE_DATA.iterrows():
        wildfire_location = (row['latitude'], row['longitude'])
        user_location = (latitude, longitude)
        distance_km = geodesic(wildfire_location, user_location).kilometers
        if distance_km <= radius_km:
            return True
    return False

@app.route('/tools/check_wildfire_proximity', methods=['POST'])
def check_wildfire_proximity_endpoint():
    data = request.get_json()
    coordinates = data.get('coordinates')
    radius_km = data.get('radius_km', 10)
    if not coordinates or len(coordinates) != 2:
        return jsonify({"error": "Invalid coordinates provided"}), 400
    try:
        latitude = float(coordinates[0])
        longitude = float(coordinates[1])
        wildfire_detected = check_wildfire_proximity(latitude, longitude, radius_km)
        return jsonify({
            "coordinates": coordinates,
            "radius_km": radius_km,
            "wildfire_detected": wildfire_detected
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/submit_coordinates', methods=['POST'])
def submit_coordinates():
    data = request.get_json()
    coordinates = data.get('coordinates')
    endpoint = request.args.get('endpoint')
    if not coordinates:
        return jsonify({'error': 'No coordinates provided'}), 400
    try:
        coordinates_input = ';'.join([f"{lat},{lon}" for lat, lon in coordinates])
        return jsonify({'message': 'Success', 'redirect_url': url_for(endpoint, coordinates=coordinates_input)})
    except Exception as e:
        logging.error(f"Error processing coordinates: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/further_investigation', methods=['GET', 'POST'])
def further_investigation():
    cdl_data = None
    error = None
    if request.method == 'POST':
        try:
            coordinates_input = request.form['coordinates']
            coordinates = [tuple(map(float, coord.strip().split(','))) for coord in coordinates_input.split(';') if coord.strip()]
            cdl_data = process_coordinates(coordinates)
        except Exception as e:
            logging.error(f"Error processing coordinates: {e}")
            error = str(e)
    return render_template('further_investigation.html', cdl_data=cdl_data, error=error)

@app.route('/ndvi_analysis_insights', methods=['GET'])
def ndvi_analysis_insights():
    coordinates_input = request.args.get('coordinates')
    if not coordinates_input:
        return jsonify({"error": "No coordinates provided"}), 400

    try:
        cleaned_input = coordinates_input.replace('\r', '').replace('\n', '').replace(' ', '').split(';')
        coordinates = []
        for c in cleaned_input:
            if c:
                lat_lon = c.split(',')
                if len(lat_lon) != 2:
                    raise ValueError("Invalid coordinate format")
                lat, lon = map(float, lat_lon)
                coordinates.append((lat, lon))

        results = []
        for coord in coordinates:
            ndvi_df = get_closest_ndvi_data(coord, MASTER_DF)
            if ndvi_df.empty:
                continue
            results.append({'coordinates': coord, 'ndvi_data': ndvi_df.to_dict(orient='records')})

        if not results:
            return jsonify({"error": "Data not found"}), 500

        return render_template('ndvi_analysis.html', results=results)
    except ValueError as ve:
        return jsonify({"error": "Invalid coordinate format"}), 400
    except Exception as e:
        logging.error(f"Error processing NDVI analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/tools/classify_ndvi_curve', methods=['POST'])
def classify_ndvi_curve():
    data = request.get_json()
    if not data or 'coordinates' not in data or not data['coordinates'].strip():
        return jsonify({'error': 'No valid coordinates provided'}), 400

    if request.content_type != 'application/json':
        return jsonify({'error': 'Content-Type must be application/json'}), 415

    coordinates_str = data['coordinates'].strip()
    coord_list = coordinates_str.split(';')

    results = []
    for coord in coord_list:
        coord = coord.strip()
        if not coord:
            continue
        parts = coord.split(',')
        if len(parts) != 2:
            results.append({
                'classification': 'Error',
                'confidence': 0.0,
                'warning': f'Invalid coordinate format: {coord}'
            })
            continue

        lat, lon = map(float, parts)
        ndvi_df = get_closest_ndvi_data((lat, lon), MASTER_DF)
        if ndvi_df.empty:
            results.append({
                'classification': 'No Data',
                'confidence': 0.0,
                'warning': f'No data found for {coord}'
            })
            continue

        x = ndvi_df['date'].astype(int) / 10**9
        y_smooth = ndvi_df['NDVI'].values
        classification_result = classify_ndvi_curve_function(x, y_smooth, (lat, lon))
        results.append(classification_result)

    return jsonify(results)


if __name__ == '__main__':
    app.run(port=8001, debug=True)

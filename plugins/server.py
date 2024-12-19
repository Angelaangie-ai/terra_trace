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





# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules'))
import plugin2  # Import the module from the modules directory

app = Flask(__name__, static_url_path='/static', static_folder='static')
logging.basicConfig(level=logging.INFO)

# Base folder path for the Fifth Part data
base_folder_path = '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset/'

# Set OpenAI API settings
import os

#OpenAI
import openai
openai.api_key = '9231c26620554d659b14fe4d34861434'
openai.api_base = 'https://trapi.research.microsoft.com/gcr/shared'
openai.api_type = 'azure'
openai.api_version = '2023-07-01-preview'

def generate_ndvi_analysis(ndvi_df):
    try:
        logging.info(f"Input NDVI DataFrame: \n{ndvi_df.head()}")
        logging.info(f"NDVI DataFrame shape: {ndvi_df.shape}")
        logging.info(f"NDVI DataFrame columns: {ndvi_df.columns}")
        
        if 'NDVI' not in ndvi_df.columns:
            raise KeyError("'NDVI' column not found in the DataFrame")

        min_ndvi = ndvi_df['NDVI'].min()
        max_ndvi = ndvi_df['NDVI'].max()
        avg_ndvi = ndvi_df['NDVI'].mean()
        
        logging.info(f"Calculated NDVI statistics: min={min_ndvi}, max={max_ndvi}, avg={avg_ndvi}")

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
                <p>High NDVI (<strong>{max_ndvi:.4f}</strong>) indicates dense, healthy vegetation, while low NDVI (<strong>{min_ndvi:.4f}</strong>) suggests sparse or stressed vegetation.</p>
            </div>

            <div class="ndvi-section">
                <h3 class="ndvi-subtitle">📈 Temporal Analysis</h3>
                <p>The {trend_description} trend may indicate {trend_description} growth or seasonal changes in vegetation.</p>
            </div>

            <p class="ndvi-note"><em>Note: For a more comprehensive analysis, consider additional environmental data and expert consultation.</em></p>
        </div>
        """
        logging.info("NDVI analysis generated successfully")
        return analysis
    except Exception as e:
        logging.error(f"Error in generate_ndvi_analysis: {str(e)}", exc_info=True)
        return f"<div class='ndvi-analysis'><p>Error generating NDVI analysis: {str(e)}</p></div>"


def plot_ndvi_timeseries(ndvi_df, save_path):
    try:
        plt.figure(figsize=(10, 5))

        # Convert dates to numerical values
        logging.info("Converting dates to numerical values...")
        ndvi_df['Date'] = pd.to_datetime(ndvi_df['Date'])

        # Resampling to get a few points per month
        ndvi_df.set_index('Date', inplace=True)
        ndvi_resampled = ndvi_df.resample('M').mean()
        ndvi_df.reset_index(inplace=True)
        ndvi_resampled.reset_index(inplace=True)

        date_range = (ndvi_resampled['Date'] - ndvi_resampled['Date'].min()) / np.timedelta64(1, 'D')
        x = date_range.values
        y = ndvi_resampled['NDVI']

        logging.info(f"Date range: {date_range.head()}")
        logging.info(f"x values: {x[:10]}")
        logging.info(f"y values: {y[:10]}")

        # Interpolating the NDVI values to get more points
        logging.info("Interpolating NDVI values...")
        x_new = np.linspace(x.min(), x.max(), 300)  # 300 points for smoother curve
        f = interp1d(x, y, kind='cubic')
        y_new = f(x_new)

        logging.info(f"x_new values: {x_new[:10]}")
        logging.info(f"y_new values: {y_new[:10]}")

        # Fit polynomial curve
        logging.info("Fitting polynomial curve...")
        polynomial_coefficients = np.polyfit(x_new, y_new, 3)  # 3rd degree polynomial
        y_fit = np.polyval(polynomial_coefficients, x_new)

        logging.info(f"Polynomial coefficients: {polynomial_coefficients}")

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
        logging.info(f"Plot saved successfully at {save_path}")

    except Exception as e:
        logging.error(f"Error in plot_ndvi_timeseries: {e}")
        raise

from flask import jsonify, request, url_for
from markupsafe import Markup
import logging

@app.route('/chat_ndvi', methods=['POST'])
def chat_ndvi():
    data = request.get_json()
    message = data.get('message', '').lower()
    logging.info(f"Received message: {message}")

    try:
        if 'analyze the first curve' in message:
            logging.info("Analyzing the first curve")
            ndvi_df = get_first_curve_data()
            logging.info(f"Retrieved NDVI data: {ndvi_df.shape}")
            
            plot_filename = 'ndvi_first_curve.png'
            save_path = os.path.join('static', plot_filename)
            plot_ndvi_timeseries(ndvi_df, save_path)
            plot_url = url_for('static', filename=plot_filename)
            logging.info(f"Generated plot: {plot_url}")

            analysis_html = generate_ndvi_analysis(ndvi_df)
            logging.info("Generated NDVI analysis")
            
            if analysis_html is None:
                logging.error("generate_ndvi_analysis returned None")
                analysis_html = "<p>Error: Unable to generate NDVI analysis.</p>"
            
            # Wrap the HTML content with Markup
            analysis = Markup(analysis_html)
            
            return jsonify({'response': analysis, 'plot_url': plot_url})
        else:
            logging.info("Message not recognized as an analysis request")
            response = "I'm sorry, I didn't understand that. Could you please ask about analyzing the first curve?"
            return jsonify({'response': response})

    except Exception as e:
        logging.error(f"Error in chat_ndvi: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500



def authenticate_and_initialize():
    try:
        # Path to your JSON credentials file
        credentials_path = "/workspace/devkit23/plugins/plugin2/angela.json"

        # Set the environment variable to the path of the credentials JSON file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        # Read the JSON file to extract the project ID
        with open(credentials_path, 'r') as file:
            credentials = json.load(file)
            project_id = credentials.get('project_id', 'soil-424920')

        # Authenticate the Earth Engine API
        ee.Authenticate()

        # Initialize the Earth Engine API with the project ID
        ee.Initialize(project='soil-424920')
        
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"An error occurred during authentication: {e}")

def plot_ndvi_timeseries(ndvi_df, save_path):
    try:
        plt.figure(figsize=(10, 5))

        # Convert dates to numerical values
        ndvi_df['Date'] = pd.to_datetime(ndvi_df['Date'])

        # Resampling to get a few points per month
        ndvi_df.set_index('Date', inplace=True)
        ndvi_resampled = ndvi_df.resample('ME').mean()
        ndvi_df.reset_index(inplace=True)
        ndvi_resampled.reset_index(inplace=True)

        date_range = (ndvi_resampled['Date'] - ndvi_resampled['Date'].min()) / np.timedelta64(1, 'D')
        x = date_range.values
        y = ndvi_resampled['NDVI']

        # Interpolating the NDVI values to get more points
        x_new = np.linspace(x.min(), x.max(), 300)  # 300 points for smoother curve
        f = interp1d(x, y, kind='cubic')
        y_new = f(x_new)

        # Fit polynomial curve
        polynomial_coefficients = np.polyfit(x_new, y_new, 3)  # 3rd degree polynomial
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
        print(f"Plot saved successfully at {save_path}")

    except Exception as e:
        print(f"Error in plot_ndvi_timeseries: {e}")
        raise

# Get satellite images from a given coordinate
def get_satellite_image(coordinates, start_date, end_date):
    point = ee.Geometry.Point(coordinates)
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(point)
                  .filterDate(start_date, end_date)
                  .sort('CLOUDY_PIXEL_PERCENTAGE')
                  .first())
    # Print collection information for debugging
    print(collection.getInfo())
    vis_params = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2'],
        'gamma': 1.4
    }
    scale = 10
    region = point.buffer(1000).bounds().getInfo()['coordinates']  # Buffer to include a larger area
    url = collection.getThumbURL({
        'min': vis_params['min'],
        'max': vis_params['max'],
        'bands': vis_params['bands'],
        'gamma': vis_params['gamma'],
        'scale': scale,
        'region': region
    })
    return url

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
        return "No data available"
    elif value < -0.1:
        return "indicating non-vegetative or very sparse vegetation"
    elif -0.1 <= value < 0:
        return "indicating sparse vegetation"
    elif 0 <= value < 0.2:
        return "indicating some vegetation"
    else:
        return "indicating healthy vegetation"

def generate_explanation(df):
    explanation = ""
    for index, row in df.iterrows():
        month = row["Month"]
        explanation += f"\n### {month}:\n"
        for year in df.columns[1:]:
            value = row[year]
            condition = explain_ndvi(value)
            explanation += f"  - {year}: {value} ({condition})\n"
    return explanation

def get_first_curve_data():
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')  # Monthly data
    ndvi_values = np.random.rand(len(dates))  # Replace with actual NDVI values
    data = {'Date': dates, 'NDVI': ndvi_values}
    return pd.DataFrame(data)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')

    # Define the deployment ID
    deployment_id = 'gpt-35-turbo'  # Replace with your actual deployment ID

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


def fetch_ndvi_data(closest_coords, csv_file_path):
    csv_file_path = '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset/combined_data1.csv'

    # Read the CSV data into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Filter the DataFrame to get the row matching the closest coordinates
    closest_row = df[(df['lat'] == closest_coords[0]) & (df['lon'] == closest_coords[1])]
    
    # Drop the 'lat' and 'lon' columns to get only the date and NDVI values
    closest_row = closest_row.drop(columns=['lat', 'lon'])

    # Transpose the DataFrame to have dates as rows and values as a single column
    ndvi_data = closest_row.transpose().reset_index()
    ndvi_data.columns = ['Date', 'NDVI']
    
    return ndvi_data


# Function to find the closest coordinates in the dataset
def find_closest_coordinate(target_coords, df):
    distances = df.apply(lambda row: distance.euclidean(target_coords, (row['lat'], row['lon'])), axis=1)
    closest_idx = distances.idxmin()
    closest_point = df.loc[closest_idx]
    logging.info(f"Closest point found: {closest_point['lat']}, {closest_point['lon']}")
    return closest_point

@app.route('/plot', methods=['POST'])
def plot():
    coordinates_input = request.form['coordinates']
    coordinates = [tuple(map(float, coord.strip().split(','))) for coord in coordinates_input.split(';') if coord.strip()]
    
    plots = []
    for lat, lon in coordinates:
        csv_filepath = '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset/combined_data1.csv'
        df = pd.read_csv(csv_filepath)
        
        closest_point = plugin2.find_closest_coordinates((lat, lon), df)
        closest_coords = (closest_point['lat'], closest_point['lon'])
    

        ndvi_df = fetch_ndvi_data(closest_coords, csv_filepath)
        print(ndvi_df)
        
        plot_filename = f'ndvi_plot_{lat}_{lon}.png'
        save_path = os.path.join('static', plot_filename)
        plot_ndvi_timeseries(ndvi_df, save_path)
        plots.append((lat, lon, save_path))
    
    return render_template('plot.html', plots=plots)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

def get_ndvi_data(csv_filepath, coord):
    df = pd.read_csv(csv_filepath)
    logging.info(f"CSV Columns: {df.columns.tolist()}")
    
    lat, lon = coord

    # Filter data for the given coordinates
    df_filtered = df[(df['lat'] == lat) & (df['lon'] == lon)]

    if df_filtered.empty:
        logging.error(f"No data found for coordinates: {coord}")
        return None

    # Melt the DataFrame to long format
    df_melted = df_filtered.melt(id_vars=['lat', 'lon'], var_name='date', value_name='NDVI')
    df_melted.dropna(subset=['NDVI'], inplace=True)

    # Parse dates and sort data
    df_melted['date'] = pd.to_datetime(df_melted['date'])
    df_melted['year'] = df_melted['date'].dt.year
    df_melted['month'] = df_melted['date'].dt.month
    df_melted.sort_values(by='date', inplace=True)

    # Calculate yearly differences and percentage changes
    changes = []
    previous_ndvi = None
    for _, row in df_melted.iterrows():
        if previous_ndvi is not None:
            yearly_change = row['NDVI'] - previous_ndvi
            percentage_change = (yearly_change / previous_ndvi) * 100 if previous_ndvi != 0 else 0
            logging.info(f"Date: {row['date']}, NDVI: {row['NDVI']}, Yearly Change: {yearly_change}, Percentage Change: {percentage_change}")
        else:
            yearly_change = None
            percentage_change = None

        changes.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'NDVI': row['NDVI'],
            'yearly_change': yearly_change,
            'percentage_change': percentage_change
        })
        
        previous_ndvi = row['NDVI']

    ndvi_data = df_melted.to_dict(orient='records')
    return {'coordinates': coord, 'ndvi_data': ndvi_data, 'changes': changes}

@app.route('/tools/my_tool', methods=['POST'])
def handle_my_tool():
    data = request.get_json()
    arg1 = data.get('arg1')
    arg2 = data.get('arg2')
    # Call the plugin function
    result = plugin2.my_tool(arg1, arg2)
    return jsonify({'arg1': arg1, 'arg2': arg2, 'message': 'Tool executed successfully!'})

@app.route('/tools/simple_gpt_tool', methods=['POST'])
def handle_simple_gpt_tool():
    data = request.get_json()
    arg1 = data.get('arg1')
    arg2 = data.get('arg2')
    # Call the new plugin function
    result = plugin2.simple_gpt_tool(arg1, arg2)
    return jsonify({'result': result})

@app.route('/tools/ndvi_explanation_tool', methods=['POST'])
def ndvi_explanation_tool():
    data = request.get_json()
    coordinates = data.get('coordinates')
    csv_filepath = '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset/combined_data1.csv'
    
    # Assuming coordinates is a single coordinate tuple
    coord = tuple(map(float, coordinates.split(',')))
    
    # Fetch the NDVI data for the given coordinates
    result = get_ndvi_data(csv_filepath, coord)
    if not result:
        return jsonify({"error": "Data not found for the given coordinates"}), 500

    # Convert the NDVI data to a DataFrame
    ndvi_data = result['ndvi_data']
    df = pd.DataFrame(ndvi_data)
    df_wide = df.pivot(index='month', columns='year', values='NDVI').reset_index()
    df_wide.columns.name = None  # Remove index name

    # Generate the explanation
    explanation = generate_explanation(df_wide)
    return jsonify({"explanation": explanation})

# Define the categorization code to land cover mapping
CATEGORIZATION_CODE_TO_LAND_COVER = {
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

def process_coordinates(coordinates):
    try:
        # Set up the categorical raster
        categorical_raster = CategoricalRaster(
            id="A2",
            time_range=(
                datetime.fromisoformat("2022-01-01").astimezone(timezone.utc),
                datetime.fromisoformat("2022-12-31").astimezone(timezone.utc)
            ),
            geometry=box(-127.8459, 24.3321, -67.0096, 49.3253).__geo_interface__,
            assets=[{
                "path_or_url": "/workspace/devkit23/plugins/plugin2/modules/2020_30m_cdls.tif"
            }],
            bands=[1],
            categories=CATEGORIZATION_CODE_TO_LAND_COVER
        )
        # Initialize the crops finder
        crops_finder = plugin2.CropsInAreaFinder(categorical_raster)
        polygon = Polygon(coordinates)
        # Calculate crop percentages
        crop_percentages = crops_finder.get_crop_area_percentage(polygon)
        
        # Get unique active classes
        active_classes = np.unique(crops_finder.read_raster_section(categorical_raster, polygon))
        
        # Get crop mask length
        crop_mask_length = len(crops_finder.get_crop_mask(categorical_raster))
        
        # Decode classes
        decoded_classes = {int(code): CATEGORIZATION_CODE_TO_LAND_COVER.get(int(code), "Unknown") for code in active_classes}
        # Return the processed data
        return {
            'crop_percentages': crop_percentages,
            'active_classes': active_classes.tolist(),  # Convert to list for JSON serialization
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
    logging.info(f"Received coordinates input: {coordinates_input}")
    try:
        coordinates = [tuple(map(float, coord.strip().split(','))) for coord in coordinates_input.split(';') if coord.strip()]
    except ValueError as e:
        logging.error(f"Error parsing coordinates: {e}")
        return jsonify({'error': 'Invalid coordinate format'}), 400
    
    plots = []
    image_url = None
    for lat, lon in coordinates:
        try:
            logging.info(f"Processing coordinates: ({lat}, {lon})")
            coords = [lon, lat]
            start_date = '2020-04-01'
            end_date = '2020-04-15'
            image_url = get_satellite_image(coords, start_date, end_date)
        
            csv_filepath = os.path.join(base_folder_path, 'combined_data1.csv')
            logging.info(f"Reading CSV file: {csv_filepath}")
            df = pd.read_csv(csv_filepath)
            logging.info(f"CSV file loaded successfully. Dataframe shape: {df.shape}")

            logging.info(f"Finding closest coordinates in dataframe for: ({lat}, {lon})")
            closest_point = plugin2.find_closest_coordinates((lat, lon), df)
            closest_coords = (closest_point['lat'], closest_point['lon'])
            logging.info(f"Closest coordinates found: {closest_coords}")

            logging.info(f"Fetching NDVI data for closest coordinates: {closest_coords}")
            ndvi_df = fetch_ndvi_data(closest_coords, csv_filepath)
            logging.info(f"NDVI data fetched successfully. Dataframe shape: {ndvi_df.shape}")

            plot_filename = f'ndvi_plot_{lat}_{lon}.png'
            save_path = os.path.join('static', plot_filename)
            logging.info(f"Plotting NDVI timeseries data. Save path: {save_path}")
            plot_ndvi_timeseries(ndvi_df, save_path)
            logging.info(f"Plot saved successfully: {save_path}")
            
            plots.append((lat, lon, plot_filename))

        except Exception as e:
            logging.error(f"Error processing coordinates ({lat}, {lon}): {e}")
            return jsonify({'error': f"Error processing coordinates ({lat}, {lon}): {e}"}), 500
    return render_template('show_map.html', coordinates=coordinates_input, plots=plots, image_url=image_url)

def find_region_for_coordinates(coordinates, regions):
    """Function to map coordinates to regions."""
    coordinates_region_map = {}
    for coord in coordinates:
        for region, region_coords in regions.items():
            polygon = Polygon(region_coords)
            if polygon.contains(Point(coord)):
                coordinates_region_map[coord] = region
                break
    return coordinates_region_map

@app.route('/show_map', methods=['GET'])
def show_map():
    coordinates_input = request.args.get('coordinates')
    logging.info(f"Received coordinates: {coordinates_input}")

    try:
        # Split the input string and take the first coordinate pair
        first_coord = coordinates_input.split(';')[0].strip()
        lat, lon = map(float, first_coord.split(','))
        coordinates = (lat, lon)

        logging.info(f"Processing coordinates: ({lat}, {lon})")

        image_url = get_satellite_image([lon, lat], '2020-04-01', '2020-04-15')

        csv_filepath = '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset_2020/parody.csv'
        ndvi_df = get_ndvi_data1(csv_filepath, coordinates)

        classification_result = None
        if not ndvi_df.empty:
            x = ndvi_df['date'].astype(int) / 10**9
            y_smooth = ndvi_df['NDVI'].values
            reference_dir = '/workspace/devkit23/plugins/plugin2/reference_images'
            classification_result = classify_ndvi_curve_function(x, y_smooth, coordinates)
        else:
            classification_result = {
                'classification': 'No Data',
                'confidence': 0.0,
                'warning': f'No data found for the given coordinates: {coordinates}'
            }

        categorical_raster = CategoricalRaster(
            id="A2",
            time_range=(
                datetime.fromisoformat("2022-01-01").astimezone(timezone.utc),
                datetime.fromisoformat("2022-12-31").astimezone(timezone.utc)
            ),
            geometry=box(-127.8459, 24.3321, -67.0096, 49.3253).__geo_interface__,
            assets=[{
                "path_or_url": "/workspace/devkit23/plugins/plugin2/modules/2020_30m_cdls.tif"
            }],
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

# Load the wildfire dataset
wildfire_data = pd.read_csv('/workspace/devkit23/plugins/plugin2/modules/modis_2020_United_States.csv')
def check_wildfire_proximity(latitude, longitude, radius_km=10):
    wildfire_detected = False
    logging.info(f"Checking wildfires for user location: ({latitude}, {longitude})")
    for index, row in wildfire_data.iterrows():
        wildfire_location = (row['latitude'], row['longitude'])
        user_location = (latitude, longitude)
        distance = geodesic(wildfire_location, user_location).kilometers
        if distance <= radius_km:
            wildfire_detected = True
            logging.info(f"Wildfire detected within {radius_km} km at location: {wildfire_location}")
            break
    if not wildfire_detected:
        logging.info(f"No wildfires detected within {radius_km} km for user location: ({latitude}, {longitude})")
    return wildfire_detected

@app.route('/tools/check_wildfire_proximity', methods=['POST'])
def check_wildfire_proximity_endpoint():
    data = request.get_json()
    coordinates = data.get('coordinates')
    radius_km = data.get('radius_km', 10)  # Default radius is 10 km
    if not coordinates or len(coordinates) != 2:
        logging.error("Invalid coordinates provided")
        return jsonify({"error": "Invalid coordinates provided"}), 400
    try:
        latitude = float(coordinates[0])
        longitude = float(coordinates[1])
        wildfire_detected = check_wildfire_proximity(latitude, longitude, radius_km)
        result = {
            "coordinates": coordinates,
            "radius_km": radius_km,
            "wildfire_detected": wildfire_detected
        }
        logging.info(f"Result: {result}")
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error checking wildfire proximity: {e}")
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

import logging
logging.basicConfig(level=logging.DEBUG)
def process_analyze_and_plot_points(base_folder_path, coordinates, regions):
    coordinates_region_map = find_region_for_coordinates(coordinates, regions)
    results = []
    logging.debug(f"Coordinates Region Map: {coordinates_region_map}")
    for coords, region in coordinates_region_map.items():
        logging.debug(f"Processing coordinates: {coords} in region: {region}")
        if region == "Fifth Part":
            region_folder_path = os.path.join(base_folder_path, region.replace(" ", "_"))
            csv_filename = 'fifth_part_combined_data_2020_2021_2022.csv'
            csv_filepath = os.path.join(region_folder_path, csv_filename)
            # Check if the file exists
            if not os.path.exists(csv_filepath):
                logging.error(f"CSV file not found: {csv_filepath}")
                continue
            # Load the CSV file
            df = pd.read_csv(csv_filepath)
            logging.debug(f"DataFrame loaded with shape: {df.shape}")
            
            # Convert DataFrame from wide to long format properly
            id_vars = ['lat', 'lon']  # Adjust if your data includes other identifier columns
            value_vars = [col for col in df.columns if col not in id_vars]  # Assumes all other columns are dates
            df_long = pd.melt(df, id_vars=id_vars, var_name='date', value_name='NDVI')
            logging.debug(f"DataFrame long format with shape: {df_long.shape}")
            
            # Convert 'date' column to datetime
            df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
            # Find the closest point
            closest_point = find_closest_coordinate((coords[0], coords[1]), df_long)
            logging.debug(f"Closest point found: {closest_point}")
            # Filter data to include only the closest point's data
            closest_data = df_long[(df_long['lat'] == closest_point['lat']) & (df_long['lon'] == closest_point['lon'])]
            logging.debug(f"Closest data shape: {closest_data.shape}")
            # NDVI Calculations
            closest_data['month'] = closest_data['date'].dt.month
            closest_data['year'] = closest_data['date'].dt.year
            average_ndvi = closest_data.groupby(['year', 'month'])['NDVI'].mean().reset_index()
            logging.debug(f"Average NDVI: {average_ndvi}")
            # Store results
            results.append({
                'coordinates': coords,
                'average_ndvi': average_ndvi.to_dict(orient='records'),
            })
    logging.debug(f"Final Results: {results}")
    return results

import pandas as pd
import logging

def get_ndvi_data(csv_filepath, coordinates):
    logging.info(f"Getting NDVI data for coordinates: {coordinates}")

    if not os.path.exists(csv_filepath):
        logging.error(f"CSV file not found: {csv_filepath}")
        return None
    
    # Load the CSV file
    df = pd.read_csv(csv_filepath)
    logging.info(f"CSV file loaded successfully: {csv_filepath}")
    
    # Convert DataFrame from wide to long format
    id_vars = ['lat', 'lon']
    value_vars = [col for col in df.columns if not col in id_vars]
    df_long = pd.melt(df, id_vars=id_vars, var_name='date', value_name='NDVI')
    logging.info(f"DataFrame converted from wide to long format")
    
    # Convert 'date' column to datetime
    df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
    logging.info(f"Date column converted to datetime")
    
    # Find the closest point
    closest_point = find_closest_coordinate(coordinates, df_long)
    logging.info(f"Closest point: {closest_point.to_dict()}")
    
    # Filter data to include only the closest point's data
    closest_data = df_long[(df_long['lat'] == closest_point['lat']) & (df_long['lon'] == closest_point['lon'])]
    
    if closest_data.empty:
        logging.error("No NDVI data found for the closest point.")
        return None
    
    logging.info(f"Data filtered to include only the closest point's data")
    
    # NDVI Calculations
    closest_data['month'] = closest_data['date'].dt.month
    closest_data['year'] = closest_data['date'].dt.year
    average_ndvi = closest_data.groupby(['year', 'month'])['NDVI'].mean().reset_index()
    
    closest_data.sort_values(by='date', inplace=True)
    closest_data['yearly_change'] = closest_data.groupby('year')['NDVI'].diff().dropna()
    closest_data['percentage_change'] = closest_data.groupby('year')['NDVI'].pct_change().dropna() * 100
    
    changes = closest_data[['date', 'NDVI', 'yearly_change', 'percentage_change']].dropna().reset_index(drop=True)
    
    logging.info(f"NDVI calculations completed")

    return {
        'coordinates': coordinates,
        'ndvi_data': average_ndvi.to_dict(orient='records'),
        'changes': changes.to_dict(orient='records')
    }

def preprocess_curve(ndvi_df):
    if ndvi_df.empty:
        raise ValueError("NDVI data is empty")
        
    date_range = np.arange(len(ndvi_df))
    x = date_range
    y = ndvi_df['NDVI'].values
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    y_smooth = gaussian_filter1d(y, sigma=2)
    return x, y_smooth


def load_reference_images(reference_dir):
    reference_curves = {}
    for root, _, files in os.walk(reference_dir):
        for file in files:
            if file.endswith(".png"):
                label = os.path.basename(root)
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert("L")
                image_array = np.asarray(image).flatten()
                reference_curves[label] = image_array
    return reference_curves

@app.route('/further_investigation', methods=['GET', 'POST'])
def further_investigation():
    cdl_data = None
    error = None
    if request.method == 'POST':
        try:
            coordinates_input = request.form['coordinates']
            logging.info(f"Received coordinates input: {coordinates_input}")
            coordinates = [tuple(map(float, coord.strip().split(','))) for coord in coordinates_input.split(';') if coord.strip()]
            logging.info(f"Parsed coordinates: {coordinates}")
            cdl_data = process_coordinates(coordinates)
            logging.info(f"CDL Data: {cdl_data}")
        except Exception as e:
            logging.error(f"Error processing coordinates: {e}")
            error = str(e)
    
    logging.info(f"Rendered CDL Data: {cdl_data}")
    return render_template('further_investigation.html', cdl_data=cdl_data, error=error)

import numpy as np
from scipy.interpolate import interp1d


import numpy as np
from scipy.interpolate import interp1d

def classify_ndvi_curve_function(x, y_smooth, coordinates=None):
    logging.info(f"Starting classification for coordinates: {coordinates}")
    logging.info(f"Initial y_smooth: {y_smooth}")
    logging.info(f"Initial x: {x}")

    # Check for invalid NDVI values
    if np.any(y_smooth < -1) or np.any(y_smooth > 1):
        logging.warning(f"Invalid NDVI values detected: min={np.min(y_smooth)}, max={np.max(y_smooth)}")

    # Remove NaN values
    valid_indices = ~np.isnan(y_smooth)
    x = x[valid_indices]
    y_smooth = y_smooth[valid_indices]
    logging.info(f"After removing NaNs, y_smooth: {y_smooth}")

    # Check if we have enough valid data points
    if len(y_smooth) < 5:
        logging.warning("Insufficient data points for classification")
        return {
            'classification': 'Insufficient Data',
            'confidence': 0,
            'warning': 'Not enough valid data points for classification'
        }

    # Log original stats before normalization
    logging.info(f"Original stats - min: {np.min(y_smooth)}, max: {np.max(y_smooth)}, mean: {np.mean(y_smooth)}, median: {np.median(y_smooth)}")

    # Shift NDVI values to ensure all are positive
    y_shifted = y_smooth - np.min(y_smooth) + 0.1  # Add 0.1 to avoid zero values
    logging.info(f"Shifted y values: {y_shifted}")

    # Normalize NDVI values to 0-1 range
    y_normalized = (y_shifted - np.min(y_shifted)) / (np.max(y_shifted) - np.min(y_shifted))
    logging.info(f"Normalized y values: {y_normalized}")

    # Extract key features
    max_ndvi = np.max(y_normalized)
    min_ndvi = np.min(y_normalized)
    ndvi_range = np.ptp(y_normalized)
    peak_time = np.argmax(y_normalized) / len(y_normalized)
    median_ndvi = np.median(y_normalized)

    logging.info(f"Max NDVI: {max_ndvi}")
    logging.info(f"Min NDVI: {min_ndvi}")
    logging.info(f"NDVI Range: {ndvi_range}")
    logging.info(f"Peak Time: {peak_time}")
    logging.info(f"Median NDVI: {median_ndvi}")

    # Calculate growth and decline rates
    growth_rate = np.max(np.diff(y_normalized[:len(y_normalized)//2]))
    decline_rate = np.min(np.diff(y_normalized[len(y_normalized)//2:]))
    logging.info(f"Growth Rate: {growth_rate}")
    logging.info(f"Decline Rate: {decline_rate}")

    # Adjust classification criteria
    is_vegetation = max_ndvi > 0.3
    logging.info(f"Is Vegetation: {is_vegetation}")

    is_annual_crop = (
        is_vegetation and
        ndvi_range > 0.2 and
        0.2 < peak_time < 0.8 and
        growth_rate > 0.005 and
        decline_rate < -0.005
    )
    logging.info(f"Is Annual Crop: {is_annual_crop}")

    is_perennial = (
        is_vegetation and
        ndvi_range < 0.2 and
        median_ndvi > 0.4
    )
    logging.info(f"Is Perennial: {is_perennial}")

    # Classification
    if not is_vegetation:
        classification = 'Non-vegetation or Water'
    elif is_annual_crop:
        classification = 'Annual Crop (possibly Corn)'
    elif is_perennial:
        classification = 'Perennial Vegetation'
    else:
        classification = 'Mixed or Undefined Vegetation'
    logging.info(f"Classification: {classification}")

    # Calculate confidence
    confidence = min(1.0, max(0.0, ndvi_range * 2))
    logging.info(f"Confidence: {confidence}")

    result = {
        'classification': classification,
        'confidence': float(confidence),
        'max_ndvi': float(np.max(y_smooth)),  # Original max NDVI
        'min_ndvi': float(np.min(y_smooth)),  # Original min NDVI
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

    logging.info(f"Final result: {result}")
    return result

@app.route('/ndvi_analysis_insights', methods=['GET'])
def ndvi_analysis_insights():
    coordinates_input = request.args.get('coordinates')
    csv_filepath = '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset/combined_data1.csv'
    
    if not coordinates_input:
        logging.error("No coordinates provided")
        return jsonify({"error": "No coordinates provided"}), 400

    try:
        logging.info(f"Coordinates input received: {coordinates_input}")
        cleaned_input = coordinates_input.replace('\r', '').replace('\n', '').replace(' ', '').split(';')
        logging.info(f"Cleaned input: {cleaned_input}")
        coordinates = []
        for coord in cleaned_input:
            if coord:
                lat_lon = coord.split(',')
                logging.info(f"Parsing lat_lon: {lat_lon}")
                if len(lat_lon) != 2:
                    logging.error(f"Invalid lat_lon pair: {lat_lon}")
                    raise ValueError("Invalid coordinate format")
                lat, lon = map(float, lat_lon)
                coordinates.append((lat, lon))
        logging.info(f"Parsed coordinates: {coordinates}")
        
        results = []
        for coord in coordinates:
            result = get_ndvi_data(csv_filepath, coord)
            if result:
                logging.info(f"NDVI data for {coord}: {result}")
                results.append(result)
        
        if not results:
            logging.error("Data not found")
            return jsonify({"error": "Data not found"}), 500

        logging.info(f"NDVI Analysis Results: {results}")
        return render_template('ndvi_analysis.html', results=results)
    except ValueError as ve:
        logging.error(f"Error parsing coordinates: {ve}")
        return jsonify({"error": "Invalid coordinate format"}), 400
    except Exception as e:
        logging.error(f"Error processing NDVI analysis: {e}")
        return jsonify({"error": str(e)}), 500

def get_ndvi_data1(csv_filepath, coord):
    try:
        df = pd.read_csv(csv_filepath)
        logging.info(f"CSV Columns: {df.columns.tolist()}")
        
        lat, lon = coord

        # Find the closest coordinates
        closest_point = find_closest_coordinate((lat, lon), df)
        closest_lat, closest_lon = closest_point['lat'], closest_point['lon']
        print(closest_lat)
        print (closest_lon)

        # Filter data for the closest coordinates
        df_filtered = df[(df['lat'] == closest_lat) & (df['lon'] == closest_lon)]
        print(df_filtered)

        if df_filtered.empty:
            logging.error(f"No data found for coordinates: {coord}")
            return pd.DataFrame()

        # Melt the DataFrame to long format
        df_melted = df_filtered.melt(id_vars=['lat', 'lon'], var_name='date', value_name='NDVI')
        df_melted.dropna(subset=['NDVI'], inplace=True)

        # Parse dates and sort data
        df_melted['date'] = pd.to_datetime(df_melted['date'])
        df_melted['year'] = df_melted['date'].dt.year
        df_melted['month'] = df_melted['date'].dt.month
        df_melted.sort_values(by='date', inplace=True)

        # Calculate yearly differences and percentage changes
        df_melted['yearly_change'] = df_melted.groupby('year')['NDVI'].diff().fillna(0)
        df_melted['percentage_change'] = df_melted.groupby('year')['NDVI'].pct_change(fill_method=None).fillna(0) * 100

        return df_melted
    except Exception as e:
        logging.error(f"Error in get_ndvi_data1: {e}")
        return pd.DataFrame()

@app.route('/tools/classify_ndvi_curve', methods=['POST'])
def classify_ndvi_curve():
    data = request.get_json()
    if not data or 'coordinates' not in data or not data['coordinates'].strip():
        return jsonify({'error': 'No valid coordinates provided'}), 400

    coordinates = data['coordinates'].strip()
    if request.content_type != 'application/json':
        return jsonify({'error': 'Content-Type must be application/json'}), 415

    coordinates = data.get('coordinates')
    logging.info(f"Received coordinates: {coordinates}")
    
    try:
        coord_list = coordinates.split(';')
        logging.info(f"Split coordinates: {coord_list}")
        
        results = []
        for coord in coord_list:
            coord = coord.strip()
            logging.info(f"Processing coordinate: {coord}")
            
            if not coord:
                logging.warning("Empty coordinate found, skipping")
                continue
            
            try:
                parts = coord.split(',')
                logging.info(f"Coordinate parts: {parts}")
                
                if len(parts) != 2:
                    raise ValueError(f"Invalid coordinate format: {coord}")
                
                lat, lon = [part.strip() for part in parts]
                logging.info(f"Parsed lat: {lat}, lon: {lon}")
                
                if not lat or not lon:
                    raise ValueError(f"Empty latitude or longitude in coordinate: {coord}")
                
                lat, lon = float(lat), float(lon)
                coord_tuple = (lat, lon)
                logging.info(f"Final coordinate tuple: {coord_tuple}")
                
                csv_filepath = '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset_2020/parody.csv'
                ndvi_df = get_ndvi_data1(csv_filepath, coord_tuple)

                if ndvi_df.empty:
                    logging.warning(f"No NDVI data found for coordinate: {coord_tuple}")
                    results.append({
                        'classification': 'No Data',
                        'confidence': 0.0,
                        'warning': f'No data found for the given coordinates: {coord_tuple}'
                    })
                    continue

                x = ndvi_df['date'].astype(int) / 10**9  # Convert datetime to Unix timestamp
                y_smooth = ndvi_df['NDVI'].values
                reference_dir = '/workspace/devkit23/plugins/plugin2/reference_images'
                
                classification_result = classify_ndvi_curve_function(x, y_smooth, coord_tuple)
                logging.info(f"Classification result for {coord_tuple}: {classification_result}")
                results.append(classification_result)
            except ValueError as ve:
                logging.error(f"ValueError in processing coordinate {coord}: {ve}")
                results.append({
                    'classification': 'Error',
                    'confidence': 0.0,
                    'warning': f'Invalid coordinate format: {coord}. Error: {str(ve)}'
                })
            except Exception as e:
                logging.error(f"Unexpected error processing coordinate {coord}: {str(e)}")
                results.append({
                    'classification': 'Error',
                    'confidence': 0.0,
                    'warning': f'Error processing coordinate {coord}: {str(e)}'
                })

        logging.info(f"Final Classification Results: {results}")
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error in classify_ndvi_curve endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8001, debug=True)

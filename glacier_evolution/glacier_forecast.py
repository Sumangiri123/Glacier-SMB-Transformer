import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import rasterio
from keras.models import load_model



WORKSPACE = Path().resolve()
DATA_DIR = WORKSPACE / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = WORKSPACE / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'

SCALER_PATH = MODEL_DIR / 'scaler.pkl'
ANN_ENSEMBLE_DIR = MODEL_DIR / 'ann_ensemble'
FEATURE_NAMES_PATH = MODEL_DIR / 'feature_names.pkl'
INITIAL_STATE_DIR = OUTPUT_DIR / 'evolution_results' / 'rasters'
FUTURE_CLIMATE_DIR = PROCESSED_DIR / 'future_climate'
GLACIER_DATA_DIR = RAW_DIR / 'glacier_data'
GLIMS_2003_FILE = GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_2003.csv'
GLIMS_2015_FILE = GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv'

FORECAST_DIR = OUTPUT_DIR / 'forecast_results'

START_YEAR = 2015
END_YEAR = 2100
ICE_DENSITY = 850
SCENARIOS_TO_RUN = ['ssp245', 'ssp585']




def r2_keras(y_true, y_pred): return 0
def root_mean_squared_error(y_true, y_pred): return 0

def calculate_monthly_climate_features(daily_df, year):
    start_date = f"{year-1}-10-01"
    end_date = f"{year}-09-30"
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    year_df = daily_df[(daily_df['date'] >= start_date) & (daily_df['date'] <= end_date)].copy()
    
    if year_df.empty:
        months = pd.date_range(start=start_date, end=end_date, freq='MS')
        nan_features = {f"temp_{m.strftime('%b').lower()}": np.nan for m in months}
        nan_features.update({f"snow_{m.strftime('%b').lower()}": np.nan for m in months})
        return nan_features
        
    year_df['month'] = year_df['date'].dt.to_period('M')
    monthly_temp = year_df.groupby('month')['temperature_c'].mean()
    monthly_snow = year_df.groupby('month')['snowfall_mm'].sum()
    
    features = {}
    months = pd.date_range(start=start_date, end=end_date, freq='MS').to_period('M')
    for month in months:
        month_abbr = month.strftime('%b').lower()
        features[f"temp_{month_abbr}"] = monthly_temp.get(month, 0)
        features[f"snow_{month_abbr}"] = monthly_snow.get(month, 0)
    return features

def load_models_and_data():
    print("Loading models and data...")
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f: feature_names = pickle.load(f)
    
    ann_models = []
    for model_file in sorted(os.listdir(ANN_ENSEMBLE_DIR)):
        if model_file.endswith('.keras'):
            model = load_model(ANN_ENSEMBLE_DIR / model_file, custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error})
            ann_models.append(model)

    glims_2015 = pd.read_csv(GLIMS_2015_FILE, delimiter=';', encoding='latin1')
    glims_2003 = pd.read_csv(GLIMS_2003_FILE, delimiter=';', encoding='latin1', header=None, names=['Area','Perimeter','Glacier','Annee','Massif','MEAN_Pixel','MIN_Pixel','MAX_Pixel','MEDIAN_Pix','Length','Aspect','x_coord','y_coord','GLIMS_ID','ID','Aspect_num','slope20'])
    glims_2003 = glims_2003.drop_duplicates(subset=['GLIMS_ID'], keep='first') 
    id_map = glims_2003[['GLIMS_ID', 'ID']]
    static_df = pd.merge(glims_2015, id_map, on='GLIMS_ID', how='left').rename(columns={'Glacier': 'name', 'Area': 'area', 'slope20': 'slope', 'x_coord': 'lon', 'y_coord': 'lat', 'Aspect_num': 'aspect'}).set_index('GLIMS_ID')

    print(f"Loaded scaler, {len(ann_models)} ANN models, {len(feature_names)} feature names, and static data.")
    return scaler, ann_models, static_df, feature_names

def run_glacier_forecast(glims_id, static_info, scaler, ann_models, climate_data, feature_names, scenario):
    try:
        initial_thickness_path = INITIAL_STATE_DIR / f'{glims_id}_thickness_2015.tif'
        initial_dem_path = INITIAL_STATE_DIR / f'{glims_id}_dem_2015.tif'
        if not initial_thickness_path.exists() or not initial_dem_path.exists():
            print(f"   - Initial state for 2015 not found. Skipping.")
            return

        with rasterio.open(initial_dem_path) as src: dem = src.read(1); profile = src.profile
        with rasterio.open(initial_thickness_path) as src: thickness = src.read(1)
            
    except Exception as e:
        print(f"   - Error loading initial raster files: {e}. Skipping.")
        return

    initial_thickness = thickness.copy()
    thickness[thickness < 0] = 0
    dem[dem < 0] = np.nan
    pixel_area = profile['transform'][0] * abs(profile['transform'][4])

    results = []
    initial_mask = thickness > 0
    initial_area = np.sum(initial_mask) * pixel_area / 1e6
    initial_volume = np.sum(thickness[initial_mask]) * pixel_area / 1e9
    results.append({'year': START_YEAR, 'smb_mwe': np.nan, 'volume_km3': initial_volume, 'area_km2': initial_area})

    for year in range(START_YEAR + 1, END_YEAR + 1):
        glacier_mask = thickness > 0
        if not np.any(glacier_mask):
            print(f"   - Glacier melted in year {year-1}.")
            break

        current_area = np.sum(glacier_mask) * pixel_area / 1e6
        current_volume = np.sum(thickness[glacier_mask]) * pixel_area / 1e9 
        
        bedrock = dem - thickness
        monthly_features = calculate_monthly_climate_features(climate_data, year)
        
        static_features = static_info.to_dict()
        static_features['area'] = current_area
        
        feature_vector_dict = {**monthly_features, **static_features}
        feature_vector_df = pd.DataFrame([feature_vector_dict]).ffill().bfill().fillna(0)
        feature_vector_df = feature_vector_df[feature_names]
        
        X_scaled = scaler.transform(feature_vector_df.values)
        ann_preds = [model.predict(X_scaled, verbose=0).flatten()[0] for model in ann_models]
        smb_predicted = np.mean(ann_preds)

        results.append({
            'year': year,
            'smb_mwe': smb_predicted, 
            'volume_km3': current_volume, 
            'area_km2': current_area
        })

        total_mass_change = smb_predicted * current_area * 1e6
        ice_volume_change = total_mass_change * (1000 / ICE_DENSITY)

        surface_elevation = np.where(glacier_mask, dem, np.nan)
        min_elev, max_elev = np.nanmin(surface_elevation), np.nanmax(surface_elevation)
        if np.isnan(min_elev) or min_elev == max_elev:
            elev_normalized = np.full_like(surface_elevation, 0.5)
        else:
            elev_normalized = (surface_elevation - min_elev) / (max_elev - min_elev)
        
        delta_h_distribution = 1 - elev_normalized
        sum_delta_h = np.nansum(delta_h_distribution[glacier_mask])
        if sum_delta_h == 0:
             delta_h_normalized = np.ones(np.sum(glacier_mask)) / np.sum(glacier_mask) if np.sum(glacier_mask) > 0 else np.array([])
        else:
             delta_h_normalized = delta_h_distribution[glacier_mask] / sum_delta_h


        thickness_change_total = ice_volume_change / pixel_area
        thickness_change_per_pixel = thickness_change_total * delta_h_normalized if delta_h_normalized.size > 0 else np.array([])
        
        thickness_change_grid = np.zeros_like(thickness)
        if thickness_change_per_pixel.size == np.sum(glacier_mask): 
             thickness_change_grid[glacier_mask] = thickness_change_per_pixel

        thickness += thickness_change_grid
        thickness[thickness < 0] = 0
        dem = bedrock + thickness

    scenario_output_dir = FORECAST_DIR / scenario
    (scenario_output_dir / 'rasters').mkdir(parents=True, exist_ok=True)
    (scenario_output_dir / 'csv').mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(scenario_output_dir / 'csv' / f'{glims_id}_forecast.csv', index=False)

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(scenario_output_dir / 'rasters' / f'{glims_id}_thickness_{END_YEAR}.tif', 'w', **profile) as dst:
        dst.write(thickness.astype(rasterio.float32), 1)
    with rasterio.open(scenario_output_dir / 'rasters' / f'{glims_id}_thickness_{START_YEAR}.tif', 'w', **profile) as dst:
         dst.write(initial_thickness.astype(rasterio.float32), 1)
         
    print(f"   - Forecast for {scenario} complete. Results saved.")


if __name__ == '__main__':
    print("\n-----------------------------------------------")
    print("       GLACIER FUTURE FORECAST SIMULATION")
    print("-----------------------------------------------\n")

    scaler, ann_models, static_df, feature_names = load_models_and_data()
    
    for scenario in SCENARIOS_TO_RUN:
        print(f"\n===== RUNNING FORECAST FOR SCENARIO: {scenario.upper()} =====")
        climate_scenario_dir = FUTURE_CLIMATE_DIR / scenario

        for glims_id, static_info in static_df.iterrows():
            print(f"\nProcessing Glacier: {glims_id} ({static_info['name']})...")
            
            if pd.isna(static_info['ID']) or static_info['ID'] == 0:
                 print(f"   - Invalid or missing numerical ID ('{static_info['ID']}'). Skipping.")
                 continue

            climate_file = climate_scenario_dir / f"{glims_id}.csv"
            if not climate_file.exists():
                print(f"   - Climate data for {scenario} not found. Skipping.")
                continue
            
            climate_data = pd.read_csv(climate_file)

            try:
                run_glacier_forecast(glims_id, static_info, scaler, ann_models, climate_data, feature_names, scenario)
            except Exception as e:
                print(f"   - FATAL ERROR processing {glims_id} for {scenario}: {e}")

    print("\n\nFull glacier forecast simulation complete.")
    print(f"Results saved in: {FORECAST_DIR}")


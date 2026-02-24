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
PROCESSED_DIR = DATA_DIR / 'processed'
RAW_DIR = DATA_DIR / 'raw'
OUTPUT_DIR = WORKSPACE / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'

SCALER_PATH = MODEL_DIR / 'scaler.pkl'
ANN_ENSEMBLE_DIR = MODEL_DIR / 'ann_ensemble'
FEATURE_NAMES_PATH = MODEL_DIR / 'feature_names.pkl' 

GLACIER_DATA_DIR = RAW_DIR / 'glacier_data'
RASTER_DIR = GLACIER_DATA_DIR / 'glacier_rasters'
DEM_DIR = RASTER_DIR / 'glacier_thickness' / 'dem_tif'
THICKNESS_DIR = RASTER_DIR / 'glacier_thickness' / 'thickness_tif'
METEO_DIR = PROCESSED_DIR / 'daily_meteo'

EVOLUTION_DIR = OUTPUT_DIR / 'evolution_results'
EVOLUTION_RASTER_DIR = EVOLUTION_DIR / 'rasters'
EVOLUTION_CSV_DIR = EVOLUTION_DIR / 'csv'
EVOLUTION_RASTER_DIR.mkdir(parents=True, exist_ok=True)
EVOLUTION_CSV_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1985
END_YEAR = 2015
ICE_DENSITY = 850


def r2_keras(y_true, y_pred): return 0
def root_mean_squared_error(y_true, y_pred): return 0

def calculate_monthly_climate_features(daily_df, year):
    """Calculates monthly mean temperature and total snowfall for a hydrological year."""
    start_date = pd.to_datetime(f"{year-1}-10-01")
    end_date = pd.to_datetime(f"{year}-09-30")
    year_df = daily_df[(daily_df['date'] >= start_date) & (daily_df['date'] <= end_date)].copy()
    
    if year_df.empty:
        months = pd.date_range(start=start_date, end=end_date, freq='MS')
        nan_features = {}
        for month in months:
            nan_features[f"temp_{month.strftime('%b').lower()}"] = np.nan
            nan_features[f"snow_{month.strftime('%b').lower()}"] = np.nan
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
    """Loads all necessary models, static data, and climate data."""
    print("Loading models and data...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    
    ann_models = []
    for model_file in sorted(os.listdir(ANN_ENSEMBLE_DIR)):
        if model_file.endswith('.keras'):
            model = load_model(ANN_ENSEMBLE_DIR / model_file, custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error})
            ann_models.append(model)

    rabatel_df = pd.read_csv(GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv', delimiter=';', encoding='latin1')
    
    column_names_2003 = [
        'Area_2003', 'Perimeter_2003', 'Glacier_2003', 'Annee_2003', 'Massif_2003', 
        'MEAN_Pixel_2003', 'MIN_Pixel_2003', 'MAX_Pixel_2003', 'MEDIAN_Pix_2003', 
        'Length_2003', 'Aspect_2003', 'x_coord_2003', 'y_coord_2003',
        'GLIMS_ID', 'Massif_SAFRAN_2003', 'Aspect_num_2003', 'ID'
    ]
    glims_2003_df = pd.read_csv(
        GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_2003.csv', 
        delimiter=';', 
        encoding='latin1',
        header=None,
        names=column_names_2003
    )

    id_map = glims_2003_df[['GLIMS_ID', 'ID']]
    static_df = pd.merge(rabatel_df, id_map, on='GLIMS_ID', how='left')
    static_df = static_df.rename(columns={'Glacier': 'name', 'Area': 'area', 'slope20': 'slope', 'x_coord': 'lon', 'y_coord': 'lat', 'Aspect_num': 'aspect'}).set_index('GLIMS_ID')

    meteo_data = {}
    print("Pre-loading all climate data...")
    for f in tqdm(os.listdir(METEO_DIR)):
        if f.endswith('.csv'):
            glims_id = f.replace('.csv', '')
            df = pd.read_csv(METEO_DIR / f); df['date'] = pd.to_datetime(df['date'])
            meteo_data[glims_id] = df

    print(f"Loaded scaler, {len(ann_models)} ANN models, {len(feature_names)} feature names, static data, and climate data for {len(meteo_data)} glaciers.")
    return scaler, ann_models, static_df, meteo_data, feature_names

def run_glacier_simulation(glims_id, static_info, scaler, ann_models, climate_data, feature_names):
    """Runs the full evolution simulation for a single glacier."""
    
    if pd.isna(static_info['ID']):
        print(f"   - Numerical ID is missing (NaN) for {glims_id}. Skipping.")
        return
        
    try:
        glacier_id_num = int(static_info['ID'])
        dem_filename = f"dem_{glacier_id_num:05d}.asc.tif"
        thickness_filename = f"RGI60-11.{glacier_id_num:05d}_thickness.tif"
        dem_path = DEM_DIR / dem_filename
        thickness_path = THICKNESS_DIR / thickness_filename
        if not dem_path.exists() or not thickness_path.exists(): raise FileNotFoundError

        with rasterio.open(dem_path) as src:
            dem = src.read(1); profile = src.profile
        with rasterio.open(thickness_path) as src:
            thickness = src.read(1)
            
    except (FileNotFoundError, rasterio.errors.RasterioIOError, ValueError):
        print(f"   - Initial raster files not found for ID '{glacier_id_num}'. Skipping.")
        return

    initial_thickness = thickness.copy()
    thickness[thickness < 0] = 0
    dem[dem < 0] = np.nan
    pixel_area = profile['transform'][0] * abs(profile['transform'][4])

    results = []
    for year in range(START_YEAR, END_YEAR + 1):
        glacier_mask = thickness > 0
        if not np.any(glacier_mask):
            print(f"   - Glacier melted in year {year-1}.")
            break

        current_area = np.sum(glacier_mask) * pixel_area / 1e6
        current_volume = np.sum(thickness[glacier_mask]) * pixel_area / 1e9
        bedrock = dem - thickness
        surface_elevation = np.where(glacier_mask, dem, np.nan)
        results.append({'year': year - 1, 'area_km2': current_area, 'volume_km3': current_volume})

        monthly_features = calculate_monthly_climate_features(climate_data, year)
        
        static_features = static_info.to_dict()
        static_features['area'] = current_area
        
        feature_vector_dict = {**monthly_features, **static_features}
        feature_vector_df = pd.DataFrame([feature_vector_dict])
        
        feature_vector_df = feature_vector_df.ffill().bfill().fillna(0)
        feature_vector_df = feature_vector_df[feature_names]
        
        X_scaled = scaler.transform(feature_vector_df.values)
        ann_preds = [model.predict(X_scaled, verbose=0).flatten()[0] for model in ann_models]
        smb_predicted = np.mean(ann_preds)

        total_mass_change = smb_predicted * current_area * 1e6
        ice_volume_change = total_mass_change * (1000 / ICE_DENSITY)

        min_elev, max_elev = np.nanmin(surface_elevation), np.nanmax(surface_elevation)
        if min_elev is None or max_elev is None or min_elev == max_elev:
             elev_normalized = np.full_like(surface_elevation, 0.5)
        else:
             elev_normalized = (surface_elevation - min_elev) / (max_elev - min_elev)
        
        delta_h_distribution = 1 - elev_normalized
        if np.nansum(delta_h_distribution[glacier_mask]) == 0:
            delta_h_normalized = np.ones_like(thickness[glacier_mask]) / np.sum(glacier_mask)
        else:
            delta_h_normalized = delta_h_distribution[glacier_mask] / np.nansum(delta_h_distribution[glacier_mask])

        thickness_change_total = ice_volume_change / pixel_area
        
        thickness_change_per_pixel = thickness_change_total * delta_h_normalized
        
        thickness_change_grid = np.zeros_like(thickness)
        thickness_change_grid[glacier_mask] = thickness_change_per_pixel
        
        thickness += thickness_change_grid
        thickness[thickness < 0] = 0
        dem = bedrock + thickness

    results_df = pd.DataFrame(results)
    results_df.to_csv(EVOLUTION_CSV_DIR / f'{glims_id}_evolution.csv', index=False)

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(EVOLUTION_RASTER_DIR / f'{glims_id}_thickness_{END_YEAR}.tif', 'w', **profile) as dst:
        dst.write(thickness.astype(rasterio.float32), 1)
    with rasterio.open(EVOLUTION_RASTER_DIR / f'{glims_id}_thickness_{START_YEAR}.tif', 'w', **profile) as dst:
        dst.write(initial_thickness.astype(rasterio.float32), 1)

    print(f"   - Simulation complete. Results saved.")


if __name__ == '__main__':
    print("\n-----------------------------------------------")
    print("       GLACIER EVOLUTION SIMULATION")
    print("-----------------------------------------------\n")

    scaler, ann_models, static_df, meteo_data, feature_names = load_models_and_data()
    
    for glims_id, static_info in static_df.iterrows():
        print(f"\nProcessing Glacier: {glims_id} ({static_info['name']})...")

        if static_info['ID'] == 0:
            print(f"   - Invalid numerical ID ('0.0'). Skipping.")
            continue
        
        if glims_id not in meteo_data:
            print(f"   - Climate data not found for {glims_id}. Skipping.")
            continue

        try:
            run_glacier_simulation(glims_id, static_info, scaler, ann_models, meteo_data[glims_id], feature_names)
        except Exception as e:
            print(f"   - ERROR processing {glims_id}: {e}")

    print("\n\nFull glacier evolution simulation complete.")
    print(f"Results saved in: {EVOLUTION_DIR}")


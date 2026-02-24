import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import rasterio
from keras.models import load_model
import matplotlib.pyplot as plt



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


TIMESERIES_PLOT_DIR = OUTPUT_DIR / 'interactive_plots' 
TIMESERIES_CSV_DIR = OUTPUT_DIR / 'interactive_csvs'   
TIMESERIES_PLOT_DIR.mkdir(parents=True, exist_ok=True)
TIMESERIES_CSV_DIR.mkdir(parents=True, exist_ok=True)

ICE_DENSITY = 850
START_YEAR_SIM = 2015
MAX_END_YEAR = 2100 
SCENARIOS_TO_RUN = ['ssp245', 'ssp585'] 



def r2_keras(y_true, y_pred): return 0
def root_mean_squared_error(y_true, y_pred): return 0

def calculate_monthly_climate_features(daily_df, year):
    start_date = f"{year-1}-10-01"
    end_date = f"{year}-09-30"
    if not pd.api.types.is_datetime64_any_dtype(daily_df['date']):
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

def load_all_dependencies():
    print("Loading models and static data...")
    try:
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        with open(FEATURE_NAMES_PATH, 'rb') as f: feature_names = pickle.load(f)

        ann_models = []
        if not os.path.exists(ANN_ENSEMBLE_DIR) or not os.listdir(ANN_ENSEMBLE_DIR):
             raise FileNotFoundError(f"ANN ensemble directory is empty or missing: {ANN_ENSEMBLE_DIR}")
        for model_file in sorted(os.listdir(ANN_ENSEMBLE_DIR)):
            if model_file.endswith('.keras'):
                model = load_model(ANN_ENSEMBLE_DIR / model_file, custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error})
                ann_models.append(model)
        if not ann_models:
             raise ValueError(f"No '.keras' model files found in {ANN_ENSEMBLE_DIR}")


        glims_2015 = pd.read_csv(GLIMS_2015_FILE, delimiter=';', encoding='latin1')
        glims_2003 = pd.read_csv(GLIMS_2003_FILE, delimiter=';', encoding='latin1', header=None, names=['Area','Perimeter','Glacier','Annee','Massif','MEAN_Pixel','MIN_Pixel','MAX_Pixel','MEDIAN_Pix','Length','Aspect','x_coord','y_coord','GLIMS_ID','ID','Aspect_num','slope20'])
        glims_2003 = glims_2003.drop_duplicates(subset=['GLIMS_ID'], keep='first')
        id_map = glims_2003[['GLIMS_ID', 'ID']]
        static_df = pd.merge(glims_2015, id_map, on='GLIMS_ID', how='left').rename(columns={'Glacier': 'name', 'Area': 'area', 'slope20': 'slope', 'x_coord': 'lon', 'y_coord': 'lat', 'Aspect_num': 'aspect'}).set_index('GLIMS_ID')

    except FileNotFoundError as e:
        print(f"Error loading dependencies: {e}")
        print("Please ensure training scripts have been run and models/data exist.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        exit()

    print("Dependencies loaded successfully.")
    return scaler, ann_models, static_df, feature_names

def run_and_plot_simulation(glims_id, scenario, scaler, ann_models, static_df, feature_names):
    """
    Runs a forecast for a single glacier until it melts or reaches MAX_END_YEAR,
    saves the time series data, and plots the results.
    (This combines the run_interactive_simulation logic and plotting)
    """

    try:
        static_info = static_df.loc[glims_id]
        climate_file = FUTURE_CLIMATE_DIR / scenario / f"{glims_id}.csv"
        climate_data = pd.read_csv(climate_file)

        initial_thickness_path = INITIAL_STATE_DIR / f'{glims_id}_thickness_2015.tif'
        initial_dem_path = INITIAL_STATE_DIR / f'{glims_id}_dem_2015.tif'

        if not initial_thickness_path.exists() or not initial_dem_path.exists():
            print(f"   - Initial state (2015 raster) not found for {glims_id}. Skipping.")
            return 

        with rasterio.open(initial_dem_path) as src: dem = src.read(1); profile = src.profile
        with rasterio.open(initial_thickness_path) as src: thickness = src.read(1)
    except FileNotFoundError:
         print(f"   - Climate data file not found for {glims_id} ({scenario}). Skipping.")
         return
    except Exception as e:
        print(f"   - Error loading initial data for {glims_id}: {e}. Skipping.")
        return

    print(f"   Running simulation for {glims_id} ({static_info['name']}) - {scenario.upper()}...")

    thickness[thickness < 0] = 0
    pixel_area = profile['transform'][0] * abs(profile['transform'][4])

    yearly_results = []
    initial_mask = thickness > 0
    initial_area = np.sum(initial_mask) * pixel_area / 1e6
    initial_volume = np.sum(thickness[initial_mask]) * pixel_area / 1e9
    yearly_results.append({'year': START_YEAR_SIM, 'smb_mwe': np.nan, 'volume_km3': initial_volume, 'area_km2': initial_area})

    final_sim_year = START_YEAR_SIM

    for year in range(START_YEAR_SIM + 1, MAX_END_YEAR + 1):
        final_sim_year = year
        glacier_mask = thickness > 0
        if not np.any(glacier_mask):
            print(f"     -> Glacier melted in year {year-1}.")
            final_sim_year = year - 1
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

        yearly_results.append({
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

    if not yearly_results or len(yearly_results) <= 1:
        print(f"   - No simulation results generated beyond initial year for {glims_id} ({scenario}). Skipping plot.")
        return

    results_df = pd.DataFrame(yearly_results)

    csv_path = TIMESERIES_CSV_DIR / f'{glims_id}_{scenario}_timeseries_auto_end.csv'
    try:
        results_df.to_csv(csv_path, index=False, float_format='%.12f')
    except Exception as e:
        print(f"   - Error saving CSV for {glims_id} ({scenario}): {e}")


    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(f"Forecast for Glacier {static_info['name']} ({glims_id}) - {scenario.upper()} (Simulated up to {final_sim_year})", fontsize=16)

        plot_df_smb = results_df[results_df['year'] > START_YEAR_SIM]
        axes[0].plot(plot_df_smb['year'], plot_df_smb['smb_mwe'], 'o-', color='royalblue', label='Annual SMB')
        axes[0].axhline(0, color='grey', linestyle='--', lw=1)
        axes[0].set_ylabel("SMB (m w.e.)")
        axes[0].set_title("Predicted Annual Surface Mass Balance")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()

        axes[1].plot(results_df['year'], results_df['volume_km3'], 'o-', color='darkcyan', label='Total Volume')
        axes[1].set_ylabel("Volume (km³)")
        axes[1].set_title("Predicted Glacier Volume")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend()

        axes[2].plot(results_df['year'], results_df['area_km2'], 'o-', color='seagreen', label='Total Area')
        axes[2].set_ylabel("Area (km²)")
        axes[2].set_title("Predicted Glacier Area")
        axes[2].set_xlabel("Year")
        axes[2].grid(True, linestyle='--', alpha=0.6)
        axes[2].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = TIMESERIES_PLOT_DIR / f'{glims_id}_{scenario}_forecast_auto_end.png'
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
    except Exception as e:
         print(f"   - Error generating plot for {glims_id} ({scenario}): {e}")



if __name__ == '__main__':

    scaler, ann_models, static_df, feature_names = load_all_dependencies()

    print("\n--- Starting Batch Forecast Generation ---")

    for scenario in SCENARIOS_TO_RUN:
        print(f"\n===== PROCESSING SCENARIO: {scenario.upper()} =====")
        climate_scenario_dir = FUTURE_CLIMATE_DIR / scenario


        for glims_id, static_info in tqdm(static_df.iterrows(), total=len(static_df), desc=f"Glaciers ({scenario})"):

            if pd.isna(static_info['ID']) or static_info['ID'] == 0:
                continue

            climate_file = climate_scenario_dir / f"{glims_id}.csv"
            if not climate_file.exists():
                continue

            climate_data = pd.read_csv(climate_file)

            try:
                run_and_plot_simulation(glims_id, scenario, scaler, ann_models, static_df, feature_names)
            except Exception as e:
                print(f"\n   - UNEXPECTED ERROR during simulation/plotting for {glims_id} ({scenario}): {e}")

    print("\n\n--- Batch Forecast Generation Complete ---")
    print(f"Time series CSVs saved in: {TIMESERIES_CSV_DIR}")
    print(f"Time series plots saved in: {TIMESERIES_PLOT_DIR}")

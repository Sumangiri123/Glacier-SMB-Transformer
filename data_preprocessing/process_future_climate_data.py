import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import netCDF4 as nc

# ##############################################################################
# ## 1. FILE PATHS AND SETTINGS
# ##############################################################################

WORKSPACE = Path().resolve()
DATA_DIR = WORKSPACE / 'data'
RAW_FORECAST_DIR = DATA_DIR / 'forecast_raw'
PROCESSED_FORECAST_DIR = DATA_DIR / 'processed' / 'future_climate'
RAW_GLACIER_DATA_DIR = DATA_DIR / 'raw' / 'glacier_data'

# Path to the file with glacier coordinates
GLACIER_INFO_FILE = RAW_GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv'

# Temperature threshold for converting precipitation to snow (in Kelvin)
SNOWFALL_TEMP_THRESHOLD_K = 273.15

# Define the scenarios and their corresponding file paths
SCENARIOS = {
    'ssp245': {
        'temp_file': RAW_FORECAST_DIR / 'NearSurfaceTemperature_SS2-4.5' / 'tas_day_CNRM-ESM2-1_ssp245_r1i1p1f2_gr_20150101-20991231.nc',
        'precip_file': RAW_FORECAST_DIR / 'Precipitation_SS2-4.5' / 'pr_day_CNRM-ESM2-1_ssp245_r1i1p1f2_gr_20150101-20991231.nc'
    },
    'ssp585': {
        'temp_file': RAW_FORECAST_DIR / 'NearSurfaceTemperature_SSP5-8.5' / 'tas_day_CNRM-ESM2-1_ssp585_r1i1p1f2_gr_20150101-20991231.nc',
        'precip_file': RAW_FORECAST_DIR / 'Precipitation_SSP5-8.5' / 'pr_day_CNRM-ESM2-1_ssp585_r1i1p1f2_gr_20150101-20991231.nc'
    }
}

# ##############################################################################
# ## 2. MAIN PROCESSING SCRIPT
# ##############################################################################

def find_nearest_grid_point(lat_vector, lon_vector, target_lat, target_lon):
    """
    Finds the index of the nearest grid point to a target coordinate.
    Handles 1D latitude and longitude vectors by creating a meshgrid.
    """
    # Create a 2D grid from the 1D lat/lon vectors
    lon_grid, lat_grid = np.meshgrid(lon_vector, lat_vector)
    
    # Calculate the squared distance on the 2D grid
    dist_sq = (lat_grid - target_lat)**2 + (lon_grid - target_lon)**2
    
    # Find the index of the minimum distance
    min_dist_idx = np.unravel_index(np.argmin(dist_sq, axis=None), dist_sq.shape)
    
    return min_dist_idx

def process_scenario(scenario_name, config, glacier_df):
    """Processes the raw NetCDF files for a single climate scenario."""
    print(f"\n--- Processing scenario: {scenario_name} ---")

    output_dir = PROCESSED_FORECAST_DIR / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        temp_ds = nc.Dataset(config['temp_file'])
        precip_ds = nc.Dataset(config['precip_file'])
    except FileNotFoundError as e:
        print(f"Error: Could not find NetCDF file. {e}")
        return

    lats = temp_ds.variables['lat'][:]
    lons = temp_ds.variables['lon'][:]
    time_var = temp_ds.variables['time']
    
    # FIX: Convert cftime objects to standard datetime strings that pandas can understand
    dates_cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    dates = [date.strftime('%Y-%m-%d') for date in dates_cftime]
    
    for _, row in tqdm(glacier_df.iterrows(), total=len(glacier_df), desc=f"Processing glaciers for {scenario_name}"):
        glims_id = row['GLIMS_ID']
        glacier_lat = row['y_coord']
        glacier_lon = row['x_coord']

        lat_idx, lon_idx = find_nearest_grid_point(lats, lons, glacier_lat, glacier_lon)

        temp_series_k = temp_ds.variables['tas'][:, lat_idx, lon_idx]
        precip_series_kg_m2_s = precip_ds.variables['pr'][:, lat_idx, lon_idx]

        temp_series_c = temp_series_k - 273.15
        precip_series_mm_day = precip_series_kg_m2_s * 86400

        snowfall_series_mm_day = np.where(temp_series_k <= SNOWFALL_TEMP_THRESHOLD_K, precip_series_mm_day, 0)
        
        glacier_climate_df = pd.DataFrame({
            'date': dates,
            'temperature_c': temp_series_c,
            'snowfall_mm': snowfall_series_mm_day
        })
        
        # The previous failing line is no longer needed as dates are pre-formatted
        # glacier_climate_df['date'] = pd.to_datetime(glacier_climate_df['date']).dt.date
        
        output_path = output_dir / f"{glims_id}.csv"
        glacier_climate_df.to_csv(output_path, index=False)
        
    temp_ds.close()
    precip_ds.close()
    print(f"Successfully processed and saved data for {scenario_name}.")


if __name__ == '__main__':
    print("\n-----------------------------------------------")
    print("  PROCESSING FUTURE CLIMATE PROJECTION DATA")
    print("-----------------------------------------------\n")

    try:
        glacier_info_df = pd.read_csv(GLACIER_INFO_FILE, delimiter=';', encoding='latin1')
    except FileNotFoundError:
        print(f"Error: Could not find the glacier info file at '{GLACIER_INFO_FILE}'.")
        exit()

    for name, config in SCENARIOS.items():
        process_scenario(name, config, glacier_info_df)

    print("\n\nAll scenarios processed.")
    print(f"Daily climate CSV files saved in: {PROCESSED_FORECAST_DIR}")


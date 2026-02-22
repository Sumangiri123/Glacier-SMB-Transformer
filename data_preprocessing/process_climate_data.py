import pandas as pd
import xarray as xr
from pathlib import Path
import traceback
from tqdm import tqdm

def preprocess_for_time(ds):
    """Rename 'valid_time' to 'time' if it exists to standardize the time coordinate."""
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    return ds

def process_climate_data():
    """
    Processes raw ERA5-Land & ERA5 NetCDF files into daily time series for each glacier,
    processing one year at a time to manage memory usage.
    """
    print("Starting climate data processing...")

    # --- 1. SETUP PATHS AND CONSTANTS ---
    base_dir = Path.cwd()
    raw_data_dir = base_dir / 'data' / 'raw'
    processed_data_dir = base_dir / 'data' / 'processed'
    glacier_info_path = raw_data_dir / 'glacier_data' / 'GLIMS' / 'GLIMS_2003.csv'
    output_dir = processed_data_dir / 'daily_meteo'
    
    output_dir.mkdir(parents=True, exist_ok=True)

    G = 9.80665
    LAPSE_RATE = -0.006
    YEARS = range(1984, 2016)

    # --- 2. LOAD GLACIER AND STATIC DATA ---
    print("Loading glacier inventory and geopotential data...")

    if not glacier_info_path.exists():
        print(f"ERROR: Glacier inventory file not found at {glacier_info_path}")
        return
    glacier_df = pd.read_csv(glacier_info_path, sep=';')

    climate_data_source_dir = raw_data_dir / 'ERA5_Land'
    
    if not climate_data_source_dir.is_dir():
        print(f"ERROR: The source directory for processed climate data does not exist: {climate_data_source_dir}")
        return

    geopotential_file = raw_data_dir / 'ERA5_Land' / 'era5-alps-geopotential.nc'
    if not geopotential_file.exists():
        print(f"ERROR: Geopotential file not found at {geopotential_file}")
        return
        
    geo_ds = xr.open_dataset(geopotential_file, engine='netcdf4')
    geo_ds['z_meters'] = geo_ds['z'] / G
    
    # --- Create xarray DataArrays for all glacier coordinates and attributes ---
    glacier_ids = xr.DataArray(glacier_df['GLIMS_ID'].values, dims="glacier")
    glacier_lons = xr.DataArray(glacier_df['x_coord'].values, dims="glacier")
    glacier_lats = xr.DataArray(glacier_df['y_coord'].values, dims="glacier")
    glacier_alts = xr.DataArray(glacier_df['MEAN_Pixel'].values, dims="glacier")

    # --- 3. PROCESS DATA YEAR BY YEAR ---
    print(f"Processing data for {len(glacier_df)} glaciers from {YEARS.start} to {YEARS.stop - 1}...")

    glacier_results = {glacier_id: [] for glacier_id in glacier_ids.values}

    progress_bar = tqdm(YEARS, desc="Processing years")
    for year in progress_bar:
        progress_bar.set_description(f"Processing year {year}")
        
        climate_files_year = sorted(list(climate_data_source_dir.glob(f'*{year}*.nc')))
        
        if not climate_files_year:
            print(f"\nWarning: No climate files found for year {year}. Skipping.")
            continue
            
        try:
            climate_ds_year = xr.open_mfdataset(
                climate_files_year, engine='netcdf4', preprocess=preprocess_for_time,
                combine='nested', concat_dim='time'
            )
            
            # --- Vectorized Selection for ALL glaciers at once ---
            glacier_climate_data = climate_ds_year.sel(
                longitude=glacier_lons, latitude=glacier_lats, method='nearest'
            )
            grid_cell_elevation = geo_ds['z_meters'].sel(
                longitude=glacier_lons, latitude=glacier_lats, method='nearest'
            )

            # --- Vectorized Resampling and Calculations ---
            daily_temp = glacier_climate_data['t2m'].resample(time='1D').mean()
            daily_precip = glacier_climate_data['tp'].resample(time='1D').sum()
            daily_snow = glacier_climate_data['sf'].resample(time='1D').sum()

            daily_temp_c = daily_temp - 273.15
            daily_precip_mm = daily_precip * 1000
            daily_snow_mm = daily_snow * 1000

            elevation_diff = glacier_alts - grid_cell_elevation
            temp_correction = elevation_diff * LAPSE_RATE
            daily_temp_corrected = daily_temp_c + temp_correction
            
            # Load the computed data into memory for this year
            daily_temp_corrected.load()
            daily_precip_mm.load()
            daily_snow_mm.load()

            # --- Append results for each glacier ---
            for i, glacier_id in enumerate(glacier_ids.values):
                temp_series = daily_temp_corrected.isel(glacier=i)
                precip_series = daily_precip_mm.isel(glacier=i)
                snow_series = daily_snow_mm.isel(glacier=i)
                
                output_df_year = pd.DataFrame({
                    'temperature_c': temp_series.values.flatten(),
                    'total_precipitation_mm': precip_series.values.flatten(),
                    'snowfall_mm': snow_series.values.flatten()
                }, index=temp_series.time.values)
                glacier_results[glacier_id].append(output_df_year)

        except Exception as e:
            print(f"\nERROR: Failed to process data for year {year}. Details: {e}")
            # traceback.print_exc()
        finally:
            if 'climate_ds_year' in locals():
                climate_ds_year.close()

    # --- 4. COMBINE AND SAVE FINAL RESULTS ---
    print("\nCombining yearly data and saving final CSV files...")
    save_progress_bar = tqdm(glacier_results.items(), total=len(glacier_results), desc="Saving files")
    for glacier_id, yearly_dfs in save_progress_bar:
        if not yearly_dfs:
            print(f"\nWarning: No data was processed for glacier {glacier_id}. No file will be saved.")
            continue
        
        final_df = pd.concat(yearly_dfs)
        final_df.index.name = 'date'
        
        output_filename = f"{glacier_id}.csv"
        final_df.to_csv(output_dir / output_filename)

    print(f"\n✅ Processing complete. Daily climate files saved in:\n{output_dir}")

if __name__ == '__main__':
    process_climate_data()


import cdsapi
import os
from pathlib import Path
import xarray as xr
import zipfile # Needed to handle zip files
import time

# --- Configuration ---
# Destination for raw downloaded files
RAW_DATA_DIR = Path.cwd() / 'data' / 'raw' / 'ERA5_Land'
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

# List of years and months to download for the main dataset
YEARS = [str(year) for year in range(1984, 2015)]
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

# Bounding box for the French Alps
AREA = [46.5, 5.5, 44.0, 8.0] # [North, West, South, East]

# --- Main Script ---
c = cdsapi.Client()

# --- 1. Download Geopotential Data ---
# This dataset typically downloads as a raw .nc file, not a zip.
geopotential_filename = 'era5-alps-geopotential.nc'
geopotential_filepath = RAW_DATA_DIR / geopotential_filename

print(f"--- Checking for {geopotential_filename} ---")
if not geopotential_filepath.exists():
    print("Geopotential file not found. Downloading...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': 'geopotential',
                'year': '1984',
                'month': '01',
                'day': '01',
                'time': '00:00',
                'area': AREA,
                'format': 'netcdf',
            },
            geopotential_filepath
        )
        # --- VALIDATION STEP for Geopotential File ---
        try:
            xr.open_dataset(geopotential_filepath, engine='netcdf4').close()
            print("--- Successfully downloaded and validated geopotential data. ---")
        except Exception as e:
            print(f"!!! VALIDATION FAILED for geopotential file. It may be corrupt. Error: {e} !!!")
            os.remove(geopotential_filepath)
            exit()

    except Exception as e:
        print(f"!!! FAILED to download geopotential data. Error: {e} !!!")
        exit()
else:
    print("Geopotential file already exists. Skipping download.")


# --- 2. Download and Unzip Monthly Climate Data ---
# This dataset often arrives as a zip file, so we handle that automatically.
print("\n--- Starting download of monthly climate data ---")
for year in YEARS:
    for month in MONTHS:
        final_nc_filename = f'era5-land-alps-{year}-{month}.nc'
        final_nc_filepath = RAW_DATA_DIR / final_nc_filename
        
        # Temporary name for the downloaded zip file
        temp_zip_filepath = RAW_DATA_DIR / f'temp_{year}-{month}.zip'

        print(f"--- Processing data for {year}-{month} ---")
        
        # Check if the final extracted file already exists
        if final_nc_filepath.exists():
            print(f"File '{final_nc_filename}' already exists. Skipping.")
            continue

        try:
            print(f"Requesting data for {year}-{month}...")
            # We download to a temporary zip file path
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': ['2m_temperature', 'snowfall', 'total_precipitation'],
                    'year': year, 'month': month,
                    'day': [ '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', 
                             '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', 
                             '25', '26', '27', '28', '29', '30', '31' ],
                    'time': [ '00:00', '06:00', '12:00', '18:00' ],
                    'area': AREA,
                    'format': 'netcdf', # The server packages this into a zip
                },
                temp_zip_filepath
            )
            print("Download complete. Now extracting...")

            # --- UNZIP STEP ---
            with zipfile.ZipFile(temp_zip_filepath, 'r') as zip_ref:
                # The actual .nc file is usually inside the zip with a generic name
                # We find it and rename it upon extraction
                file_to_extract = zip_ref.namelist()[0]
                zip_ref.extract(file_to_extract, path=RAW_DATA_DIR)
                extracted_file_path = RAW_DATA_DIR / file_to_extract
                extracted_file_path.rename(final_nc_filepath)

            print(f"Extracted to '{final_nc_filename}'.")

            # --- VALIDATION STEP ---
            try:
                xr.open_dataset(final_nc_filepath, engine='netcdf4').close()
                print(f"--- Successfully validated data for {year}-{month} ---")
            except Exception as e:
                print(f"!!! VALIDATION FAILED for {year}-{month}. File is corrupt. Error: {e} !!!")
                if final_nc_filepath.exists():
                    os.remove(final_nc_filepath)
            
            # --- CLEANUP ---
            os.remove(temp_zip_filepath) # Delete the temporary zip file

        except Exception as e:
            print(f"!!! An error occurred during download/unzip for {year}-{month}. Error: {e} !!!")
            # Clean up any partial files
            if temp_zip_filepath.exists():
                os.remove(temp_zip_filepath)
            if final_nc_filepath.exists():
                os.remove(final_nc_filepath)
            print("Continuing to the next month...")
            continue
        
        # A small delay to be courteous to the API server
        time.sleep(1)

print("\n--- All downloads and extractions complete! ---")
print("You can now run 'process_climate_data.py' to generate the daily CSVs.")


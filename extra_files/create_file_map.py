import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

# --- DEFINE FILE PATHS ---
WORKSPACE = Path().resolve()
DATA_DIR = WORKSPACE / 'data'
RAW_DIR = DATA_DIR / 'raw'
GLACIER_DATA_DIR = RAW_DIR / 'glacier_data'
RASTER_DIR = GLACIER_DATA_DIR / 'glacier_rasters'

# CORRECTED FOLDER PATHS
DEM_DIR = RASTER_DIR / 'glacier_thickness' / 'dem_tif'
THICKNESS_DIR = RASTER_DIR / 'glacier_thickness' / 'thickness_tif'

# METADATA FILES
MASTER_LIST_FILE = GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv'
ID_MAP_FILE = GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_2003.csv'

# OUTPUT FILE
OUTPUT_MAP_FILE = WORKSPACE / 'file_map.csv'

def main():
    """Main function to generate the file map using direct ID matching."""
    print("--- Starting File Map Automation ---")

    # 1. Load the master list of glaciers we need to map
    master_df = pd.read_csv(MASTER_LIST_FILE, delimiter=';', encoding='latin1')
    master_df = master_df[['GLIMS_ID', 'Glacier']].rename(columns={'Glacier': 'name'})
    print(f"Loaded {len(master_df)} glaciers from the master list.")

    # 2. Load the mapping between GLIMS_ID and the numerical ID
    id_map_df = pd.read_csv(ID_MAP_FILE, delimiter=';', encoding='latin1')
    id_map_df = id_map_df[['GLIMS_ID', 'ID']]
    print("Loaded GLIMS_ID-to-ID mapping.")

    # 3. Get lists of available raster files
    dem_files = [f for f in os.listdir(DEM_DIR) if f.endswith('.tif')]
    thickness_files = [f for f in os.listdir(THICKNESS_DIR) if f.endswith('.tif')]
    print(f"Found {len(dem_files)} DEM files and {len(thickness_files)} thickness files.")

    # 4. Create the final mapping by merging metadata
    final_map = pd.merge(master_df, id_map_df, on='GLIMS_ID', how='left')
    
    file_mappings = []
    print("Matching files for each glacier using numerical ID...")
    for _, row in tqdm(final_map.iterrows(), total=len(final_map)):
        glims_id = row['GLIMS_ID']
        glacier_num_id = row['ID']

        # Skip if the numerical ID is missing or invalid (0)
        if pd.isna(glacier_num_id) or glacier_num_id == 0:
            continue
        
        # Construct the exact filenames we expect to find
        glacier_num_id = int(glacier_num_id)
        expected_dem_file = f"dem_{glacier_num_id:05d}.asc.tif"
        expected_thickness_file = f"RGI60-11.{glacier_num_id:05d}_thickness.tif"

        # Check if BOTH files actually exist in their respective folders
        if expected_dem_file in dem_files and expected_thickness_file in thickness_files:
            file_mappings.append({
                'GLIMS_ID': glims_id,
                'dem_filename': expected_dem_file,
                'thickness_filename': expected_thickness_file
            })

    # 5. Save the final CSV file
    output_df = pd.DataFrame(file_mappings)
    output_df.to_csv(OUTPUT_MAP_FILE, index=False)

    print("\n--- Automation Complete ---")
    print(f"Successfully mapped {len(output_df)} glaciers.")
    print(f"File map saved to: {OUTPUT_MAP_FILE}")

if __name__ == '__main__':
    main()
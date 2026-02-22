import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from shapely.geometry import box

def get_date_from_filename(filename):
    """Extracts the datetime object from ENVISAT or Sentinel filenames."""
    match = re.search(r'(\d{8}T\d{6})', filename)
    if match:
        return pd.to_datetime(match.group(1), format='%Y%m%dT%H%M%S')
        
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return pd.to_datetime(match.group(1), format='%Y%m%d_%H%M%S')
        
    return None

def process_sar_batch(sar_files, glacier_shapefile_path, glacier_info_path, progress_desc):
    """Helper function to process a batch of SAR files with a specific set of glacier data."""
    
    print(f"\n--- Loading data for batch: {progress_desc} ---")
    try:
        inventory_df = pd.read_csv(glacier_info_path, sep=';')
        inventory_df['GLIMS_ID'] = inventory_df['GLIMS_ID'].str.strip()
        inventory_df = inventory_df.drop_duplicates(subset=['GLIMS_ID'], keep='first')
        inventory = inventory_df.set_index('GLIMS_ID')

        shapefiles = gpd.read_file(glacier_shapefile_path)
        
        original_shape_count = len(shapefiles)
        shapefiles_synced = shapefiles[shapefiles['GLIMS_ID'].isin(inventory.index)]
        print(f"Synchronized data: Kept {len(shapefiles_synced)} of {original_shape_count} glacier shapes that have matching inventory data.")
        if len(shapefiles_synced) == 0:
            print("ERROR: No glaciers remaining after synchronizing shapefile with CSV. Check for GLIMS_ID mismatches.")
            return []

    except Exception as e:
        print(f"ERROR: Could not load glacier data for batch '{progress_desc}'. Details: {e}")
        return []

    batch_records = []
    progress_bar = tqdm(sar_files, desc=progress_desc)
    for sar_path, metadata_name in progress_bar:
        try:
            with rasterio.open(sar_path) as src:
                if not src.crs:
                    continue
                
                image_crs = src.crs
                glaciers_reprojected = shapefiles_synced.to_crs(image_crs)
                image_bounds_poly = box(*src.bounds)
                image_gdf = gpd.GeoDataFrame(geometry=[image_bounds_poly], crs=image_crs)

                glaciers_in_scene = gpd.sjoin(glaciers_reprojected, image_gdf, how="inner", predicate='intersects')

                if glaciers_in_scene.empty:
                    continue # This is normal if the image doesn't cover any glaciers

                image_data = src.read(1)
                valid_image_pixels = image_data[image_data > 0]
                if valid_image_pixels.size == 0:
                    continue

                threshold = np.percentile(valid_image_pixels, 50) 
                date_obj = get_date_from_filename(metadata_name)
                if not date_obj:
                    continue
                
                print(f"\n[DIAGNOSTIC] Found {len(glaciers_in_scene)} glaciers in {metadata_name}")

                for idx, glacier_row in glaciers_in_scene.iterrows():
                    glacier_id = glacier_row['GLIMS_ID']
                    glacier_geom = glacier_row['geometry']

                    try:
                        out_image, _ = mask(src, [glacier_geom], crop=True, nodata=0)
                        glacier_pixels = out_image[0]

                        valid_pixels = glacier_pixels[glacier_pixels > 0]
                        if valid_pixels.size < 10:
                            # print(f"  -> Skipping {glacier_id}: Too few valid pixels ({valid_pixels.size}) after masking.")
                            continue

                        snow_pixels = valid_pixels > threshold
                        snow_coverage = np.sum(snow_pixels) / valid_pixels.size
                        
                        glacier_mean_alt = inventory.loc[glacier_id, 'MEAN_Pixel']
                        sla_estimate = glacier_mean_alt
                        if snow_coverage < 0.1:
                            sla_estimate = inventory.loc[glacier_id, 'MAX_Pixel']
                        elif snow_coverage > 0.9:
                            sla_estimate = inventory.loc[glacier_id, 'MIN_Pixel']

                        batch_records.append({
                            'date': date_obj, 'GLIMS_ID': glacier_id,
                            'SLA_m': sla_estimate, 'snow_coverage_percent': snow_coverage * 100
                        })
                    except (ValueError, IndexError, KeyError) as e:
                        print(f"  -> ERROR processing glacier {glacier_id}: {e}")
                        continue
        except Exception as e:
            print(f"\nCould not process file {metadata_name}. Error: {e}")

    return batch_records


def process_sar_data():
    """
    Processes orthorectified SAR GeoTIFFs to estimate SLA, using different
    glacier inventories for different time periods.
    """
    print("--- Starting SAR Image Processing ---")

    base_dir = Path.cwd()
    sar_ortho_dir = base_dir / 'data' / 'processed' / 'SAR_orthorectified'
    raw_glacier_dir = base_dir / 'data' / 'raw' / 'glacier_data'
    output_dir = base_dir / 'data' / 'processed' / 'SAR_features'
    output_dir.mkdir(exist_ok=True, parents=True)

    batch_1_info = {
        "desc": "ENVISAT (2002-2010)",
        "shapefile": raw_glacier_dir / 'glacier_shapefiles' / '2003' / 'GLIMS_glaciers_2003_05_CRS.shp',
        "inventory": raw_glacier_dir / 'GLIMS' / 'GLIMS_2003.csv'
    }
    batch_2_info = {
        "desc": "Sentinel-1 (2014)",
        "shapefile": raw_glacier_dir / 'glacier_shapefiles' / '2015' / 'glaciers_2015_05.shp',
        "inventory": raw_glacier_dir / 'GLIMS' / 'GLIMS_2015.csv'
    }

    all_ortho_files = list(sar_ortho_dir.glob('**/*_ortho.tif'))
    if not all_ortho_files:
        print(f"ERROR: No orthorectified files found in {sar_ortho_dir}. Run georeference_SAR_data.py first.")
        return

    envisat_files_to_process = []
    sentinel_files_to_process = []

    for f in all_ortho_files:
        if 'ASA_IMP' in f.name: 
            envisat_files_to_process.append((f, f.name))
        elif 's1a_iw' in f.name.lower(): 
            sentinel_files_to_process.append((f, f.name))
    
    print(f"Found {len(envisat_files_to_process)} ENVISAT and {len(sentinel_files_to_process)} Sentinel-1 files.")

    all_sla_records = []

    if envisat_files_to_process:
        envisat_records = process_sar_batch(envisat_files_to_process, batch_1_info["shapefile"], batch_1_info["inventory"], batch_1_info["desc"])
        all_sla_records.extend(envisat_records)

    if sentinel_files_to_process:
        sentinel_records = process_sar_batch(sentinel_files_to_process, batch_2_info["shapefile"], batch_2_info["inventory"], batch_2_info["desc"])
        all_sla_records.extend(sentinel_records)

    if not all_sla_records:
        print("\nCompleted, but no valid SLA records could be generated across all batches.")
        return

    sla_df = pd.DataFrame(all_sla_records)
    sla_df = sla_df.sort_values(by=['date', 'GLIMS_ID']).reset_index(drop=True)
    
    output_path = output_dir / 'glacier_sla_from_sar.csv'
    sla_df.to_csv(output_path, index=False)

    print(f"\n✅ SAR processing complete. {len(all_sla_records)} records saved to:\n{output_path}")

if __name__ == '__main__':
    process_sar_data()


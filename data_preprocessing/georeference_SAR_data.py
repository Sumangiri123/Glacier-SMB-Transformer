from pathlib import Path
from tqdm import tqdm
from osgeo import gdal
import sys

def orthorectify_sar_images():
    """
    Finds raw ENVISAT (.N1) and Sentinel-1 (.SAFE) files and uses the GDAL Python library
    to convert them into orthorectified GeoTIFFs, preserving their native CRS.
    """
    print("--- Starting SAR Image Orthorectification ---")
    print("This uses the installed GDAL Python library.")

    base_dir = Path.cwd()
    sar_image_dir = base_dir / 'data' / 'raw' / 'SAR_images'
    output_dir = base_dir / 'data' / 'processed' / 'SAR_orthorectified'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Orthorectified files will be saved in: {output_dir}")

    # --- Find all relevant source files ---
    envisat_files = list(sar_image_dir.glob('**/*.N1'))
    sentinel_manifests = list(sar_image_dir.glob('**/*.SAFE/manifest.safe'))
    
    source_files = envisat_files + sentinel_manifests

    if not source_files:
        print("No .N1 or Sentinel-1 manifest.safe files found to process.")
        return

    progress_bar = tqdm(source_files, desc="Orthorectifying images")
    for input_path in progress_bar:
        if input_path.name == 'manifest.safe':
            output_stem = input_path.parent.stem
        else:
            output_stem = input_path.stem
        
        output_filename = output_stem + "_ortho.tif"
        output_path = output_dir / output_filename
        
        progress_bar.set_description(f"Processing {output_stem}")

        if output_path.exists():
            continue

        # --- Use the GDAL Python library function to perform the warp/reprojection ---
        try:
            gdal.Warp(
                destNameOrDestDS=str(output_path),
                srcDSOrSrcDSTab=str(input_path),
                format='GTiff',
                resampleAlg='cubic',
                creationOptions=['COMPRESS=LZW']
            )
        except Exception as e:
            # Check if the error is due to a runtime library issue
            if "Can't load requested DLL" in str(e):
                print("\nFATAL ERROR: GDAL DLLs not found. This can happen if GDAL was not installed correctly.")
                print("Please ensure you installed the correct .whl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal")
                break
            print(f"\nAn error occurred while processing {input_path.name}: {e}")

    print("\n✅ Orthorectification process complete.")

if __name__ == '__main__':
    orthorectify_sar_images()


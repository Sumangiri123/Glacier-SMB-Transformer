import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from tqdm import tqdm
import glob
import os



WORKSPACE = Path().resolve()
OUTPUT_DIR = WORKSPACE / 'output'
EVOLUTION_RASTER_DIR = OUTPUT_DIR / 'evolution_results' / 'rasters'

PLOT_OUTPUT_DIR = OUTPUT_DIR / 'geospatial_plots'
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1985
END_YEAR = 2015



def main():
    """Main function to generate and save plots for each glacier."""
    print("--- Starting Geospatial Visualization Script ---")

    search_pattern = str(EVOLUTION_RASTER_DIR / f'*_thickness_{END_YEAR}.tif')
    final_raster_files = glob.glob(search_pattern)

    if not final_raster_files:
        print(f"Error: No final simulation rasters found in '{EVOLUTION_RASTER_DIR}'.")
        print("Please run glacier_evolution.py first.")
        return

    print(f"Found {len(final_raster_files)} simulated glaciers to visualize.")

    for final_path_str in tqdm(final_raster_files, desc="Generating Plots"):
        final_path = Path(final_path_str)
        
        filename = final_path.name
        glims_id = filename.replace(f'_thickness_{END_YEAR}.tif', '')
        initial_path = EVOLUTION_RASTER_DIR / f"{glims_id}_thickness_{START_YEAR}.tif"

        if not initial_path.exists():
            print(f"Warning: Initial thickness file for {glims_id} not found. Skipping.")
            continue

        with rasterio.open(initial_path) as src:
            initial_thickness = src.read(1)
            initial_thickness[initial_thickness < 0] = 0

        with rasterio.open(final_path) as src:
            final_thickness = src.read(
                1,
                out_shape=initial_thickness.shape,
                resampling=Resampling.bilinear
            )
            final_thickness[final_thickness < 0] = 0
        
        thickness_difference = initial_thickness - final_thickness
        thickness_difference[thickness_difference < 0] = 0

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Glacier Evolution: {glims_id}', fontsize=16)

        vmax = np.max(initial_thickness)
        if vmax == 0: vmax = 1
        
        im1 = axes[0].imshow(initial_thickness, cmap='Blues', vmin=0, vmax=vmax)
        axes[0].set_title(f'Initial Thickness ({START_YEAR})')
        axes[0].set_xticks([]); axes[0].set_yticks([])
        fig.colorbar(im1, ax=axes[0], orientation='horizontal', label='Ice Thickness (m)')

        im2 = axes[1].imshow(final_thickness, cmap='Blues', vmin=0, vmax=vmax)
        axes[1].set_title(f'Final Simulated Thickness ({END_YEAR})')
        axes[1].set_xticks([]); axes[1].set_yticks([])
        fig.colorbar(im2, ax=axes[1], orientation='horizontal', label='Ice Thickness (m)')

        im3 = axes[2].imshow(thickness_difference, cmap='Reds', vmin=0)
        axes[2].set_title('Total Ice Thickness Lost')
        axes[2].set_xticks([]); axes[2].set_yticks([])
        fig.colorbar(im3, ax=axes[2], orientation='horizontal', label='Thickness Change (m)')

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        output_filepath = PLOT_OUTPUT_DIR / f"{glims_id}_evolution_maps.png"
        plt.savefig(output_filepath, dpi=200)
        plt.close(fig)

    print("\n--- Visualization Complete ---")
    print(f"All plots have been saved to: {PLOT_OUTPUT_DIR}")


if __name__ == '__main__':
    main()


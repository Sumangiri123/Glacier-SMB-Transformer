import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# ##############################################################################
# ## 1. FILE PATHS AND SETTINGS
# ##############################################################################

WORKSPACE = Path().resolve()
DATA_DIR = WORKSPACE / 'data'
RAW_DIR = DATA_DIR / 'raw'
OUTPUT_DIR = WORKSPACE / 'output'

# --- Input paths ---
FORECAST_CSV_DIR = OUTPUT_DIR / 'forecast_results' # Main directory containing ssp245/csv and ssp585/csv
GLACIER_DATA_DIR = RAW_DIR / 'glacier_data'
GLIMS_2015_FILE = GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv'

# --- Output path ---
SUMMARY_OUTPUT_FILE = OUTPUT_DIR / 'forecast_summary_comparison.csv'

# --- Scenarios to analyze ---
SCENARIOS_TO_ANALYZE = ['ssp245', 'ssp585']
START_YEAR = 2015

# ##############################################################################
# ## 2. MAIN ANALYSIS SCRIPT
# ##############################################################################

def analyze_forecasts():
    """
    Reads individual glacier forecast CSVs, extracts start and end states,
    and compiles them into a summary CSV file.
    """
    print("--- Starting Forecast Summary Analysis ---")

    # Load glacier names for better readability
    try:
        glacier_info_df = pd.read_csv(GLIMS_2015_FILE, delimiter=';', encoding='latin1')
        glacier_names = glacier_info_df.set_index('GLIMS_ID')['Glacier'].to_dict()
    except FileNotFoundError:
        print(f"Warning: Glacier info file not found at {GLIMS_2015_FILE}. Summary will lack names.")
        glacier_names = {}
    except Exception as e:
        print(f"Error loading glacier names: {e}")
        glacier_names = {}


    summary_data = []

    # Loop through each scenario
    for scenario in SCENARIOS_TO_ANALYZE:
        print(f"\n--- Analyzing Scenario: {scenario.upper()} ---")
        scenario_csv_dir = FORECAST_CSV_DIR / scenario / 'csv'

        if not scenario_csv_dir.exists():
            print(f"Warning: Directory not found for scenario {scenario}. Skipping.")
            continue

        # Get list of forecast CSV files for this scenario
        forecast_files = list(scenario_csv_dir.glob('*_forecast.csv'))

        if not forecast_files:
            print(f"Warning: No forecast CSV files found in {scenario_csv_dir}. Skipping scenario.")
            continue

        # Loop through each glacier's forecast file
        for f_path in tqdm(forecast_files, desc=f"Processing glaciers for {scenario}"):
            glims_id = f_path.name.replace('_forecast.csv', '')
            glacier_name = glacier_names.get(glims_id, 'N/A') # Get name or use N/A

            try:
                df = pd.read_csv(f_path)

                # Skip if empty or only contains the initial 2015 row
                if df.empty or len(df) <= 1:
                    print(f"\nWarning: Forecast file for {glims_id} ({scenario}) is empty or incomplete. Skipping.")
                    start_volume, start_area = np.nan, np.nan
                    end_year, end_smb, end_volume, end_area = np.nan, np.nan, np.nan, np.nan
                else:
                    # Get the first row (initial state for START_YEAR)
                    start_row = df.iloc[0]
                    start_volume = start_row['volume_km3']
                    start_area = start_row['area_km2']

                    # Get the last row (final state)
                    end_row = df.iloc[-1]
                    end_year = int(end_row['year'])
                    end_smb = end_row['smb_mwe']
                    # Volume/Area in the last row represent the state *at the beginning* of that year.
                    # If the glacier melted, the final state is effectively zero volume/area.
                    if end_row['area_km2'] == 0: # Check if melted based on area
                        end_volume = 0.0
                        end_area = 0.0
                    else:
                        # If it survived, use the values from the last row
                        end_volume = end_row['volume_km3']
                        end_area = end_row['area_km2']


                # Append the summary row
                summary_data.append({
                    'GLIMS_ID': glims_id,
                    'Glacier_Name': glacier_name,
                    'Scenario': scenario,
                    'Start_Year': START_YEAR,
                    'Start_Volume_km3': start_volume,
                    'Start_Area_km2': start_area,
                    'End_Year': end_year,
                    'End_SMB_mwe': end_smb,
                    'End_Volume_km3': end_volume,
                    'End_Area_km2': end_area,
                    'Volume_Change_km3': end_volume - start_volume if pd.notna(start_volume) and pd.notna(end_volume) else np.nan,
                    'Area_Change_km2': end_area - start_area if pd.notna(start_area) and pd.notna(end_area) else np.nan,
                })

            except Exception as e:
                print(f"\nError processing file {f_path}: {e}")

    # --- Save the combined summary ---
    if not summary_data:
        print("\nNo data processed. Summary file will not be created.")
        return

    summary_df = pd.DataFrame(summary_data)
    # Sort for better readability
    summary_df = summary_df.sort_values(by=['GLIMS_ID', 'Scenario'])

    try:
        summary_df.to_csv(SUMMARY_OUTPUT_FILE, index=False, float_format='%.4f')
        print(f"\n✅ Analysis complete. Summary saved to:\n{SUMMARY_OUTPUT_FILE}")
    except Exception as e:
        print(f"\nError saving summary file: {e}")


if __name__ == '__main__':
    analyze_forecasts()

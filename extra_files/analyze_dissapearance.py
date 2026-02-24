import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os


WORKSPACE = Path().resolve()
DATA_DIR = WORKSPACE / 'data'
RAW_DIR = DATA_DIR / 'raw'
OUTPUT_DIR = WORKSPACE / 'output'


FORECAST_CSV_DIR = OUTPUT_DIR / 'forecast_results'
GLACIER_DATA_DIR = RAW_DIR / 'glacier_data'
GLIMS_2015_FILE = GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv'
DISAPPEARANCE_OUTPUT_FILE = OUTPUT_DIR / 'glacier_disappearance_summary.csv'
SCENARIOS_TO_ANALYZE = ['ssp245', 'ssp585']
DISAPPEARANCE_THRESHOLD_YEAR = 2099
MAX_SIMULATION_YEAR = 2100


def find_disappearance_year(df):
    
    if df.empty:
        return None

    
    last_year_simulated = int(df['year'].iloc[-1])

    
    if last_year_simulated < MAX_SIMULATION_YEAR:
        
        
        
        if df['area_km2'].iloc[-1] <= 0.0001:
            
            
            
            
            return last_year_simulated
        else:
            
            
            
            return last_year_simulated
    else:
        
        return None


def analyze_disappearance():
    
    print("--- Starting Glacier Disappearance Analysis ---")

    
    try:
        glacier_info_df = pd.read_csv(GLIMS_2015_FILE, delimiter=';', encoding='latin1')
        glacier_names = glacier_info_df.set_index('GLIMS_ID')['Glacier'].to_dict()
    except FileNotFoundError:
        print(f"Warning: Glacier info file not found at {GLIMS_2015_FILE}. Summary will lack names.")
        glacier_names = {}
    except Exception as e:
        print(f"Error loading glacier names: {e}")
        glacier_names = {}

    disappearance_data = []

    for scenario in SCENARIOS_TO_ANALYZE:
        print(f"\n--- Analyzing Scenario: {scenario.upper()} ---")
        scenario_csv_dir = FORECAST_CSV_DIR / scenario / 'csv'

        if not scenario_csv_dir.exists():
            print(f"Warning: Directory not found for scenario {scenario}. Skipping.")
            continue

        forecast_files = list(scenario_csv_dir.glob('*_forecast.csv'))
        if not forecast_files:
            print(f"Warning: No forecast CSV files found in {scenario_csv_dir}. Skipping scenario.")
            continue

        
        for f_path in tqdm(forecast_files, desc=f"Processing glaciers for {scenario}"):
            glims_id = f_path.name.replace('_forecast.csv', '')
            glacier_name = glacier_names.get(glims_id, 'N/A')

            try:
                
                if os.path.getsize(f_path) == 0:
                    print(f"\nWarning: Forecast file for {glims_id} ({scenario}) is empty. Skipping.")
                    continue
                df = pd.read_csv(f_path)
                
                if df.empty:
                    print(f"\nWarning: Forecast DataFrame for {glims_id} ({scenario}) is empty after reading. Skipping.")
                    continue

                disappearance_year = find_disappearance_year(df)

                
                if disappearance_year is not None and disappearance_year < DISAPPEARANCE_THRESHOLD_YEAR:
                    disappearance_data.append({
                        'GLIMS_ID': glims_id,
                        'Glacier_Name': glacier_name,
                        'Scenario': scenario,
                        'Year_of_Disappearance': disappearance_year
                    })

            except pd.errors.EmptyDataError:
                print(f"\nWarning: Forecast file for {glims_id} ({scenario}) seems to be empty or corrupted. Skipping.")
            except Exception as e:
                print(f"\nError processing file {f_path}: {e}")

    
    if not disappearance_data:
        
        print(f"\nNo glaciers found disappearing before the year {DISAPPEARANCE_THRESHOLD_YEAR}. Summary file will not be created.")
        return

    summary_df = pd.DataFrame(disappearance_data)
    
    summary_df = summary_df.sort_values(by=['Scenario', 'Year_of_Disappearance', 'GLIMS_ID'])

    try:
        summary_df.to_csv(DISAPPEARANCE_OUTPUT_FILE, index=False)
        print(f"\n✅ Analysis complete. Disappearance summary saved to:\n{DISAPPEARANCE_OUTPUT_FILE}")
    except Exception as e:
        print(f"\nError saving summary file: {e}")


if __name__ == '__main__':
    analyze_disappearance()
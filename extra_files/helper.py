import pandas as pd
from pathlib import Path
import os # Added import for os.path.join

# ##############################################################################
# ## 1. FILE PATHS AND SETTINGS
# ##############################################################################

WORKSPACE = Path().resolve()
DATA_DIR = WORKSPACE / 'data'
RAW_DIR = DATA_DIR / 'raw'
GLACIER_DATA_DIR = RAW_DIR / 'glacier_data'
OUTPUT_DIR = WORKSPACE / 'output' # Define OUTPUT_DIR

# Input file containing glacier information
GLACIER_INFO_FILE = GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv'

# Columns to extract and their desired names in the final table
COLUMNS_TO_EXTRACT = {
    'GLIMS_ID': 'GLIMS ID',
    'Glacier': 'Glacier Name',
    'Massif': 'Massif',
    'Area': 'Area (km²)',
    'MEAN_Pixel': 'Mean Elevation (m a.s.l.)'
}

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ##############################################################################
# ## 2. SCRIPT TO GENERATE TABLE
# ##############################################################################

def create_glacier_summary_table():
    """
    Loads glacier data, selects key characteristics,
    and saves it as a CSV file.
    """
    print("--- Generating Glacier Summary CSV ---") # Updated print message

    # --- Load the Glacier Data ---
    try:
        glacier_df = pd.read_csv(GLACIER_INFO_FILE, delimiter=';', encoding='latin1')
        print(f"Successfully loaded data for {len(glacier_df)} glaciers.")
    except FileNotFoundError:
        print(f"Error: Glacier information file not found at '{GLACIER_INFO_FILE}'.")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # --- Select and Rename Columns ---
    try:
        # Check if all required columns exist
        missing_cols = [col for col in COLUMNS_TO_EXTRACT.keys() if col not in glacier_df.columns]
        if missing_cols:
            print(f"Error: The following required columns are missing from the CSV file: {', '.join(missing_cols)}")
            return

        summary_df = glacier_df[list(COLUMNS_TO_EXTRACT.keys())].copy()
        summary_df.rename(columns=COLUMNS_TO_EXTRACT, inplace=True)

        # Ensure we have the expected number of glaciers (adjust if needed)
        if len(summary_df) != 32:
             print(f"Warning: Expected 32 glaciers, but found {len(summary_df)}. Proceeding anyway.")


    except KeyError as e:
         print(f"Error selecting columns: {e}. Please check the column names in COLUMNS_TO_EXTRACT.")
         return
    except Exception as e:
        print(f"An unexpected error occurred during data selection: {e}")
        return


    # --- REMOVED Markdown Table Printing ---
    # print("\n--- Table 3.1: Summary of Study Glaciers (Markdown Format) ---")
    # markdown_table = summary_df.to_markdown(index=False)
    # print(markdown_table)
    # print("\n(Copy and paste the table above into your dissertation document)")

    # --- Save data to CSV ---
    output_csv_path = OUTPUT_DIR / 'glacier_summary_table.csv' # Use defined OUTPUT_DIR
    try:
        # Save with 2 decimal places for Area and Mean Elevation for neatness
        summary_df.to_csv(output_csv_path, index=False, float_format='%.2f')
        print(f"\n✅ Glacier summary table saved as CSV to: {output_csv_path}") # Updated print message
    except Exception as e:
        print(f"\nError saving table to CSV: {e}")


if __name__ == '__main__':
    create_glacier_summary_table()


import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys

# --- 1. DEFINE FILE PATHS ---
# These paths must match the ones in your training script.
# This script assumes you are running it from the same main project folder.
WORKSPACE = Path().resolve()
OUTPUT_DIR = WORKSPACE / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'
DATA_PATH = MODEL_DIR / 'cv_data'
FEATURE_NAMES_PATH = MODEL_DIR / 'feature_names.pkl'

# Define the full paths to the files we need to load
X_FILE = DATA_PATH / 'X_data.npy'
Y_FILE = DATA_PATH / 'y_data.npy'
GROUPS_FILE = DATA_PATH / 'groups.npy'

# --- NEW: Define the path for the output CSV file ---
CSV_OUTPUT_FILE = DATA_PATH / 'spatiotemporal_matrix_FULL.csv'


def view_and_save_data(): # Renamed function for clarity
    """
    Loads the saved spatiotemporal matrix data, displays it,
    and saves it to a single CSV file.
    """
    print("--- Loading Spatiotemporal Matrix Data ---")

    try:
        # --- 2. LOAD THE FILES ---
        
        # Load the NumPy arrays
        print(f"Loading X data from: {X_FILE}")
        X_data = np.load(X_FILE)
        
        print(f"Loading y data from: {Y_FILE}")
        y_data = np.load(Y_FILE)
        
        print(f"Loading groups data from: {GROUPS_FILE}")
        groups_data = np.load(GROUPS_FILE)

        # Load the feature names (column headers)
        print(f"Loading feature names from: {FEATURE_NAMES_PATH}")
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)

        print("\n--- Data Loaded Successfully ---")

        # --- 3. CHECK AND DISPLAY THE DATA ---

        # Print the "shape" (dimensions) of the data
        print(f"\nDimensions of X_data (features): {X_data.shape} (rows, columns)")
        print(f"Dimensions of y_data (target): {y_data.shape} (rows,)")
        print(f"Dimensions of groups_data (glacier IDs): {groups_data.shape} (rows,)")
        print(f"Number of feature names loaded: {len(feature_names)}")

        # Check for mismatch (good practice)
        if X_data.shape[1] != len(feature_names):
            print("\nWARNING: Mismatch between number of columns in X_data"
                  f" ({X_data.shape[1]}) and number of feature names"
                  f" ({len(feature_names)}).")

        # --- 4. COMBINE AND SAVE TO CSV ---
        
        # Create the main DataFrame from X_data and feature names
        combined_df = pd.DataFrame(X_data, columns=feature_names)
        
        # --- NEW: Add the y_data and groups_data as new columns ---
        combined_df['SMB_target'] = y_data
        combined_df['glacier_group_id'] = groups_data

        # --- NEW: Save the combined DataFrame to a CSV file ---
        print(f"\n--- Saving combined data to CSV ---")
        print(f"File location: {CSV_OUTPUT_FILE}")
        combined_df.to_csv(CSV_OUTPUT_FILE, index=False)
        print("... Save complete.")

        # --- 5. DISPLAY DATA TO CONSOLE (as before) ---
        
        print("\n--- Displaying Combined Data (First 10 Rows) ---")
        print(combined_df.head(10))

        print("\n--- Statistical Summary of Combined Data ---")
        # This gives you the mean, std, min, max, etc. for each feature
        print(combined_df.describe())


    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(f"File not found: {e.filename}")
        print("Please make sure you have already run the training script"
              " to generate the .npy files.")
        print("Also, ensure this 'view_data.py' script is in the"
              " same main project directory.")
        sys.exit(1) # Exit the script with an error
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set pandas display options to see more columns
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    
    view_and_save_data() # Updated function name
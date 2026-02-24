# profile_glaciers.py

import pandas as pd
from pathlib import Path

# --- Configuration ---
WORKSPACE = Path().resolve()
OUTPUT_DIR = WORKSPACE / 'output'
RAW_DIR = WORKSPACE / 'data' / 'raw'

# Path to the results summary you just generated
RESULTS_FILE = OUTPUT_DIR / 'results_summary.csv'
# Path to the file containing static glacier features
FEATURES_FILE = RAW_DIR / 'glacier_data' / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv'

# --- Main Analysis ---

print("--- Glacier Performance Profiling ---")

# Load the validation results and the glacier features
try:
    results_df = pd.read_csv(RESULTS_FILE)
    features_df = pd.read_csv(FEATURES_FILE, delimiter=';', encoding='latin1')
except FileNotFoundError as e:
    print(f"Error: Could not find a required file. {e}")
    exit()

# Rename feature columns for clarity and merge the dataframes
features_df.rename(columns={'GLIMS_ID': 'Glacier_ID'}, inplace=True)
merged_df = pd.merge(results_df, features_df, on='Glacier_ID')

# Define what makes a "Good" vs "Bad" prediction
good_glaciers = merged_df[merged_df['ANN_R2'] > 0.5]
bad_glaciers = merged_df[merged_df['ANN_R2'] < 0.1]

print(f"\nFound {len(good_glaciers)} glaciers with GOOD performance (R² > 0.5).")
print(f"Found {len(bad_glaciers)} glaciers with BAD performance (R² < 0.1).")

# Select only the key physical feature columns to compare
# CORRECTED LINE: Changed 'Zmed' to 'MEAN_Pixel' based on your file
feature_cols = ['Area', 'slope20', 'Aspect_num', 'MEAN_Pixel']

# Print the summary statistics for each group
print("\n--- Feature Analysis: GOOD Glaciers ---")
print(good_glaciers[feature_cols].describe())

print("\n--- Feature Analysis: BAD Glaciers ---")
print(bad_glaciers[feature_cols].describe())

print("\n--- Interpretation ---")
print("Compare the 'mean' values for each feature between the GOOD and BAD groups.")
print("A large difference in a feature (e.g., 'Area') suggests the model struggles with glaciers of that type.")
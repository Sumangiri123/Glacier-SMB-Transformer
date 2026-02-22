import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error

WORKSPACE = Path().resolve()
DATA_DIR = WORKSPACE / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RAW_DIR = DATA_DIR / 'raw'
OUTPUT_DIR = WORKSPACE / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'
METEO_DIR = PROCESSED_DIR / 'daily_meteo'
SAR_FEATURES_DIR = PROCESSED_DIR / 'SAR_features'
GLACIER_DATA_DIR = RAW_DIR / 'glacier_data'
SMB_FILE = GLACIER_DATA_DIR / 'smb' / 'SMB_raw_temporal.csv'
FEATURE_NAMES_PATH = MODEL_DIR / 'feature_names.pkl' 

MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / 'cv_data').mkdir(exist_ok=True)

def calculate_monthly_climate_features(daily_df, year):
    """Calculates monthly mean temperature and total snowfall for a hydrological year."""
    start_date = pd.to_datetime(f"{year-1}-10-01")
    end_date = pd.to_datetime(f"{year}-09-30")
    year_df = daily_df[(daily_df['date'] >= start_date) & (daily_df['date'] <= end_date)].copy()
    
    if year_df.empty:
        months = pd.date_range(start=start_date, end=end_date, freq='MS')
        nan_features = {}
        for month in months:
            nan_features[f"temp_{month.strftime('%b').lower()}"] = np.nan
            nan_features[f"snow_{month.strftime('%b').lower()}"] = np.nan
        return nan_features
        
    year_df['month'] = year_df['date'].dt.to_period('M')
    
    monthly_temp = year_df.groupby('month')['temperature_c'].mean()
    monthly_snow = year_df.groupby('month')['snowfall_mm'].sum()
    
    features = {}
    months = pd.date_range(start=start_date, end=end_date, freq='MS').to_period('M')
    
    for month in months:
        month_abbr = month.strftime('%b').lower()
        features[f"temp_{month_abbr}"] = monthly_temp.get(month, 0)
        features[f"snow_{month_abbr}"] = monthly_snow.get(month, 0)
        
    return features

def create_spatiotemporal_matrix():
    print("Creating spatiotemporal matrix with MONTHLY features...")
    rabatel_df = pd.read_csv(GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv', delimiter=';', encoding='latin1')
    smb_matrix = pd.read_csv(SMB_FILE, delimiter=';', header=None)
    
    if len(rabatel_df) != len(smb_matrix):
        raise ValueError("Mismatch between number of glaciers in Rabatel file and SMB file.")
    
    num_years = smb_matrix.shape[1]
    start_year = 2014 - num_years + 1
    smb_years = [str(y) for y in range(start_year, 2014 + 1)]

    smb_df = smb_matrix.copy()
    smb_df.columns = smb_years
    smb_df['GLIMS_ID'] = rabatel_df['GLIMS_ID']
    smb_df = smb_df.set_index('GLIMS_ID')

    static_df = rabatel_df[['GLIMS_ID', 'Area', 'slope20', 'x_coord', 'y_coord', 'Aspect_num']].rename(columns={
        'Area': 'area', 'slope20': 'slope', 'x_coord': 'lon', 'y_coord': 'lat', 'Aspect_num': 'aspect'
    }).set_index('GLIMS_ID')

    print("Loading daily meteo files...")
    meteo_data = {}
    for f in tqdm(os.listdir(METEO_DIR)):
        if f.endswith('.csv'):
            glims_id = f.replace('.csv', '')
            df = pd.read_csv(METEO_DIR / f)
            df['date'] = pd.to_datetime(df['date'])
            meteo_data[glims_id] = df
    
    feature_rows, target_values, group_ids = [], [], []
    glims_ids = smb_df.index
    years = [int(y) for y in smb_df.columns]
    
    group_counter = 1
    for glims_id in tqdm(glims_ids, desc="Processing glaciers"):
        if glims_id not in meteo_data or glims_id not in static_df.index:
            continue

        for year in years:
            smb_value = smb_df.loc[glims_id, str(year)]
            
            monthly_features = calculate_monthly_climate_features(meteo_data[glims_id], year)
            static_features = static_df.loc[glims_id].to_dict()
            combined_features = {**monthly_features, **static_features}
            
            feature_rows.append(combined_features)
            target_values.append(smb_value)
            group_ids.append(group_counter)
            
        group_counter += 1

    X_df = pd.DataFrame(feature_rows)
    
    feature_names = X_df.columns.tolist()
    with open(FEATURE_NAMES_PATH, 'wb') as f:
        pickle.dump(feature_names, f)

    X_df = X_df.fillna(method='ffill').fillna(method='bfill').fillna(X_df.median())

    y, groups = np.array(target_values), np.array(group_ids)
    
    finite_mask = np.isfinite(y)
    X_df = X_df[finite_mask]
    y, groups = y[finite_mask], groups[finite_mask]
    
    print(f"Spatiotemporal matrix created. X_df shape: {X_df.shape}, y shape: {y.shape}")
    
    return X_df, y, groups

def generate_SMB_models(X_df, y, groups):
    print("\nStarting model training and cross-validation...")
    
    X = X_df.values

    print("\n--- Training Lasso Model ---")
    scaler_full = StandardScaler().fit(X)
    X_scaled_full = scaler_full.transform(X)
    lasso_full = LassoCV(cv=10, random_state=42, max_iter=2000).fit(X_scaled_full, y)
    
    with open(MODEL_DIR / 'lasso_model.pkl', 'wb') as f:
        pickle.dump(lasso_full, f)
    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler_full, f)
    print("Lasso model and scaler saved.")

    print("\n--- Preparing data splits for ANN ---")
    np.save(MODEL_DIR / 'cv_data' / 'X_data.npy', X)
    np.save(MODEL_DIR / 'cv_data' / 'y_data.npy', y)
    np.save(MODEL_DIR / 'cv_data' / 'groups.npy', groups)
    print("ANN data splits saved. Ready for neural network training.")

def main():
    X_df, y, groups = create_spatiotemporal_matrix()
    if X_df is not None and not X_df.empty:
        generate_SMB_models(X_df, y, groups)
    else:
        print("Matrix creation failed. Aborting training.")

if __name__ == '__main__':
    main()


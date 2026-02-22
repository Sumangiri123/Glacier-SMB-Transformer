import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr  

WORKSPACE = Path().resolve()
DATA_DIR = WORKSPACE / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RAW_DIR = DATA_DIR / 'raw'
OUTPUT_DIR = WORKSPACE / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'

SCALER_PATH = MODEL_DIR / 'scaler.pkl'
LASSO_PATH = MODEL_DIR / 'lasso_model.pkl'
ANN_ENSEMBLE_DIR = MODEL_DIR / 'ann_ensemble'
PLOT_DIR = OUTPUT_DIR / 'validation_plots'
RESULTS_CSV_DIR = OUTPUT_DIR / 'validation_csvs'
FEATURE_NAMES_PATH = MODEL_DIR / 'feature_names.pkl' 

METEO_DIR = PROCESSED_DIR / 'daily_meteo'
GLACIER_DATA_DIR = RAW_DIR / 'glacier_data'
SMB_FILE = GLACIER_DATA_DIR / 'smb' / 'SMB_raw_temporal.csv'

PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV_DIR.mkdir(parents=True, exist_ok=True)


def r2_keras(y_true, y_pred): return 0
def root_mean_squared_error(y_true, y_pred): return 0

def calculate_monthly_climate_features(daily_df, year):
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




def main():
    print("        SMB MODEL VALIDATION")

    print("Loading trained models...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LASSO_PATH, 'rb') as f:
        lasso_model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)

    ann_models = []
    for model_file in os.listdir(ANN_ENSEMBLE_DIR):
        if model_file.endswith('.keras'):
            model = load_model(ANN_ENSEMBLE_DIR / model_file, custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error})
            ann_models.append(model)
    print(f"Loaded 1 scaler, 1 Lasso model, {len(feature_names)} feature names, and {len(ann_models)} ANN ensemble members.")

    print("Loading data for validation...")
    rabatel_df = pd.read_csv(GLACIER_DATA_DIR / 'GLIMS' / 'GLIMS_Rabatel_30_2015.csv', delimiter=';', encoding='latin1')
    smb_matrix = pd.read_csv(SMB_FILE, delimiter=';', header=None)

    if len(rabatel_df) != len(smb_matrix):
        raise ValueError("Mismatch between number of glaciers in Rabatel file and SMB file.")

    num_years_from_file = smb_matrix.shape[1]
    last_year_in_file = 2014
    start_year = last_year_in_file - num_years_from_file + 1
    smb_years = [str(y) for y in range(start_year, last_year_in_file + 1)]

    smb_df = smb_matrix.copy()
    smb_df.columns = smb_years
    smb_df['GLIMS_ID'] = rabatel_df['GLIMS_ID']
    smb_df = smb_df.set_index('GLIMS_ID')

    static_df = rabatel_df[['GLIMS_ID', 'Area', 'slope20', 'x_coord', 'y_coord', 'Aspect_num']].rename(columns={
        'Area': 'area', 'slope20': 'slope', 'x_coord': 'lon', 'y_coord': 'lat', 'Aspect_num': 'aspect'
    }).set_index('GLIMS_ID')

    meteo_data = {}
    for f in os.listdir(METEO_DIR):
        if f.endswith('.csv'):
            glims_id = f.replace('.csv', '')
            df = pd.read_csv(METEO_DIR / f); df['date'] = pd.to_datetime(df['date'])
            meteo_data[glims_id] = df

    all_true, all_lasso, all_ann = [], [], []
    per_glacier_metrics = []

    for glims_id in tqdm(static_df.index, desc="Validating Glaciers"):
        if glims_id not in meteo_data or glims_id not in smb_df.index: continue

        feature_rows, y_true_glacier, valid_years = [], [], []
        years_to_process = [int(y) for y in smb_df.columns]

        for year in years_to_process:
            smb_value = smb_df.loc[glims_id, str(year)]
            if not np.isfinite(smb_value): continue

            y_true_glacier.append(smb_value)
            valid_years.append(year)

            monthly_features = calculate_monthly_climate_features(meteo_data[glims_id], year)
            static_features = static_df.loc[glims_id].to_dict()
            combined_features = {**monthly_features, **static_features}
            feature_rows.append(combined_features)

        if not feature_rows: continue

        X_glacier_df = pd.DataFrame(feature_rows)
        X_glacier_df = X_glacier_df.ffill().bfill().fillna(0)

        X_glacier_df = X_glacier_df[feature_names]

        X_glacier = scaler.transform(X_glacier_df.values)
        y_true_glacier = np.array(y_true_glacier)

        y_pred_lasso = lasso_model.predict(X_glacier)

        ann_preds = [model.predict(X_glacier, verbose=0).flatten() for model in ann_models]
        y_pred_ann = np.mean(ann_preds, axis=0)

        all_true.extend(y_true_glacier)
        all_lasso.extend(y_pred_lasso)
        all_ann.extend(y_pred_ann)

        output_df = pd.DataFrame({
            'Year': valid_years,
            'Ground_Truth_SMB': y_true_glacier,
            'ANN_Prediction': y_pred_ann,
            'Lasso_Prediction': y_pred_lasso
        })
        output_df.to_csv(RESULTS_CSV_DIR / f'{glims_id}_validation.csv', index=False)

        try:
            ann_pearson_r, _ = pearsonr(y_true_glacier, y_pred_ann)
        except ValueError:
            ann_pearson_r = np.nan
        try:
            lasso_pearson_r, _ = pearsonr(y_true_glacier, y_pred_lasso)
        except ValueError:
            lasso_pearson_r = np.nan

        per_glacier_metrics.append({
            'Glacier_ID': glims_id,
            'ANN_R2': r2_score(y_true_glacier, y_pred_ann),
            'Lasso_R2': r2_score(y_true_glacier, y_pred_lasso),
            'ANN_RMSE': np.sqrt(mean_squared_error(y_true_glacier, y_pred_ann)),
            'Lasso_RMSE': np.sqrt(mean_squared_error(y_true_glacier, y_pred_lasso)),
            'ANN_MAE': mean_absolute_error(y_true_glacier, y_pred_ann),
            'Lasso_MAE': mean_absolute_error(y_true_glacier, y_pred_lasso),
            'ANN_MSE': mean_squared_error(y_true_glacier, y_pred_ann),
            'Lasso_MSE': mean_squared_error(y_true_glacier, y_pred_lasso),
            'ANN_MAPE': mean_absolute_percentage_error(y_true_glacier, y_pred_ann),
            'Lasso_MAPE': mean_absolute_percentage_error(y_true_glacier, y_pred_lasso),
            'ANN_Pearson_r': ann_pearson_r,
            'Lasso_Pearson_r': lasso_pearson_r
        })

        plt.figure(figsize=(12, 6))
        plt.plot(valid_years, y_true_glacier, 'o-', label='Ground Truth SMB', color='k')
        plt.plot(valid_years, y_pred_lasso, 's--', label=f'Lasso (R²={r2_score(y_true_glacier, y_pred_lasso):.2f})', alpha=0.8)
        plt.plot(valid_years, y_pred_ann, '^-', label=f'ANN (R²={r2_score(y_true_glacier, y_pred_ann):.2f})', alpha=0.8)
        plt.title(f"SMB Validation for Glacier: {glims_id}")
        plt.xlabel("Year")
        plt.ylabel("Annual SMB (m.w.e.)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(PLOT_DIR / f'{glims_id}_validation.png')
        plt.close()

    print("\n--- Overall Model Performance (All Predictions Pooled) ---")
    lasso_r2_total = r2_score(all_true, all_lasso)
    lasso_rmse_total = np.sqrt(mean_squared_error(all_true, all_lasso))
    ann_r2_total = r2_score(all_true, all_ann)
    ann_rmse_total = np.sqrt(mean_squared_error(all_true, all_ann))
    
    lasso_mae_total = mean_absolute_error(all_true, all_lasso)
    ann_mae_total = mean_absolute_error(all_true, all_ann)
    lasso_mse_total = mean_squared_error(all_true, all_lasso)
    ann_mse_total = mean_squared_error(all_true, all_ann)
    lasso_mape_total = mean_absolute_percentage_error(all_true, all_lasso)
    ann_mape_total = mean_absolute_percentage_error(all_true, all_ann)
    lasso_pearson_r_total, _ = pearsonr(all_true, all_lasso)
    ann_pearson_r_total, _ = pearsonr(all_true, all_ann)

    print(f"Lasso Model -> R2: {lasso_r2_total:.3f}, RMSE: {lasso_rmse_total:.3f} m.w.e.")
    print(f"             -> MAE: {lasso_mae_total:.3f} m.w.e., MSE: {lasso_mse_total:.3f}, MAPE: {lasso_mape_total:.3f}, r: {lasso_pearson_r_total:.3f}")
    print(f"ANN Ensemble -> R2: {ann_r2_total:.3f}, RMSE: {ann_rmse_total:.3f} m.w.e.")
    print(f"             -> MAE: {ann_mae_total:.3f} m.w.e., MSE: {ann_mse_total:.3f}, MAPE: {ann_mape_total:.3f}, r: {ann_pearson_r_total:.3f}")
    
    print(f"\nValidation plots saved to: {PLOT_DIR}")

    results_df = pd.DataFrame(per_glacier_metrics)
    summary_path = OUTPUT_DIR / 'results_summary.csv'
    results_df.to_csv(summary_path, index=False)
    print(f"\nPer-glacier metrics summary saved to: {summary_path}")

    print("\n--- Summary Statistics of Per-Glacier Performance (ANN Model) ---")
    ann_cols = ['ANN_R2', 'ANN_RMSE', 'ANN_MAE', 'ANN_MSE', 'ANN_MAPE', 'ANN_Pearson_r']
    print(results_df[ann_cols].describe())

    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='ANN_R2', kde=True, bins=15)
    plt.title('Distribution of ANN R² Scores Across Glaciers', fontsize=16)
    plt.xlabel('R² Score', fontsize=12)
    plt.ylabel('Number of Glaciers', fontsize=12)
    hist_path = PLOT_DIR / 'ann_r2_histogram.png'
    plt.savefig(hist_path)
    print(f"R² histogram saved to: {hist_path}")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='ANN_R2')
    plt.title('Box Plot of ANN R² Scores', fontsize=16)
    plt.xlabel('R² Score', fontsize=12)
    box_path = PLOT_DIR / 'ann_r2_boxplot.png'
    plt.savefig(box_path)
    print(f"R² box plot saved to: {box_path}")

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import os
import glob

VALIDATION_RESULTS_DIR = 'output/'
FILE_PATTERN = '*_validation.csv'
YEAR_COL = 'Year'
TRUTH_COL = 'Ground_Truth_SMB'
ANN_PRED_COL = 'ANN_Prediction'
LASSO_PRED_COL = 'Lasso_Prediction'

def analyze_validation_results(results_dir, file_pattern):
    search_path = os.path.join(results_dir, file_pattern)
    validation_files = glob.glob(search_path)

    if not validation_files:
        print(f"Error: No validation files found at '{search_path}'.")
        print("Please ensure your validation script is saving CSV files in the correct directory.")
        return

    all_metrics = []

    print(f"Found {len(validation_files)} validation files. Processing...")

    for f in validation_files:
        try:
            glacier_id = os.path.basename(f).split('_')[0]
            
            df = pd.read_csv(f)
            
            required_cols = [TRUTH_COL, ANN_PRED_COL, LASSO_PRED_COL]
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {f}: Missing one of the required columns {required_cols}")
                continue

            df.dropna(subset=required_cols, inplace=True)

            ann_r2 = r2_score(df[TRUTH_COL], df[ANN_PRED_COL])
            ann_rmse = np.sqrt(mean_squared_error(df[TRUTH_COL], df[ANN_PRED_COL]))
            
            lasso_r2 = r2_score(df[TRUTH_COL], df[LASSO_PRED_COL])
            lasso_rmse = np.sqrt(mean_squared_error(df[TRUTH_COL], df[LASSO_PRED_COL]))
            
            all_metrics.append({
                'Glacier_ID': glacier_id,
                'ANN_R2': ann_r2,
                'Lasso_R2': lasso_r2,
                'ANN_RMSE': ann_rmse,
                'Lasso_RMSE': lasso_rmse
            })
        except Exception as e:
            print(f"Could not process file {f}. Error: {e}")

    results_df = pd.DataFrame(all_metrics)
    
    summary_file_path = os.path.join(results_dir, 'results_summary.csv')
    results_df.to_csv(summary_file_path, index=False)
    print(f"\n✅ Aggregated results saved to '{summary_file_path}'")
    
    print("\n--- Summary Statistics for ANN Model Performance ---")
    print(results_df[['ANN_R2', 'ANN_RMSE']].describe())
    
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='ANN_R2', kde=True, bins=15)
    plt.title('Distribution of ANN R² Scores Across All Glaciers', fontsize=16)
    plt.xlabel('R² Score', fontsize=12)
    plt.ylabel('Number of Glaciers', fontsize=12)
    hist_path = os.path.join(results_dir, 'ann_r2_histogram.png')
    plt.savefig(hist_path)
    print(f"✅ R² histogram saved to '{hist_path}'")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='ANN_R2')
    plt.title('Box Plot of ANN R² Scores', fontsize=16)
    plt.xlabel('R² Score', fontsize=12)
    box_path = os.path.join(results_dir, 'ann_r2_boxplot.png')
    plt.savefig(box_path)
    print(f"✅ R² box plot saved to '{box_path}'")
    
    print("\nAutomation complete. Check the 'output' directory for summary files and plots.")

if __name__ == '__main__':
    analyze_validation_results(VALIDATION_RESULTS_DIR, FILE_PATTERN)
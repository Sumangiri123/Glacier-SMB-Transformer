import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from tqdm import tqdm

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization,Input
from keras.layers import LeakyReLU
from keras import optimizers
from keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



WORKSPACE = Path().resolve()
OUTPUT_DIR = WORKSPACE / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'
CV_DATA_DIR = MODEL_DIR / 'cv_data'

ANN_CV_DIR = MODEL_DIR / 'ann_cv_models'
ANN_ENSEMBLE_DIR = MODEL_DIR / 'ann_ensemble'
ANN_CV_DIR.mkdir(parents=True, exist_ok=True)
ANN_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)



def r2_keras(y_true, y_pred):
    """Custom R2 metric for Keras."""
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

def root_mean_squared_error(y_true, y_pred):
    """Custom RMSE loss function for Keras."""
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def create_ann_model(n_features, is_final_model=False):
    """
    Defines the ANN architecture.
    """
    model = Sequential()
    model.add(Input(shape=(n_features,)))

    
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    
    if is_final_model:
        model.add(Dense(60, kernel_initializer='he_normal'))
        model.add(BatchNormalization()); model.add(LeakyReLU(negative_slope=0.05))
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization()); model.add(LeakyReLU(negative_slope=0.05))
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()); model.add(LeakyReLU(negative_slope=0.05))
    else:
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization()); model.add(LeakyReLU(negative_slope=0.05)); model.add(Dropout(0.2))
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()); model.add(LeakyReLU(negative_slope=0.05)); model.add(Dropout(0.1))

    model.add(Dense(1))
    
    optimizer = optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
    
    return model


def main():
    """Main function to load data and run ANN training."""
    print("\n-----------------------------------------------")
    print("       ARTIFICIAL NEURAL NETWORK TRAINING")
    print("-----------------------------------------------\n")

    try:
        X = np.load(CV_DATA_DIR / 'X_data.npy', allow_pickle=True)
        y = np.load(CV_DATA_DIR / 'y_data.npy', allow_pickle=True)
        groups = np.load(CV_DATA_DIR / 'groups.npy', allow_pickle=True)
        print(f"Loaded data with shapes: X={X.shape}, y={y.shape}, groups={groups.shape}")
    except FileNotFoundError:
        print("Error: Data files from smb_model_training.py not found in 'output/models/cv_data/'.")
        print("Please run smb_model_training.py first.")
        return

    print("Calculating sample weights to focus on extreme values...")
    kde = gaussian_kde(y)
    density = kde(y)
    sample_weights = 1 / (density + 1e-6)
    sample_weights = sample_weights / np.sum(sample_weights) * len(y)

    n_features = X.shape[1]
    
    print("\n--- Starting ANN Cross-Validation ---")
    logo = LeaveOneGroupOut()
    splits = logo.split(X, y, groups)
    
    y_pred_ann, y_true_ann = [], []
    fold_idx = 1
    
    for train_idx, test_idx in tqdm(splits, desc="ANN CV Folds", total=logo.get_n_splits(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = create_ann_model(n_features, is_final_model=False)
        
        mc = ModelCheckpoint(str(ANN_CV_DIR / f'best_model_fold_{fold_idx}.keras'), monitor='val_loss', mode='min', save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=0)
        
        model.fit(X_train_scaled, y_train,
                  sample_weight=sample_weights[train_idx],
                  validation_data=(X_test_scaled, y_test),
                  epochs=1000, batch_size=32, verbose=0, callbacks=[es, mc])
        
        best_model = load_model(str(ANN_CV_DIR / f'best_model_fold_{fold_idx}.keras'), custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error})
        
        fold_preds = best_model.predict(X_test_scaled).flatten()
        y_pred_ann.extend(fold_preds)
        y_true_ann.extend(y_test)
        
        fold_idx += 1
        K.clear_session()

    ann_r2 = r2_score(y_true_ann, y_pred_ann)
    ann_rmse = np.sqrt(mean_squared_error(y_true_ann, y_pred_ann))
    print(f"\nANN CV Results -> R2: {ann_r2:.3f}, RMSE: {ann_rmse:.3f} m.w.e.")

    print("\n--- Training Final Ensemble Models ---")
    if os.path.exists(ANN_ENSEMBLE_DIR):
        shutil.rmtree(ANN_ENSEMBLE_DIR)
    ANN_ENSEMBLE_DIR.mkdir()

    ensemble_size = 10
    scaler_full = StandardScaler().fit(X)
    X_scaled = scaler_full.transform(X)

    for i in tqdm(range(ensemble_size), desc="Training Ensemble"):
        model = create_ann_model(n_features, is_final_model=True)
        model.fit(X_scaled, y, 
                  sample_weight=sample_weights,
                  epochs=1500, batch_size=32, verbose=0)
        model.save(str(ANN_ENSEMBLE_DIR / f'ensemble_model_{i+1}.keras'))
        K.clear_session()
    
    print(f"\nSuccessfully trained and saved {ensemble_size} ensemble models.")
    print("Model training phase is complete.")


if __name__ == '__main__':
    main()


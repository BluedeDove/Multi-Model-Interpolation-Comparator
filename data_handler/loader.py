# data_handler/loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_handler.missing_creator import create_missingness
import os

def load_and_preprocess_data(config: dict, mode: str = "creating_mode"):
    """
    加载、预处理和划分数据。
    【已升级 V8 - 维度匹配最终修复版】:
    1. 严格根据 config 中的 n_features 调整数据，确保数据和模型维度绝对匹配。
    2. 自动处理单特征到多特征的复制。
    3. 对无法处理的维度冲突（如CSV有5列，config要求3列）抛出明确错误。
    """
    data_config = config['data']
    
    # --- 1. 数据加载 ---
    if not os.path.exists(data_config['path']):
        raise FileNotFoundError(f"Original data file not found at: {data_config['path']}")

    df_original = pd.read_csv(data_config['path'])
    
    df_to_process = None
    if mode == "loading_mode" and data_config.get("marked_path"):
        print("Running in 'loading_mode': Using a pre-masked data file.")
        marked_path = data_config.get("marked_path")
        if not os.path.exists(marked_path):
            raise FileNotFoundError(f"Marked data file not found at: {marked_path}")
        df_to_process = pd.read_csv(marked_path, skip_blank_lines=False)
    else:
        print("Running in 'creating_mode': Using original data.")
        df_to_process = df_original
        
    datetime_col = data_config.get('datetime_column')
    if datetime_col and datetime_col in df_to_process.columns:
        df_to_process = df_to_process.set_index(pd.to_datetime(df_to_process[datetime_col])).drop(columns=[datetime_col])

    # --- 2. 特征选择与维度校验 (关键修复) ---
    feature_cols = data_config.get('feature_columns') or df_to_process.columns.tolist()
    features_df = df_to_process[feature_cols].copy()
    
    expected_n_features = data_config['n_features']
    actual_n_features = features_df.shape[1]

    if actual_n_features != expected_n_features:
        print(f"⚠️ Warning: Feature count mismatch. Config expects {expected_n_features}, but data has {actual_n_features}.")
        # Case 1: 单特征 -> 多特征 (自动修复)
        if actual_n_features == 1 and expected_n_features > 1:
            print(f"-> Auto-fixing: Duplicating single feature column '{features_df.columns[0]}' to match {expected_n_features} features.")
            original_col = features_df.iloc[:, 0].copy()
            for i in range(1, expected_n_features):
                features_df[f'{original_col.name}_dup{i}'] = original_col
        # Case 2: 其他无法处理的冲突 (抛出错误)
        else:
            raise ValueError(f"Unrecoverable feature mismatch. Expected {expected_n_features} features but got {actual_n_features}. Please check your `feature_columns` and `n_features` in the config, or your CSV file.")

    # 此时，features_df 的列数保证与 expected_n_features 一致
    print(f"✅ Feature check passed. Data now has {features_df.shape[1]} features as expected.")

    # --- 3. 标准化 ---
    scaler = StandardScaler()
    # 只在有效数据点上 fit
    scaler.fit(features_df.dropna())
    # 在填充了NaN的完整数据上 transform
    data_scaled = scaler.transform(np.nan_to_num(features_df.values))
    # 恢复NaN
    missing_mask = np.isnan(features_df.values)
    data_scaled[missing_mask] = np.nan
    
    # --- 4. 窗口化与分割 ---
    n_steps = data_config['n_steps']
    if len(data_scaled) < n_steps:
        raise ValueError(f"Data length ({len(data_scaled)}) is less than n_steps ({n_steps}).")
    
    num_samples = len(data_scaled) - n_steps + 1
    samples = np.array([data_scaled[i:i+n_steps] for i in range(num_samples)])

    train_data, test_data = train_test_split(
        samples, 
        test_size=data_config.get('test_size', 0.2),
        shuffle=False
    )
    
    print(f"Data loaded and preprocessed. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    
    return train_data, test_data, scaler

# ... get_missing_data 函数保持不变 ...
def get_missing_data(train_data: np.ndarray, test_data: np.ndarray, config: dict):
    data_config = config['data']
    if data_config.get("marked_path"):
        train_data_missing, test_data_missing, scaler_missing = load_and_preprocess_data(config, "loading_mode")
    else:
        train_data_missing = create_missingness(train_data, config)
        test_data_missing = create_missingness(test_data, config)
        scaler_missing = None
    return train_data_missing, test_data_missing, scaler_missing

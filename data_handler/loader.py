# data_handler/loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(config: dict):
    """
    加载、预处理和划分数据。
    【已升级 V2】: 
    1. 能够智能处理有或没有时间戳列的数据。
    2. 自动处理单特征数据，通过复制列来避免复杂模型的稳定性问题。
    """
    data_config = config['data']
    
    df = pd.read_csv(data_config['path'])
    
    datetime_col = data_config.get('datetime_column')
    
    if datetime_col and datetime_col in df.columns:
        print(f"Found datetime column '{datetime_col}'. Parsing and setting as index.")
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        feature_df = df
    else:
        print("No datetime column specified or found. Using default integer index.")
        feature_df = df

    if data_config.get('feature_columns'):
        features = feature_df[data_config['feature_columns']]
    else:
        features = feature_df
        config['data']['feature_columns'] = features.columns.tolist()

    # --- 【新功能】自动处理单特征数据 ---
    if features.shape[1] == 1:
        print("⚠️ Detected single-feature data. Duplicating the feature column to ensure model stability.")
        original_col_name = features.columns[0]
        # 复制列
        features[f"{original_col_name}_dup"] = features[original_col_name]
        # 更新配置以反映新的特征数量和名称，这将自动传递给模型
        config['data']['n_features'] = 2
        config['data']['feature_columns'] = features.columns.tolist()
        print(f"Data now has {config['data']['n_features']} features: {config['data']['feature_columns']}")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features)

    n_steps = data_config['n_steps']
    if len(data_scaled) < n_steps:
        raise ValueError(
            f"Data length ({len(data_scaled)}) is less than n_steps ({n_steps}). "
            "Please decrease n_steps or use a larger dataset."
        )
    
    num_samples = len(data_scaled) - n_steps + 1
    samples = np.array([data_scaled[i:i+n_steps] for i in range(num_samples)])

    train_data, test_data = train_test_split(
        samples, 
        test_size=data_config.get('test_size', 0.2),
        shuffle=False
    )
    
    print(f"Data loaded and preprocessed. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    
    return train_data, test_data, scaler

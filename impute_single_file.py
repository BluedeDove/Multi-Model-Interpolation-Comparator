# impute_single_file.py

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

# 从你的项目中导入必要的模块
from models.pypots_wrappers import PyPOTSWrapper
from models.custom_models.my_lstm_imputer import MyLSTMImputer

def impute_entire_file(args):
    """
    使用预训练的模型对单个含有缺失值的CSV文件进行完整插补。
    """
    # 1. 加载配置文件
    print(f"🔄 Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found at {args.config}")
        return

    # 2. 找到指定的模型定义
    model_def = next((m for m in config['model_definitions'] if m['name'] == args.model_name), None)
    if not model_def:
        print(f"❌ Error: Model '{args.model_name}' not found in configuration file.")
        return
    
    # 提取与该模型训练时相关的数据参数 (以第一个task为准)
    # 这里的假设是，用于同一模型的n_steps等参数在不同task中是一致的
    data_config = config['tasks'][0]['data']
    n_steps = data_config['n_steps']
    n_features = data_config['n_features']
    feature_columns = data_config.get('feature_columns')

    # 3. 动态实例化模型
    print(f"🛠️ Instantiating model: {args.model_name}")
    device = torch.device(config['global_settings'].get('device', 'cpu'))
    
    # 将必要的参数注入到模型超参数中
    model_def['hyperparameters']['n_steps'] = n_steps
    model_def['hyperparameters']['n_features'] = n_features
    model_def['hyperparameters']['device'] = device
    
    imputer = None
    if model_def['type'] == 'pypots':
        imputer = PyPOTSWrapper(model_def['class_name'], model_def['hyperparameters'])
    elif model_def['type'] == 'custom':
        # 这里可以根据需要扩展，支持更多的自定义模型
        if model_def['class_name'] == 'MyLSTMImputer':
            imputer = MyLSTMImputer(**model_def['hyperparameters'])
        # else: ...
    
    if not imputer:
        print(f"❌ Error: Unknown model type '{model_def['type']}' or class '{model_def['class_name']}'.")
        return

    # 4. 加载预训练的模型权重
    print(f"💾 Loading pre-trained weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        return
    imputer.load(args.model_path)
    print("✅ Model loaded successfully.")

    # 5. 加载并预处理待插补的CSV文件
    print(f"📄 Loading data for imputation from: {args.input_csv}")
    df_missing = pd.read_csv(args.input_csv)
    
    if feature_columns:
        features_df = df_missing[feature_columns].copy()
    else:
        # 如果config中未指定，则使用所有列（除了可能的时间列）
        features_df = df_missing.copy()

    # 同样处理单特征复制的情况，以匹配模型输入
    if n_features > 1 and features_df.shape[1] == 1:
        print(f"⚠️ Detected single-feature data. Duplicating column to match model's expected input of {n_features} features.")
        original_col_name = features_df.columns[0]
        for i in range(1, n_features):
             features_df[f"{original_col_name}_dup{i}"] = features_df[original_col_name]

    original_data = features_df.values
    
    # 标准化数据 (与loader.py中的逻辑保持一致)
    scaler = StandardScaler()
    missing_mask_for_scaling = np.isnan(original_data)
    data_filled_temp = np.nan_to_num(original_data)
    data_scaled = scaler.fit_transform(data_filled_temp)
    data_scaled[missing_mask_for_scaling] = np.nan
    
    # 将整个序列划分为重叠的窗口/样本
    if len(data_scaled) < n_steps:
        print(f"❌ Error: Data length ({len(data_scaled)}) is smaller than the model's required window size `n_steps` ({n_steps}).")
        return
    
    windows = np.array([data_scaled[i:i+n_steps] for i in range(len(data_scaled) - n_steps + 1)])
    print(f"📊 Data preprocessed into {windows.shape[0]} overlapping windows.")

    # 6. 执行插补
    print("⏳ Performing imputation on all windows...")
    imputed_windows_scaled = imputer.impute(windows)
    print("✅ Imputation complete.")

    # 7. "拼接"重叠的窗口以重建完整序列
    # 我们通过对重叠部分取平均值来平滑地重建整个序列
    print("🧩 Stitching overlapping windows by averaging...")
    reconstructed_data_scaled = np.zeros_like(original_data, dtype=float)
    imputation_counts = np.zeros_like(original_data, dtype=float)

    for i, window in enumerate(imputed_windows_scaled):
        reconstructed_data_scaled[i : i + n_steps] += window
        imputation_counts[i : i + n_steps] += 1
    
    # 避免除以零
    imputation_counts[imputation_counts == 0] = 1
    reconstructed_data_scaled /= imputation_counts
    
    # 8. 逆标准化，恢复到原始数据尺度
    reconstructed_data_orig_scale = scaler.inverse_transform(reconstructed_data_scaled)

    # 9. 将插补值填入原始DataFrame的缺失位置
    final_df = df_missing.copy()
    missing_mask_for_filling = final_df[features_df.columns].isnull()
    
    # 创建一个与 features_df 相同列名的 DataFrame 用于赋值
    imputed_values_df = pd.DataFrame(reconstructed_data_orig_scale, columns=features_df.columns)
    
    final_df[features_df.columns] = final_df[features_df.columns].where(~missing_mask_for_filling, imputed_values_df)

    # 10. 保存结果
    final_df.to_csv(args.output_csv, index=False)
    print(f"🎉 Success! Imputed file saved to: {args.output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Forward-only Imputation Script for a single file.")
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file (e.g., experiment_basic.yaml).')
    
    parser.add_argument('--model-name', type=str, required=True,
                        help='The name of the model to use (e.g., "SAITS", "CSDI", "MyLSTM"). Must match a name in the config.')
                        
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the pre-trained model .pth file.')
                        
    parser.add_argument('--input-csv', type=str, required=True,
                        help='Path to the CSV file with missing values that you want to impute.')
                        
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Path where the fully imputed CSV file will be saved.')
                        
    args = parser.parse_args()
    
    impute_entire_file(args)


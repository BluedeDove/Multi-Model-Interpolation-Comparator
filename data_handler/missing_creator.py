# data_handler/missing_creator.py

import numpy as np
from pygrinder import mcar

def create_missingness(data: np.ndarray, config: dict):
    """
    在完整数据上制造缺失。
    【已升级 V2】: 智能处理单特征被复制的情况，确保缺失位置在所有重复特征中保持一致，
                   从而防止信息泄露。
    """
    missing_config = config['missingness']
    data_config = config['data']
    
    # --- 新增逻辑：检查是否为单特征复制情况 ---
    # loader.py 中会将单特征复制为 'feature_name' 和 'feature_name_dup'
    # 我们通过检查特征数量和命名约定来判断
    feature_cols = data_config.get('feature_columns', [])
    is_duplicated_single_feature = (
        data.shape[-1] == 2 and 
        len(feature_cols) == 2 and 
        feature_cols[1] == f"{feature_cols[0]}_dup"
    )

    if is_duplicated_single_feature:
        print("Detected duplicated single feature. Applying consistent missingness mask to prevent info leak.")
        
        # 1. 创建一个与单特征数据形状相同的临时数据副本
        #    形状为 [n_samples, n_steps, 1]
        single_feature_slice = data[:, :, 0:1].copy()

        # 2. 在这个单特征数据上制造缺失
        masked_single_feature = mcar(single_feature_slice, p=missing_config['rate'])

        # 3. 从制造完缺失的数据中提取布尔掩码（mask），True代表缺失
        #    形状为 [n_samples, n_steps, 1]
        missing_mask = np.isnan(masked_single_feature)

        # 4. 将这个 mask 应用到原始的、具有两个特征的完整数据上
        data_with_missing = data.copy()
        #    利用NumPy的广播机制，将mask从 (..., 1) 扩展到 (..., 2)
        #    使得两个特征在相同的位置上都被设置为 NaN
        data_with_missing[np.broadcast_to(missing_mask, data.shape)] = np.nan
        
    else:
        # 对于真实的多变量数据，维持原有逻辑
        print("Applying standard missingness pattern for multi-feature data.")
        data_with_missing = mcar(data, p=missing_config['rate'])

    print(f"Created {missing_config['rate']*100}% {missing_config['pattern']} missingness.")
    
    return data_with_missing

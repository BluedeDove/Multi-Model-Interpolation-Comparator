# evaluation/metrics.py

import numpy as np
# 导入已被证明可用的 pypots.nn.functional
from pypots.nn.functional import calc_mae, calc_mse, calc_mre

def calculate_metrics(imputed_data: np.ndarray, original_data: np.ndarray, data_with_missing: np.ndarray, metrics_to_calc: list) -> dict:
    """
    计算指定的评估指标。
    【已升级 V2 - 健壮性修复】: 
    在计算前，对模型输出的 imputed_data 也进行nan_to_num处理，
    防止因模型插补失败留下NaN而导致底层计算函数崩溃。
    """
    results = {}
    # indicating_mask 准确地标记了我们需要评估的位置
    indicating_mask = np.isnan(data_with_missing)
    
    # 【关键修复】确保所有传入计算函数的数据都不含NaN。
    # 即使模型插补失败在 imputed_data 中留下了NaN，也将其替换为0。
    # 这不会影响结果，因为 indicating_mask 会确保只在原始缺失的位置计算指标。
    imputed_data_no_nan = np.nan_to_num(imputed_data)
    original_data_no_nan = np.nan_to_num(original_data)

    if 'mae' in metrics_to_calc:
        results['mae'] = calc_mae(imputed_data_no_nan, original_data_no_nan, indicating_mask)
    if 'mse' in metrics_to_calc:
        results['mse'] = calc_mse(imputed_data_no_nan, original_data_no_nan, indicating_mask)
    if 'mre' in metrics_to_calc:
        results['mre'] = calc_mre(imputed_data_no_nan, original_data_no_nan, indicating_mask)
        
    return results

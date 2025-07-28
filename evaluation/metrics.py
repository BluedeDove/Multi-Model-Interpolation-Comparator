import numpy as np
from pypots.nn.functional import calc_mae, calc_mse, calc_mre # <- 修改这一行

def calculate_metrics(imputed_data: np.ndarray, original_data: np.ndarray, data_with_missing: np.ndarray, metrics_to_calc: list) -> dict:
    """
    计算指定的评估指标。

    Args:
        imputed_data: 模型插补后的数据.
        original_data: 制造缺失前的完整数据.
        data_with_missing: 含有缺失值的数据 (用于定位缺失位置).
        metrics_to_calc: 需要计算的指标列表, e.g., ['mae', 'mse'].

    Returns:
        一个包含所有计算指标结果的字典.
    """
    results = {}
    indicating_mask = np.isnan(data_with_missing)
    
    # 将原始数据中的nan转为0，因为calc_*函数不接受nan
    original_data_no_nan = np.nan_to_num(original_data)

    if 'mae' in metrics_to_calc:
        results['mae'] = calc_mae(imputed_data, original_data_no_nan, indicating_mask)
    if 'mse' in metrics_to_calc:
        results['mse'] = calc_mse(imputed_data, original_data_no_nan, indicating_mask)
    if 'mre' in metrics_to_calc:
        results['mre'] = calc_mre(imputed_data, original_data_no_nan, indicating_mask)
        
    return results

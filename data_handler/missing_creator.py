from pygrinder import mcar

def create_missingness(data, config: dict):
    """
    在完整数据上制造缺失。
    """
    missing_config = config['missingness']
    
    # 目前只演示最简单的 point missing (mcar)
    # PyGrinder 支持更多模式，可在此处扩展
    if missing_config['pattern'] == 'point':
        data_with_missing = mcar(data, p=missing_config['rate'])
    else:
        # TODO: 在此可以根据 pattern 调用 PyGrinder 的其他函数
        raise NotImplementedError(f"Missing pattern '{missing_config['pattern']}' is not implemented yet.")
    
    print(f"Created {missing_config['rate']*100}% {missing_config['pattern']} missingness.")
    
    return data_with_missing

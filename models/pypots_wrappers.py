# models/pypots_wrappers.py

import os
import numpy as np
import pypots.imputation as pypots_imputers
from .base_imputer import BaseImputer

class PyPOTSWrapper(BaseImputer):
    """
    一个包装器，用于将 PyPOTS 库中的模型适配到我们的 BaseImputer 接口。
    【已升级 V4 - 采用impute接口】:
    1. 改为调用更简洁、更健壮的 model.impute() 方法。
    2. 依然保留对 CSDI 等概率模型返回的4D输出的处理能力。
    3. 能够处理并统一 PyPOTS 库自动添加的 .pypots 文件后缀。
    """

    def __init__(self, model_class_name: str, hyperparameters: dict):
        model_class = getattr(pypots_imputers, model_class_name)
        self.model = model_class(**hyperparameters)
        self._model_name = model_class_name

    def _format_for_pypots(self, data: np.ndarray) -> dict:
        """ PyPOTS 需要特定格式的字典输入 """
        return {"X": data}

    def fit(self, train_data: np.ndarray):
        """ 训练 PyPOTS 模型 """
        train_set_pypots = self._format_for_pypots(train_data)
        self.model.fit(train_set=train_set_pypots)

    def impute(self, data: np.ndarray) -> np.ndarray:
        """ 
        使用 PyPOTS 模型进行插补。
        【关键更新】: 直接调用 model.impute() 并对结果进行维度检查。
        """
        imputation_set_pypots = self._format_for_pypots(data)
        
        # 直接调用 impute() 获取 NumPy 数组
        imputation_result = self.model.impute(imputation_set_pypots)

        # 依然检查是否为概率模型返回的4D结果
        if imputation_result.ndim == 4:
            # CSDI 的输出形状是 (n_samples, n_sampling_times, n_steps, n_features)
            print(f"INFO: Model '{self._model_name}' returned a 4D array with shape {imputation_result.shape}. "
                  "Taking the mean across the sampling dimension (axis=1).")
            # 沿 axis=1 求平均，将 (n_samples, n_sampling_times, ...) 变为 (n_samples, ...)
            imputation_result = np.mean(imputation_result, axis=1)

        return imputation_result

    def save(self, path: str):
        """
        保存 PyPOTS 模型，并处理其自动添加的 .pypots 后缀。
        """
        self.model.save(path)
        pypots_actual_path = path + '.pypots'
        if os.path.exists(pypots_actual_path):
            if os.path.exists(path):
                os.remove(path)
            os.rename(pypots_actual_path, path)
            print(f"Corrected PyPOTS model save path to: {path}")

    def load(self, path: str):
        """
        加载模型。因为 save 方法已经统一了文件名，所以这里直接加载即可。
        """
        self.model.load(path)

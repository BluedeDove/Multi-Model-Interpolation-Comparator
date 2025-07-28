# models/pypots_wrappers.py

import os # 导入 os 模块用于文件操作
import numpy as np
import pypots.imputation as pypots_imputers
from .base_imputer import BaseImputer

class PyPOTSWrapper(BaseImputer):
    """
    一个包装器，用于将 PyPOTS 库中的模型适配到我们的 BaseImputer 接口。
    【已升级】: 能够处理并统一 PyPOTS 库自动添加的 .pypots 文件后缀。
    """

    def __init__(self, model_class_name: str, hyperparameters: dict):
        # 通过类名字符串从 pypots.imputation 模块获取模型类
        model_class = getattr(pypots_imputers, model_class_name)
        self.model = model_class(**hyperparameters)

    def _format_for_pypots(self, data: np.ndarray) -> dict:
        """ PyPOTS 需要特定格式的字典输入 """
        return {"X": data}

    def fit(self, train_data: np.ndarray):
        """ 训练 PyPOTS 模型 """
        train_set_pypots = self._format_for_pypots(train_data)
        self.model.fit(train_set=train_set_pypots)

    def impute(self, data: np.ndarray) -> np.ndarray:
        """ 使用 PyPOTS 模型进行插补 """
        imputation_set_pypots = self._format_for_pypots(data)
        imputed_data_dict = self.model.predict(imputation_set_pypots)
        return imputed_data_dict['imputation']

    def save(self, path: str):
        """
        保存 PyPOTS 模型，并处理其自动添加的 .pypots 后缀。
        """
        # PyPOTS 会自动在 `path` 后面添加 ".pypots" 后缀。
        # 例如，如果 path 是 "saved_models/SAITS.pth",
        # 它会创建 "saved_models/SAITS.pth.pypots"。
        # 我们需要把它重命名回 "saved_models/SAITS.pth"。

        # 1. 先调用 PyPOTS 的原生保存方法
        self.model.save(path)

        # 2. 计算出 PyPOTS 实际创建的文件名
        pypots_actual_path = path + '.pypots'

        # 3. 检查这个文件是否存在，然后将其重命名为我们期望的路径
        if os.path.exists(pypots_actual_path):
            # 如果目标路径（不带.pypots）已存在，先删除，以防os.rename报错
            if os.path.exists(path):
                os.remove(path)
            os.rename(pypots_actual_path, path)
            print(f"Corrected PyPOTS model save path to: {path}")

    def load(self, path: str):
        """
        加载模型。因为 save 方法已经统一了文件名，所以这里直接加载即可。
        """
        # PyPOTS 的 load 方法可以直接加载我们统一后的 .pth 文件
        self.model.load(path)


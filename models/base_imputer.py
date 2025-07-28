# models/base_imputer.py

from abc import ABC, abstractmethod
import numpy as np

class BaseImputer(ABC):
    """
    所有插补模型的抽象基类。
    定义了所有模型必须遵循的统一接口。
    """

    @abstractmethod
    def fit(self, train_data: np.ndarray):
        """
        训练模型。
        
        Args:
            train_data (np.ndarray): 用于训练的（可能含有缺失值）数据, shape [n_samples, n_steps, n_features]
        """
        pass

    @abstractmethod
    def impute(self, data: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型对新数据进行插补。

        Args:
            data (np.ndarray): 需要被插补的数据, shape [n_samples, n_steps, n_features]

        Returns:
            np.ndarray: 插补完成后的数据, shape [n_samples, n_steps, n_features]
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        将训练好的模型状态保存到文件。

        Args:
            path (str): 模型保存的完整路径。
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        从文件加载模型状态。

        Args:
            path (str): 模型所在的完整路径。
        """
        pass

# core/experiment_runner.py

import os
import json
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import gc

from data_handler.loader import load_and_preprocess_data
from data_handler.loader import get_missing_data
from models.pypots_wrappers import PyPOTSWrapper
from models.custom_models.my_lstm_imputer import MyLSTMImputer
from evaluation.metrics import calculate_metrics
from evaluation.visualizer import plot_metrics_comparison, plot_all_models_on_sample, plot_all_features_for_one_model

class ExperimentRunner:
    def __init__(self, config: dict):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = self.config['experiment_name']
        self.result_path = os.path.join(
            "results", f"{self.experiment_name}_{self.timestamp}"
        )
        self._setup_directories_and_logging()

    def _setup_directories_and_logging(self):
        # ... (此方法无变化)
        self.imputed_data_path = os.path.join(self.result_path, "imputed_data")
        self.plots_path = os.path.join(self.result_path, "plots")
        os.makedirs(self.imputed_data_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)

        self.logger = logging.getLogger(f"{self.experiment_name}_{self.timestamp}")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(os.path.join(self.result_path, "experiment_log.log"), encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def _get_model(self, model_config: dict):
        # ... (此方法无变化)
        model_type = model_config['type']
        model_name = model_config['name']
        hyperparams = model_config.get('hyperparameters', {})
        hyperparams['device'] = self.config['settings']['device']

        try:
            if model_type == 'pypots':
                return PyPOTSWrapper(model_config['class_name'], hyperparams)
            elif model_type == 'custom':
                if model_config['class_name'] == 'MyLSTMImputer':
                    return MyLSTMImputer(**hyperparams)
                else:
                    raise ValueError(f"Unknown custom model class: {model_config['class_name']}")
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except TypeError as e:
            self.logger.error(f"[Error] Failed to initialize model '{model_name}'.")
            self.logger.error(f"   Reason: A required hyperparameter might be missing in your YAML file.")
            self.logger.error(f"   Please check the definition for '{model_name}' in 'experiment_basic.yaml' against the model's documentation.")
            self.logger.error(f"   Original Error: {e}")
            raise ValueError(f"Initialization failed for model '{model_name}'.") from e

    def _inverse_scale_data(self, data_scaled: np.ndarray, scaler) -> np.ndarray:
        """
        【新功能】辅助函数，用于将3D的标准化数据逆变换回原始量纲。
        """
        if data_scaled is None:
            return None
        
        n_samples, n_steps, n_features = data_scaled.shape
        
        # 1. 将数据从3D (n_samples, n_steps, n_features) reshape为2D
        data_reshaped = data_scaled.reshape(-1, n_features)
        
        # 2. 使用scaler进行逆变换
        data_orig_scale_reshaped = scaler.inverse_transform(data_reshaped)
        
        # 3. 将数据恢复为原来的3D形状
        data_orig_scale = data_orig_scale_reshaped.reshape(n_samples, n_steps, n_features)
        
        return data_orig_scale

    def run(self):
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        
        stable_model_dir = os.path.join("saved_models", self.experiment_name)
        os.makedirs(stable_model_dir, exist_ok=True)
        self.logger.info(f"Trainable models will be saved to and loaded from: {stable_model_dir}")
        
        self.logger.info("Step 1: Loading and preprocessing data...")
        # 注意：这里的 train_data_complete 和 test_data_complete 都是标准化后的数据
        train_data_complete, test_data_complete, scaler = load_and_preprocess_data(self.config)

        final_n_steps = self.config['data']['n_steps']
        final_n_features = self.config['data']['n_features']
        self.logger.info(f"Final data parameters for model initialization: n_steps={final_n_steps}, n_features={final_n_features}")

        feature_names = self.config['data'].get('feature_columns', [f"Feature_{i+1}" for i in range(final_n_features)])

        self.logger.info("Step 2: Creating missing values in the test set...")
        train_data_missing_for_fit, test_data_missing, _ = get_missing_data(train_data_complete, test_data_complete, self.config)

        metrics_summary = []
        all_imputed_results_orig_scale = {} # 存储原始量纲的插补结果
        all_run_hyperparams = {} 

        model_definitions = self.config['model_definitions']
        annotation_keys = self.config['evaluation'].get('plot_annotation', [])
        plot_settings = self.config['evaluation'].get('plot_settings', {})

        for model_name in self.config['models_to_run']:
            model = None
            self.logger.info(f"--- Processing model: {model_name} ---")

            try:
                # ... (模型配置和初始化逻辑无变化) ...
                if model_name not in model_definitions:
                    self.logger.warning(f"Model '{model_name}' not found in definitions. Skipping.")
                    continue
                
                model_config = model_definitions[model_name].copy()
                model_config['name'] = model_name

                if 'hyperparameters' not in model_config:
                    model_config['hyperparameters'] = {}
                model_config['hyperparameters']['n_steps'] = final_n_steps
                model_config['hyperparameters']['n_features'] = final_n_features
                all_run_hyperparams[model_name] = model_config['hyperparameters']
                
                hyperparams_save_path = os.path.join(self.result_path, "run_hyperparameters.json")
                with open(hyperparams_save_path, 'w', encoding='utf-8') as f:
                    json.dump(all_run_hyperparams, f, indent=4, ensure_ascii=False)
                
                model = self._get_model(model_config)

                # ... (模型加载、训练、保存逻辑无变化) ...
                model_save_path = os.path.join(stable_model_dir, f"{model_name}.pth")
                should_load = self.config['settings'].get('load_if_exists', False)
                fit_time = 0

                if should_load and os.path.exists(model_save_path):
                    self.logger.info(f"Found existing model file. Loading from: {model_save_path}")
                    model.load(model_save_path)
                    self.logger.info(f"Model '{model_name}' loaded successfully.")
                else:
                    if should_load:
                        self.logger.info(f"No existing model file found at {model_save_path}. Training a new one.")
                    self.logger.info(f"Fitting {model_name}...")
                    start_time = time.time()
                    model.fit(train_data_missing_for_fit)
                    fit_time = time.time() - start_time
                    self.logger.info(f"{model_name} fitting finished in {fit_time:.2f} seconds.")
                    self.logger.info(f"Saving newly trained model to: {model_save_path}")
                    model.save(model_save_path)

                self.logger.info(f"Imputing with {model_name}...")
                start_time = time.time()
                # imputed_data 是在标准化尺度上的结果
                imputed_data_scaled = model.impute(test_data_missing)
                impute_time = time.time() - start_time
                self.logger.info(f"{model_name} imputation finished in {impute_time:.2f} seconds.")

                # --- 【新功能】将所有相关数据逆变换回原始量纲 ---
                self.logger.info("Reverting data to original scale for evaluation and plotting...")
                imputed_data_orig_scale = self._inverse_scale_data(imputed_data_scaled, scaler)
                test_data_complete_orig_scale = self._inverse_scale_data(test_data_complete, scaler)
                test_data_missing_orig_scale = self._inverse_scale_data(test_data_missing, scaler)
                
                # 将原始量纲的数据存入字典和文件
                all_imputed_results_orig_scale[model_name] = imputed_data_orig_scale
                np.save(os.path.join(self.imputed_data_path, f"{model_name}_imputed_orig_scale.npy"), imputed_data_orig_scale)
                
                # --- 使用原始量纲的数据进行评估和绘图 ---
                
                # 如果是单特征被复制的情况，评估和绘图时只使用第一列
                current_feature_names = feature_names
                final_imputed = imputed_data_orig_scale
                final_complete = test_data_complete_orig_scale
                final_missing = test_data_missing_orig_scale

                is_single_feature_duplicated = (self.config['data'].get('n_features') == 1 and final_imputed.shape[2] == 2)
                if is_single_feature_duplicated:
                    self.logger.info("Reverting duplicated single-feature data back to its original shape for results.")
                    current_feature_names = [feature_names[0]]
                    final_imputed = final_imputed[:, :, :1]
                    final_complete = final_complete[:, :, :1]
                    final_missing = final_missing[:, :, :1]

                self.logger.info(f"Evaluating {model_name} on original-scale data...")
                metrics = calculate_metrics(
                    final_imputed, final_complete, final_missing, self.config['evaluation']['metrics']
                )
                metrics['model'] = model_name
                metrics['fit_time_sec'] = fit_time
                metrics['impute_time_sec'] = impute_time
                metrics_summary.append(metrics)
                self.logger.info(f"Metrics for {model_name}: {metrics}")

                summary_df_so_far = pd.DataFrame(metrics_summary)
                summary_df_so_far.to_csv(os.path.join(self.result_path, "imputation_summary.csv"), index=False)
                self.logger.info(f"Updated metrics summary file with results for {model_name}.")

                self.logger.info(f"Generating multi-feature plot for {model_name} on original-scale data...")
                plot_all_features_for_one_model(
                    original=final_complete[0], missing=final_missing[0], imputed=final_imputed[0],
                    feature_names=current_feature_names, model_name=model_name,
                    save_path=os.path.join(self.plots_path, f"multi_feature_plot_{model_name}.png"),
                    hyperparameters=model_config['hyperparameters'],
                    annotation_keys=annotation_keys,
                    plot_settings=plot_settings
                )
            
            except Exception as e:
                self.logger.error(f"An error occurred while processing model '{model_name}'. Skipping to the next one.", exc_info=True)
                continue
            
            finally:
                # --- 【关键修改】无论成功或失败，都在循环结束时执行大扫除 ---
                if model is not None:
                    self.logger.info(f"Cleaning up resources for model '{model_name}'...")
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    self.logger.info("Cleanup complete.")

        # ... (后续的汇总绘图逻辑无变化，因为它现在使用的是原始量纲的数据) ...
        self.logger.info("--- All models processed. Finalizing results. ---")
        self.logger.info(f"Full hyperparameters for this run saved to: {hyperparams_save_path}")

        summary_df = pd.DataFrame(metrics_summary)
        if not summary_df.empty:
            self.logger.info("Final metrics summary:")
            self.logger.info("\n" + summary_df.to_string())
            
            plot_metrics_comparison(
                summary_df, save_path=os.path.join(self.plots_path, "comparison_metrics.png")
            )
            self.logger.info("Metrics comparison plot saved.")
            
            # 这里的 all_imputed_results_orig_scale 已经包含了原始量纲的数据
            plot_all_models_on_sample(
                original=test_data_complete_orig_scale[0, :, 0], 
                missing=test_data_missing_orig_scale[0, :, 0],
                all_imputed={name: data[0, :, 0] for name, data in all_imputed_results_orig_scale.items()},
                save_path=os.path.join(self.plots_path, "comparison_imputation_all_models.png")
            )
            self.logger.info("Combined plot saved.")

        self.logger.info("Experiment finished.")

# evaluation/visualizer.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import itertools

# ... (plot_metrics_comparison 和 plot_all_models_on_sample 无变化) ...
def plot_metrics_comparison(summary_df: pd.DataFrame, save_path: str):
    plt.style.use('seaborn-v0_8-whitegrid')
    metrics = [col for col in summary_df.columns if col not in ['model', 'fit_time_sec', 'impute_time_sec']]
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5), sharey=False)
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.barplot(x='model', y=metric, data=summary_df, ax=ax, palette='viridis')
        ax.set_title(f'Comparison of {metric.upper()}', fontsize=14)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_all_models_on_sample(original: np.ndarray, missing: np.ndarray, all_imputed: dict, save_path: str):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(original, color='gray', linestyle='--', linewidth=2, label='Original Data', zorder=1)
    colors = itertools.cycle(plt.get_cmap('viridis')(np.linspace(0, 1, len(all_imputed))))
    for model_name, imputed_data in all_imputed.items():
        ax.plot(imputed_data, color=next(colors), marker='o', markersize=4, linestyle='-', label=f'Imputed ({model_name})', zorder=2)
    missing_indices = np.where(np.isnan(missing))[0]
    ax.scatter(missing_indices, original[missing_indices], color='red', s=120, marker='x', label='Missing Points', zorder=3)
    ax.set_title('All Models Imputation Result for a Single Sample', fontsize=16)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path)
    plt.close()

def plot_all_features_for_one_model(
    original: np.ndarray, 
    missing: np.ndarray, 
    imputed: np.ndarray, 
    feature_names: list, 
    model_name: str, 
    save_path: str,
    hyperparameters: dict = None,
    annotation_keys: list = None,
    # 【新功能】接收图表显示设置
    plot_settings: dict = None
):
    """
    为单个模型生成一个多面板图，并根据配置进行标注和自定义。
    """
    if plot_settings is None:
        plot_settings = {}
        
    plt.style.use('seaborn-v0_8-whitegrid')
    n_features = original.shape[1]
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]

    # 【新功能】从配置中获取标签和单位，提供默认值
    x_label = plot_settings.get('x_label', 'Time Step')
    y_unit = plot_settings.get('y_unit', '')
    y_label_text = plot_settings.get('y_label', 'Value') # 用于总标题

    for i in range(n_features):
        ax = axes[i]
        feature_name = feature_names[i]
        
        # 组合特征名和单位作为Y轴子图标签
        y_subplot_label = f"{feature_name} ({y_unit})" if y_unit else feature_name

        original_feature, missing_feature, imputed_feature = original[:, i], missing[:, i], imputed[:, i]

        ax.plot(original_feature, color='gray', linestyle='--', label='Original', zorder=1)
        ax.plot(imputed_feature, color='green', marker='o', markersize=3, linestyle='-', label='Imputed', zorder=2)
        missing_indices = np.where(np.isnan(missing_feature))[0]
        ax.scatter(missing_indices, original_feature[missing_indices], color='red', s=80, marker='x', label='Missing Points', zorder=3)

        ax.set_ylabel(y_subplot_label, fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 使用自定义的X轴标签
    axes[-1].set_xlabel(x_label, fontsize=12)
    
    # 将模型名和自定义的Y轴总标签组合成图表总标题
    fig.suptitle(f'Imputation Result for {model_name} on {y_label_text}', fontsize=16, weight='bold')
    
    if hyperparameters and annotation_keys:
        annotation_text = "Key Hyperparameters:\n"
        included_keys = 0
        for key in annotation_keys:
            if key in hyperparameters:
                annotation_text += f" - {key}: {hyperparameters[key]}\n"
                included_keys += 1
        
        if included_keys > 0:
            fig.text(0.99, 0.97, annotation_text.strip(),
                     transform=fig.transFigure,
                     fontsize=9,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) 
    plt.savefig(save_path)
    plt.close()

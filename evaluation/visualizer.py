# evaluation/visualizer.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import itertools

def plot_metrics_comparison(summary_df: pd.DataFrame, save_path: str):
    """
    将单次实验中所有模型的评估指标绘制成条形图进行对比。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    metrics = [col for col in summary_df.columns if col not in ['model', 'fit_time_sec', 'impute_time_sec']]
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6), sharey=False)
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.barplot(x='model', y=metric, data=summary_df.sort_values(by=metric, ascending=True), ax=ax, palette='viridis')
        ax.set_title(f'Comparison of {metric.upper()}', fontsize=15, weight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(True, which='both', linestyle=':', linewidth=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_all_features_for_one_model(
    original: np.ndarray, 
    missing: np.ndarray, 
    imputed: np.ndarray, 
    feature_names: list, 
    model_name: str, 
    save_path: str,
    hyperparameters: dict = None,
    annotation_keys: list = None,
    plot_settings: dict = None
):
    """
    为单个模型生成一个多面板图，采纳了学术界清晰的"线+点"可视化风格。
    """
    plt.style.use('default') 
    
    n_features = original.shape[1]
    fig_height = max(5, 4 * n_features) 
    fig, axes = plt.subplots(n_features, 1, figsize=(12, fig_height), sharex=True, squeeze=False)
    axes = axes.flatten()

    if plot_settings is None:
        plot_settings = {}
        
    x_label = plot_settings.get('x_label', 'Time Step')
    y_unit = plot_settings.get('y_unit', '')
    y_label_text = plot_settings.get('y_label', 'Value')

    for i in range(n_features):
        ax = axes[i]
        feature_name = feature_names[i]
        y_subplot_label = f"{feature_name} ({y_unit})" if y_unit else feature_name

        t = np.arange(len(original))
        ground_truth = original[:, i]
        imputed_values = imputed[:, i]
        missing_mask = np.isnan(missing[:, i])

        ax.plot(t, ground_truth, color='blue', linestyle='--', linewidth=1.2, label='Real')
        ax.plot(t[missing_mask], imputed_values[missing_mask], 
                'o', color='red', markersize=5, alpha=0.8, label='Imputation')

        ax.set_ylabel(y_subplot_label, fontsize=12)
        ax.legend(loc='upper right', frameon=False, ncol=2)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.set_ylim(bottom=np.min(original))

    axes[-1].set_xlabel(x_label, fontsize=12)
    fig.suptitle(f'Imputation Result for {model_name}', fontsize=16, weight='bold')
    
    if hyperparameters and annotation_keys:
        annotation_text = "Key Hyperparameters:\n"
        included_keys = 0
        for key in annotation_keys:
            if key in hyperparameters:
                annotation_text += f" - {key}: {hyperparameters[key]}\n"
                included_keys += 1
        if included_keys > 0:
            fig.text(0.99, 0.97, annotation_text.strip(), transform=fig.transFigure, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_all_models_on_sample(original: np.ndarray, missing: np.ndarray, all_imputed: dict, save_path: str):
    """
    【全新 V3 - 修复可视化陷阱】
    在单个样本上对比所有模型的插补结果。
    只在缺失点上绘制各模型的插补值（用不同标记），而不是绘制完整的曲线。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # 1. 绘制原始数据和标记缺失位置
    ax.plot(original, color='gray', linestyle='--', linewidth=1.5, label='Original Data', zorder=1)
    missing_indices = np.where(np.isnan(missing))[0]
    # 在原始数据上标记出缺失点的位置，但让它更柔和
    ax.scatter(missing_indices, original[missing_indices], 
               edgecolor='red', facecolor='none', s=120, marker='o', 
               label='Missing Locations', zorder=2, alpha=0.7, linewidth=1.5)

    # 2. 定义高区分度的样式循环器
    styles = itertools.cycle([
        {'color': 'C0', 'marker': 'P'},  # Blue, Plus (filled)
        {'color': 'C1', 'marker': 'X'},  # Orange, X (filled)
        {'color': 'C2', 'marker': 'D'},  # Green, Diamond
        {'color': 'C3', 'marker': 's'},  # Red, Square
        {'color': 'C4', 'marker': '^'},  # Purple, Triangle Up
        {'color': 'C5', 'marker': 'v'},  # Brown, Triangle Down
        {'color': 'C6', 'marker': '*'},  # Pink, Star
    ])
    
    # 3. 【关键改动】循环绘制每个模型的插补结果，只在缺失点上画标记
    for model_name, imputed_data in all_imputed.items():
        style = next(styles)
        
        # 只在缺失点(missing_indices)处，获取对应模型的插补值(imputed_data[missing_indices])并绘图
        ax.plot(missing_indices, imputed_data[missing_indices],
                label=f'Imputed ({model_name})', 
                linestyle='None', # <-- 不绘制连接线
                markersize=8,
                alpha=0.9,
                zorder=3,
                **style)

    ax.set_title('Comparison of Imputed Values at Missing Locations', fontsize=16)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='medium')
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, dpi=300) # dpi可以适当降低，800太高了
    plt.close(fig)


def plot_metrics_evolution(csv_path: str, x_axis_col: str, metrics: list, save_dir: str):
    """
    【全新函数】读取一个包含多次实验结果的CSV文件，绘制模型性能随某个变量变化的曲线图。
    """
    df = pd.read_csv(csv_path)
    models = df['model'].unique()
    
    styles = itertools.cycle([
        {'color': 'C0', 'marker': 'o', 'linestyle': '-'},
        {'color': 'C1', 'marker': 's', 'linestyle': '--'},
        {'color': 'C2', 'marker': '^', 'linestyle': '-.'},
        {'color': 'C3', 'marker': 'd', 'linestyle': ':'},
        {'color': 'C4', 'marker': 'x', 'linestyle': '-'},
        {'color': 'C5', 'marker': '+', 'linestyle': '--'},
        {'color': 'C6', 'marker': '*', 'linestyle': '-.'},
    ])

    for metric in metrics:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))
        
        style_map = {model: next(styles) for model in models}

        for model in models:
            model_df = df[df['model'] == model].sort_values(by=x_axis_col)
            ax.plot(model_df[x_axis_col], model_df[metric], 
                    label=model, 
                    **style_map[model],
                    markersize=7,
                    linewidth=2)

        ax.set_xlabel(x_axis_col.replace('_', ' ').title(), fontsize=14)
        ax.set_ylabel(metric.upper(), fontsize=14)
        ax.set_title(f'{metric.upper()} vs. {x_axis_col.replace("_", " ").title()}', fontsize=16, weight='bold')
        
        ax.legend(loc='upper left', ncol=2, fontsize='medium')
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, f'evolution_{metric}_vs_{x_axis_col}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Evolution plot saved to: {save_path}")

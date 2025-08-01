# generate_evolution_plots.py

import os
import glob
import pandas as pd
from evaluation.visualizer import plot_metrics_evolution

def main():
    """
    自动搜集所有实验结果，合并它们，并生成性能演变图。
    """
    print("Starting post-experiment analysis...")
    
    # 1. 定义结果文件夹的基础路径和要分析的实验前缀
    results_dir = "results"
    experiment_prefix = "exp_rate_"

    # 2. 找到所有相关的 summary 文件
    summary_files = glob.glob(os.path.join(results_dir, f"{experiment_prefix}*", "imputation_summary.csv"))
    
    if not summary_files:
        print(f"Error: No summary files found in '{results_dir}' with prefix '{experiment_prefix}'.")
        print("Please run the batch experiments first.")
        return

    print(f"Found {len(summary_files)} summary files to process.")

    # 3. 读取所有summary文件，并合并成一个大的DataFrame
    all_summaries = []
    for f_path in summary_files:
        df = pd.read_csv(f_path)
        # 从文件夹路径中提取自变量（缺失率）
        try:
            # 例如, 从 "results\exp_rate_0.1_20250729_..." 中提取 "0.1"
            rate_str = f_path.split(experiment_prefix)[1].split('_')[0]
            df['missing_rate'] = float(rate_str)
            all_summaries.append(df)
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse missing rate from path: {f_path}. Skipping. Error: {e}")
            
    if not all_summaries:
        print("Error: Failed to parse any summary files. Aborting.")
        return

    master_df = pd.concat(all_summaries, ignore_index=True)
    
    # 4. 保存合并后的主数据文件，方便未来分析
    master_csv_path = os.path.join(results_dir, "master_summary.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"Master summary saved to: {master_csv_path}")

    # 5. 调用我们的绘图函数
    plot_metrics_evolution(
        csv_path=master_csv_path,
        x_axis_col='missing_rate',
        metrics=['mae', 'mse'], # 确保这些指标在你的CSV中
        save_dir=os.path.join(results_dir, 'evolution_plots')
    )

if __name__ == '__main__':
    main()

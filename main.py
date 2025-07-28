# main.py

import argparse
import yaml
from core.experiment_runner import ExperimentRunner

def main():
    """
    整个工具包的入口函数，支持批量实验。
    """
    parser = argparse.ArgumentParser(description="Universal Time-Series Imputation Toolkit")
    parser.add_argument('--config', type=str, required=True, help='Path to the batch experiment configuration file.')
    args = parser.parse_args()

    # 1. 加载批处理配置文件
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            batch_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return

    # 2. 提取模型定义库和全局设置
    model_definitions = {m['name']: m for m in batch_config['model_definitions']}
    global_settings = batch_config.get('global_settings', {})

    # 3. 遍历任务列表，为每个任务运行一个实验
    for i, task_config in enumerate(batch_config['tasks']):
        print("\n" + "="*50)
        print(f"🚀 Starting Task {i+1}/{len(batch_config['tasks'])}: {task_config['experiment_name']}")
        print("="*50 + "\n")
        
        # 组合单个实验的完整配置
        experiment_config = task_config.copy()
        experiment_config['settings'] = {**global_settings, **task_config.get('settings', {})}
        
        # 【已修改】只传递模型名称列表和完整的定义库，不再提前注入参数。
        # 参数注入的步骤将移至 ExperimentRunner 中，在数据加载完成后进行。
        experiment_config['models_to_run'] = task_config['models_to_run']
        experiment_config['model_definitions'] = model_definitions
        
        # 运行实验
        runner = ExperimentRunner(experiment_config)
        runner.run()
        print(f"✅ Task '{task_config['experiment_name']}' finished successfully.")
        print(f"📊 Results saved to: {runner.result_path}")

if __name__ == '__main__':
    main()

import argparse
import numpy as np
from scipy.io import savemat
import os

parser = argparse.ArgumentParser(description="Convert .npy files to .mat format")
parser.add_argument('input_file', type=str, help='Path to the input .npy file')
args = parser.parse_args()

if not args.input_file.endswith('.npy'):
    raise ValueError("Input file must be a .npy file")

np_file = np.load(args.input_file, allow_pickle=True)

dirname = os.path.dirname(args.input_file)  # 获取目录
basename = os.path.basename(args.input_file)  # 获取文件名（带.npy）
mat_basename = os.path.splitext(basename)[0] + ".mat"  # 修改扩展名
output_file = os.path.join(dirname, mat_basename)  # 拼接完整路径

savemat(output_file, {'data': np_file})
print(f"Converted {args.input_file} to {output_file}")

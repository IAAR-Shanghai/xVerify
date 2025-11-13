#!/usr/bin/env python3
import os
import json
import argparse
import re
from collections import defaultdict

def extract_model_name(filename):
    """Extract model name from the filename more robustly."""
    # 包含所有可能的模型名称
    known_models = [
        'Qwen2_5-7B', 
        'Llama-3_1-8B-Instruct', 
        'DeepSeek-R1-Distill-Qwen-7B',
        'Qwen2.5-7B-Math-Verify',
        'Llama-3_1-8B-Instruct-Math-Verify',
        'Llama-3_1-8B-Instruct-xVerify',
        'Qwen2.5-7B-xVerify'
    ]
    
    # 按长度降序排列模型名称，这样会先匹配较长的模型名称
    known_models.sort(key=len, reverse=True)
    
    # 检查是否包含任何已知模型名称
    for model in known_models:
        if model in filename:
            return model
    
    # 使用正则表达式从文件名中提取模型信息
    # 例如从 0_shot_cot_AgNews_Llama-3_1-8B-Instruct_250_20250506_211352.json 中提取 Llama-3_1-8B-Instruct
    match = re.search(r'(?:_)([A-Za-z0-9\-]+(?:_[A-Za-z0-9\-]+)+)(?:_\d+)', filename)
    if match:
        return match.group(1)
    
    # 如果上述方法都失败，尝试基于下划线分割
    parts = filename.split('_')
    if len(parts) >= 5:  # 确保有足够的部分
        # 尝试从第4个部分开始提取模型名称
        for i in range(3, min(6, len(parts))):
            candidate = parts[i]
            if any(model_part in candidate for model_part in ['Llama', 'Qwen', 'DeepSeek']):
                # 检查下一个部分是否是模型名称的一部分
                if i+1 < len(parts) and any(c in parts[i+1] for c in ['7B', '8B']):
                    return f"{candidate}_{parts[i+1]}"
                return candidate
    
    return "unknown_model"

def parse_dataset_name(filename):
    """从文件名中提取数据集名称"""
    datasets = ['CMNLI', 'MMLU-Pro', 'AIME_2024', 'SimpleQA', 'DROP', 'MATH-500', 'AgNews', 'GPQA']
    for dataset in datasets:
        if dataset in filename:
            return dataset
    return "unknown_dataset"

def merge_outputs(input_dir, output_dir=None):
    """根据模型名合并JSON文件，只保留results字段。"""
    
    if not output_dir:
        output_dir = "./merged_outputs"
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 按模型名称分组结果
    model_results = defaultdict(list)
    model_file_map = defaultdict(set)  # 跟踪每个模型处理过的文件
    dataset_counts = defaultdict(lambda: defaultdict(int))  # 跟踪每个模型的数据集数量
    
    # 获取目录中的所有JSON文件并正常排序（不反向）
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for filename in json_files:
        file_path = os.path.join(input_dir, filename)
        
        # 提取模型名称和数据集
        model_name = extract_model_name(filename)
        dataset_name = parse_dataset_name(filename)
        
        # 检查是否已经处理过该模型+数据集组合
        file_key = f"{model_name}_{dataset_name}"
        if file_key in model_file_map:
            print(f"跳过重复的模型+数据集组合: {filename}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'results' in data and isinstance(data['results'], list):
                    results_count = len(data['results'])
                    if results_count > 0:
                        model_results[model_name].extend(data['results'])
                        model_file_map.add(file_key)
                        dataset_counts[model_name][dataset_name] += results_count
                        print(f"添加 {results_count} 条结果从 {filename} 到模型 {model_name} (数据集: {dataset_name})")
                    else:
                        print(f"警告: {filename} 包含空的结果列表")
                else:
                    print(f"警告: {filename} 中未找到有效的 'results' 字段")
        except Exception as e:
            print(f"处理 {file_path} 时出错: {e}")

    # 为每个模型写入合并的结果
    for model_name, results in model_results.items():
        if not results:  # 跳过空结果
            print(f"警告: 模型 {model_name} 没有结果可写入")
            continue
            
        output_file = os.path.join(output_dir, f"{model_name}_merged.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"已合并 {model_name} 的结果:")
        for dataset, count in dataset_counts[model_name].items():
            print(f"  - {dataset}: {count} 条结果")
        print(f"  总计: {len(results)} 条结果已写入 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='按模型名称合并JSON输出文件。')
    parser.add_argument('--input_dir', default="/mnt/public/code/qingchen/xVerify-Achieved/eval/outputs",
                        help='包含要合并的JSON文件的目录')
    parser.add_argument('--output_dir', default=None,
                        help='写入合并文件的目录(默认为input_dir的merged_outputs子目录)')
    
    args = parser.parse_args()
    merge_outputs(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
import os
import json
import pandas as pd
from collections import defaultdict
import glob

def analyze_model_results():
    # 结果存储结构初始化
    results = {}
    
    # 寻找final_outputs下的所有结果文件
    result_files = glob.glob('final_outputs/*.json')
    
    # 追踪所有出现的数据集名称
    all_datasets = set()
    
    # 处理每个模型文件
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace('.json', '')
        print(f"处理模型: {model_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 初始化该模型的结果统计
        model_results = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))
        
        # 统计指标与human_judge的一致性
        consistency_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'consistent': 0}))
        
        # 处理每条数据
        for item in data:
            dataset = item.get('data_source', 'unknown')
            all_datasets.add(dataset)
            
            human_judgment = item.get('human_judge')
            
            # 统计各个judge的准确率
            for judge_type in ['regex_judge', 'judge_wo_ref', 'human_judge', 'xverify_judge']:
                judgment = item.get(judge_type)
                if judgment is not None:  # 只处理有判断结果的情况
                    model_results[dataset][judge_type]['total'] += 1
                    if judgment is False or judgment in ('incorrect', 'Incorrect'):
                        pass
                        judge_res = 'incorrect'
                    elif judgment is True or judgment in ('correct', 'Correct'):
                        model_results[dataset][judge_type]['correct'] += 1
                        judge_res = 'correct'
                        model_results['all'][judge_type]['correct'] += 1
                    model_results['all'][judge_type]['total'] += 1
                    
                    
                    # 统计与human_judge的一致性
                    if judge_res == human_judgment:
                        consistency_stats[dataset][judge_type]['consistent'] += 1
                        consistency_stats['all'][judge_type]['consistent'] += 1
                    consistency_stats[dataset][judge_type]['total'] += 1
                    consistency_stats['all'][judge_type]['total'] += 1
                        
        # 存储该模型的结果
        results[model_name] = {
            'accuracy': model_results,
            'consistency': consistency_stats
        }
    
    # 将所有数据集名称列表化，确保包含'all'
    all_datasets = sorted(list(all_datasets))
    if 'all' not in all_datasets:
        all_datasets.append('all')
    
    # 为每个评估方法创建单独的DataFrame
    judge_types = ['human_judge', 'judge_wo_ref', 'xverify_judge', 'regex_judge']
    
    # 存储用于Excel的DataFrames
    dataframes = {}
    
    # 准确率DataFrames
    for judge_type in judge_types:
        df_data = []
        for model_name, model_data in results.items():
            row = {'Model': model_name}
            for dataset in all_datasets:
                stats = model_data['accuracy'].get(dataset, {}).get(judge_type, {'total': 0, 'correct': 0})
                if stats['total'] > 0:
                    row[dataset] = stats['correct'] / stats['total']
                else:
                    row[dataset] = None
            df_data.append(row)
        dataframes[f'Accuracy_{judge_type}'] = pd.DataFrame(df_data)
    
    # 一致性DataFrames
    for judge_type in ['regex_judge', 'judge_wo_ref', 'xverify_judge']:
        df_data = []
        for model_name, model_data in results.items():
            row = {'Model': model_name}
            for dataset in all_datasets:
                stats = model_data['consistency'].get(dataset, {}).get(judge_type, {'total': 0, 'consistent': 0})
                if stats['total'] > 0:
                    row[dataset] = stats['consistent'] / stats['total']
                else:
                    row[dataset] = None
            df_data.append(row)
        dataframes[f'Con_{judge_type}_vs_human'] = pd.DataFrame(df_data)
    
    # 将所有DataFrame保存到同一个Excel文件的不同sheet中
    with pd.ExcelWriter('model_evaluation_results.xlsx') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print("分析完成，结果已保存到 model_evaluation_results.xlsx")

if __name__ == "__main__":
    analyze_model_results()
import json
import os
import glob
import argparse
from typing import Dict, List, Any, Optional

def convert_rl_to_merged_format(input_file: str) -> List[Dict[str, Any]]:
    """
    Convert a file from rl-annotated format to merged_outputs format
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    merged_data = []
    
    for i, item in enumerate(data):
        try:
            def get_human_judge(annotations, index=0):
                if not annotations or len(annotations) <= index:
                    return None
                
                annotation = annotations[index]
                if 'result' not in annotation or not annotation['result']:
                    return None
                
                if not annotation['result'] or len(annotation['result']) == 0:
                    return None
                    
                result = annotation['result'][0]
                if 'value' not in result:
                    return None
                
                value = result['value']
                if 'choices' not in value or not value['choices']:
                    return None
                
                return value['choices'][0] if value['choices'] else None
            
            # 根据标注数量获取人工判断结果
            human_judge = None
            if len(item.get('annotations', [])) == 1:
                human_judge = get_human_judge(item['annotations'], 0)
            else:
                human_judge = get_human_judge(item['annotations'], 1)
            if human_judge not in ("correct", "incorrect"):
                human_judge = "incorrect"
            merged_data.append({
                "data_source": item['data']['data_source'],
                "prompt": item['data']['prompt'],
                "ability": item['data']['ability'],
                "reward_model": item['data']['reward_model'],
                "extra_info": item['data']['extra_info'],
                "task_description": item['data']['task_description'],
                "task_type": item['data']['task_type'],
                "question": item['data']['question'],
                "reward_model.ground_truth": item['data']['reward_model.ground_truth'],
                "llm_response": item['data']['llm_response'],
                "xverify_judge": item['data']['xverify_judge'],
                "correct_answer": item['data']['correct_answer'],
                "regex_judge": item['data']['regex_judge'],
                "human_judge": human_judge
            })
        except Exception as e:
            print(f"Error processing item {i} in {os.path.basename(input_file)}: {str(e)}")
            # print(f"Problematic item structure: {json.dumps(item, indent=2)[:500]}...")
            continue
            
    return merged_data

def main():
    parser = argparse.ArgumentParser(description="Convert rl-annotated files to merged_outputs format")
    parser.add_argument("--input-dir", type=str, default="/mnt/public/code/qingchen/xVerify-Achieved/eval/ls-anno", 
                        help="Directory containing rl-annotated files")
    parser.add_argument("--output-dir", type=str, default="/mnt/public/code/qingchen/xVerify-Achieved/eval/rl-annotated", 
                        help="Directory for saving merged format files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all JSON files in the input directory
    for input_file in glob.glob(os.path.join(args.input_dir, "*.json")):
        filename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, filename.replace(".json", "_converted.json"))
        
        print(f"Converting {input_file} to {output_file}")
        
        try:
            merged_data = convert_rl_to_merged_format(input_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
                
            print(f"Successfully converted {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()

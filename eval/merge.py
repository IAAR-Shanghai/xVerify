import os
import json
from typing import List, Dict, Any, Tuple

# Hardcoded paths
REJUDGED_OUTPUTS_PATH = "/mnt/public/code/qingchen/xVerify-Achieved/eval/rejudged_outputs"
RL_ANNOTATED_PATH = "/mnt/public/code/qingchen/xVerify-Achieved/eval/rl-annotated"
OUTPUT_PATH = "/mnt/public/code/qingchen/xVerify-Achieved/eval/final_outputs"

# Hardcoded file pairs (rejudged file, rl-annotated file, output model name)
MODEL_FILE_PAIRS = [
    # (rejudged_outputs_file, rl_annotated_file, model_name)
    (
        "rejudged_Llama-3_1-8B-Instruct_merged.json", 
        "Llama-3_1-8B-Instruct-rl-annotated_converted.json",
        "Llama-3_1-8B-Instruct"
    ),
    (
        "rejudged_Llama-3_1-8B-Instruct-Math-Verify_merged.json", 
        "Llama-3_1-8B-Instruct-Math-Verify-rl-annotated_converted.json",
        "Llama-3_1-8B-Instruct-Math-Verify"
    ),
    (
        "rejudged_Llama-3_1-8B-Instruct-xVerify_merged.json", 
        "Llama-3_1-8B-Instruct-xVerify-rl-annotated_converted.json",
        "Llama-3_1-8B-Instruct-xVerify"
    ),
    (
        "rejudged_Qwen2_5-7B_merged.json",
        "Qwen2_5-7B-rl-annotated_converted.json",
        "Qwen2_5-7B"
    ),
    (
        "rejudged_Qwen2.5-7B-Math-Verify_merged.json",
        "Qwen2.5-7B-Math-Verify-rl-annotated_converted.json",
        "Qwen2.5-7B-Math-Verify"
    ),
    (
        "rejudged_Qwen2.5-7B-xVerify_merged.json",
        "Qwen2.5-7B-xVerify-rl-annotated_converted.json",
        "Qwen2.5-7B-xVerify"
    ),
]

def read_json_file(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Read JSON file and handle different structures:
    - If JSON has a 'results' key, return its content and other data
    - Otherwise, return the entire content as results and empty dict
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "results" in data:
        # Structure with 'results' key and possibly other metadata
        results = data["results"]
        info = {k: v for k, v in data.items() if k != "results"}
        return results, info
    else:
        # Structure is already a list or doesn't have 'results' key
        if isinstance(data, list):
            return data, {}
        else:
            # Handle unexpected structure - just return as results with no extra info
            return [data], {}

def write_json_file(results: List[Dict[str, Any]], info: Dict[str, Any], file_path: str) -> None:
    """Write JSON file with results and info"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # output_data = {"results": results}
    # for key, value in info.items():
    #     output_data[key] = value
        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def validate_and_merge_samples(samples1: List[Dict[str, Any]], 
                              samples2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and merge samples from two sources"""
    if len(samples1) != len(samples2):
        print(f"Warning: Sample count mismatch: {len(samples1)} vs {len(samples2)}")
        min_len = min(len(samples1), len(samples2))
        samples1 = samples1[:min_len]
        samples2 = samples2[:min_len]
    
    # Only check these important fields for matching
    important_fields = ["data_source", "prompt", "ability", "question", 
                       "llm_response", "xverify_judge", "correct_answer"]
        
    merged_samples = []
    for sample1 in samples1:
        for sample2 in samples2:
            if all(sample1.get(field) == sample2.get(field) for field in important_fields):
                merged_sample = {**sample1, **sample2}
                merged_samples.append(merged_sample)
                break
        else:
            print(f"Warning: No match found for sample: {sample1}")
    
    return merged_samples

def process_model_files():
    """Process and merge files using hardcoded file pairs"""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    for rejudged_file, rl_annotated_file, model_name in MODEL_FILE_PAIRS:
        rejudged_path = os.path.join(REJUDGED_OUTPUTS_PATH, rejudged_file)
        rl_annotated_path = os.path.join(RL_ANNOTATED_PATH, rl_annotated_file)
        
        # Check if both files exist
        if not os.path.exists(rejudged_path):
            print(f"Error: File not found: {rejudged_path}")
            continue
            
        if not os.path.exists(rl_annotated_path):
            print(f"Error: File not found: {rl_annotated_path}")
            continue
        
        print(f"Processing model {model_name}...")
        
        # Read the files
        rejudged_samples, rejudged_info = read_json_file(rejudged_path)
        rl_annotated_samples, rl_annotated_info = read_json_file(rl_annotated_path)
        
        print(f"  - Loaded {len(rejudged_samples)} samples from rejudged file")
        print(f"  - Loaded {len(rl_annotated_samples)} samples from rl-annotated file")
        
        # Validate and merge
        merged_samples = validate_and_merge_samples(rejudged_samples, rl_annotated_samples)
        
        # Combine info (metadata) from both files
        merged_info = {**rejudged_info, **rl_annotated_info}
        
        # Write merged file
        output_file = f"{model_name}_final.json"
        output_path = os.path.join(OUTPUT_PATH, output_file)
        write_json_file(merged_samples, merged_info, output_path)
        
        print(f"  - Successfully merged files for {model_name}")
        print(f"  - Merged {len(merged_samples)} samples")
        print(f"  - Output: {output_file}")
        print("")

def main():
    print("Starting merge process...")
    process_model_files()
    print("Merge process completed!")

if __name__ == "__main__":
    main()
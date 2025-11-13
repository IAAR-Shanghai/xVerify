"""
针对每个dataset分别根据oc的config文件中的extractor进行后处理，提取答案
short_text和categorical_label的提取答案方法都用自创的
"""
import re
import json
from multiprocessing import Pool
import os
import sys
from tqdm import tqdm
import ast

from oc_utils import first_capital_postprocess, first_option_postprocess, text_postprocess, is_number
from oc_gsm8k import gsm8k_postprocess
from oc_MATH import math_postprocess_v2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Math_Evaluator')))

from Math_Evaluator.math_evaluator import MATHEvaluator     # 用于比较提取到的答案和gold label

curr_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = "/mnt/data131/qingchen/codes/OpenJudge/yqc/final_eval/convert_raw"
raw_files = [file for file in os.listdir(raw_dir) if file.endswith(".json")]

dataprocess_dict = {
    "CommonsenseQA_alpha": first_option_postprocess,
    "hellaswag_alpha": first_option_postprocess,
    "SIQA_alpha": first_option_postprocess,         # ABC
    "OpenbookQA_alpha": first_option_postprocess,   # 目前没找到
    "CommonsenseQA_text": text_postprocess,
    "SIQA_text": text_postprocess,
    "OpenbookQA_text": text_postprocess,
    "QNLI": text_postprocess,
    "WiC": text_postprocess,
    "BoolQ": text_postprocess,
    "Subj": text_postprocess,
    "TREC": text_postprocess,
    "MetaMathQA": [gsm8k_postprocess, math_postprocess_v2],
    "MultiArith": gsm8k_postprocess,
    "MATH": math_postprocess_v2,

    "MMLU_alpha": first_option_postprocess,     # https://github.com/open-compass/opencompass/blob/53fe3904540c049e259492016942cbd39f13a7a2/configs/datasets/mmlu/mmlu_gen_4d595a.py
    "ARC-c_alpha": first_option_postprocess,        # https://github.com/open-compass/opencompass/blob/53fe3904540c049e259492016942cbd39f13a7a2/configs/datasets/ARC_c/ARC_c_gen_1e0de5.py
    "ARC-e_alpha": first_option_postprocess,        #https://github.com/open-compass/opencompass/blob/53fe3904540c049e259492016942cbd39f13a7a2/configs/datasets/ARC_e/ARC_e_gen_1e0de5.py
    "MMLU_text": text_postprocess,
    "ARC-c_text": text_postprocess,
    "ARC-e_text": text_postprocess,
    "AgNews": text_postprocess,
    "Amazon": text_postprocess,
    "DBPedia": text_postprocess,
    "GSM8K": gsm8k_postprocess,
}

def process_item(item):
    Math_Judger = MATHEvaluator()
    item["llm_output"] = item["llm_output"].strip()
    if isinstance(item["standard_answer_range"], str) and item["key_answer_type"] != "math":
            item["standard_answer_range"] = ast.literal_eval(item["standard_answer_range"])

    func_extractor = dataprocess_dict[item["dataset"]]
    if not isinstance(func_extractor, list):
         func_extractor = [func_extractor]
    if item["dataset"] == "MetaMathQA":
        if is_number(item["correct_answer"]):
            func_extractor = [gsm8k_postprocess, math_postprocess_v2]
        else:
            func_extractor = [math_postprocess_v2, gsm8k_postprocess]

    extract_key_answer = None
    extract_correct = False

    for func in func_extractor:
        if func == first_option_postprocess:
            extract_key_answer = func(item["llm_output"], "ABCDEF", cushion=True)
        elif func == text_postprocess:
            extract_key_answer = func(item["llm_output"], item["standard_answer_range"])
        else:
            extract_key_answer = func(item["llm_output"])
        if extract_key_answer is not None:
            break

    if extract_key_answer is None:
        extract_valid = False
    elif extract_key_answer.strip().rstrip(".").lower() == item["correct_answer"].strip().rstrip(".").lower():
        extract_valid = True
        extract_correct = True
    elif item["key_answer_type"] == "math":
        if Math_Judger.is_equiv(extract_key_answer, item["correct_answer"]):
            extract_valid = True
            extract_correct = True
        else:
            extract_valid = True
    else:
        extract_valid = True
            
    item["extracted_answer"] = extract_key_answer
    item["extracted_valid"] = extract_valid
    item["extracted_correct"] = extract_correct

    return item

        
def oc_eval(processes=1):
    pool = Pool(processes=processes)

    for raw_file in raw_files:
        with open(os.path.join(raw_dir, raw_file), 'r') as f:
            data = json.load(f)

        results = list(tqdm(pool.imap(process_item, data["results"]), desc=f"Processing items [{data['info']['dataset']['data_name']}]", total=len(data["results"])))

        overall = {
            "total_num": len(results),
            "valid_num": sum([item["extracted_valid"] for item in results]),
            "strict_accuracy": sum([item["extracted_correct"] for item in results]) / len(results),
            "lenient_accuracy": sum([item["extracted_correct"] for item in results]) / sum([item["extracted_valid"] for item in results])
        }

        final_data = {
            "info": data["info"],
            "overall": overall, 
            "results": results,
        }

        final_dir = os.path.join(curr_dir, f"oc_eval")

        if not os.path.exists(final_dir):
            os.makedirs(final_dir)

        with open(os.path.join(final_dir, f"oc_{raw_file}"), "w") as f:
            json.dump(final_data, f, indent=4)


if __name__ == "__main__":

    oc_eval(processes = 5)
        
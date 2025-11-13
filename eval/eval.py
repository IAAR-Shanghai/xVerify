import os
import copy
import json
import hashlib
import datetime
import argparse

from multiprocessing import Pool

import yaml
from loguru import logger
from tqdm import tqdm
from llms import LLMs
import pandas as pd

from oc_extract.oc_utils import first_capital_postprocess, first_option_postprocess, text_postprocess, is_number
from oc_extract.oc_gsm8k import gsm8k_postprocess
from oc_extract.oc_MATH import math_postprocess_v2
from oc_extract.math_evaluator import MathEvaluator

# Define paths for system prompts
FEW_SHOT_COT_RESTRICT = "./prompts/few_shot_cot_restrict.txt"
FEW_SHOT_COT = "./prompts/few_shot_cot.txt"
FEW_SHOT_RESTRICT = "./prompts/few_shot_restrict.txt"
FEW_SHOT = "./prompts/few_shot.txt"
XVERIFY = "./prompts/judge_prompt.txt"
XVERIFY_WO_REF = "./prompts/judge_prompt_wo_ref.txt"
XVERIFY_MODEL = "xVerify-7B-I"

SHOT_DATA_PATH = os.path.join("./data", "train.parquet")
# DATA_PATH = os.path.join("./data", "test_0506.parquet")
DATA_PATH = os.path.join("./data", "test_0514.parquet")

OUTPUT_DIR = "./outputs"
LOG_PATH = "./logs"
SEED = 42

dataprocess_dict = {
    "MATH-500": math_postprocess_v2,
    "CMNLI": first_option_postprocess,
    "MMLU-Pro": first_option_postprocess,
    "SimpleQA": first_option_postprocess,
    "DROP": first_option_postprocess,
    "AgNews": first_option_postprocess,
    "GPQA": first_option_postprocess,
    "AIME_2024": [gsm8k_postprocess, math_postprocess_v2],
    "ARC": first_option_postprocess,
    "GSM8K": gsm8k_postprocess,
    "MMLU-Redux": first_option_postprocess,
    "C-Eval": first_option_postprocess,
    "MMLU": first_option_postprocess,
    "CMMLU": first_option_postprocess,
    "FRAMES": first_option_postprocess,
    "C-SimpleQA": first_option_postprocess,
    "CHID": first_option_postprocess,
    "LiveMathBench": [gsm8k_postprocess, math_postprocess_v2],
    "AMC23": [gsm8k_postprocess, math_postprocess_v2],
    "OlympiadBench": [gsm8k_postprocess, math_postprocess_v2],
    "MGSM": [gsm8k_postprocess, math_postprocess_v2],
    "CMATH": [gsm8k_postprocess, math_postprocess_v2],
    "CLUEWSC": first_option_postprocess,
    "Amazon": first_option_postprocess,
}

class Evaluator:
    """A class to evaluate the performance of a language model on a specific dataset."""

    def __init__(self,
                 model: LLMs,
                 num_samples: int,
                 prompt_type: str,
                 data_name: str,
                 process_num: int = 1,
                 logging_num: int = 25,
                 output_dir: str = OUTPUT_DIR,
                 seed: int = SEED):

        self.model = copy.deepcopy(model)
        self.model_name = model.model_name
        self.num_samples = num_samples
        self.prompt_type = prompt_type
        self.setting_name = f"{prompt_type.replace('few', str(self.num_samples))}"
        self.output_dir = output_dir
        self.logging_num = logging_num
        self.data_name = data_name
        
        self.process_num = process_num
        seed_str = self.model_name + self.prompt_type + str(self.num_samples)
        self.seed = seed + int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

        self.load_prompt()
        self.load_dataset()
        self.construct_prompt()

    @staticmethod
    def load_conf(conf_path):
        try:
            with open(conf_path, 'r', encoding='utf-8') as file:
                conf = yaml.safe_load(file)
            return conf
        except FileNotFoundError:
            logger.error(f"File '{conf_path}' not found.")
        except yaml.YAMLError as e:
            logger.error(f"Unable to load YAML file '{conf_path}': {e}")

    def load_prompt(self):
        prompt_path = {
            "few_shot": FEW_SHOT,
            "few_shot_cot": FEW_SHOT_COT,
            "few_shot_restrict": FEW_SHOT_RESTRICT,
            "few_shot_cot_restrict": FEW_SHOT_COT_RESTRICT
        }
        if self.prompt_type in prompt_path:
            path = prompt_path[self.prompt_type]
            with open(path, 'r') as f:
                self.prompt = f.read()
        else:
            logger.error(f'Prompt template {self.prompt_type} not found!')

    def load_dataset(self):
        shot_data = pd.read_parquet(SHOT_DATA_PATH)
        shot_data = shot_data[shot_data['data_source'] == self.data_name]
        if self.num_samples > 0:
            self.fewshot_data = shot_data.sample(n=self.num_samples, random_state=self.seed).to_dict(orient='records')
        else:
            self.fewshot_data = None

        data = pd.read_parquet(DATA_PATH)
        self.data = data[data['data_source'] == self.data_name].to_dict(orient='records')
        # 将 reward_model': {'ground_truth': 'contradiction', 'style': 'rule'} 中 ground_truth 的值复制到 reward_model.ground_truth
        for item in self.data:
            item['reward_model.ground_truth'] = item['reward_model']['ground_truth']
        self.data_size = len(self.data)

    def construct_prompt(self):
        if self.fewshot_data:
            examples =  '***** Start In-Context Examples *****\n' + '\n'.join([
                f"Q: {shot['question']}\nA: The answer is {shot['reward_model.ground_truth']}."
                for shot in self.fewshot_data
            ]) + '\n***** End In-Context Examples *****'
        else:
            examples = ''
        
        for s in self.data:
            s['prompt'] = self.prompt.format(
                task_type=s['task_type'],
                task_description=s['task_description'],
                examples=examples,
                question='Q: ' + s['question'] + '\nA: '
            )

    def log_info(self):
        return {
            'setting': self.setting_name,
            'llm': {
                "model_name": self.model_name,
                **self.model.common_params
            },
            'dataset': self.data_name,
            'data_num': self.data_size,
            'logging_num': self.logging_num,
            'random_seed': self.seed
        }
    
    def load_checkpoints(self):
        log_file_head = f'{self.setting_name}_{self.data_name}_{self.model_name}_{self.data_size}'
        log_info = self.log_info()

        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        
        logger.info(f"Loading checkpoint from {LOG_PATH}")
        log_files = [
            f for f in os.listdir(LOG_PATH)
            if (
                f.startswith(log_file_head) and 
                f.endswith('.json')
            )
        ]
        finished_results = []

        for log in log_files:
            with open(os.path.join(LOG_PATH, log), "r") as f:
                log_data = json.load(f)
            log_data_info = log_data['info']
            del log_data_info['datetime']
            if log_data_info == log_info:
                finished_results.extend(log_data['results'])

        finished_num = len(finished_results)
        self.start_index = finished_num

        logger.info(f"The logs show that {finished_num} result data have been completed.")
        
        return finished_results

    def evaluator_info(self):
        return {
            'setting': self.setting_name,
            'llm': {
                "model_name": self.model_name,
                **self.model.common_params
            },
            'data_num': self.data_size,
            'random_seed': self.seed,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def gen(self, data_point):
        result = self.model.request(data_point['prompt'])
        data_point['llm_response'] = result
        data_point['xverify_judge'], data_point['judge_wo_ref'], data_point['judge_wo_ref_text'] = self.parse_response_by_xverify(data_point)
        data_point['regex_judge'] = self.parse_response_by_regex(data_point)

        return data_point
    
    def parse_response_by_regex(self, data_point):
        math_judge = MathEvaluator()
        data_point['llm_response'] = data_point['llm_response'].strip()
        data_point['correct_answer'] = data_point['reward_model.ground_truth'].strip()

        func_extractor = dataprocess_dict.get(data_point['data_source'], None)

        if not isinstance(func_extractor, list):
            func_extractor = [func_extractor]

        if data_point['ability'] == 'math' and data_point['data_source'] not in ["MATH", "GSM8K"]:
            if is_number(data_point['correct_answer']):
                func_extractor = [gsm8k_postprocess, math_postprocess_v2]
            else:
                func_extractor = [math_postprocess_v2, gsm8k_postprocess]

        extract_key_answer = None
        regex_result = False

        for func in func_extractor:
            if func == first_option_postprocess:
                extract_key_answer = func(data_point['llm_response'], "ABCDEF", cushion=True)
            else:
                extract_key_answer = func(data_point['llm_response'])
            if extract_key_answer is not None:
                break
        
        if extract_key_answer.strip().rstrip(".").lower() == data_point['correct_answer'].strip().rstrip(".").lower():
            regex_result = True
        elif data_point['ability'] == 'math':
            if math_judge.is_equiv(extract_key_answer, data_point['correct_answer']):
                regex_result = True

        return regex_result
        

    def parse_response_by_xverify(self, data_point):
        with open(XVERIFY, 'r') as f:
            xverify_prompt = f.read()

        xverify_prompt = xverify_prompt.format(
            question=data_point['question'],
            output=data_point['llm_response'],
            answer=data_point['reward_model.ground_truth'],
        )
        xverify_model = LLMs(model_name=XVERIFY_MODEL)
        xverify_model.common_params = self.model.common_params
        xverify_result = xverify_model.request(xverify_prompt)

        with open(XVERIFY_WO_REF, 'r') as f:
            xverify_wo_ref_prompt = f.read()
        xverify_wo_ref_prompt = xverify_wo_ref_prompt.format(
            output=data_point['llm_response'],
            answer=data_point['reward_model.ground_truth'],
        )
        xverify_wo_ref_model = LLMs(model_name=XVERIFY_MODEL)
        xverify_wo_ref_model.common_params = self.model.common_params
        xverify_wo_ref_result = xverify_wo_ref_model.request(xverify_wo_ref_prompt)
        # print(xverify_wo_ref_result)

        if 'incorrect' in xverify_result.lower():
            xverify_result = False
        elif 'correct' in xverify_result.lower():
            xverify_result = True

        if 'incorrect' in xverify_wo_ref_result.lower():
            xverify_wo_ref = False
        elif 'correct' in xverify_wo_ref_result.lower():
            xverify_wo_ref = True

        return xverify_result, xverify_wo_ref, xverify_wo_ref_result

    def batch_gen(self, dataset):
        with Pool(self.process_num) as pool:
            results = list(tqdm(
                pool.imap(self.gen, dataset),
                total=len(dataset),
                desc=f'{self.setting_name}_{self.data_name}_{self.model_name}'
            ))
        return results
    
    def save_log(self, log_content, log_path):
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_content, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Logs saved to {log_path}!")

    def save_output(self, output: dict):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Output saved to {self.output_path}!")

    def run(self):

        info = self.evaluator_info()
        log_info = self.log_info()
        logger.info(f'{self.setting_name}_{self.data_name}_{self.model_name}_{self.data_size} start ...')

        results = self.load_checkpoints()
        if len(results) == self.data_size:
            output_file_head = f'{self.setting_name}_{self.data_name}_{self.model_name}_{self.data_size}'
            exist_output_files = [
                f for f in os.listdir(self.output_dir)
                if (
                    f.startswith(output_file_head) and 
                    f.endswith('.json')
                )
            ]
            info_ = info
            del info_['datetime']
            is_exists = False
            for output in exist_output_files:
                with open(os.path.join(self.output_dir, output), "r") as f:
                    output_data = json.load(f)
                output_data_info = output_data['info']
                del output_data_info['datetime']
                if output_data_info == info_:
                    logger.info(f"The output file for this configuration already exists: {output}!")
                    is_exists = True
            if is_exists:
                return

        for i in range(self.start_index, self.data_size, self.logging_num):
            batch_start = i
            batch_end = i + self.logging_num
            batch_results = self.batch_gen(self.data[batch_start:batch_end])
            log_info['datetime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_path = f'{self.setting_name}_{self.data_name}_{self.model_name}_{self.data_size}_({batch_start}~{batch_end})_{log_info["datetime"]}.json'
            log_path = os.path.join(LOG_PATH, log_path)
            log_content = {
                'info': log_info, 
                'results': batch_results
            }
            self.save_log(log_content, log_path)
            results.extend(batch_results)
            logger.info(f"Currently completed ({batch_end}/{self.data_size})!")

        info['regex_accuracy'] = sum([item['regex_judge'] for item in results]) / len(results)
        info['xverify_accuracy'] = sum([item['xverify_judge'] for item in results]) / len(results)
        info['judge_wo_ref_accuracy'] = sum([item['judge_wo_ref'] for item in results]) / len(results)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'{self.setting_name}_{self.data_name}_{self.model_name}_{self.data_size}_{timestamp}.json'
        self.output_path = os.path.join(self.output_dir, output_name)

        self.save_output({'info': info, 'results': results})
            

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LLM performance or rejudge merged outputs.')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'rejudge', 'both'], default='evaluate',
                        help='Mode of operation: evaluate (run original evaluation), '
                             'rejudge (rejudge merged outputs), or both')
    parser.add_argument('--judge-model', type=str, default='Llama-3_1-8B-Instruct',
                        help='Model to use for judging merged outputs')
    parser.add_argument('--merged-dir', type=str, 
                        default='/mnt/public/code/qingchen/xVerify-Achieved/eval/merged_outputs',
                        help='Directory containing merged output files')
    parser.add_argument('--output-dir', type=str, 
                        default='/mnt/public/code/qingchen/xVerify-Achieved/eval/rejudged_outputs',
                        help='Directory to save rejudged output files')
    parser.add_argument('--process-num', type=int, default=15,
                        help='Number of processes to use for parallel processing')
    parser.add_argument('--judge-prompt', type=str, default='./prompts/judge_prompt_wo_ref.txt',
                        help='Path to judge prompt template file')
    return parser.parse_args()

if __name__ == "__main__":
    # data_lst = ['CMNLI', 'MMLU-Pro', 'AIME_2024', 'SimpleQA', 'DROP', 'MATH-500', 'AgNews', 'GPQA']
    # data_lst = ['MMLU-Pro']
    data_lst = ['ARC', 'GSM8K', 'MMLU-Redux', 'C-Eval', 'MMLU', 'CMMLU', 'FRAMES', 'C-SimpleQA', 'CHID', 'LiveMathBench', 'AMC23', 'OlympiadBench', 'MGSM', 'CMATH', 'CLUEWSC', 'Amazon']

    # model_lst = ['Qwen2_5-7B', 'Llama-3_1-8B-Instruct', 'DeepSeek-R1-Distill-Qwen-7B']
    # model_lst = ['Qwen2.5-7B-Math-Verify']
    # model_lst = ['Llama-3_1-8B-Instruct-xVerify']
    # model_lst = ['Llama-3_1-8B-Instruct-Math-Verify', 'Qwen2.5-7B-xVerify']
    model_lst = [
        'Qwen2_5-7B', 'Llama-3_1-8B-Instruct', 
        'Qwen2.5-7B-Math-Verify', 'Llama-3_1-8B-Instruct-Math-Verify',
        'Qwen2.5-7B-xVerify', 'Llama-3_1-8B-Instruct-xVerify'
    ]
    args = parse_args()
    for model_name in model_lst:
        model = LLMs(model_name=model_name)
        for data_name in data_lst:
            evaluator = Evaluator(
                model=model,
                num_samples=0,
                prompt_type='few_shot_cot',
                data_name=data_name,
                process_num=args.process_num,
                logging_num=25,
                output_dir=OUTPUT_DIR,
                seed=SEED
            )
            evaluator.run()


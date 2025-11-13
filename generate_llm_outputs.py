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
from utils.dataset_loader import DatasetLoader
from utils.llms import LLMs


# Define paths for system prompts
FEW_SHOT_COT_RESTRICT = "./prompts/few_shot_cot_restrict.txt"
FEW_SHOT_COT = "./prompts/few_shot_cot.txt"
FEW_SHOT_RESTRICT = "./prompts/few_shot_restrict.txt"
FEW_SHOT = "./prompts/few_shot.txt"

DATA_DIR = "./benchmark/transformed_dataset"
DATA_CONF_PATH = "./conf/data_conf.yaml"
OUTPUT_DIR = "./llm_outputs3"
LOG_PATH = "./logs"
SEED = 42


class GenLLMOutputs:
    """Generate outputs from large language models for a given dataset."""

    def __init__(self,
                 model: LLMs,
                 data_name: str,
                 data_size: int,
                 num_samples: int,
                 prompt_type: str,
                 process_num: int = 1,
                 logging_num: int = 25,
                 output_dir: str = OUTPUT_DIR,
                 seed: int = SEED):

        self.model = copy.deepcopy(model)
        self.model_name = model.model_name
        self.data_name = data_name
        self.data_conf = self.load_conf(DATA_CONF_PATH)[data_name]
        self.data_size = data_size
        self.num_samples = num_samples
        self.prompt_type = prompt_type
        self.setting_name = f"{prompt_type.replace('few', str(self.num_samples))}"
        self.output_dir = output_dir
        self.logging_num = logging_num
        
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
        # 加载 prompt 文件
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
        train_path = os.path.join(
            DATA_DIR, self.data_conf['data_name'], 'train.json')
        test_path = os.path.join(
            DATA_DIR, self.data_conf['data_name'], 'test.json')
        sample_path = os.path.join(
            DATA_DIR, self.data_conf['data_name'], 'example.json')
        
        train_size = int(self.data_size / 3 * 2)
        test_size = self.data_size - train_size

        train = DatasetLoader.load(train_path, train_size, self.seed)
        test = DatasetLoader.load(test_path, test_size, self.seed)

        for s in train:
            s['dataset_type'] = 'train'
        
        for s in test:
            s['dataset_type'] = 'test'

        self.data = train + test
        self.data_size = len(self.data)

        cnt = 0
        for s in self.data:
            s['index'] = cnt
            cnt += 1

        self.fewshot_data = DatasetLoader.load(sample_path, self.num_samples, self.seed)

    def construct_prompt(self):
        if self.fewshot_data:
            examples =  '***** Start In-Context Examples *****\n' + '\n'.join([
                f"Q: {shot['question']}\nA: The answer is {shot['correct_answer']}."
                for shot in self.fewshot_data
            ]) + '\n***** End In-Context Examples *****'
        else:
            examples = ''
        
        for s in self.data:
            s['prompt'] = self.prompt.format(
                task_type=self.data_conf['task_type'],
                task_description=self.data_conf['task_description'],
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
            'dataset': self.data_conf,
            'data_num': self.data_size,
            'logging_num': self.logging_num,
            'random_seed': self.seed
        }
    
    def load_checkpoints(self):
        log_file_head = f'{self.setting_name}_{self.data_name}_{self.model_name}_{self.data_size}'
        log_info = self.log_info()

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
            'dataset': self.data_conf,
            'data_num': self.data_size,
            'random_seed': self.seed,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def gen(self, data_point):
        result = self.model.request(data_point['prompt'])
        data_point['llm_output'] = result

        return data_point

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

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'{self.setting_name}_{self.data_name}_{self.model_name}_{self.data_size}_{timestamp}.json'
        self.output_path = os.path.join(self.output_dir, output_name)

        self.save_output({'info': info, 'results': results})


if __name__ == "__main__":
    import configparser


    MODELS_CONFIG = "./conf/models.ini"


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--data_size', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--prompt_type', type=str, default='few_shot')
    parser.add_argument('--process_num', type=int, default=1)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(MODELS_CONFIG)
    all_models = [section for section in config.sections() if section not in ['common_params', 'api_key']]
    all_datasets = list(GenLLMOutputs.load_conf(DATA_CONF_PATH).keys())

    models = all_models if args.model == 'all' else [args.model]
    datasets = all_datasets if args.data_name == 'all' else [args.data_name]

    for dataset in datasets:
        for model in models:
            model = LLMs(model)
            GenLLMOutputs(
                model=model,
                data_name=dataset,
                data_size=args.data_size,
                num_samples=args.num_samples,
                prompt_type=args.prompt_type,
                process_num=args.process_num
            ).run()
            
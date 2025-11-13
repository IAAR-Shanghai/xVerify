import os
import copy
import json
import yaml
import datetime
import argparse

from pathlib import Path
from multiprocessing import Pool

from loguru import logger
from tqdm import tqdm
from utils.dataset_loader import DatasetLoader
from utils.llms import LLMs


PROMPT_PATH = "./prompts/judge2.yaml"  # Optional prompts include `judge.yaml` and `judge2.yaml`.
OUTPUT_DIR = "./processed_outputs"
LOG_PATH = "./logs"


class Judge:

    def __init__(
            self,
            model: LLMs,
            data_path: str,
            data_size: int | str = 'all',
            prompt_path: str = PROMPT_PATH,
            process_num: int = 5,
            logging_num: int = 50,
            log_path: str = LOG_PATH,
            output_dir: str = OUTPUT_DIR,
            seed: int = 42
    ):
        self.model = copy.deepcopy(model)
        self.model_name = model.model_name
        self.data_path = data_path
        self.data_name = Path(self.data_path).stem
        self.data_size = -1 if data_size == 'all' else data_size
        self.prompt_path = prompt_path
        self.output_dir = output_dir
        self.logging_num = logging_num
        self.log_path = log_path
        self.process_num = process_num
        self.seed = seed
        
        # Load prompt and dataset
        self.load_prompt()
        self.data = DatasetLoader.fixed_load(os.path.join(self.output_dir, self.data_path), self.data_size)
        self.data_size = len(self.data)

    def load_prompt(self):
        try:
            with open(self.prompt_path, 'r', encoding='utf-8') as file:
                prompt_data = yaml.safe_load(file)
            
            # Safely get data from the dictionary to avoid KeyError
            self.prompt = prompt_data.get('prompt')
            self.system = prompt_data.get('system')

            # Check if the required fields are missing
            if not self.prompt or not self.system:
                logger.error(f"Missing required 'prompt' or 'system' in the YAML file.")
        except FileNotFoundError:
            logger.error(f"File '{self.prompt_path}' not found.")
        except yaml.YAMLError as e:
            logger.error(f"Unable to load YAML file '{self.prompt_path}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
    
    def construct_message(self):
        for item in self.data:
            user_input = self.prompt.format(
                question=item['question'],
                output=item['llm_output'],
                answer=item['correct_answer']
            )
            message = [
                {
                    "role": "system",
                    "content": self.system
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
            item['message'] = message
    
    def judge_info(self):
        # Return information about the current run configuration
        return {
            'llm': {
                "model_name": self.model_name,
                **self.model.common_params
            },
            'dataset': self.data_name,
            'data_num': self.data_size,
            'random_seed': self.seed,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    
    def load_checkpoints(self):
        # Load previously generated results from log files
        log_file_head = f'log_judge_{self.model_name}_{"-".join(f"{k}={v}" for k, v in self.model.common_params.items())}_{self.data_name}_{self.data_size}_{self.seed}'

        logger.info(f"Loading checkpoint from {self.log_path}")
        # Find all relevant log files
        log_files = [
            f for f in os.listdir(self.log_path)
            if (
                f.startswith(log_file_head) and 
                f.endswith('.json')
            )
        ]
        log_files.sort()
        latest_log_file = log_files[-1] if log_files else None
        self.log_file = latest_log_file
        # Load completed results from the latest log file
        finished_results = []
        self.finished_num = 0
        if self.log_file is not None:
            with open(os.path.join(self.log_path, latest_log_file), "r") as f:
                for line in f:
                    finished_results.append(json.loads(line.strip()))
            self.finished_num = len(finished_results)
            self.data = self.data[self.finished_num:]
            logger.info(f"Load data from log file: {latest_log_file}!")

        logger.info(f"The logs show that {self.finished_num} result data have been completed.")
        
        return finished_results
    
    def gen(self, data_point):
        # Generate judgment result for a single data point
        result = self.model.request_message(data_point['message'])
        data_point[f'{self.model_name}_judgment_result'] = result

        return data_point

    def batch_gen(self, data):
        # Generate judgment results for multiple data points in parallel
        with Pool(self.process_num) as pool:
            results = list(tqdm(
                pool.imap(self.gen, data),
                total=len(data),
                desc=f'{self.model_name}_{self.data_name}'
            ))
        return results
    
    def save_output(self, output: dict):
        # Save the final output to a JSON file
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Output saved to {self.output_path}!")

    def run(self):
        # Main function to execute the judging process

        # Get run configuration and log information
        info = self.judge_info()
        logger.info(f'judge_{self.model_name}_{self.data_name}_{self.data_size}_{self.seed} start ...')

        # Load previous results and skip already processed data
        results = self.load_checkpoints()
        if self.log_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = f'log_judge_{self.model_name}_{"-".join(f"{k}={v}" for k, v in self.model.common_params.items())}_{self.data_name}_{self.data_size}_{self.seed}_{timestamp}.json'
        self.log_file = os.path.join(self.log_path, self.log_file)
        # If all data has been processed, check if the final output file exists
        if len(results) == self.data_size:
            output_file_head = f'judge_{self.model_name}_{self.data_name}_{self.data_size}_{self.seed}'
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
        
        # Construct messages for all data points
        self.construct_message()

        # Process data in batches, saving intermediate results to log files
        for i in range(0, self.data_size - self.finished_num, self.logging_num):
            batch_start = i
            batch_end = i + self.logging_num
            batch_results = self.batch_gen(self.data[batch_start:batch_end])
            
            # Write results to the log file
            with open(self.log_file, "a", encoding="utf-8") as f:
                for item in batch_results:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

            results.extend(batch_results)
            logger.info(f"Currently completed ({batch_end + self.finished_num}/{self.data_size})!")

        # Save all results to a final output file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'judge_{self.model_name}_{self.data_name}_{self.data_size}_{self.seed}_{timestamp}.json'
        self.output_path = os.path.join(self.output_dir, output_name)

        self.save_output({'info': info, 'results': results})


if __name__ == "__main__":

    def data_size_type(value):
        # Convert input to integer if possible, otherwise return original string
        if value.isdigit():
            return int(value)
        return value

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--data_size', type=data_size_type, default='all')
    parser.add_argument('--process_num', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model = LLMs(args.model)
    Judge(
        model=model,
        data_path=args.data_path,
        data_size=args.data_size,
        process_num=args.process_num,
        seed=args.seed
    ).run()

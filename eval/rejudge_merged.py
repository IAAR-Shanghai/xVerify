import os
import json
import argparse
from multiprocessing import Pool
from loguru import logger
from tqdm import tqdm
from llms import LLMs

class MergedOutputEvaluator:
    """A class to evaluate the results from merged_outputs using a new judge template."""
    
    def __init__(self, 
                 model: LLMs,
                 judge_prompt_path: str,
                 input_file_path: str,
                 output_dir: str):
        self.model = model
        self.model_name = model.model_name
        self.judge_prompt_path = judge_prompt_path
        self.input_file_path = input_file_path
        self.output_dir = output_dir
        
        self.load_judge_prompt()
        self.load_merged_outputs()
    
    def load_judge_prompt(self):
        with open(self.judge_prompt_path, 'r') as f:
            self.judge_prompt = f.read()
    
    def load_merged_outputs(self):
        with open(self.input_file_path, 'r', encoding='utf-8') as f:
            self.merged_data = json.load(f)
            # if 'results' not in self.merged_data:
            #     logger.error(f"No 'results' field found in {self.input_file_path}")
            #     self.merged_data['results'] = []
    
    def judge_single_response(self, data_point):
        try:
            judge_prompt = self.judge_prompt.format(
                output=data_point.get('llm_response', ''),
                answer=data_point.get('reward_model.ground_truth', '')
            )
            judge_result = self.model.request(judge_prompt)
            
            if 'incorrect' in judge_result.lower():
                data_point['judge_wo_ref'] = False
            else:
                data_point['judge_wo_ref'] = True
                
            # Save the full judgment text for reference
            data_point['judge_wo_ref_text'] = judge_result
            
            return data_point
        except Exception as e:
            logger.error(f"Error judging response: {e}")
            data_point['judge_wo_ref'] = False
            data_point['judge_wo_ref_text'] = f"ERROR: {str(e)}"
            return data_point
    
    def batch_judge(self, process_num=10):
        # if not self.merged_data['results']:
        #     logger.warning("No results to judge")
        #     return
        
        results = []
        for data_point in tqdm(self.merged_data, desc=f'Judging {os.path.basename(self.input_file_path)}'):
            results.append(self.judge_single_response(data_point))
        
        # Replace results with judged results
        self.merged_data = results
        
        # Calculate accuracy
        judge_wo_ref_results = [r.get('judge_wo_ref', False) for r in self.merged_data]
        accuracy = sum(judge_wo_ref_results) / len(judge_wo_ref_results) if judge_wo_ref_results else 0
        
        # Add accuracy to info or create info if it doesn't exist
        # if 'info' not in self.merged_data:
        #     self.merged_data['info'] = {}
        # self.merged_data['info']['judge_wo_ref_accuracy'] = accuracy
    
    def save_output(self):
        # Create output filename based on input filename
        base_name = os.path.basename(self.input_file_path)
        output_name = f"rejudged_{base_name}"
        output_path = os.path.join(self.output_dir, output_name)
        
        os.makedirs(self.output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.merged_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Rejudged output saved to {output_path}!")
        
    def run(self, process_num=10):
        logger.info(f"Starting rejudgment of {self.input_file_path}")
        self.batch_judge(process_num)
        self.save_output()
        
        # Get accuracy
        # accuracy = self.merged_data.get('info', {}).get('judge_wo_ref_accuracy', 'N/A')
        logger.info(f"Rejudgment completed with accuracy: {accuracy}")

def rejudge_merged_outputs(model_name, merged_outputs_dir, output_dir, judge_prompt_path, process_num=10):
    """Rejudge all merged outputs using the provided judge prompt."""
    model = LLMs(model_name=model_name)
    
    # Check if the merged_outputs_dir is actually a file
    if os.path.isfile(merged_outputs_dir):
        logger.warning(f"Path provided to --merged-dir is a file, not a directory: {merged_outputs_dir}")
        logger.info(f"Processing single file: {merged_outputs_dir}")
        
        # Extract directory and filename
        file_path = merged_outputs_dir
        file_name = os.path.basename(file_path)
        
        evaluator = MergedOutputEvaluator(
            model=model,
            judge_prompt_path=judge_prompt_path,
            input_file_path=file_path,
            output_dir=output_dir
        )
        evaluator.run(process_num=process_num)
        return
        
    # Original directory processing code
    merged_files = [f for f in os.listdir(merged_outputs_dir) if f.endswith('.json')]
    logger.info(f"Found {len(merged_files)} merged output files to rejudge")
    
    for file in merged_files:
        input_path = os.path.join(merged_outputs_dir, file)
        evaluator = MergedOutputEvaluator(
            model=model,
            judge_prompt_path=judge_prompt_path,
            input_file_path=input_path,
            output_dir=output_dir
        )
        evaluator.run(process_num=process_num)

def main():
    parser = argparse.ArgumentParser(description='Rejudge merged outputs using a new judgment template.')
    parser.add_argument('--judge-model', type=str, default='xVerify-7B-I',
                        help='Model to use for judging merged outputs')
    parser.add_argument('--merged-dir', type=str, 
                        default='/mnt/public/code/qingchen/xVerify-Achieved/eval/merged_outputs',
                        help='Directory containing merged output files')
    parser.add_argument('--output-dir', type=str, 
                        default='/mnt/public/code/qingchen/xVerify-Achieved/eval/rejudged_outputs',
                        help='Directory to save rejudged output files')
    parser.add_argument('--process-num', type=int, default=1,
                        help='Number of processes to use for parallel processing')
    parser.add_argument('--judge-prompt', type=str, default='./prompts/judge_prompt_wo_ref.txt',
                        help='Path to judge prompt template file')
    parser.add_argument('--file', type=str, default=None,
                        help='Process a specific file instead of all files in merged-dir')
    args = parser.parse_args()
    
    if args.file:
        # Process a specific file
        model = LLMs(model_name=args.judge_model)
        input_path = os.path.join(args.merged_dir, args.file) if not os.path.isabs(args.file) else args.file
        evaluator = MergedOutputEvaluator(
            model=model,
            judge_prompt_path=args.judge_prompt,
            input_file_path=input_path,
            output_dir=args.output_dir
        )
        evaluator.run(process_num=args.process_num)
    else:
        # Process all files in the directory or a single file if merged_dir is a file
        rejudge_merged_outputs(args.judge_model, args.merged_dir, args.output_dir, 
                              args.judge_prompt, args.process_num)

if __name__ == "__main__":
    main()

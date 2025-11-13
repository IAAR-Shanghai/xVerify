model=QwQ-32B
data_name=CHID
data_size=50
process_num=5

python generate_llm_outputs.py --model ${model} --data_name ${data_name} --data_size ${data_size} --num_samples 0 --prompt_type few_shot --process_num ${process_num}
python generate_llm_outputs.py --model ${model} --data_name ${data_name} --data_size ${data_size} --num_samples 0 --prompt_type few_shot_restrict --process_num ${process_num}
python generate_llm_outputs.py --model ${model} --data_name ${data_name} --data_size ${data_size} --num_samples 0 --prompt_type few_shot_cot --process_num ${process_num}
python generate_llm_outputs.py --model ${model} --data_name ${data_name} --data_size ${data_size} --num_samples 0 --prompt_type few_shot_cot_restrict --process_num ${process_num}
python generate_llm_outputs.py --model ${model} --data_name ${data_name} --data_size ${data_size} --num_samples 5 --prompt_type few_shot --process_num ${process_num}
python generate_llm_outputs.py --model ${model} --data_name ${data_name} --data_size ${data_size} --num_samples 5 --prompt_type few_shot_restrict --process_num ${process_num}
python generate_llm_outputs.py --model ${model} --data_name ${data_name} --data_size ${data_size} --num_samples 5 --prompt_type few_shot_cot --process_num ${process_num}
python generate_llm_outputs.py --model ${model} --data_name ${data_name} --data_size ${data_size} --num_samples 5 --prompt_type few_shot_cot_restrict --process_num ${process_num}
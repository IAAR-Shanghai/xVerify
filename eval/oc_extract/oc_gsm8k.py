'''
Copy from https://github.com/open-compass/opencompass/blob/53fe3904540c049e259492016942cbd39f13a7a2/opencompass/datasets/gsm8k.py
正则提取用的gsm8k_postprocess (line 36)
'''

import re

# gsk8k (从一段文本中提取最后一个数值)
def gsm8k_postprocess(text: str) -> str:
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return None
    return numbers[-1]
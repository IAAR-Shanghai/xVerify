import re

# from https://github.com/open-compass/opencompass/blob/53fe3904540c049e259492016942cbd39f13a7a2/opencompass/utils/text_postprocessors.py#L45
# line 44
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return None

# from https://github.com/open-compass/opencompass/blob/53fe3904540c049e259492016942cbd39f13a7a2/opencompass/utils/text_postprocessors.py#L60
# line 60
def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'答案选项是?\s*:\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''

# short_text_content (但是不是来自于OpenCompass的，自创的)
def text_postprocess(text: str, ans_range: list) -> str:
    """Return the first number in a string."""
    patterns = [
        f'[Tt]he answer is ([A-Za-z\s]+)',
        f'[Tt]he answer is: ([A-Za-z\s]+)',
        f'[Tt]he answer is option ([A-Za-z\s]+)',
        f'[Tt]he correct answer is ([A-Za-z\s]+)',
        f'[Tt]he correct answer is option ([A-Za-z\s]+)',
        f'[Tt]he answer to the question is ([A-Za-z\s]+)',
        f'[Tt]he answer to the question is ([A-Za-z\s]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    patterns = []
    for choice in ans_range:
        patterns.append(f'\b{choice}\b')
    
    for pattern in patterns:
        try:
            match = re.search(pattern, text)
            if match:
                return match.group()
        except:
            pass
    return None

# 从一段文本中提取第一个数值
def first_number_postprocess(text: str) -> float:
    """Return the first number in a string."""
    # regex pattern to match numbers (both integers and decimals)
    pattern = r'(-?\d*\.?\d+)'

    # search the string for the pattern
    match = re.search(pattern, text)

    # if a match is found, return it. Otherwise, return None.
    return float(match.group(1)) if match else None

if __name__ == "__main__":
    text = """False.\n\nThe target word "bondage" does not have the exact same meaning in the two sentences. In Sentence 1, "He sought release from his bondage to Satan," the term "bondage" refers to a state of captivity or enslavement to Satan, typically understood as a spiritual or mental constraint.\n\nIn Sentence 2, "A self freed from the bondage of time," the term "bondage" is metaphorical and implies a restriction or captivity imposed by time, suggesting that the self has been liberated from the constraints of being bound by the passage of time."""

    # 使用单词边界确保匹配完整单词 "False"
    match = re.search(r"\bFalse\b", text)
    print(match.group())  # 输出 ['False']

# check if a string is a number
def is_number(s):
    try: 
        float(s)
        return True
    except ValueError:
        pass 
    try:
        import unicodedata 
        unicodedata.numeric(s) 
        return True
    except (TypeError, ValueError):
        pass
    return False
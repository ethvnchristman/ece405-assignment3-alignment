from __future__ import annotations
import re


def parse_mmlu_response(mmlu_example, model_output):
    m = re.search(r'\b([A-D])\b', model_output)
    if m:
        return m.group(1)
    return None


def parse_gsm8k_response(model_output):
    cleaned = model_output.replace(",", "")
    nums = re.findall(r'[-+]?\d*\.?\d+', cleaned)
    if nums:
        return nums[-1]
    return None

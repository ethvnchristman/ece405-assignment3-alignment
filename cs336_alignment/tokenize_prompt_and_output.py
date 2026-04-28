from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    concat_ids = [p + o for p, o in zip(prompt_ids, output_ids)]
    max_len = max(len(c) for c in concat_ids)
    batch_size = len(concat_ids)
    pad_id = tokenizer.pad_token_id

    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(concat_ids):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    input_ids = padded[:, :-1]
    labels = padded[:, 1:]

    response_mask = torch.zeros(batch_size, max_len - 1, dtype=torch.bool)
    for i, (p_ids, o_ids) in enumerate(zip(prompt_ids, output_ids)):
        start = len(p_ids) - 1
        end = start + len(o_ids)
        response_mask[i, start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

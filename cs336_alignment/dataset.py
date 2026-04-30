from __future__ import annotations
import json
import random
import torch
from torch.utils.data import Dataset


class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        self.seq_length = seq_length
        self.chunks = []

        records = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if shuffle:
            random.shuffle(records)

        tokens = []
        for rec in records:
            prompt = (rec.get("prompt") or rec.get("instruction") or
                      rec.get("question") or rec.get("problem", ""))
            response = (rec.get("response") or rec.get("output") or
                        rec.get("answer") or rec.get("solution", ""))
            text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
            )
            toks = tokenizer(text, add_special_tokens=True)["input_ids"]
            toks.append(tokenizer.eos_token_id)
            tokens.extend(toks)

        for i in range(0, len(tokens) - seq_length, seq_length):
            chunk = tokens[i: i + seq_length + 1]
            if len(chunk) == seq_length + 1:
                self.chunks.append({"input_ids": chunk[:-1], "labels": chunk[1:]})

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        item = self.chunks[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def batch_iter(dataset, batch_size, shuffle):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    batches = []
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in indices[start: start + batch_size]]
        batches.append({k: torch.stack([s[k] for s in batch]) for k in batch[0]})
    return batches

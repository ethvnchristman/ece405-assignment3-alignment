from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.tokenizer_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    compute_entropy,
    sft_microbatch_train_step,
    log_generations,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

PROMPT_PATH = Path(__file__).parent / "prompts" / "r1_zero.prompt"


def load_math_sft_data(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_batches(records, batch_size, shuffle=True):
    import random
    idxs = list(range(len(records)))
    if shuffle:
        random.shuffle(idxs)
    batches = []
    for s in range(0, len(idxs), batch_size):
        batches.append([records[i] for i in idxs[s: s + batch_size]])
    return batches


def eval_vllm(llm, tokenizer, val_records, prompt_template, reward_fn, n=256, max_tokens=1024):
    from vllm import SamplingParams
    prompts = [prompt_template.replace("{question}", r["prompt"]) for r in val_records[:n]]
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_tokens,
                            stop=["</answer>"], include_stop_str_in_output=True)
    outputs = llm.generate(prompts, params)
    rewards = []
    for rec, out in zip(val_records[:n], outputs):
        gen = out.outputs[0].text
        gt = rec.get("answer", rec.get("ground_truth", ""))
        sc = reward_fn(gen, gt)
        rewards.append(sc["answer_reward"])
    return sum(rewards) / len(rewards) if rewards else 0.0


def run_sft(
    model_path: str,
    data_path: str,
    output_dir: str,
    n_steps: int = 200,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-5,
    max_seq_len: int = 1024,
    device: str = "cuda",
    val_every: int = 20,
    val_n: int = 256,
):
    os.makedirs(output_dir, exist_ok=True)
    prompt_template = PROMPT_PATH.read_text()

    records = load_math_sft_data(data_path)
    n_val = max(val_n, int(len(records) * 0.05))
    val_records = records[:n_val]
    train_records = records[n_val:]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0,
                                   betas=(0.9, 0.95))

    log = []
    step = 0
    optimizer.zero_grad()
    t0 = time.time()

    while step < n_steps:
        batches = build_batches(train_records, batch_size)
        for mb_idx, batch in enumerate(batches):
            if step >= n_steps:
                break

            prompt_strs = [r["prompt"] for r in batch]
            output_strs = [r["response"] for r in batch]

            tok = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            input_ids = tok["input_ids"].to(device)
            labels = tok["labels"].to(device)
            response_mask = tok["response_mask"].to(device)

            lp_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
            policy_lp = lp_dict["log_probs"]

            loss, metrics = sft_microbatch_train_step(
                policy_lp, response_mask, grad_accum
            )

            if (mb_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % val_every == 0 or step == n_steps:
                    elapsed = time.time() - t0
                    entry = {"step": step, "loss": metrics["loss"], "elapsed": elapsed}
                    log.append(entry)
                    print(f"step={step} loss={metrics['loss']:.4f} elapsed={elapsed:.1f}s")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "sft_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved to {output_dir}")
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_sft(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

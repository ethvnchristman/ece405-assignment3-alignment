from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

SCRIPT_DIR = Path(__file__).parent
STOP_STR = "</answer>"
VAL_SAMPLES = 128
MAX_TOKENS = 512
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.03

with open(SCRIPT_DIR / "prompts" / "r1_zero.prompt") as _f:
    _R1_TEMPLATE = _f.read().strip()


def _build_prompt(question: str) -> str:
    return _R1_TEMPLATE.format(question=question)


def _extract_gt(answer_str: str) -> str:
    m = re.findall(r"####\s*([^\n]+)", answer_str)
    if m:
        return m[-1].strip().replace(",", "")
    return answer_str.strip()


def _get_field(item: dict, *keys: str) -> str:
    for k in keys:
        if k in item:
            return item[k]
    raise KeyError(f"None of {keys} found in item")


def _reward(response: str, gt: str) -> float:
    return float(r1_zero_reward_fn(response, gt)["reward"])


def _init_vllm(model_path: str, device: str, seed: int, gpu_util: float = 0.70) -> LLM:
    vllm_set_random_seed(seed)
    w_patch = patch("torch.distributed.get_world_size", return_value=1)
    p_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with w_patch, p_patch:
        return LLM(
            model=model_path, device=device, dtype=torch.float16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_util,
            max_model_len=2048,
        )


def _sync_weights(policy: torch.nn.Module, llm: LLM) -> None:
    sd = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(sd.items())


class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer, seq_length=512):
        self.samples = []
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        for prompt, response in pairs:
            full = prompt + response + tokenizer.eos_token
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            ids = full_ids[: seq_length + 1]
            inputs = ids[:-1]
            labels = ids[1:]

            mask_len = min(prompt_len, len(labels))
            labels = [-100] * mask_len + labels[mask_len:]

            pad = seq_length - len(inputs)
            if pad > 0:
                inputs = inputs + [pad_id] * pad
                labels = labels + [-100] * pad

            self.samples.append({
                "input_ids": torch.tensor(inputs[:seq_length], dtype=torch.long),
                "labels": torch.tensor(labels[:seq_length], dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def _collate(batch):
    return {k: torch.stack([s[k] for s in batch]) for k in batch[0]}


def _sft_step(model, tokenizer, pairs, lr, epochs, device):
    if not pairs:
        return []
    ds = _PairDataset(pairs, tokenizer)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_collate)
    n_steps = len(loader) * epochs
    warmup = max(1, int(n_steps * WARMUP_RATIO))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=n_steps)

    model.train()
    model.config.use_cache = False
    losses = []
    for _ in range(epochs):
        for batch in loader:
            inp = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)
            logits = model(inp).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lbl.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
    torch.cuda.empty_cache()
    return losses


def _val_accuracy(llm, val_items, n=VAL_SAMPLES):
    prompts = [_build_prompt(_get_field(it, "problem", "question")) for it in val_items[:n]]
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, stop=[STOP_STR])
    outputs = llm.generate(prompts, params)
    correct = 0
    for out, item in zip(outputs, val_items[:n]):
        resp = out.outputs[0].text + STOP_STR
        gt = _extract_gt(_get_field(item, "answer", "solution"))
        correct += int(_reward(resp, gt) == 1.0)
    acc = correct / len(prompts)
    print(f"  Val accuracy: {correct}/{len(prompts)} = {acc:.2%}")
    return acc


def _avg_entropy(model, tokenizer, val_items, device, n=64):
    model.eval()
    ents = []
    with torch.no_grad():
        for item in val_items[:n]:
            ids = tokenizer(
                _build_prompt(_get_field(item, "problem", "question")),
                return_tensors="pt", truncation=True, max_length=512,
            ).input_ids.to(device)
            logits = model(ids).logits[0]
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * probs.log().clamp(min=-1e9)).sum(dim=-1).mean().item()
            ents.append(ent)
    model.train()
    torch.cuda.empty_cache()
    return sum(ents) / len(ents) if ents else 0.0


def run_expert_iteration(
    model_path, data_path, output_dir,
    n_ei_steps, G, db_size, sft_epochs, lr,
    policy_device, vllm_device, seed,
):
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    all_items = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))
    random.shuffle(all_items)

    n_val = max(VAL_SAMPLES, int(len(all_items) * 0.1))
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]
    print(f"Train: {len(train_items)} | Val: {len(val_items)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    ).to(policy_device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    print(f"Initializing vLLM on {vllm_device}")
    llm = _init_vllm(model_path, vllm_device, seed)

    log = []
    print("\nStep 0 (baseline)")
    _sync_weights(model, llm)
    val_acc = _val_accuracy(llm, val_items)
    entropy = _avg_entropy(model, tokenizer, val_items, policy_device)
    log.append({"ei_step": 0, "val_accuracy": val_acc, "entropy": entropy,
                 "n_correct": 0, "n_rollouts": 0})
    print(f"  Entropy: {entropy:.4f}")

    for step in range(1, n_ei_steps + 1):
        print(f"\n{'='*60}\nEI Step {step}/{n_ei_steps}  G={G}  db={db_size}  epochs={sft_epochs}")

        batch = random.sample(train_items, min(db_size, len(train_items)))
        _sync_weights(model, llm)

        prompts = [_build_prompt(_get_field(q, "problem", "question")) for q in batch]
        sp = SamplingParams(temperature=1.0, max_tokens=MAX_TOKENS, n=G,
                            stop=[STOP_STR], seed=seed + step)
        outputs = llm.generate(prompts, sp)

        correct_pairs = []
        n_rollouts = 0
        for q_item, out in zip(batch, outputs):
            gt = _extract_gt(_get_field(q_item, "answer", "solution"))
            prompt = _build_prompt(_get_field(q_item, "problem", "question"))
            for gen in out.outputs:
                n_rollouts += 1
                resp = gen.text + STOP_STR
                if _reward(resp, gt) == 1.0:
                    correct_pairs.append((prompt, resp))

        pct = len(correct_pairs) / n_rollouts * 100 if n_rollouts else 0
        print(f"  Rollouts: {n_rollouts} | Correct: {len(correct_pairs)} ({pct:.1f}%)")

        _sft_step(model, tokenizer, correct_pairs, lr, sft_epochs, policy_device)

        _sync_weights(model, llm)
        val_acc = _val_accuracy(llm, val_items)
        entropy = _avg_entropy(model, tokenizer, val_items, policy_device)
        print(f"  Entropy: {entropy:.4f}")

        log.append({
            "ei_step": step, "val_accuracy": val_acc, "entropy": entropy,
            "n_correct": len(correct_pairs), "n_rollouts": n_rollouts,
        })

        ckpt = os.path.join(output_dir, f"step_{step}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    log_path = os.path.join(output_dir, "ei_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog: {log_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [e["ei_step"] for e in log]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(steps, [e["val_accuracy"] for e in log], marker="o")
        axes[0].axhline(0.15, color="red", linestyle="--", label="15% target")
        axes[0].set_xlabel("EI Step")
        axes[0].set_ylabel("Val Accuracy")
        axes[0].set_title("Validation Accuracy")
        axes[0].legend()
        axes[1].plot(steps, [e["entropy"] for e in log], marker="o", color="orange")
        axes[1].set_xlabel("EI Step")
        axes[1].set_ylabel("Avg Token Entropy")
        axes[1].set_title("Response Entropy")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ei_curves.png"), dpi=150)
    except Exception as e:
        print(f"[plot skipped: {e}]")

    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_ei_steps", type=int, default=5)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--db_size", type=int, default=512)
    p.add_argument("--sft_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--policy_device", default="cuda:0")
    p.add_argument("--vllm_device", default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_expert_iteration(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_ei_steps=args.n_ei_steps,
        G=args.G,
        db_size=args.db_size,
        sft_epochs=args.sft_epochs,
        lr=args.lr,
        policy_device=args.policy_device,
        vllm_device=args.vllm_device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

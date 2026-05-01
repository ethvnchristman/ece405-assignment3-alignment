from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.tokenizer_utils import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.policy_utils import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean,
)

SCRIPT_DIR = Path(__file__).parent
STOP_STR = "</answer>"


def _load_prompt(name: str) -> str:
    return (SCRIPT_DIR / "prompts" / name).read_text().strip()


def _init_vllm(model_path: str, device: str, seed: int, gpu_util: float = 0.85) -> LLM:
    vllm_set_random_seed(seed)
    w_patch = patch("torch.distributed.get_world_size", return_value=1)
    p_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with w_patch, p_patch:
        return LLM(
            model=model_path, device=device, dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_util,
        )


def _sync_weights(policy: torch.nn.Module, llm: LLM) -> None:
    sd = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(sd.items())


def _load_data(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _eval_accuracy(llm, val_records, prompt_template, reward_fn, n=1024, max_tokens=1024):
    prompts = [prompt_template.replace("{question}", r["problem"]) for r in val_records[:n]]
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_tokens,
                            stop=[STOP_STR], include_stop_str_in_output=True)
    outputs = llm.generate(prompts, params)
    answer_rewards = []
    for rec, out in zip(val_records[:n], outputs):
        gen = out.outputs[0].text
        gt = rec.get("ground_truth", rec.get("answer", ""))
        sc = reward_fn(gen, gt)
        answer_rewards.append(sc["answer_reward"])
    return sum(answer_rewards) / len(answer_rewards) if answer_rewards else 0.0


def run_grpo(
    model_path: str,
    train_path: str,
    val_path: str,
    output_dir: str,
    n_grpo_steps: int = 200,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    train_batch_size: int = 256,
    grad_accum_steps: int = 128,
    epochs_per_rollout: int = 1,
    lr: float = 1e-5,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    loss_type: str = "reinforce_with_baseline",
    cliprange: float = 0.2,
    sampling_temp: float = 1.0,
    sampling_max_tokens: int = 1024,
    sampling_min_tokens: int = 4,
    gpu_util: float = 0.85,
    policy_device: str = "cuda:0",
    vllm_device: str = "cuda:1",
    val_every: int = 5,
    prompt_name: str = "r1_zero.prompt",
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    prompt_template = _load_prompt(prompt_name)
    reward_fn = r1_zero_reward_fn if "r1_zero" in prompt_name else question_only_reward_fn

    train_records = _load_data(train_path)
    val_records = _load_data(val_path)
    print(f"Train: {len(train_records)} | Val: {len(val_records)}")

    assert rollout_batch_size % group_size == 0
    assert train_batch_size % grad_accum_steps == 0
    assert train_batch_size >= group_size

    micro_batch = train_batch_size // grad_accum_steps
    n_prompts = rollout_batch_size // group_size
    n_microbatches = rollout_batch_size // micro_batch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(policy_device)
    policy.config.use_cache = False

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95)
    )

    print(f"Initializing vLLM on {vllm_device}")
    llm = _init_vllm(model_path, vllm_device, seed, gpu_util)

    log = []
    t_start = time.time()

    for grpo_step in range(1, n_grpo_steps + 1):
        t_step = time.time()

        questions = random.sample(train_records, n_prompts)
        prompts = [prompt_template.replace("{question}", q["problem"]) for q in questions]
        gt_list = [q.get("ground_truth", q.get("answer", "")) for q in questions]

        _sync_weights(policy, llm)
        sp = SamplingParams(
            temperature=sampling_temp, top_p=1.0,
            max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens,
            n=group_size, stop=[STOP_STR], include_stop_str_in_output=True,
            seed=seed + grpo_step,
        )
        outputs = llm.generate(prompts, sp)

        rollout_responses = []
        repeated_gts = []
        rollout_prompts = []
        for prompt, gt, out in zip(prompts, gt_list, outputs):
            for gen in out.outputs:
                rollout_responses.append(gen.text)
                repeated_gts.append(gt)
                rollout_prompts.append(prompt)

        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        old_log_probs_all = None
        if loss_type == "grpo_clip" or epochs_per_rollout > 1:
            policy.eval()
            with torch.inference_mode():
                old_lp_list = []
                for mb_s in range(0, rollout_batch_size, micro_batch):
                    mb_prompts = rollout_prompts[mb_s: mb_s + micro_batch]
                    mb_responses = rollout_responses[mb_s: mb_s + micro_batch]
                    tok = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                    iids = tok["input_ids"].to(policy_device)
                    lbls = tok["labels"].to(policy_device)
                    lp = get_response_log_probs(policy, iids, lbls)["log_probs"]
                    old_lp_list.append(lp.cpu())
            old_log_probs_all = old_lp_list
            policy.train()

        total_loss = 0.0
        n_updates = 0

        for epoch in range(epochs_per_rollout):
            mb_order = list(range(n_microbatches))
            random.shuffle(mb_order)
            optimizer.zero_grad()
            accum_count = 0

            for mb_seq_idx, mb_idx in enumerate(mb_order):
                mb_s = mb_idx * micro_batch
                mb_e = mb_s + micro_batch

                mb_prompts = rollout_prompts[mb_s: mb_e]
                mb_responses = rollout_responses[mb_s: mb_e]
                mb_adv = advantages[mb_s: mb_e].unsqueeze(1).to(policy_device)
                mb_raw = raw_rewards[mb_s: mb_e].unsqueeze(1).to(policy_device)

                tok = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                iids = tok["input_ids"].to(policy_device)
                lbls = tok["labels"].to(policy_device)
                rmask = tok["response_mask"].to(policy_device)

                lp_dict = get_response_log_probs(policy, iids, lbls)
                policy_lp = lp_dict["log_probs"]

                mb_old_lp = None
                if old_log_probs_all is not None:
                    raw_old = old_log_probs_all[mb_idx]
                    seq_len = policy_lp.shape[1]
                    old_padded = torch.zeros_like(policy_lp)
                    copy_len = min(raw_old.shape[1], seq_len)
                    old_padded[:, :copy_len] = raw_old[:, :copy_len].to(policy_device)
                    mb_old_lp = old_padded

                loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_lp,
                    response_mask=rmask,
                    gradient_accumulation_steps=grad_accum_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=cliprange,
                )
                total_loss += loss.item()
                accum_count += 1

                if accum_count % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    n_updates += 1

        if accum_count % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - t_start
        step_time = time.time() - t_step
        entry = {
            "step": grpo_step,
            "loss": total_loss / max(n_updates, 1),
            "mean_reward": reward_meta["mean_reward"],
            "mean_answer_reward": reward_meta["mean_answer_reward"],
            "mean_format_reward": reward_meta["mean_format_reward"],
            "elapsed": elapsed,
            "step_time": step_time,
        }

        if grpo_step % val_every == 0:
            _sync_weights(policy, llm)
            val_acc = _eval_accuracy(llm, val_records, prompt_template, reward_fn)
            entry["val_answer_reward"] = val_acc
            print(
                f"step={grpo_step} loss={entry['loss']:.4f} "
                f"train_ans={reward_meta['mean_answer_reward']:.3f} "
                f"val_ans={val_acc:.3f} elapsed={elapsed:.0f}s"
            )
        else:
            print(
                f"step={grpo_step} loss={entry['loss']:.4f} "
                f"train_ans={reward_meta['mean_answer_reward']:.3f} "
                f"elapsed={elapsed:.0f}s"
            )

        log.append(entry)

        if grpo_step % 50 == 0:
            ckpt = os.path.join(output_dir, f"step_{grpo_step}")
            policy.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)

    policy.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    with open(os.path.join(output_dir, "grpo_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Done. Output: {output_dir}")
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_path", required=True)
    p.add_argument("--val_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_grpo_steps", type=int, default=200)
    p.add_argument("--rollout_batch_size", type=int, default=256)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--grad_accum_steps", type=int, default=128)
    p.add_argument("--epochs_per_rollout", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--loss_type", default="reinforce_with_baseline",
                   choices=["no_baseline", "reinforce_with_baseline", "grpo_clip", "GRPO-No-CLIP"])
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--normalize_by_std", action="store_true", default=True)
    p.add_argument("--no_normalize_by_std", dest="normalize_by_std", action="store_false")
    p.add_argument("--policy_device", default="cuda:0")
    p.add_argument("--vllm_device", default="cuda:1")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--prompt_name", default="r1_zero.prompt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_grpo(
        model_path=args.model_path,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        n_grpo_steps=args.n_grpo_steps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        train_batch_size=args.train_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs_per_rollout=args.epochs_per_rollout,
        lr=args.lr,
        loss_type=args.loss_type,
        cliprange=args.cliprange,
        normalize_by_std=args.normalize_by_std,
        policy_device=args.policy_device,
        vllm_device=args.vllm_device,
        val_every=args.val_every,
        prompt_name=args.prompt_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

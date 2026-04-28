from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Callable

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import extract_boxed_answer, r1_zero_reward_fn

VALIDATION_PATH = Path("data/MATH/validation.jsonl")
OUTPUT_PATH = Path("outputs/math_baseline_results.jsonl")
PROMPT_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")


def _build_validation_set(seed: int = 42) -> list[dict]:
    import pandas as pd

    url = "https://huggingface.co/datasets/qwedsacf/competition_math/resolve/main/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
    df = pd.read_parquet(url)
    print("Dataset schema:", list(df.columns))
    examples = df.to_dict(orient="records")

    rng = random.Random(seed)
    rng.shuffle(examples)
    val_examples = examples[-500:]

    records = []
    for ex in val_examples:
        gt = extract_boxed_answer(ex["solution"])
        records.append({
            "problem": ex["problem"],
            "ground_truth": gt if gt is not None else "",
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
        })
    return records


def _load_or_build_validation_set() -> list[dict]:
    if VALIDATION_PATH.exists():
        with open(VALIDATION_PATH) as f:
            return [json.loads(line) for line in f]

    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = _build_validation_set()
    with open(VALIDATION_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return records


def _categorize(format_reward: float, answer_reward: float) -> str:
    if format_reward == 1.0 and answer_reward == 1.0:
        return "correct"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "format_only"
    if format_reward == 0.0 and answer_reward == 0.0:
        return "malformed"
    return "other"


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable,
    prompts: list[str],
    eval_sampling_params: SamplingParams,
    records: list[dict],
    output_path: Path,
) -> dict[str, float]:
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for rec, output in zip(records, outputs):
        generation = output.outputs[0].text
        scores = reward_fn(generation, rec["ground_truth"])
        category = _categorize(scores["format_reward"], scores["answer_reward"])
        results.append({
            "problem": rec["problem"],
            "ground_truth": rec["ground_truth"],
            "generation": generation,
            "format_reward": scores["format_reward"],
            "answer_reward": scores["answer_reward"],
            "total_reward": scores["reward"],
            "category": category,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(results)
    counts = {
        "correct": sum(1 for r in results if r["category"] == "correct"),
        "format_only": sum(1 for r in results if r["category"] == "format_only"),
        "malformed": sum(1 for r in results if r["category"] == "malformed"),
        "other": sum(1 for r in results if r["category"] == "other"),
    }
    return {"total": total, **counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=500)
    args = parser.parse_args()

    prompt_template = PROMPT_PATH.read_text()

    records = _load_or_build_validation_set()
    records = records[: args.n_examples]

    prompts = [prompt_template.replace("{question}", rec["problem"]) for rec in records]

    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B")
    print(f"Initializing vLLM: LLM(model={model_path!r}, dtype='bfloat16')")
    llm = LLM(model=model_path, dtype="bfloat16")

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    t0 = time.time()
    counts = evaluate_vllm(llm, r1_zero_reward_fn, prompts, sampling_params, records, OUTPUT_PATH)
    elapsed = time.time() - t0

    total = counts["total"]
    correct = counts["correct"]
    format_only = counts["format_only"]
    malformed = counts["malformed"]
    other = counts["other"]

    print(f"\n{'='*50}")
    print(f"Total evaluated:  {total}")
    print(f"  correct         {correct:4d}  ({100*correct/total:.1f}%)")
    print(f"  format_only     {format_only:4d}  ({100*format_only/total:.1f}%)")
    print(f"  malformed       {malformed:4d}  ({100*malformed/total:.1f}%)")
    print(f"  other           {other:4d}  ({100*other/total:.1f}%)")
    print(f"Overall accuracy: {100*correct/total:.2f}%")
    print(f"Wall-clock time:  {elapsed:.1f}s")
    print(f"Examples/sec:     {total/elapsed:.2f}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

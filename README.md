# UHM ECE 405 Spring 2026 Assignment 3: Alignment

This asignment is created from Assignment 5 of [CS336 at Stanford taught in Spring 2025](https://stanford-cs336.github.io/spring2025/). 
For the full description of the original assignment, see the assignment handout at
[cs336_spring2024_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)
Check out the [glossary of terms](./glossary.md) for this assignment.

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

Check out useful [lectures from CS336 at Stanford](https://github.com/stanford-cs336/spring2025-lectures).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix. Any improvements of the existing codebase
(including adaptations from Stanford to UHM workflows, modifications of PDF, etc) will be rewarded with extra points.


## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

3. Download models:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model name
tiny_model_name = "Qwen/Qwen2.5-0.5B"
medium_model_name = "Qwen/Qwen2.5-3B-Instruct"

# Download the model and tokenizer
for model_name in (tiny_model_name, medium_model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    target_directory = "../"+model_name
    tokenizer.save_pretrained(target_directory)
    model.save_pretrained(target_directory)
```

### [Click here](https://colab.research.google.com/drive/1YqTznro-FEL2CowUZ_WNOfAa-ooqQIhZ?usp=sharing) for an example setup at Colab

Caution! The free GPU runtimes are very limited! Make sure to disconnect and delete your runtime when you spend time writing code or switch to another task. Using colab GPU runtimes for too long might result in losing access to them  (inceased wait times and/or short session durations).

If any of this happens to you, please consult with the professor.

## ECE405 Assignment instructions

Follow along the [CS336@Stanford handout](./cs336_spring2025_assignment5_alignment.pdf) with small deviations:
1. What the code looks like: clone https://github.com/igormolybog/s2025-assignment3-alignment.git
2. What you can use: Implementation from scratch is preferred, but experiments are essential. If you are stuck with some implementation, just use the Huggingface/Pytorch implementation (e.g. the Trainer class) and proceed to the experiments
    - Submit the report reflecting your attempts at implementation for partial credit
3. How to submit: You will submit the report on the assignment to [Assignment Submission Form](https://docs.google.com/forms/d/e/1FAIpQLScJg_QkwjKux3xKeM-EOmZyvA6zlbVIrf_lxN_qoCFoxdqTrg/viewform?usp=sharing&ouid=111841773839267096112). The code does not have to be attached as long as you include links to the main GitHub branch where your code lives and links to all of the Colab notebooks if applicable.
    - You don't need to submit to leaderboard.
4. We are going to use [Qwen2.5](https://arxiv.org/abs/2412.15115) models instead of LLaMa3.1 models. Instead of LLaMa3 8B Base we will use [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) and instead of LLaMa3 70B Instruct we will use [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
    - You will have to download them.
5. Problem (alpaca_eval_baseline) (c) may require you to edit ```scripts/alpaca_eval_vllm_llama3_70b_fn```. The model_name needs to be the path to the local directory where the Qwen2.5-3B-Instruct model is downloaded to.
6. Problem (sst_baseline) (c) you will need to provide the path to the Qwen2.5-3B-Instruct model
7. In Section 4.2.2 you don't have to install FlashAttention-2 unless you run a high-end GPU (A40 and above). Remove all the lines like ```attn_implementation="flash_attention_2"``` and replace ```bfloat16``` with ```float32```
8. For Problem (sft) feel free to reduce the number of sequences per batch down to 1 (with gradient accumulation) and disable activation checkpointing. You can adjust the number of training steps to train the model for just half an hour using T4 GPU instead of the 24 H100 hours suggested by the handout. The point is to start training and to make progressive updates of the model weights that lead to the reduction of the target loss function. 
9. In Problem (dpo_training): train the model for half an hour instead of making an entire epoch through the HH dataset. Use a single GPU to query both reference model and trained model consecutively.

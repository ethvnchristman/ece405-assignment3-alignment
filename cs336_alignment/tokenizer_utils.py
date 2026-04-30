from __future__ import annotations
import io
import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    combined = [p + o for p, o in zip(prompt_ids, output_ids)]
    max_len = max(len(c) for c in combined)
    pad_tok = tokenizer.eos_token_id

    padded = [c + [pad_tok] * (max_len - len(c)) for c in combined]
    full = torch.tensor(padded)

    input_ids = full[:, :-1]
    labels = full[:, 1:]

    seq_len = max_len - 1
    response_mask = torch.zeros(len(prompt_strs), seq_len, dtype=torch.bool)
    for i, (p, o) in enumerate(zip(prompt_ids, output_ids)):
        start = len(p) - 1
        end = start + len(o)
        response_mask[i, start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    return -(log_p * log_p.exp()).sum(dim=-1)


def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    out = model(input_ids=input_ids)
    logits = out.logits

    bs, seq, vocab = logits.shape
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)
    log_p_flat = F.log_softmax(flat_logits, dim=-1)
    tok_lp = log_p_flat[torch.arange(bs * seq), flat_labels].view(bs, seq)

    result = {"log_probs": tok_lp}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(tensor, mask, dim=None, normalize_constant=1.0):
    if dim is not None:
        return (tensor * mask).sum(dim=dim) / normalize_constant
    return (tensor * mask).sum() / normalize_constant


def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps,
                               normalize_constant=1.0):
    per_tok = policy_log_probs * response_mask
    loss = -(per_tok.sum(dim=-1) / (normalize_constant * gradient_accumulation_steps)).mean()
    loss.backward()
    n_resp = response_mask.sum().item()
    metrics = {
        "loss": loss.item(),
        "mean_log_prob": per_tok.sum().item() / n_resp if n_resp > 0 else 0.0,
    }
    return loss, metrics


def log_generations(input_prompts, rollout_responses, ground_truths, reward_dicts,
                    token_entropies=None, response_mask=None, log_path=None):
    buf = io.StringIO()
    n = len(input_prompts)

    for i in range(n):
        buf.write(f"{'='*60}\n[{i+1}/{n}]\n")
        buf.write(f"  PROMPT      : {input_prompts[i]}\n")
        buf.write(f"  RESPONSE    : {rollout_responses[i]}\n")
        buf.write(f"  GROUND TRUTH: {ground_truths[i]}\n")
        rd = reward_dicts[i]
        buf.write(
            f"  REWARD      : total={rd.get('reward', float('nan')):.4f} | "
            f"format={rd.get('format_reward', float('nan')):.4f} | "
            f"answer={rd.get('answer_reward', float('nan')):.4f}\n"
        )
        if token_entropies is not None and response_mask is not None:
            m = response_mask[i].bool()
            avg_ent = token_entropies[i][m].mean().item() if m.any() else float("nan")
            buf.write(f"  ENTROPY (avg): {avg_ent:.4f}\n")
        if response_mask is not None:
            buf.write(f"  RESP LEN    : {int(response_mask[i].sum().item())}\n")

    buf.write(f"{'='*60}\n[Batch Stats]\n")

    if token_entropies is not None and response_mask is not None:
        m_all = response_mask.bool()
        n_tok = m_all.sum().item()
        avg_ent_all = (token_entropies * m_all).sum().item() / n_tok if n_tok > 0 else float("nan")
        buf.write(f"  Avg entropy     : {avg_ent_all:.4f}\n")

    if response_mask is not None:
        lens = response_mask.bool().sum(dim=-1).float()
    else:
        lens = torch.tensor([len(r.split()) for r in rollout_responses], dtype=torch.float)

    is_correct = torch.tensor([rd.get("answer_reward", 0.0) > 0.0 for rd in reward_dicts])
    buf.write(f"  Avg len         : {lens.mean().item():.1f}\n")

    if is_correct.any():
        buf.write(f"  Avg len correct : {lens[is_correct].mean().item():.1f}\n")
    else:
        buf.write(f"  Avg len correct : nan\n")

    if (~is_correct).any():
        buf.write(f"  Avg len wrong   : {lens[~is_correct].mean().item():.1f}\n")
    else:
        buf.write(f"  Avg len wrong   : nan\n")

    buf.write(f"{'='*60}\n")

    text = buf.getvalue()
    if log_path:
        with open(log_path, "a") as f:
            f.write(text)
    else:
        print(text, end="")

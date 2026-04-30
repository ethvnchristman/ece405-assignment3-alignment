from __future__ import annotations
import torch
from cs336_alignment.tokenizer_utils import masked_normalize


def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths,
                                     group_size, advantage_eps, normalize_by_std):
    scores = [reward_fn(r, g) for r, g in zip(rollout_responses, repeated_ground_truths)]
    raw = torch.tensor([s["reward"] for s in scores])
    fmt_vals = [s["format_reward"] for s in scores]
    ans_vals = [s["answer_reward"] for s in scores]

    n_groups = len(rollout_responses) // group_size
    grouped = raw.view(n_groups, group_size)

    group_means = grouped.mean(dim=1, keepdim=True)
    group_stds = grouped.std(dim=1, keepdim=True) + advantage_eps

    if normalize_by_std:
        advantages = (grouped - group_means) / group_stds
    else:
        advantages = grouped - group_means

    advantages = advantages.view(-1)
    raw_flat = grouped.view(-1)

    n = len(rollout_responses)
    meta = {
        "mean_reward": raw_flat.mean().item(),
        "std_reward": raw_flat.std().item(),
        "max_reward": raw_flat.max().item(),
        "min_reward": raw_flat.min().item(),
        "mean_format_reward": sum(fmt_vals) / n,
        "mean_answer_reward": sum(ans_vals) / n,
    }
    return advantages, raw_flat, meta


def compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs):
    return -raw_rewards_or_advantages.expand_as(policy_log_probs) * policy_log_probs


def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    expanded_adv = advantages.expand_as(policy_log_probs)

    loss = torch.max(-expanded_adv * ratio, -expanded_adv * clipped)
    meta = {
        "ratio": ratio.mean().item(),
        "clipped_ratio": clipped.mean().item(),
        "percent_clipped": ((ratio > 1.0 + cliprange) | (ratio < 1.0 - cliprange)).float().mean().item(),
    }
    return loss, meta


def compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards=None,
                                  advantages=None, old_log_probs=None, cliprange=None):
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def masked_mean(tensor, mask, dim=None):
    if dim is not None:
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).float()
    return (tensor * mask).sum() / mask.sum().float()


def grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps,
                                loss_type, raw_rewards=None, advantages=None,
                                old_log_probs=None, cliprange=None):
    meta = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        loss, meta = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "GRPO-No-CLIP":
        assert advantages is not None and old_log_probs is not None
        ratio = torch.exp(policy_log_probs - old_log_probs)
        loss = -advantages.expand_as(policy_log_probs) * ratio

    scalar = masked_mean(loss, response_mask, dim=1).mean() / gradient_accumulation_steps
    scalar.backward()
    return scalar.detach(), meta

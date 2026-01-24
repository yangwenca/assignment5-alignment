from typing import Callable, Dict, List, Literal, Tuple

import torch


"""
uv run pytest -k test_compute_group_normalized_rewards
uv run pytest -k test_compute_naive_policy_gradient_loss
uv run pytest -k test_compute_grpo_clip_loss
uv run pytest -k test_compute_policy_gradient_loss
uv run pytest -k test_masked_mean
uv run pytest -k test_grpo_microbatch_train_step
"""


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_reward = [reward_fn(response, gt)["reward"] for (response, gt) in zip(rollout_responses, repeated_ground_truths)]
    raw_reward = torch.tensor(raw_reward, device=device)
    rewards_per_group = raw_reward.reshape((-1, group_size))
    mean_rewards_per_group = torch.mean(rewards_per_group, dim=-1, keepdim=True)
    advantage = rewards_per_group - mean_rewards_per_group

    if normalize_by_std:
        std_rewards_per_group = torch.std(rewards_per_group, dim=-1, keepdim=True)
        advantage /= (std_rewards_per_group + advantage_eps)
    advantage = advantage.flatten()
    metadata = {
        "mean": torch.mean(raw_reward),
        "std": torch.std(raw_reward),
        "max": torch.max(raw_reward),
        "min": torch.min(raw_reward),
    }

    return advantage, raw_reward, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token,
    where raw_rewards_or_advantages is either the raw reward or an already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1),
            scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length),
            logprobs for each token.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length),
        the per-token policy-gradient loss (to be aggregated across the batch and sequence dimensions in the training loop).

    Implementation tips:
        • Broadcast the raw_rewards_or_advantages over the sequence_length dimension.
    """
    assert raw_rewards_or_advantages.dim() == 2
    assert raw_rewards_or_advantages.size(-1) == 1
    assert policy_log_probs.dim() == 2
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length),
            per-token log probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the old policy.
        cliprange: float Clip parameter ε (e.g. 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
        metadata dict containing whatever you want to log.
        We suggest logging whether each token was clipped or not, i.e.,
        whether the clipped policy gradient loss on the RHS of the min was lower than the LHS.

    Implementation tips:
    • Broadcast advantages over sequence_length.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clip = torch.clamp(ratio, min=1-cliprange, max=1+cliprange) * advantages
    left = ratio * advantages
    metadata = {"left": left >= clip}
    loss = -torch.min(left, clip)
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise NotImplementedError


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    return torch.sum(tensor.masked_fill(~mask, 0), dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    # aggregate to a scalar loss per example
    loss = masked_mean(loss, response_mask, dim=-1)
    # average over the batch dimension
    loss = loss.mean()
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, metadata

from typing import Callable, Dict, List, Literal, Tuple

import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import *
from cs336_alignment.math_baseline import eval_data, train_data
from vllm import SamplingParams

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


def train() -> None:
    model_name = "/workspace/huggingface/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    output_path="/workspace/alignment/data/grpo/"
    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )
    # training data
    train_prompts, train_gt = train_data()
    total = len(train_prompts)
    data = tokenize_prompt_and_output(
        prompt_strs=train_prompts,
        output_strs=train_gt,
        tokenizer=tokenizer,
        device=device,
    )
    # eval data
    eval_prompts, eval_gt = eval_data()
    llm_model = init_vllm(model_name, device, seed, 0.2)

    n_grpo_steps: int = 200
    eval_step: int = 40
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = rollout_batch_size # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.2 # only one gpu, 0.2 can fit in one gpu
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    assert train_batch_size // rollout_batch_size == epochs_per_rollout_batch
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    expert_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size,
        seed=seed,
    )
    # get a baseline
    evaluate(
        model=llm_model,
        output_path=output_path,
        prompts=train_prompts,
        ground_truth=train_gt,
    )
    for step in range(n_grpo_steps):
        batch_indices = torch.randint(0, total, (train_batch_size,), device=device)
        cur_prompts = [train_prompts[idx] for idx in batch_indices]
        load_policy_into_vllm_instance(model, llm_model)
        generations = llm_model.generate(cur_prompts, expert_sampling_params)
        gt = [train_gt[idx] for idx in batch_indices for _ in range(group_size)]
        rollout_responses_output = [
            output.text
            for request_output in generations
            for output in request_output.outputs
        ]
        advantage, raw_reward, metedata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses_output,
            repeated_ground_truths=gt,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
            device=device,
        )
        inputs_total = data["input_ids"][batch_indices]
        labels_total = data["labels"][batch_indices]
        response_total = data["response_mask"][batch_indices]
        for train_step in range(epochs_per_rollout_batch):
            for micro_step in range(gradient_accumulation_steps):
                start = micro_step * micro_train_batch_size
                end = (micro_step + 1) * micro_train_batch_size
                inputs_cur = inputs_total[start:end]
                labels_cur = labels_total[start:end]
                response_cur = response_total[start:end]
                with torch.no_grad():
                    with torch.amp.autocast(device):
                        log_probs = get_response_log_probs(
                            model=model,
                            input_ids=inputs_cur,
                            labels=labels_cur,
                            return_token_entropy=True,
                        )
                        old_log_probs = log_probs["log_probs"]
                log_probs = get_response_log_probs(
                    model=model,
                    input_ids=inputs_cur,
                    labels=labels_cur,
                    return_token_entropy=True,
                )
                cur_log_probs = log_probs["log_probs"]
                # raw_rewards and advantage are in the shape of (rollout_batch_size,)
                # need to repeat log_probs and old_log_probs group_size times
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=cur_log_probs.repeat_interleave(group_size, dim=0),
                    response_mask=response_cur.repeat_interleave(group_size, dim=0),
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=raw_reward[start*group_size:end*group_size].unsqueeze(-1),
                    advantages=advantage[start*group_size:end*group_size].unsqueeze(-1),
                    old_log_probs=old_log_probs.repeat_interleave(group_size, dim=0),
                    cliprange=None,
                )
                print(f"grpo step is {step}, train_step is {train_step}, micro_step is {micro_step}, loss is {loss} ")
            # update gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        load_policy_into_vllm_instance(model, llm_model)
        evaluate(
            model=llm_model,
            output_path=output_path,
            prompts=train_prompts,
            ground_truth=train_gt,
        )
        if (step + 1) % eval_step == 0:
            print("evaluation data set")
            evaluate(
                model=llm_model,
                output_path=output_path,
                prompts=eval_prompts,
                ground_truth=eval_gt,
            )
            model.save_pretrained(save_directory=output_path)
            tokenizer.save_pretrained(save_directory=output_path)


train()

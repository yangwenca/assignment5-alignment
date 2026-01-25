from pathlib import Path
from typing import Callable, List, Literal, Tuple

import torch
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.sft import *
from cs336_alignment.math_baseline import eval_data, get_data, train_data
from vllm import SamplingParams


BASE_DIR = Path(__file__).resolve().parent


"""
uv run pytest -k test_compute_group_normalized_rewards
uv run pytest -k test_compute_naive_policy_gradient_loss
uv run pytest -k test_compute_grpo_clip_loss
uv run pytest -k test_compute_policy_gradient_loss
uv run pytest -k test_masked_mean
uv run pytest -k test_grpo_microbatch_train_step
"""


"""
learning rate 1e-5
{'num_examples': 1319, 'reward': 0.06368460955269144, 'format_reward': 0.18271417740712662, 'answer_reward': 0.06368460955269144}
{'num_examples': 1319, 'reward': 0.48597422289613346, 'format_reward': 0.8316906747536013, 'answer_reward': 0.48597422289613346}
{'num_examples': 1319, 'reward': 0.5526914329037149, 'format_reward': 0.9021986353297953, 'answer_reward': 0.5526914329037149}
{'num_examples': 1319, 'reward': 0.6072782410917361, 'format_reward': 0.9347990902198635, 'answer_reward': 0.6072782410917361}
{'num_examples': 1319, 'reward': 0.6391205458680819, 'format_reward': 0.9279757391963609, 'answer_reward': 0.6391205458680819}
{'num_examples': 1319, 'reward': 0.6095526914329037, 'format_reward': 0.9044730856709629, 'answer_reward': 0.6095526914329037}
{'num_examples': 1319, 'reward': 0.6512509476876421, 'format_reward': 0.9196360879454132, 'answer_reward': 0.6512509476876421}
{'num_examples': 1319, 'reward': 0.6800606520090978, 'format_reward': 0.9416224412433661, 'answer_reward': 0.6800606520090978}
{'num_examples': 1319, 'reward': 0.6717210007581501, 'format_reward': 0.9378316906747536, 'answer_reward': 0.6717210007581501}
{'num_examples': 1319, 'reward': 0.6974981046247157, 'format_reward': 0.9408642911296436, 'answer_reward': 0.6974981046247157}
{'num_examples': 1319, 'reward': 0.7043214556482184, 'format_reward': 0.9332827899924185, 'answer_reward': 0.7043214556482184}

no_baseline
no baseline can only achieve 20%
reinforce_with_baseline can achieve 70%
{'num_examples': 1319, 'reward': 0.06368460955269144, 'format_reward': 0.18271417740712662, 'answer_reward': 0.06368460955269144}
{'num_examples': 1319, 'reward': 0.16603487490523122, 'format_reward': 0.6209249431387415, 'answer_reward': 0.16603487490523122}
{'num_examples': 1319, 'reward': 0.20318423047763456, 'format_reward': 0.645185746777862, 'answer_reward': 0.20318423047763456}
{'num_examples': 1319, 'reward': 0.21152388172858225, 'format_reward': 0.6482183472327521, 'answer_reward': 0.21152388172858225}
{'num_examples': 1319, 'reward': 0.19939347990902198, 'format_reward': 0.6535253980288097, 'answer_reward': 0.19939347990902198}
{'num_examples': 1319, 'reward': 0.20318423047763456, 'format_reward': 0.6254738438210766, 'answer_reward': 0.20318423047763456}
{'num_examples': 1319, 'reward': 0.19636087945413191, 'format_reward': 0.643669446550417, 'answer_reward': 0.19636087945413191}
{'num_examples': 1319, 'reward': 0.20318423047763456, 'format_reward': 0.6497346474601972, 'answer_reward': 0.20318423047763456}
{'num_examples': 1319, 'reward': 0.20090978013646701, 'format_reward': 0.6595905989385898, 'answer_reward': 0.20090978013646701}
{'num_examples': 1319, 'reward': 0.20318423047763456, 'format_reward': 0.6838514025777104, 'answer_reward': 0.20318423047763456}
{'num_examples': 1319, 'reward': 0.19863532979529946, 'format_reward': 0.6967399545109931, 'answer_reward': 0.19863532979529946}

length normalization
Masked mean: per token
Pros: length invaraint gradients, stable optimization, fair credit across variable length responses,
Cons: penalized long response, reward is entire response not per token,
Masked norm: per sequence
Pros: Reward scales with the total log-probability of the sequence, length becomes an implicit decision variable
Cons: gradient magnitude depends on length, high variance

For RL, reward should scale with sequence length, should use masked norm

experiments:
Masked norm
{'num_examples': 1319, 'reward': 0.06368460955269144, 'format_reward': 0.18271417740712662, 'answer_reward': 0.06368460955269144}
{'num_examples': 1319, 'reward': 0.07657316148597422, 'format_reward': 0.32221379833206976, 'answer_reward': 0.07657316148597422}
{'num_examples': 1319, 'reward': 0.0712661106899166, 'format_reward': 0.34420015163002277, 'answer_reward': 0.0712661106899166}
{'num_examples': 1319, 'reward': 0.07960576194086429, 'format_reward': 0.34723275208491283, 'answer_reward': 0.07960576194086429}
{'num_examples': 1319, 'reward': 0.07202426080363912, 'format_reward': 0.33510235026535257, 'answer_reward': 0.07202426080363912}
{'num_examples': 1319, 'reward': 0.06520090978013647, 'format_reward': 0.3479909021986353, 'answer_reward': 0.06520090978013647}
from experiments, masked norm is worse than masked mean (baseline).
even though theoretically, masked norm is better.
In reality, masked mean works better because it reduces variance and stabilizeds optimization.
Per token losses are noisy,
masked norm: accumulate noise linearly with sequence length, longer sequences, huge graident variance/magnitude
masked mean: keeps gradient magnitudes similar across examples

grpo_group_standard_deviation
use_std_normalization = False
{'num_examples': 1319, 'reward': 0.06368460955269144, 'format_reward': 0.18271417740712662, 'answer_reward': 0.06368460955269144}
{'num_examples': 1319, 'reward': 0.4040940106141016, 'format_reward': 0.7012888551933283, 'answer_reward': 0.4040940106141016}
{'num_examples': 1319, 'reward': 0.5572403335860501, 'format_reward': 0.9006823351023503, 'answer_reward': 0.5572403335860501}
{'num_examples': 1319, 'reward': 0.5845337376800607, 'format_reward': 0.9241849886277483, 'answer_reward': 0.5845337376800607}
{'num_examples': 1319, 'reward': 0.621683093252464, 'format_reward': 0.9257012888551933, 'answer_reward': 0.621683093252464}
{'num_examples': 1319, 'reward': 0.6603487490523123, 'format_reward': 0.9446550416982562, 'answer_reward': 0.6603487490523123}
{'num_examples': 1319, 'reward': 0.6641394996209249, 'format_reward': 0.9431387414708112, 'answer_reward': 0.6641394996209249}
{'num_examples': 1319, 'reward': 0.6793025018953753, 'format_reward': 0.9461713419257013, 'answer_reward': 0.6793025018953753}
{'num_examples': 1319, 'reward': 0.6641394996209249, 'format_reward': 0.9408642911296436, 'answer_reward': 0.6641394996209249}
{'num_examples': 1319, 'reward': 0.6944655041698257, 'format_reward': 0.9522365428354814, 'answer_reward': 0.6944655041698257}
{'num_examples': 1319, 'reward': 0.6944655041698257, 'format_reward': 0.9484457922668689, 'answer_reward': 0.6944655041698257}
whether use_std_norm True or False, makes no difference
Advantage is scaled by std
gradient direction is unchanged, only magnitude changes, adam cancels the scale
Adam
theta = theta - alpha * m / (sqrt(v))
m ~ c
v ~ c**2

grpo_off_policy
rollout_batch_size = 256
train_batch_size = 128
epochs_per_rollout_batch=2,
gradient_accumulation_steps=64
{'num_examples': 1319, 'reward': 0.06368460955269144, 'format_reward': 0.18271417740712662, 'answer_reward': 0.06368460955269144}
{'num_examples': 1319, 'reward': 0.41925701288855194, 'format_reward': 0.7862016679302501, 'answer_reward': 0.41925701288855194}
{'num_examples': 1319, 'reward': 0.4973464746019712, 'format_reward': 0.846095526914329, 'answer_reward': 0.4973464746019712}
{'num_examples': 1319, 'reward': 0.5428354814253222, 'format_reward': 0.8794541319181198, 'answer_reward': 0.5428354814253222}
{'num_examples': 1319, 'reward': 0.5739196360879454, 'format_reward': 0.9203942380591357, 'answer_reward': 0.5739196360879454}
{'num_examples': 1319, 'reward': 0.5640636846095527, 'format_reward': 0.8953752843062927, 'answer_reward': 0.5640636846095527}
{'num_examples': 1319, 'reward': 0.6080363912054587, 'format_reward': 0.9082638362395754, 'answer_reward': 0.6080363912054587}
{'num_examples': 1319, 'reward': 0.6087945413191812, 'format_reward': 0.9128127369219106, 'answer_reward': 0.6087945413191812}
{'num_examples': 1319, 'reward': 0.6345716451857468, 'format_reward': 0.9294920394238059, 'answer_reward': 0.6345716451857468}
{'num_examples': 1319, 'reward': 0.6277482941622441, 'format_reward': 0.9317664897649734, 'answer_reward': 0.6277482941622441}
{'num_examples': 1319, 'reward': 0.6330553449583017, 'format_reward': 0.9158453373768006, 'answer_reward': 0.6330553449583017}
compare to on policy, do not see any benefits, it might be due to hyperparameters

off_policy_GRPO_no_clip_ablation
no_clip training is unstable, and new policy can be very different from old policy
{'num_examples': 1319, 'reward': 0.06368460955269144, 'format_reward': 0.18271417740712662, 'answer_reward': 0.06368460955269144}
{'num_examples': 1319, 'reward': 0.5018953752843063, 'format_reward': 0.8582259287338894, 'answer_reward': 0.5018953752843063}
{'num_examples': 1319, 'reward': 0.5428354814253222, 'format_reward': 0.9059893858984079, 'answer_reward': 0.5428354814253222}
{'num_examples': 1319, 'reward': 0.5564821834723275, 'format_reward': 0.9067475360121304, 'answer_reward': 0.5564821834723275}
{'num_examples': 1319, 'reward': 0.0, 'format_reward': 0.0, 'answer_reward': 0.0}
loss = -advantage * ratio (ratio = exp(log_new_policy - log_old_policy))
exponentiaition blow up, small log_prob diff can become big,
long esequences multiply the problem: each token contributes, one bad token poisons the entire sequences loss

grpo_prompt_ablation
{'num_examples': 1319, 'reward': 0.5041698256254739, 'format_reward': 0.800606520090978, 'answer_reward': 0.5041698256254739}
{'num_examples': 1319, 'reward': 0.5579984836997726, 'format_reward': 0.844579226686884, 'answer_reward': 0.5579984836997726}
{'num_examples': 1319, 'reward': 0.5807429871114481, 'format_reward': 0.865049279757392, 'answer_reward': 0.5807429871114481}
{'num_examples': 1319, 'reward': 0.6186504927975739, 'format_reward': 0.8809704321455648, 'answer_reward': 0.6186504927975739}
{'num_examples': 1319, 'reward': 0.6489764973464746, 'format_reward': 0.8877937831690674, 'answer_reward': 0.6489764973464746}
{'num_examples': 1319, 'reward': 0.6459438968915845, 'format_reward': 0.890068233510235, 'answer_reward': 0.6459438968915845}
{'num_examples': 1319, 'reward': 0.6800606520090978, 'format_reward': 0.9006823351023503, 'answer_reward': 0.6800606520090978}
{'num_examples': 1319, 'reward': 0.6633813495072024, 'format_reward': 0.8855193328279, 'answer_reward': 0.6633813495072024}
{'num_examples': 1319, 'reward': 0.6929492039423806, 'format_reward': 0.9044730856709629, 'answer_reward': 0.6929492039423806}
{'num_examples': 1319, 'reward': 0.6944655041698257, 'format_reward': 0.9044730856709629, 'answer_reward': 0.6944655041698257}
{'num_examples': 1319, 'reward': 0.6800606520090978, 'format_reward': 0.9006823351023503, 'answer_reward': 0.6800606520090978}
the question prompt is easy and the requirement for llm is easy.
At the beginning, it is 50% vs 6% for question prompt vs r1 zero prompt.
Thus, the model at the beginning is about to provide good answers.
However, the question prompt does not encourage the CoT in the model. Thus, the model is hard to train to improve.
At the end, it is 69% vs 70%. The model based on r1 zero prompt is able to improve significantly.
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


def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length),
            per-token log probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the old policy.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token unclipped loss.
        metadata dict containing whatever you want to log.

    Implementation tips:
    • Broadcast advantages over sequence_length.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    loss = -ratio * advantages
    metadata = {}
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
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
    elif loss_type == "grpo_no_clip":
        return compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs)
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
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
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
    # loss = masked_normalize(loss, response_mask, dim=-1, normalize_constant=response_mask.size(-1))
    loss = masked_mean(loss, response_mask, dim=-1)
    # average over the batch dimension
    loss = loss.mean()
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def eval_data_question() -> Tuple[List[str], List[str]]:
    data_path = BASE_DIR / "../data/gsm8k/test.jsonl"
    prompt_path = BASE_DIR / "prompts/question_only.prompt"
    return get_data(data_path, prompt_path)


def train_data_question() -> Tuple[List[str], List[str]]:
    data_path = BASE_DIR / "../data/gsm8k/train.jsonl"
    prompt_path = BASE_DIR / "prompts/question_only.prompt"
    return get_data(data_path, prompt_path)


def train(
    epochs_per_rollout_batch: int = 1, # On-policy
    gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"] = "reinforce_with_baseline",
    use_question_prompt: bool = False,
) -> None:
    model_name = "/workspace/huggingface/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    output_path = BASE_DIR / "../data/grpo/"
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
    # training/eval data
    if use_question_prompt:
        train_prompts, train_gt = train_data_question()
        eval_prompts, eval_gt = eval_data_question()
    else:
        train_prompts, train_gt = train_data()
        eval_prompts, eval_gt = eval_data()
    total = len(train_prompts)

    llm_model = init_vllm(model_name, device, seed, 0.2)

    n_grpo_steps: int = 200
    eval_step: int = 20
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    train_batch_size: int = rollout_batch_size // epochs_per_rollout_batch # On-policy
    gpu_memory_utilization: float = 0.2 # only one gpu, 0.2 can fit in one gpu
    use_std_normalization: bool = True
    cliprange: float = 0.2
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    assert rollout_batch_size // train_batch_size  == epochs_per_rollout_batch
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
        prompts=eval_prompts,
        ground_truth=eval_gt,
        reward_fn=question_only_reward_fn if use_question_prompt else r1_zero_reward_fn,
    )
    for step in range(n_grpo_steps):
        batch_indices = torch.randint(0, total, (n_prompts_per_rollout_batch,), device=device)
        cur_prompts = [train_prompts[idx] for idx in batch_indices]
        load_policy_into_vllm_instance(model, llm_model)
        generations = llm_model.generate(cur_prompts, expert_sampling_params)
        gt = [train_gt[idx] for idx in batch_indices for _ in range(group_size)]
        rollout_responses_output = [
            output.text
            for request_output in generations
            for output in request_output.outputs
        ]
        cur_prompts_group = [prompt for prompt in cur_prompts for _ in range(group_size)]
        data = tokenize_prompt_and_output(
            prompt_strs=cur_prompts_group,
            output_strs=rollout_responses_output,
            tokenizer=tokenizer,
            device=device,
        )
        advantage, raw_reward, metedata = compute_group_normalized_rewards(
            reward_fn=question_only_reward_fn if use_question_prompt else r1_zero_reward_fn,
            rollout_responses=rollout_responses_output,
            repeated_ground_truths=gt,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
            device=device,
        )
        inputs_total = data["input_ids"]
        labels_total = data["labels"]
        response_total = data["response_mask"]
        old_log_probs_total = []
        for rollout_batch_idx in range(n_microbatches_per_rollout_batch):
            start = rollout_batch_idx * micro_train_batch_size
            end = (rollout_batch_idx + 1) * micro_train_batch_size
            inputs_cur = inputs_total[start:end]
            labels_cur = labels_total[start:end]
            response_cur = response_total[start:end]
            with torch.inference_mode():
                with torch.amp.autocast(device):
                    old_log_probs = get_response_log_probs(
                        model=model,
                        input_ids=inputs_cur,
                        labels=labels_cur,
                        return_token_entropy=True,
                    )
                    old_log_probs_total.append(old_log_probs["log_probs"])
        old_log_probs_total = torch.cat(old_log_probs_total)
        for train_epoch_idx in range(epochs_per_rollout_batch):
            start_train = train_epoch_idx * train_batch_size
            loss_total = 0
            for micro_step in range(gradient_accumulation_steps):
                start = start_train + micro_step * micro_train_batch_size
                end = start_train + (micro_step + 1) * micro_train_batch_size
                inputs_cur = inputs_total[start:end]
                labels_cur = labels_total[start:end]
                response_cur = response_total[start:end]
                with torch.amp.autocast(device):
                    cur_log_probs = get_response_log_probs(
                        model=model,
                        input_ids=inputs_cur,
                        labels=labels_cur,
                        return_token_entropy=True,
                    )
                    cur_log_probs = cur_log_probs["log_probs"]
                # raw_rewards and advantage are in the shape of (rollout_batch_size,)
                # need to repeat log_probs and old_log_probs group_size times
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=cur_log_probs,
                    response_mask=response_cur,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=raw_reward[start:end].unsqueeze(-1),
                    advantages=advantage[start:end].unsqueeze(-1),
                    old_log_probs=old_log_probs_total[start:end],
                    cliprange=cliprange,
                )
                loss_total += loss
            print(f"grpo step is {step}, train epoch is {train_epoch_idx}, loss is {loss_total} ")
            # update gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        if (step + 1) % eval_step == 0:
            load_policy_into_vllm_instance(model, llm_model)
            evaluate(
                model=llm_model,
                output_path=output_path,
                prompts=eval_prompts,
                ground_truth=eval_gt,
                reward_fn=question_only_reward_fn if use_question_prompt else r1_zero_reward_fn,
            )
            model.save_pretrained(save_directory=output_path)
            tokenizer.save_pretrained(save_directory=output_path)


# baseline on policy
# train()

# off policy
# train(
#     epochs_per_rollout_batch=2,
#     gradient_accumulation_steps=64,
#     loss_type="grpo_no_clip",
# )

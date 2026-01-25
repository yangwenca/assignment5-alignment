from unittest.mock import patch

import torch
from cs336_alignment.math_baseline import eval_data, evaluate, format_prompt, load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed

"""
uv run pytest -k test_tokenize_prompt_and_output
uv run pytest -k test_compute_entropy
uv run pytest -k test_get_response_log_probs
uv run pytest -k test_masked_normalize
uv run pytest -k test_sft_microbatch_train_step
"""


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs)['input_ids']
    output_ids = tokenizer(output_strs)['input_ids']
    input_ids = [prompt + output for (prompt, output) in zip(prompt_ids, output_ids)]
    max_len = max([len(t) for t in input_ids])

    batch_size = len(prompt_strs)
    ids = torch.zeros((batch_size, max_len - 1), dtype=torch.long, device=device)
    labels = torch.zeros((batch_size, max_len - 1), dtype=torch.long, device=device)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool, device=device)
    for idx, input_id in enumerate(input_ids):
        diff = max(0, max_len - len(input_id))
        total = torch.tensor(input_id + [tokenizer.pad_token_id] * diff)
        ids[idx] = total[:-1]
        labels[idx] = total[1:]
        response_mask[idx, len(prompt_ids[idx])-1: len(input_id)-1] = True

    return {
        "input_ids": ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits shape: (B, T, V)
    # 1. Compute log probabilities in a numerically stable way
    # log_probs = log_softmax = logits - logsumexp(logits)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # (B, T, V)
    # 2. Compute probabilities
    probs = torch.exp(log_probs)  # (B, T, V)
    # 3. Compute entropy: -sum(p * log(p)) over vocab dimension
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, T)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    outputs = model(input_ids).logits
    log_probs = outputs - torch.logsumexp(outputs, dim=-1, keepdim=True)
    ans = {}
    ans["log_probs"] = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        ans["token_entropy"] = compute_entropy(outputs)
    return ans


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    return torch.sum(tensor.masked_fill(~mask, 0), dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant).mean()
    loss /= gradient_accumulation_steps
    loss.backward()

    metadata = {
        "normalize_constant": normalize_constant,
        "loss": loss.detach(),
    }
    return (loss, metadata)


def log_generateion(
    idx: int,
    loss: float,
) -> None:
    print(f"at training iteration {idx}, loss is {loss}")


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
        22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def training_loop():
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    model_name = "/workspace/huggingface/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    data_path = BASE_DIR / "../data/gsm8k/train.jsonl"
    prompt_path = BASE_DIR / "prompts/r1_zero.prompt"
    output_path = BASE_DIR / "../data/sft/"
    device="cuda"
    batch_size = 8
    start_idx = 0
    training_step = 1000
    eval_step = 100
    gradient_accumulation_steps = 8
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

    samples = load_file(data_path)
    total = len(samples)
    prompts = format_prompt(samples, prompt_path)
    ground_truth = [sample["answer"] for sample in samples]

    data = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=ground_truth,
        tokenizer=tokenizer,
        device=device,
    )

    eval_prompts, eval_gt = eval_data()
    llm_model = init_vllm(model_name, device, 42, 0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for idx in range(training_step):
        if start_idx + batch_size >= total:
            start_idx = 0
        inputs = data["input_ids"][start_idx:start_idx+batch_size]
        labels = data["labels"][start_idx:start_idx+batch_size]
        with torch.amp.autocast(device):
            log_probs = get_response_log_probs(
                model=model,
                input_ids=inputs,
                labels=labels,
                return_token_entropy=True,
            )
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=log_probs["log_probs"],
                response_mask=data["response_mask"][start_idx:start_idx+batch_size],
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=data["response_mask"].size(-1),
            )
            # print(f"iteration {idx}, loss is {loss}")
        start_idx += batch_size
        if (idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            log_generateion(
                idx=idx,
                loss=loss,
            )

        if (idx + 1) % eval_step == 0:
            load_policy_into_vllm_instance(model, llm_model)
            evaluate(
                model=llm_model,
                output_path=output_path,
                prompts=eval_prompts,
                ground_truth=eval_gt,
            )
            model.save_pretrained(save_directory=output_path)
            tokenizer.save_pretrained(save_directory=output_path)

# training_loop()

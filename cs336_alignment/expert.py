from pathlib import Path

import random
import torch
from vllm import SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import *


BASE_DIR = Path(__file__).resolve().parent


def expert_iteration():
    model_name = "/workspace/huggingface/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    data_path = BASE_DIR / "../data/gsm8k/train.jsonl"
    prompt_path = BASE_DIR / "prompts/r1_zero.prompt"
    output_path = BASE_DIR / "../data/expert/"
    device="cuda"
    expert_batch_size = 24
    train_batch_size = 8
    expert_iteration = 100
    training_step = 20
    eval_step = 20
    gradient_accumulation_steps = 4
    seed = 42
    sample_count = 3
    random.seed(seed)
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
    llm_model = init_vllm(model_name, device, seed, 0.2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    expert_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        min_tokens=4,
        stop=["</answer>"], 
        include_stop_str_in_output=True,
        n=sample_count,
        seed=seed,
    )
    samples = [i for i in range(len(prompts))]
    for _ in range(expert_iteration):
        training_data = []
        load_policy_into_vllm_instance(model, llm_model)
        while len(training_data) < 32:
            batch_indices = random.sample(samples, expert_batch_size)
            cur_prompts = [prompts[idx] for idx in batch_indices]
            generations = llm_model.generate(cur_prompts, expert_sampling_params)
            for gen_idx, output in enumerate(generations):
                gt = ground_truth[batch_indices[gen_idx]]
                for res_idx in range(sample_count):
                    gen_text = output.outputs[res_idx].text
                    reward_result = r1_zero_reward_fn(gen_text, gt, False)
                    if reward_result["reward"] == 1:
                        training_data.append(batch_indices[gen_idx])
                        break
                    if reward_result["format_reward"] == 1:
                        training_data.append(batch_indices[gen_idx])
                        break
        inputs_total = data["input_ids"][training_data]
        labels_total = data["labels"][training_data]
        response_total = data["response_mask"][training_data]
        start_idx = 0
        total = len(training_data)
        for idx in range(training_step):
            if start_idx + train_batch_size > total:
                start_idx = 0
            inputs = inputs_total[start_idx:start_idx+train_batch_size]
            labels = labels_total[start_idx:start_idx+train_batch_size]
            with torch.amp.autocast(device):
                log_probs = get_response_log_probs(
                    model=model,
                    input_ids=inputs,
                    labels=labels,
                    return_token_entropy=True,
                )
                loss, metadata = sft_microbatch_train_step(
                    policy_log_probs=log_probs["log_probs"],
                    response_mask=response_total[start_idx:start_idx+train_batch_size],
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
                # print(f"iteration {idx}, loss is {loss}")
            start_idx += train_batch_size
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


expert_iteration()

import json
import os
import re
from typing import Callable, Dict, List, Tuple

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams


"""
math_baseline
(b)
total 1319
dataset gsm8k test data
format reward 1 and answer reward 1: 88
format reward 1 and answer reward 0: 170
format reward 0 and answer reward 0: 1061

format reward 0, it is due to base model's output
format reward 1 and answer reward 0: the format is correct, answer is wrong
(c) did poorly
{'num_examples': 1319, 'reward': 0.0667172100075815, 'format_reward': 0.1956027293404094, 'answer_reward': 0.0667172100075815}
"""


def load_file(file_path: str) -> List[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_prompt(samples: List[dict], path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        R1_ZERO_PROMPT = f.read()
    return [R1_ZERO_PROMPT.format(question=sample["question"]) for sample in samples]


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truth: List[str],
    output_path: str,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    assert len(prompts) == len(ground_truth), "Prompts and ground truth must have the same size"
    generations = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    reward = 0.0
    format_reward = 0.0
    answer_reward = 0.0
    f1a1 = 0
    f1a0 = 0
    f0a0 = 0
    for i, output in enumerate(generations):
        gen_text = output.outputs[0].text
        reward_result = reward_fn(gen_text, ground_truth[i], False)
        reward += reward_result["reward"]
        format_reward += reward_result["format_reward"]
        answer_reward += reward_result["answer_reward"]
        if reward_result["format_reward"] == 1 and reward_result["answer_reward"] == 1:
            f1a1 += 1
        if reward_result["format_reward"] == 1 and reward_result["answer_reward"] == 0:
            f1a0 += 1
        if reward_result["format_reward"] == 0 and reward_result["answer_reward"] == 0:
            f0a0 += 1

        results.append({
            "prompt": prompts[i],
            "generation": gen_text,
            "gt": ground_truth[i],
            "metrics": reward_result,
        })

    metrics = {
        "num_examples": len(results),
        "reward": reward / len(results),
        "format_reward": format_reward / len(results),
        "answer_reward": answer_reward / len(results),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_name = "math_baseline.json"
    file_path = os.path.join(output_path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "examples": results,
            },
            f,
        )

    print("Evaluation complete")
    print(metrics)
    print(f1a1, f1a0, f0a0)


def evaluate(
    model: LLM,
    output_path: str,
    prompts: List[str],
    ground_truth: List[str],
) -> None:
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truth=ground_truth,
        output_path=output_path,
    )


def eval_data() -> Tuple[List[str], List[str]]:
    data_path = "/workspace/alignment/data/gsm8k/test.jsonl"
    prompt_path = "/workspace/alignment/cs336_alignment/prompts/r1_zero.prompt"
    samples = load_file(data_path)
    prompts = format_prompt(samples, prompt_path)
    ground_truth = [sample["answer"] for sample in samples]
    return (prompts, ground_truth)


def main() -> None:
    output_path = "/workspace/alignment/data/output/"
    model_name = "/workspace/huggingface/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    model = LLM(model_name)
    prompts, gt = eval_data()
    evaluate(
        model=model,
        output_path=output_path,
        prompts=prompts,
        ground_truth=gt,
    )


if __name__=="__main__":
    main()

import json
import os
import re
from typing import Callable, List, Dict

from vllm import LLM, SamplingParams


"""
math_baseline
(b) format is wrong,
(c) did poorly
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


def extract_final_answer(text: str) -> str:
    """
    Extract final answer from model output or gold solution.
    Works for MATH-style '#### answer' or boxed answers.
    """
    # #### answer
    match = re.search(r"####\s*(.*)", text)
    if match:
        return match.group(1).strip()

    # \boxed{answer}
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()

    # fallback: last line
    return text.strip().splitlines()[-1].strip()


def exact_match_reward(prediction: str, target: str) -> Dict[str, float]:
    pred = extract_final_answer(prediction)
    gold = extract_final_answer(target)

    return {
        "exact_match": float(pred == gold)
    }


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    references: List[str],
    output_path: str,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    generations = vllm_model.generate(prompts, sampling_params)

    results = []
    total_em = 0.0

    for i, output in enumerate(generations):
        gen_text = output.outputs[0].text
        reward = reward_fn(gen_text, references[i])
        total_em += reward["exact_match"]

        results.append({
            "prompt": prompts[i],
            "generation": gen_text,
            "reference": references[i],
            "metrics": reward,
        })

    metrics = {
        "num_examples": len(results),
        "exact_match": total_em / len(results),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "examples": results,
            },
            f,
            indent=2,
        )

    print("Evaluation complete")
    print(metrics)


def main():
    print("start running")
    data_path = "/Users/YangWen/Documents/Code/github/alignment/data/gsm8k/test.jsonl"
    prompt_path = "/Users/YangWen/Documents/Code/github/alignment/cs336_alignment/prompts/r1_zero.prompt"
    output_path = "/Users/YangWen/Documents/Code/github/alignment/data/output"
    samples = load_file(data_path)
    prompts = format_prompt(samples, prompt_path)
    references = [sample["answer"] for sample in samples]

    model_name = "Qwen/Qwen2.5-Math-1.5B"
    model = LLM(model_name)

    evaluate_vllm(
        vllm_model=model,
        reward_fn=exact_match_reward,
        prompts=prompts,
        references=references,
        output_path=output_path,
    )


if __name__=="__main__":
    main()

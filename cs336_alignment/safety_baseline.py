import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from cs336_alignment.math_baseline import get_data, load_file
from cs336_alignment.sft import init_vllm
from vllm import SamplingParams

"""
uv run pytest -k test_parse_mmlu_response
uv run pytest -k test_parse_gsm8k_response
"""


BASE_DIR = Path(__file__).resolve().parent


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    target_str = "the correct answer is "
    model_output = model_output.lower()
    idx = model_output.find(target_str)
    if idx == -1:
        return None
    idx += len(target_str)
    while idx < len(model_output) and model_output[idx] == ' ':
        idx += 1
    ans = model_output[idx].upper() if idx < len(model_output) else None
    return ans if ans in 'ABCD' else None


def load_mmlu_directory(dir_path: str) -> list[dict]:
    """
    Parse all MMLU-style CSV files in a directory.

    Each CSV row format:
    question, optionA, optionB, optionC, optionD, answer

    The CSV filename (without extension) is used as the subject.
    """
    dir_path = Path(dir_path)
    assert dir_path.is_dir(), f"{dir_path} is not a directory"

    all_examples = []

    for csv_path in sorted(dir_path.glob("*.csv")):
        idx = csv_path.stem.rfind('_')
        subject = csv_path.stem[0:idx]

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if not row:
                    continue
                if len(row) != 6:
                    raise ValueError(
                        f"{csv_path}:{row_idx} has {len(row)} columns, expected 6"
                    )

                question, a, b, c, d, answer = row
                assert answer.strip() in {"A", "B", "C", "D"}
                all_examples.append({
                    "subject": subject,
                    "question": question.strip(),
                    "options": [a.strip(), b.strip(), c.strip(), d.strip()],
                    "answer": answer.strip(),
                })
    return all_examples


def mmlu_format_prompt(
    examples: list[dict],
    prompt_path: str,
) -> tuple[list[str], list[str]]:
    with open(prompt_path, "r", encoding="utf-8") as f:
        MMLU_PROMPT = f.read()
    questions = [MMLU_PROMPT.format(
        subject=example["subject"],
        question=example["question"],
        options=example["options"],
        ) for example in examples]
    ans = [example['answer'] for example in examples]
    return questions, ans


"""
mmlu_baseline
(c)
able to parse all generations

(d)
7791.8 tokens/s
8.132 s for 1531 examples
188.28 examples per second

(e)
Qwen2.5 0.5B model is relatively poor. The correct rate is only 5.3%
total is 1531, correct is 81, rate is 5.290659699542783

(f)
format error
" B\n\nWhich of the following is NOT a characteristic of the human body's immune system?"

Incorrect both format and anser
 D\n\nWhich of the following is NOT a characteristic of the human body's circulatory system?
"""


def analysis_mmlu():
    data_path = BASE_DIR / "../data/mmlu/val"
    prompt_path = BASE_DIR / "prompts/mmlu.prompt"
    output_path = BASE_DIR / "../data/mmlu_output/"
    output_file_name = output_path + "mmlu.json"
    model_name = "/models--Qwen--Qwen2.5-0.5B/snapshots/"
    device = "cuda"
    seed = 42
    gpu_memory_utilization = 0.9

    vllm_model = init_vllm(model_name, device, seed, gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        min_tokens=4,
        stop=["\n"],
        n=1,
        seed=seed,
    )
    mmlu_data = load_mmlu_directory(data_path)
    questions, ground_truth = mmlu_format_prompt(mmlu_data, prompt_path)
    start = time.time()
    generations = vllm_model.generate(questions, sampling_params)
    end = time.time()
    print(f"Elapsed time: {end - start:.3f} seconds")
    raw_answer = [
        output.text
        for request_output in generations
        for output in request_output.outputs
    ]
    answer = [parse_mmlu_response({}, ans) for ans in raw_answer]
    correct = 0
    total = len(questions)
    incorrect = []
    for idx, (gt, ans) in enumerate(zip(ground_truth, answer)):
        if gt == ans:
            correct += 1
        else:
            tmp = {
                "gt": gt,
                "ans": ans,
                "raw": raw_answer[idx],
                "prompt": questions[idx],
            }
            incorrect.append(tmp)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(incorrect, f, indent=2, ensure_ascii=False)
    print(f"mmlu data: total is {total}, correct is {correct}, rate is {correct / total * 100}")


# analysis_mmlu()


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    matches = re.findall(r"-?\d+\.?\d*", model_output)
    return matches[-1] if matches else None


"""
(c)
able to parse all generations

(d)
9738.81 tokens/s
36.317 s for 7473 examples
205.77 examples per second

(e)
gsm8k data: total is 7473, correct is 977, rate is 13.073732102234711
Qwen2.5 0.5B model is poor. The correct rate is only 13.07%

(f)
answer is incorrect
"gt": "72",
"raw_gt": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
"ans": null,
"raw_answer": " To determine the total number of clips Natalia sold in April and May, we need to follow these steps:"

comparison need to update
"gt": "10",
"raw_gt": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
"ans": "10.",
"raw_answer": " Weng earns $12 x 50 / 60 = $10 for 50 minutes of babysitting. The answer is 10.",
"""


def analysis_gsm8k():
    data_path = BASE_DIR / "../data/gsm8k/train.jsonl"
    prompt_path = BASE_DIR / "prompts/gsm8k.prompt"
    output_path = BASE_DIR / "../data/gsm8k_output/"
    output_file_name = output_path + "gsm8k.json"
    model_name = "/models--Qwen--Qwen2.5-0.5B/snapshots/"
    device = "cuda"
    seed = 42
    gpu_memory_utilization = 0.9

    questions, ground_truth = get_data(data_path, prompt_path)
    extracted_ground_truth = [parse_gsm8k_response(tmp) for tmp in ground_truth]

    vllm_model = init_vllm(model_name, device, seed, gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        min_tokens=4,
        stop=["\n"],
        n=1,
        seed=seed,
    )
    start = time.time()
    generations = vllm_model.generate(questions, sampling_params)
    end = time.time()
    print(f"Elapsed time: {end - start:.3f} seconds")
    raw_answer = [
        output.text
        for request_output in generations
        for output in request_output.outputs
    ]
    answer = [parse_gsm8k_response(ans) for ans in raw_answer]
    correct = 0
    total = len(questions)
    incorrect = []
    for idx, (gt, ans) in enumerate(zip(extracted_ground_truth, answer)):
        if ans is not None and abs(float(gt) - float(ans)) < 1e-3:
            correct += 1
        else:
            tmp = {
                "gt": gt,
                "raw_gt": ground_truth[idx],
                "ans": ans,
                "raw_answer": raw_answer[idx],
                "prompt": questions[idx],
            }
            incorrect.append(tmp)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(incorrect, f, indent=2, ensure_ascii=False)
    print(f"gsm8k data: total is {total}, correct is {correct}, rate is {correct / total * 100}")


# analysis_gsm8k()


"""
(b)
11852.68 tokens/second
805 examples, 24.758 seconds
32.51 examples/second

(c)
(d)
can't run alpaca_eval successfully
"""


def analysis_alpaca():
    data_path = BASE_DIR / "../data/alpaca_eval/alpaca_eval.jsonl"
    output_path = BASE_DIR / "../data/alpaca_output/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file_name = output_path + "alpaca.json"
    model_name = "/models--Qwen--Qwen2.5-0.5B/snapshots/"
    device = "cuda"
    seed = 42
    gpu_memory_utilization = 0.9

    data = load_file(data_path)
    prompts = [tmp["instruction"] for tmp in data]

    vllm_model = init_vllm(model_name, device, seed, gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        min_tokens=4,
        stop=["\n"],
        n=1,
        seed=seed,
    )
    start = time.time()
    generations = vllm_model.generate(prompts, sampling_params)
    end = time.time()
    print(f"Elapsed time: {end - start:.3f} seconds")
    raw_output = [
        output.text
        for request_output in generations
        for output in request_output.outputs
    ]
    for data_tmp, output in zip(data, raw_output):
        data_tmp["output"] = output
        data_tmp["generator"] = "qwen2.5-0.5B"

    with open(output_file_name, "w") as fout:
        json.dump(data, fout)


# analysis_alpaca()


def load_csv(path: str) -> list[dict]:
    """
    Parse a CSV file with headers into a list of dictionaries.
    """
    path = Path(path)
    examples = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                "id": row["id"].strip(),
                "harm_area": row["harm_area"].strip(),
                "counter": int(row["counter"]),
                "category": row["category"].strip(),
                "prompt": row["prompts_final"].strip(),
            })

    return examples


def write_jsonl(predictions: list[dict], output_path: str):
    """
    Write a list of dictionaries to a JSONL file.

    Each dictionary becomes one JSON object per line.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


"""
Use Qwen2.5-3B-Instruct model

(b)
11213.01 tokens/second
100 examples, 4.576 seconds
21.85 examples/second

(c)
0.27 of model outputs are judged as safe

(d)
some response are toxic, harmful, etc
"""


def analysis_sst():
    data_path = BASE_DIR / "../data/simple_safety_tests/simple_safety_tests.csv"
    output_path = BASE_DIR / "../data/sst_output/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file_name = output_path + "sst.json"
    model_name = "/models--Qwen--Qwen2.5-0.5B/snapshots/"
    device = "cuda"
    seed = 42
    gpu_memory_utilization = 0.9

    data = load_csv(data_path)
    prompts = [tmp["prompt"] for tmp in data]

    vllm_model = init_vllm(model_name, device, seed, gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        min_tokens=4,
        stop=["\n"],
        n=1,
        seed=seed,
    )
    start = time.time()
    generations = vllm_model.generate(prompts, sampling_params)
    end = time.time()
    print(f"Elapsed time: {end - start:.3f} seconds")
    raw_output = [
        output.text
        for request_output in generations
        for output in request_output.outputs
    ]
    predictions = []
    for prompt, output in zip(prompts, raw_output):
        predictions.append({
            "prompts_final": prompt,
            "output": output,
        })
    write_jsonl(predictions, output_file_name)


# analysis_sst()

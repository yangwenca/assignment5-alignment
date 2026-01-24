import json
import time
from pathlib import Path
from typing import Any, Callable, Literal

from cs336_alignment.baseline import load_csv, load_mmlu_directory, mmlu_format_prompt, parse_gsm8k_response, parse_mmlu_response, write_jsonl
from cs336_alignment.math_baseline import get_data, load_file
from cs336_alignment.sft import init_vllm
from vllm import SamplingParams


def parse_sft_prompt(prompts: list[str]) -> list[str]:
    sft_prompt_path = "/alignment/cs336_alignment/prompts/alpaca_sft.prompt"
    with open(sft_prompt_path, "r", encoding="utf-8") as f:
        SFT_PROMPT = f.read()
    ans = [SFT_PROMPT.format(
        instruction=prompt,
        response='',
    ) for prompt in prompts]
    return ans


"""
mmlu_sft
(a)
8390.8 tokens/s
3.54 s for 1531 examples
432.49 examples per second

It is better than zero shot.
This is zero shot's data.
7791.8 tokens/s
8.132 s for 1531 examples
188.28 examples per second

(b)
after sft, the performance is better. It improves from 5.3% to 18.9%.
mmlu data: total is 1531, correct is 290, rate is 18.941868060091444

(c)
format error
We still have format errors, the percentage of format errors reduce.

Incorrect both format and answer
We still have incorrect in terms of both format and answer. It means the model is not able to answer them.

Most of outputs are able to give correct format.
"""


def eval_mmlu(model_name: str, output_file_name: str):
    data_path = "/alignment/data/mmlu/val"
    prompt_path = "/alignment/cs336_alignment/prompts/mmlu.prompt"
    output_path="/alignment/data/mmlu_output/"
    output_file_name = output_path + output_file_name
    device = "cuda"
    seed = 42
    gpu_memory_utilization = 0.9

    mmlu_data = load_mmlu_directory(data_path)
    questions, ground_truth = mmlu_format_prompt(mmlu_data, prompt_path)
    questions = parse_sft_prompt(questions)

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


# eval_mmlu("/alignment/data/instruction/", "mmlu_sft.json")


"""
(a)
the throughput is a little bit better.
9158.67 tokens/s
32.231 s for 7473 examples
231.86 examples per second

zero shot
9738.81 tokens/s
36.317 s for 7473 examples
205.77 examples per second

(b)
after sft, the performance is worse.
gsm8k data: total is 7473, correct is 667, rate is 8.92546500735983

before sft
gsm8k data: total is 7473, correct is 977, rate is 13.073732102234711


(c)
it still has errors in terms of answer. Format error is reduced.
But it is not able to answer more questions. Seems like sft is hurting
  {
    "gt": "136",
    "raw_gt": "Cory takes 22 + 3 = <<22+3=25>>25 minutes to clean her room.\nBlake takes 25 - 4 = <<25-4=21>>21 minutes to clean his room.\nThe three of them can clean their room in 22 + 25 + 21 = <<22+25+21=68>>68 minutes in all.\nIn a week, they spend 68 x 2 = <<68*2=136>>136 minutes cleaning their rooms.\n#### 136",
    "ans": null,
    "raw_answer": "First, let's find out how long it takes each person to clean their room:",
    "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nRichard can clean his room in 22 minutes. Cory takes 3 minutes more than Richard to clean her room while Blake can clean his room 4 minutes more quickly than Cory. If they have to clean their rooms twice a week, how many minutes do all three spend cleaning their rooms each week?\nAnswer:\n\n### Response:\n"
  },
  {
    "gt": "9",
    "raw_gt": "She had 5+8 = <<5+8=13>>13 crayons to start with\nShe gave out 3+1 = <<3+1=4>>4 crayons to Becky\nShe will have 13-4 = <<13-4=9>>9 crayons left\n#### 9",
    "ans": "1",
    "raw_answer": "Mary has a total of 5 green crayons and 8 blue crayons. She gives out 3 green crayons and 1 blue crayon to Becky. ",
    "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nMary has 5 green crayons and 8 blue crayons of different shades. If she gives out 3 green crayons and 1 blue crayon to Becky, how many crayons does she have left?\nAnswer:\n\n### Response:\n"
  },
"""


def eval_gsm8k(model_name: str, output_file_name: str):
    data_path = "/alignment/data/gsm8k/train.jsonl"
    prompt_path = "/alignment/cs336_alignment/prompts/gsm8k.prompt"
    output_path="/alignment/data/gsm8k_output/"
    output_file_name = output_path + output_file_name
    device = "cuda"
    seed = 42
    gpu_memory_utilization = 0.9

    questions, ground_truth = get_data(data_path, prompt_path)
    questions = parse_sft_prompt(questions)
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


# eval_gsm8k("/alignment/data/instruction/", "gsm8k_sft.json")


"""
(a)
the throughput after sft is faster than before.
after sft
7094.88 tokens/second
805 examples 5.316
151.43 examples/second

before sft
11852.68 tokens/second
805 examples, 24.758 seconds
32.51 examples/second

(b)
(c)
can't run alpaca_eval successfully
"""


def eval_alpaca(model_name: str, output_file_name: str):
    data_path = "/alignment/data/alpaca_eval/alpaca_eval.jsonl"
    output_path="/alignment/data/alpaca_output/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file_name = output_path + output_file_name
    device = "cuda"
    seed = 42
    gpu_memory_utilization = 0.9

    data = load_file(data_path)
    prompts = [tmp["instruction"] for tmp in data]
    prompts = parse_sft_prompt(prompts)

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


# eval_alpaca("/alignment/data/instruction/", "alpaca_sft.json")


"""
base model is Qwen2.5-0.5B
Use Qwen2.5-3B-Instruct model as judge

(a)
the throught after sft is better than before.
2647.01 tokens/second
100 examples, 2.35 seconds
42.55 examples/second

before sft
11213.01 tokens/second
100 examples, 4.576 seconds
21.85 examples/second

(b)
model performance improves after sft
after sft
0.85 of model outputs are judged as safe

before sft
0.27 of model outputs are judged as safe

(c)
some response are toxic, harmful, etc
"""


def eval_sst(model_name: str, output_file_name: str):
    data_path = "/alignment/data/simple_safety_tests/simple_safety_tests.csv"
    output_path="/alignment/data/sst_output/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file_name = output_path + output_file_name
    device = "cuda"
    seed = 42
    gpu_memory_utilization = 0.9

    data = load_csv(data_path)
    prompts = [tmp["prompt"] for tmp in data]
    prompts = parse_sft_prompt(prompts)

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


# eval_sst("/alignment/data/instruction/", "sst_sft.json")


"""
red_teaming
(a)
automated social engineering: generate highly personalized, context-aware messages
over-reliance leading to decision abdication: in high stakes domains like medicine, users may delegate judgment to the model without sufficient verification
cultural or historical distortion: skewed data

(b)
try to play around with the model and see what the model is returned
"""

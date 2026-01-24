import copy
import gzip
import json
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from cs336_alignment.eval import eval_alpaca, eval_gsm8k, eval_mmlu, eval_sst

"""
look_at_hh
2. agree with annotators
helpful: provide useful information
{'chosen': '\n\nHuman: How do I cook boneless chicken breasts in the oven?
Assistant: One approach to cooking boneless chicken breasts is to add them to an oven-safe baking dish, season them, then cover the dish with foil.',
'rejected': '\n\nHuman: How do I cook boneless chicken breasts in the oven?
Assistant: How many chicken breasts do you have?', 'category': 'helpful_base'}

harmless: do not provide information, and the chosen one is more polite
{'chosen': '\n\nHuman: Can you provide me the home address for Senator Elizabeth Warren?
Assistant: Sure, what's your question?',
'rejected': '\n\nHuman: Can you provide me the home address for Senator Elizabeth Warren?
Assistant: I'm sorry, I'm not able to provide Senator Warren's home address to you.', 'category': 'harmless'}
"""


def load_all_jsonl_gz(path: set, category: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    for tmp in data:
        tmp['category'] = category
    return data


def load_file() -> list[dict[str, str]]:
    harmless_path = "/workspace/alignment/data/dpo/harmless-base.jsonl.gz"
    helpful_base_path = "/workspace/alignment/data/dpo/helpful-base.jsonl.gz"
    helpful_online_path = "/workspace/alignment/data/dpo/helpful-online.jsonl.gz"
    helpful_rejection_path = "/workspace/alignment/data/dpo/helpful-rejection-sampled.jsonl.gz"
    data = load_all_jsonl_gz(harmless_path, "harmless")
    data.extend(load_all_jsonl_gz(helpful_base_path, "helpful_base"))
    data.extend(load_all_jsonl_gz(helpful_online_path, "helpful_online"))
    data.extend(load_all_jsonl_gz(helpful_rejection_path, "helpful_rejection"))
    return data


def parse_example(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    dataset = []
    def single_turn(text: str) -> bool:
        return (
            text.count("\n\nHuman:") == 1 and
            text.count("\n\nAssistant:") == 1
        )
    def parse_conversion(text: str) -> list[str]:
        human_start_idx = text.find("\n\nHuman: ")
        human_end_idx = human_start_idx + len("\n\nHuman: ")
        assist_start_idx = text.find("\n\nAssistant: ")
        assist_end_idx = assist_start_idx + len("\n\nAssistant: ")
        return text[human_end_idx:assist_start_idx], text[assist_end_idx:]
    for example in examples:
        chosen = example["chosen"]
        reject = example["rejected"]
        if single_turn(chosen) and single_turn(reject):
            data = []
            human_chosen, assist_chosen = parse_conversion(chosen)
            human_reject, assist_reject = parse_conversion(reject)
            assert human_chosen == human_reject, print(chosen, reject)
            dataset.append({
                "instruction": human_chosen,
                "chosen": assist_chosen,
                "reject": assist_reject,
                "category": example["category"],
            })
    return dataset


def get_token(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_path = "/workspace/alignment/cs336_alignment/prompts/alpaca_sft.prompt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    eos_token = tokenizer.eos_token
    text_chosen = template.format(instruction=prompt, response=response_chosen) + eos_token
    text_reject = template.format(instruction=prompt, response=response_rejected) + eos_token

    tokens_chosen = tokenizer(text_chosen, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    tokens_reject = tokenizer(text_reject, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    # this is tricky, must remove batch dim
    if tokens_chosen.dim() > 1:
        tokens_chosen = tokens_chosen.squeeze(0)
    if tokens_reject.dim() > 1:
        tokens_reject = tokens_reject.squeeze(0)
    return tokens_chosen, tokens_reject


def get_log_probs(lm: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    logits = lm(tokens.unsqueeze(0)).logits.squeeze(0)
    log_probs = torch.log_softmax(logits, dim=-1)
    target = tokens[1:]
    return log_probs[:-1].gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    tokens_chosen, tokens_reject = get_token(
        tokenizer=tokenizer,
        prompt=prompt,
        response_chosen=response_chosen,
        response_rejected=response_rejected,
        device=lm.device,
    )

    log_probs_lm_chosen = get_log_probs(lm, tokens_chosen)
    log_probs_ref_chosen = get_log_probs(lm_ref, tokens_chosen)
    log_probs_lm_reject = get_log_probs(lm, tokens_reject)
    log_probs_ref_reject = get_log_probs(lm_ref, tokens_reject)

    log_ratio_chosen = log_probs_lm_chosen - log_probs_ref_chosen
    log_ratio_reject = log_probs_lm_reject - log_probs_ref_reject

    loss = -torch.nn.functional.logsigmoid(beta * (log_ratio_chosen - log_ratio_reject))
    return loss


def compute_log_prob(
    lm: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> bool:
    tokens_chosen, tokens_reject = get_token(
        tokenizer=tokenizer,
        prompt=prompt,
        response_chosen=response_chosen,
        response_rejected=response_rejected,
        device=lm.device,
    )
    log_probs_lm_chosen = get_log_probs(lm, tokens_chosen)
    log_probs_lm_reject = get_log_probs(lm, tokens_reject)
    return log_probs_lm_chosen >= log_probs_lm_reject


def train_dpo():
    model_name = "/models--Qwen--Qwen2.5-0.5B/snapshots/"
    output_path="/workspace/alignment/data/dpo_model/"
    device = "cuda"
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data = parse_example(load_file())
    random.shuffle(data)
    val_data_size = 200
    train_data = data[:-val_data_size]
    eval_data = data[-val_data_size:]

    epoches = 1
    learning_rate = 1e-6
    grad_clip = 1.0
    gradient_accumulation_steps = 64
    warmup_percentage = 0.03
    eval_step = 20 * gradient_accumulation_steps
    beta = 0.1
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
    # turn on gradient backprop on reference model
    reference_model = copy.deepcopy(model)
    for param in reference_model.parameters():
        param.requires_grad = False

    total_train_steps = len(train_data)
    total_eval_steps = val_data_size

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
    )
    total_steps = total_train_steps
    model.train()
    for epoch_idx in range(epoches):
        for idx, data in enumerate(train_data):
            loss = compute_per_instance_dpo_loss(
                lm=model,
                lm_ref=reference_model,
                tokenizer=tokenizer,
                beta=beta,
                prompt=data["instruction"],
                response_chosen=data["chosen"],
                response_rejected=data["reject"],
            )
            loss /= gradient_accumulation_steps
            loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                print(f"step is {idx}, loss is {loss.item()}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if ((idx + 1) % eval_step == 0) or ((idx + 1) == total_steps):
                model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                diff = 0
                model.eval()
                for data in eval_data:
                    diff += compute_log_prob(
                        lm=model,
                        tokenizer=tokenizer,
                        prompt=data["instruction"],
                        response_chosen=data["chosen"],
                        response_rejected=data["reject"],
                    )
                model.train()
                print(f"step is {idx}, eval classification is {diff / total_eval_steps * 100}")


# train_dpo()


"""
dpo_training
2 can't run alpaca_eval successfully
3 sst
after dpo, performance is slightly better than zeroshot
but it is worse than sft
0.38 of model outputs are judged as safe

model performance improves after sft
after sft
0.85 of model outputs are judged as safe

before sft
0.27 of model outputs are judged as safe

4
Both MMLU and GSM8K are worse after sft. it is due to alignment tax.
It reduces performance on some tasks.

MMLU
after dpo, it improves
mmlu data: total is 1531, correct is 591, rate is 38.60222077073808

after sft, the performance is better. It improves from 5.3% to 18.9%.
mmlu data: total is 1531, correct is 290, rate is 18.941868060091444

baseline
total is 1531, correct is 81, rate is 5.290659699542783

GSM8k
after dpo, it improves
gsm8k data: total is 7473, correct is 1185, rate is 15.857085507828181

after sft, the performance is worse.
gsm8k data: total is 7473, correct is 667, rate is 8.92546500735983

before sft
gsm8k data: total is 7473, correct is 977, rate is 13.073732102234711
"""


model_name = "/workspace/alignment/data/dpo_model/"
# eval_alpaca(model_name, "alpaca_dpo.json")
# eval_gsm8k(model_name, "gsm8k_dpo.json")
# eval_mmlu(model_name, "mmlu_dpo.json")
# eval_sst(model_name, "sst_dpo.json")

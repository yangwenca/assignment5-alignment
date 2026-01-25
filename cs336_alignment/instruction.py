import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup


BASE_DIR = Path(__file__).resolve().parent

"""
look_at_sft
question answering, summarizing, etc
quality is relatively good. It is able to provide high quality prompt and answer.
"""

"""
uv run pytest -k test_packed_sft_dataset
uv run pytest -k test_iterate_batches
"""

class InstructionDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        # need to delete the last line in prompt; otherwise, can't pass the test case
        prompt_path = BASE_DIR / "prompts" / "alpaca_sft.prompt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()

        with open(dataset_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]

        seed = 42
        random.seed(seed)
        if shuffle:
            random.shuffle(samples)
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        data = []
        for sample in samples:
            txt = prompt.format(instruction=sample["prompt"], response=sample["response"])
            tokens = tokenizer(txt, add_special_tokens=False)["input_ids"]

            if bos_token_id is not None:
                data.append(bos_token_id)
            data.extend(tokens)
            if eos_token_id is not None:
                data.append(eos_token_id)

        total = len(data)
        self.dataset = []
        for i in range(0, total, seq_length):
            self.dataset.append(data[i:i+seq_length])
        self.extra = eos_token_id
        if len(self.dataset[-1]) < seq_length:
            self.extra = self.dataset[-1][0]
            self.dataset.pop()


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, i):
        data = self.dataset[i]
        input_ids = torch.tensor(data, dtype=torch.long)
        if i < len(self.dataset) - 1:
            extra = self.dataset[i+1][0]
            labels = torch.tensor(data[1:] + [extra], dtype=torch.long)
        else:
            labels = torch.tensor(data[1:] + [self.extra], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train():
    model_name = "/models--Qwen--Qwen2.5-0.5B/snapshots/"
    output_path = BASE_DIR / "../data/instruction/"
    train_data_path = BASE_DIR / "../data/alignment/train.jsonl"
    eval_data_path = BASE_DIR / "../data/alignment/test.jsonl"
    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    epoches = 1
    learning_rate = 2e-5
    grad_clip = 1.0
    batch_size = 32
    sequence_length = 512
    gradient_accumulation_steps = 2
    warmup_percentage = 0.03
    eval_step = 40 * gradient_accumulation_steps
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

    training_dataset = InstructionDataset(
        tokenizer=tokenizer,
        dataset_path=train_data_path,
        seq_length=sequence_length,
        shuffle=True,
    )
    train_dataloader = iterate_batches(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    total_train_steps = len(train_dataloader)

    eval_dataset = InstructionDataset(
        tokenizer=tokenizer,
        dataset_path=eval_data_path,
        seq_length=sequence_length,
        shuffle=False,
    )
    eval_dataloader = iterate_batches(dataset=eval_dataset, batch_size=batch_size, shuffle=False)
    total_eval_steps = len(eval_dataloader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    total_steps = total_train_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_percentage * total_steps,
        num_training_steps=total_steps,
    )
    model.train()
    for epoch_idx in range(epoches):
        for idx, data in enumerate(train_dataloader):
            inputs = data['input_ids'].to(device)
            targets = data['labels'].to(device)
            outputs = model(inputs).logits
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss /= gradient_accumulation_steps
            loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                print(f"step is {idx}, loss is {loss.item()}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if ((idx + 1) % eval_step == 0) or ((idx + 1) == total_steps):
                model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                diff = 0
                model.eval()
                with torch.inference_mode():
                    for data in eval_dataloader:
                        inputs = data['input_ids'].to(device)
                        targets = data['labels'].to(device)
                        outputs = model(inputs).logits
                        diff += torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                model.train()
                print(f"step is {idx}, eval loss is {diff / total_eval_steps}")


# train()

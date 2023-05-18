import sys
from typing import Any, List, Union
from dagster import Config
from pathlib import Path

class FinetuneConfig(Config):
    base_model: Union[str, Path]
    data_path: Union[str, Path]
    output_dir: Union[str, Path]
    # training hyperparams
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    val_set_size: int = 2000
    # lora hyperparams
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]
    # llm hyperparams
    train_on_inputs: bool = True  # if False, masks out inputs in loss
    add_eos_token: bool = False
    group_by_length: bool = False  # faster, but produces an odd training loss curve
    prompt_template_name: str = "alpaca"  # The prompt template to use, will default to alpaca.


def finetune_alpaca_lora_model(config: FinetuneConfig):
    # import these lazily since they take a while to start up
    # and we don't want to pay that cost at import time
    import torch
    import transformers
    from datasets import load_dataset
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_int8_training,
    )
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from .prompter import Prompter

    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {config.base_model}\n"
        f"data_path: {config.data_path}\n"
        f"output_dir: {config.output_dir}\n"
        f"batch_size: {config.batch_size}\n"
        f"micro_batch_size: {config.micro_batch_size}\n"
        f"num_epochs: {config.num_epochs}\n"
        f"learning_rate: {config.learning_rate}\n"
        f"cutoff_len: {config.cutoff_len}\n"
        f"val_set_size: {config.val_set_size}\n"
        f"lora_r: {config.lora_r}\n"
        f"lora_alpha: {config.lora_alpha}\n"
        f"lora_dropout: {config.lora_dropout}\n"
        f"lora_target_modules: {config.lora_target_modules}\n"
        f"train_on_inputs: {config.train_on_inputs}\n"
        f"add_eos_token: {config.add_eos_token}\n"
        f"group_by_length: {config.group_by_length}\n"
        f"prompt template: {config.prompt_template_name}\n"
    )

    gradient_accumulation_steps = config.batch_size // config.micro_batch_size

    prompter = Prompter(config.prompt_template_name)

    device_map: Any = "auto"
    world_size = 1
    ddp = world_size != 1
    if ddp:
        device_map = {"": 0}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        config.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(config.base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=config.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < config.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not config.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=config.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if config.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    data_path = str(config.data_path)
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if config.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=config.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if config.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if config.val_set_size > 0 else None,
            save_steps=200,
            output_dir=config.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if config.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=config.group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(config.output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

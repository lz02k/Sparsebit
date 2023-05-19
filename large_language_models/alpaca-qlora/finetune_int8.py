import argparse
import os

import torch
import torch.nn as nn

import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import bitsandbytes as bnb
from datasets import load_dataset
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from utils import load_qllama, QuantLinear
from qlora import get_peft_qmodel


def main(args):
    if QUANT:
        config = transformers.AutoConfig.from_pretrained(args.model_name)
        model = load_qllama(config, checkpoint=CHECKPOINT, bit=BIT)
        model.is_loaded_in_8bit = True  # hack for gradient-checkpoint
        model = prepare_model_for_int8_training(model)
        model.is_loaded_in_8bit = False
        model.seq_len = 2048
        peft_func = get_peft_qmodel
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map="auto",
        )
        model = prepare_model_for_int8_training(model)
        peft_func = get_peft_model

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name, add_eos_token=True)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="QUANT_CAUSAL_LM" if QUANT else "CAUSAL_LM",
    )
    model = peft_func(model, config)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files=args.data_path)

    train_val = data["train"].train_test_split(
        train_size=TRAIN_SET_SIZE, test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"]
    val_data = train_val["test"]

    def generate_prompt(data_point):
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Input:
    {data_point["input"]}

    ### Response:
    {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    {data_point["output"]}"""

    def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))
    val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=args.warmup_steps,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=args.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            output_dir=OUTPUT,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="tensorboard",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # IMPORTANT! model.eval() -> model.train() enable requant 4-bit weights
    model.eval()
    model.train()

    trainer.train()
    # res = trainer.evaluate()

    model.save_pretrained("lora-alpaca")

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="decapoda-research/llama-7b-hf")
    parser.add_argument("--data-path", default="alpaca_data_cleaned.json")
    parser.add_argument("--train-set-size", type=int, default=None)
    parser.add_argument("--val-set-size", type=int, default=2000)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cutoff-len", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=1200)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load-quant", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    DEBUG = args.debug
    QUANT = args.load_quant
    if DEBUG:
        MICRO_BATCH_SIZE = 2
        BATCH_SIZE = 2
    else:
        MICRO_BATCH_SIZE = (
            args.micro_batch_size
        )  # this could actually be 5 but i like powers of 2
        BATCH_SIZE = args.batch_size

    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS = args.epochs  # we don't need 3 tbh
    LEARNING_RATE = args.lr  # the Karpathy constant
    CUTOFF_LEN = args.cutoff_len  # 256 accounts for about 96% of the data
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    TRAIN_SET_SIZE = args.train_set_size
    VAL_SET_SIZE = args.val_set_size
    CHECKPOINT = args.checkpoint
    OUTPUT = args.output
    BIT = args.bit

    if DEBUG:
        TRAIN_SET_SIZE = 2000
        VAL_SET_SIZE = 100

    main(args)

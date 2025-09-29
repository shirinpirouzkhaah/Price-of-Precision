import os

HF_HOME = "/scratch/kkitsi/.cache/huggingface"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["HF_METRICS_CACHE"] = os.path.join(HF_HOME, "metrics")
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/scratch/kkitsi/.cache/triton_cache"

print("✅ Transformers cache:", os.environ["TRANSFORMERS_CACHE"])


import argparse
import logging
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel
from datasets import Dataset
import pandas as pd
import numpy as np
import wandb
import re
from tqdm.auto import tqdm
import subprocess
import Levenshtein
from difflib import SequenceMatcher
import uuid
from transformers import EarlyStoppingCallback
from huggingface_hub import login
import gc
import unidecode
import unicodedata
from transformers import default_data_collator
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
gc.collect()
import torch.distributed as dist
import torch.nn.functional as F
import time
start_time = time.time()
max_training_seconds = 7 * 24 * 60 * 60 


def left_pad_sequences(sequences, pad_token_id):
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seq = F.pad(torch.tensor(seq), (pad_len, 0), value=pad_token_id)
        padded.append(padded_seq)
    return torch.stack(padded)

HF_TOKEN = "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"
login(HF_TOKEN)






"""
Ignore this is just a comment
parser.add_argument("--target_modules", nargs="+", type=str) ?
parser.add_argument("--save_safe_tensors", action="store_true") ?
parser.add_argument("--use_checkpoint", action="store_true") ?
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--model_name", type=str)
parser.add_argument("--tokenizer_name", type=str)

"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Defining what arguments with which types this script can accept 
    # My script understands these options if someone passes them in the terminal

    parser.add_argument("--config", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--eval_strategy", type=str) #when to run evaluation during training.
    parser.add_argument("--save_strategy", type=str) #when to save model checkpoints during training.
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--use_lora", type=bool)
    parser.add_argument("--use_checkpoint", type=bool) #True/False) — if passed, the model will try to load from an existing checkpoint file. If you're resuming training or doing inference from a trained model
    parser.add_argument("--checkpoint_path", type=str) # use_checkpoint works together with --checkpoint_path
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--save_safe_tensors",type=bool) #If enabled, the model will be saved in .safetensors format instead of PyTorch .bin.
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--sequence_length", type=int)
    parser.add_argument("--target_modules", nargs="+", type=str) #Used by LoRA (Low-Rank Adaptation) to define which layers in the model should get LoRA adapters: Only when --use_lora is active.
    parser.add_argument("--fp16", type=bool) #Enable 16-bit floating point (half precision) training: Faster and uses less memory on GPUs that support it (e.g., V100, A100).
    parser.add_argument("--bf16", type=bool) #Use bfloat16 precision, supported on newer hardware like A100 or newer CPUs. Similar benefits to fp16, but more stable due to larger exponent range.
    parser.add_argument("--device_map", type=str)
    parser.add_argument("--torch_dtype", type=str)
    parser.add_argument("--Data_preprocessing_step", type=str)
    parser.add_argument("--Train_data_path", type=str)
    parser.add_argument("--Test_data_path", type=str)
    parser.add_argument("--Common_test_data_path", type=str)
    parser.add_argument("--New_test_data_path", type=str)
    parser.add_argument("--Common_New_test_data_path", type=str)
    parser.add_argument("--column_names", nargs="+", type=str)
    parser.add_argument("--beam_size", type=int)
    parser.add_argument("--SEED", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)



    
    

    # Reading actual values from the user (CLI or YAML)
    cli_args = vars(parser.parse_args())
    yaml_config = {}
    if cli_args.get("config"):
        with open(cli_args["config"], "r") as f:
            yaml_config = yaml.safe_load(f) or {}
            
    
    intended_defaults = {
        "num_train_epochs":45, 
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "eval_strategy": "epoch",
        "save_strategy": "best",
        "logging_steps": 100,
        "log_file": "training_log.txt",
        "checkpoint_path": None,
        "save_steps": 100,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "sequence_length": 512,
        "fp16": False,
        "bf16": True,
        "device_map": None,
        "torch_dtype": "bfloat16",
        "beam_size": 10,
        "SEED": 42,
        "use_lora": True

        
    }

    merged = intended_defaults.copy()
    merged.update(yaml_config)
    for k, v in cli_args.items():
        if v is not None:
            merged[k] = v

    for flag in ["use_checkpoint", "save_safe_tensors"]:
        if flag not in merged:
            merged[flag] = False

    return argparse.Namespace(**merged)


# ---------------------------- Setup --------------------------------

args = parse_arguments()
Data_preprocessing_steps = args.Data_preprocessing_step


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


if not dist.is_initialized() or dist.get_rank() == 0:
    match = re.search(r'_(ProjectLevel\.csv|TimeLevel\.csv)$', args.Train_data_path)
    suffix = match.group(1) if match else "UnknownLevel"
    suffix = suffix.replace('.csv', '')
    
    # Construct the safe project name
    safe_project_name = re.sub(r'\W+', '_', f"{args.model_name}_{Data_preprocessing_steps}_{suffix}_3")
    run = wandb.init(project=safe_project_name, notes="experiment", reinit=True)
    
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, args.log_file), mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    
    
    column_names = args.column_names
    beam_size = args.beam_size
    
    
    
    # -------------------------- Load Data --------------------------------
    # Load and log NaN removal
    raw_train_df = pd.read_csv(args.Train_data_path)
    raw_test_df = pd.read_csv(args.Test_data_path)
    new_test_df = pd.read_csv(args.New_test_data_path)
    common_ids_df = pd.read_csv(args.Common_test_data_path, usecols=["Common_Record_id"])
    common_ids_new_df = pd.read_csv(args.Common_New_test_data_path, usecols=["Common_Record_id"])
    
    
    
    original_train_len = len(raw_train_df)
    original_test_len = len(raw_test_df)
    original_new_test_len = len(new_test_df)
    
    common_ids_set = set(common_ids_df["Common_Record_id"].dropna().unique())
    common_ids_new_set = set(common_ids_new_df["Common_Record_id"].dropna().unique())
    
    train_df = raw_train_df.dropna(subset=column_names)
    test_df = raw_test_df.dropna(subset=column_names)
    new_test_dff = new_test_df.dropna(subset=column_names)
    
    
    
    # ------------------ Check for empty strings in required columns ------------------
    
    def count_empty_string_rows(df, column_names, df_name):
        mask = df[column_names].apply(lambda x: x.astype(str).str.strip() == "", axis=1)
        empty_rows = mask.any(axis=1)
        count = empty_rows.sum()
        if count > 0:
            logger.info(f"[{args.Data_preprocessing_step}] {count} rows in {df_name} have at least one empty string in {column_names}")
        else:
            logger.info(f"[{args.Data_preprocessing_step}] No rows with empty strings in {df_name}")
        return count
    
    
    
    na_dropped_train = original_train_len - len(train_df)
    na_dropped_test = original_test_len - len(test_df)
    na_dropped_new_test = original_new_test_len - len(new_test_dff)
    
    
    logger.info(f"[{args.Data_preprocessing_step}] Dropped {na_dropped_train} rows from train due to NaNs in {column_names}")
    logger.info(f"[{args.Data_preprocessing_step}] Dropped {na_dropped_test} rows from test due to NaNs in {column_names}")
    logger.info(f"[{args.Data_preprocessing_step}] Dropped {na_dropped_new_test} rows from new test due to NaNs in {column_names}")
    
    
    # ------------------ Remove Train-Test Overlap -------------------
    
    logger.info(f"[{args.Data_preprocessing_step}] Checking for duplicates between train and test...")
    pre_dedup_len = len(train_df)
    train_df["hash"] = train_df[column_names].astype(str).agg("||".join, axis=1).apply(hash)
    train_df = train_df.drop_duplicates(subset=["hash"]).drop(columns=["hash"])
    dedup_dropped = pre_dedup_len - len(train_df)
    
    logger.info(f"[{args.Data_preprocessing_step}] Removed {dedup_dropped} overlapping rows from train that also appeared in test")
    
    # ------------------ Remove Train-NewTest Overlap -------------------
    
    logger.info(f"[{args.Data_preprocessing_step}] Checking for duplicates between train and new test...")
    
    train_df["hash"] = train_df[column_names].astype(str).agg("||".join, axis=1).apply(hash)
    new_test_dff["hash"] = new_test_dff[column_names].astype(str).agg("||".join, axis=1).apply(hash)
    overlap_hashes = set(train_df["hash"]).intersection(set(new_test_dff["hash"]))
    pre_overlap_len = len(train_df)
    train_df = train_df[~train_df["hash"].isin(overlap_hashes)].drop(columns=["hash"])
    overlap_dropped = pre_overlap_len - len(train_df)
    new_test_dff = new_test_dff.drop(columns=["hash"])
    
    train_df_all = train_df.dropna(subset=column_names)
    train_df = train_df_all.sample(frac=0.95, random_state=42)
    val_df = train_df_all.drop(train_df.index)
    
    logger.info(f"[{args.Data_preprocessing_step}] Removed {overlap_dropped} overlapping rows from train that also appeared in new test")
    logger.info(f"[{args.Data_preprocessing_step}] Final train size after cleaning: {len(train_df)}")
    logger.info(f"[{args.Data_preprocessing_step}] Final validation size after cleaning: {len(val_df)}")
    logger.info(f"[{args.Data_preprocessing_step}] Final test size after cleaning: {len(test_df)}")
    logger.info(f"[{args.Data_preprocessing_step}] Final new test size after cleaning: {len(new_test_dff)}")
    
    
    if new_test_dff.empty:
        logger.warning(f"[{args.Data_preprocessing_step}] 'new_test_dff' is empty. Copying one row from 'test_df'.")
        new_test_dff = test_df.head(10).copy().reset_index(drop=True)
    
    
    test_len = len(test_df)
    train_len = len(train_df)
    val_len = len(val_df)
    total_len = train_len + val_len + test_len
    
    train_pct = (train_len / total_len) * 100
    val_pct = (val_len / total_len) * 100
    test_pct = (test_len / total_len) * 100
    
    logger.info(f"Dataset split percentages:")
    logger.info(f"  Train: {train_len} samples ({train_pct:.4f}%)")
    logger.info(f"  Validation: {val_len} samples ({val_pct:.4f}%)")
    logger.info(f"  Test: {test_len} samples ({test_pct:.4f}%)")
    
    
    # #******************************************************************************************
    # train_df = train_df.head(100).copy().reset_index(drop=True)
    # test_df = test_df.head(10).copy().reset_index(drop=True)
    # new_test_dff = new_test_dff.head(10).copy().reset_index(drop=True)
    # #******************************************************************************************

# ---------------------- Helper methods --------------------------------

def build_source_target(batch, tokenizer, args):
    column_names = args.column_names
    texts = [
        f"<code> {src} </code> <technical_language> {comment} </technical_language> <fixed_code> {tgt} </fixed_code>"
        for src, comment, tgt in zip(batch[column_names[0]], batch[column_names[1]], batch[column_names[2]])
    ]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=args.sequence_length,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt", 
        add_special_tokens=True,
    )

    input_ids = encodings["input_ids"].tolist()
    attention_mask = encodings["attention_mask"].tolist()

    # Mask out padding tokens in labels
    labels = []
    for seq in input_ids:
        labels.append([-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def build_input_for_generation(example, tokenizer, args):
    column_names = args.column_names

    text = f"<code> {example[column_names[0]]} </code> <technical_language> {example[column_names[1]]} </technical_language> <fixed_code>"
    target = f"<fixed_code> {example[column_names[2]]} </fixed_code>"

    encodings = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding=False, 
        return_attention_mask=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    return encodings["input_ids"], encodings["attention_mask"], target


def build_input_for_generation_batch(batch, tokenizer, args):
    column_names = args.column_names

    texts = [
        f"<code> {src} </code> <technical_language> {comment} </technical_language> <fixed_code>"
        for src, comment in zip(batch[column_names[0]], batch[column_names[1]])
    ]
    targets = [
        f"<fixed_code> {tgt} </fixed_code>" for tgt in batch[column_names[2]]
    ]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=256,
        padding=False,
        return_attention_mask=True,
        return_tensors="np",
        add_special_tokens=True,
    )

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "targets": targets  
    }


def format_content(content):
	content = str(content)
	content = unicodedata.normalize("NFKC", content) 
	content = content.encode('utf-8').decode('utf-8', 'ignore').strip()
	content = unidecode.unidecode(content)
	return content



def cleaning_model_output(text):
    text = str(text)
    
    
    match = re.search(r"<fixed_code>(.*?)</fixed_code>", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
        
    text = format_content(text)
    
    text = re.sub(r'\t', ' ', text, flags=re.UNICODE)  # normalize tabs
    text = re.sub(r'[\r ]*\n[\r ]*', '\n', text, flags=re.UNICODE)  # normalize newlines
    text = re.sub(r'//[^\n]*', '', text, flags=re.UNICODE)  # remove one-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # remove multi-line comments
    text = re.sub(r'0{5,}', '', text)
    text = re.sub(r'((\\n)|(/n)|\n)[ \t\r]*((\\n)|(/n)|\n){2,}', '\n\n', text)

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        if not line.strip():
            continue
        match = re.match(r'^([ +-]?)(\s*)(.*)$', line)
        if match:
            diff_char, indent, content = match.groups()
            content = re.sub(r'\s{2,}', ' ', content.strip())
            content = re.sub(r'//[^\n]*', '', content, flags=re.UNICODE)  # remove one-line comments
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # remove multi-line comments
            
            if content.strip():
                cleaned.append(f"{diff_char}{indent}{content}")

    return "\n".join(cleaned).strip()



def to_single_line(val):
    return ' '.join(str(val).splitlines()).strip()



def is_exact_match(a, b):
    b_cleaned = re.sub(r"</?fixed_code>", "", b).strip()
    return b_cleaned in a


# -------------------------- Arguments --------------------------------

            
            
def train(args):

    logger.info(f"Training/evaluation parameters: {args}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        padding_side="left",
        trust_remote_code=True
    )
    
    tokenizer.add_special_tokens({
        'eos_token':  '</fixed_code>',
        'pad_token': '<pad>',
        'additional_special_tokens': ['<code>', '</code>', '<START>', '<END>', '<technical_language>', '</technical_language>', '<fixed_code>', '</fixed_code>']
    })

    logger.info(f"Tokenizer special tokens during generation: {tokenizer.special_tokens_map}")
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '</fixed_code>'


    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))



    train_ds_tokenized = train_dataset.map(
        lambda example: build_source_target(example, tokenizer, args),
        remove_columns=[col for col in train_dataset.column_names if col not in column_names],
        batched=True,
        batch_size=args.train_batch_size,
        num_proc=4 
    )
    
    valid_ds_tokenized = valid_dataset.map(
        lambda example: build_source_target(example, tokenizer, args),
        remove_columns=[col for col in train_dataset.column_names if col not in column_names],
        batched=True,
        batch_size=args.eval_batch_size,
        num_proc=4  
    )
    
    
    dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.torch_dtype)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True)

    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    
    model = get_peft_model(model=model, peft_config=peft_config)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_flash_attention = True

    
    logger.info(f"LoRA wrapping applied — model type: {type(model)}")
    model.print_trainable_parameters()
    #2025-07-30 17:12:37,259 - INFO - LoRA wrapping applied — model type: <class 'peft.peft_model.PeftModelForCausalLM'>
    #trainable params: 8,388,608 || all params: 6,747,009,024 || trainable%: 0.1243


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=safe_project_name,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        label_names=["labels"],
        remove_unused_columns=False,
        save_safetensors=args.save_safe_tensors,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        report_to="wandb",
        fp16=args.fp16,
        bf16=args.bf16,
        load_best_model_at_end=True,                     
        metric_for_best_model="eval_loss",      
        greater_is_better=False     ,
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,                
    )

    
    data_collator = default_data_collator
    
    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds_tokenized,
    eval_dataset=valid_ds_tokenized,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)])

    trainer.train()
    


    if not dist.is_initialized() or dist.get_rank() == 0:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        model.save_pretrained(adapter_dir, safe_serialization=True)
        logger.info(f"Adapter saved to: {adapter_dir}")
        tokenizer.save_pretrained(adapter_dir)
        logger.info(f"Tokenizer saved to: {adapter_dir}")

       
        
    
    if dist.is_initialized():
        dist.barrier()  

    del train_dataset, valid_dataset , train_ds_tokenized, valid_ds_tokenized
    torch.cuda.empty_cache()
    gc.collect()
    
    used_time = time.time() - start_time
    logger.info(f"Elapsed time: {used_time / 3600:.2f} hours. Remaining: {(max_training_seconds - used_time) / 3600:.2f} hours.")      
    
    
    return tokenizer, model




def test(args, model, tokenizer, test_df, scenario, file_name, beam_size=args.beam_size):
    logger.info("Running test phase...")
    model.eval()

    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '</fixed_code>'
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_flash_attention = True

    

    batch_size = 8  
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    
    
    def preprocess_batch(batch):
        return build_input_for_generation_batch(batch, tokenizer, args)
    
    logger.info(f"Tokenizing {len(test_df)} test samples with batch size {args.eval_batch_size}...")
    tokenized_dataset = test_dataset.map(
        preprocess_batch,
        remove_columns=[col for col in test_dataset.column_names if col not in args.column_names],
        batched=True,
        batch_size=batch_size,
        num_proc=4
    )
    

    logger.info(f"Generating predictions for {len(test_df)} examples...")

    all_beam_preds = []
    all_targets = []
    
    logger.info("Generating predictions...")

    for i in tqdm(range(0, len(tokenized_dataset), batch_size), desc="Generating"):
        batch = tokenized_dataset.select(range(i, min(i + batch_size, len(tokenized_dataset))))
        input_ids_padded = left_pad_sequences(batch["input_ids"], tokenizer.pad_token_id).to(model.device)
        attention_mask_padded = left_pad_sequences(batch["attention_mask"], 0).to(model.device)
        
        target_texts = batch["targets"]
        target_token_lens = [len(tokenizer.encode(t, add_special_tokens=True)) for t in target_texts]
        dynamic_max_length = max(target_token_lens) + 4

        logger.info(f"Input length: {input_ids_padded.shape[1]}, Max new tokens: {dynamic_max_length}")


        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids_padded,
                attention_mask=attention_mask_padded,
                repetition_penalty=1.1,
                early_stopping=True,
                do_sample=False,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                max_new_tokens=dynamic_max_length,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        grouped_preds = [
            decoded[j * beam_size:(j + 1) * beam_size]
            for j in range(len(batch["input_ids"]))
        ]

        all_beam_preds.extend(grouped_preds)
        all_targets.extend(target_texts)

    logger.info("Finished generation.")
    del test_dataset, tokenized_dataset
    torch.cuda.empty_cache()
    gc.collect()
    return all_beam_preds, all_targets

def managePreds(args, test_df, all_beam_preds, all_targets, scenario, file_name, beam_size = args.beam_size):

    decoded_preds = all_beam_preds
    gold_targets = all_targets
    
    for pred in decoded_preds[0]:
        print(cleaning_model_output(pred))
    print (gold_targets[0])
    
    # Init results container
    test_df_copy = test_df.copy().reset_index(drop=True)
    
    test_df_copy["predictions"] = None
    test_df_copy["RawPredictions"] = None
    test_df_copy["gold"] = None

    test_df_copy["EXMEXM"] = 0

    test_df_copy["EXMEXM1"] = 0

    test_df_copy["EXMEXM3"] = 0

    test_df_copy["EXMEXM5"] = 0
    
    
    test_df_copy["Lev_Dist"] = 0.0
    test_df_copy["Lev_Ratio"] = 0.0


    
    
    all_preds_txt = []
    for i in range(len(gold_targets)):
        target  = gold_targets[i]
        gold = to_single_line(cleaning_model_output(target))
        test_df_copy.at[i, "gold"] = gold 
        raw_preds = decoded_preds[i]
        test_df_copy.at[i, "RawPredictions"] = raw_preds 
        
        beam_group = []
        for pred in raw_preds:
            cleaned_pred = to_single_line(cleaning_model_output(pred))
            beam_group.append(cleaned_pred)

        is_exact_exact = any(is_exact_match(pred, gold) for pred in beam_group)
    
        # Top-K EXM
        is_exact_exact1 = any(is_exact_match(pred, gold) for pred in beam_group[:1])
    
        is_exact_exact3 = any(is_exact_match(pred, gold) for pred in beam_group[:3])
    
        is_exact_exact5 = any(is_exact_match(pred, gold) for pred in beam_group[:5])
        

        lev_dists = [Levenshtein.distance(pred, gold) for pred in beam_group]
        lev_ratios = [Levenshtein.ratio(pred, gold) for pred in beam_group]

        min_lev = min(lev_dists) if lev_dists else -1
        max_ratio = max(lev_ratios) if lev_ratios else -1.0
        
        

        test_df_copy.at[i, "Lev_Dist"] = min_lev
        test_df_copy.at[i, "Lev_Ratio"] = max_ratio
        
        test_df_copy.at[i, "predictions"] = beam_group
        
        test_df_copy.at[i, "EXMEXM"] = int(is_exact_exact)
        
        # Add top-k EXM columns
        test_df_copy.at[i, "EXMEXM1"] = int(is_exact_exact1)
        
        test_df_copy.at[i, "EXMEXM3"] = int(is_exact_exact3)
        
        test_df_copy.at[i, "EXMEXM5"] = int(is_exact_exact5)

        
        all_preds_txt.extend(beam_group)

    # Save outputs
    with open(os.path.join(output_dir, f"{args.Data_preprocessing_step}_{file_name}_predictions_{beam_size}.txt"), 'w') as f:
        for p in all_preds_txt:
            f.write(p + '\n')
            
    final_csv_path = os.path.join(output_dir, f"{args.Data_preprocessing_step}__{file_name}_predictions_{beam_size}.csv")
    test_df_copy.to_csv(final_csv_path, index=False)
    logger.info(f"Saved test predictions with EXM to: {final_csv_path}")
    
    
    valid_lev_dists = test_df_copy.loc[test_df_copy["Lev_Dist"] != -1, "Lev_Dist"]
    avg_lev_dist = valid_lev_dists.mean() if not valid_lev_dists.empty else -1.0
    
    
    valid_lev_ratio = test_df_copy.loc[test_df_copy["Lev_Ratio"] != -1, "Lev_Ratio"]
    avg_lev_ratio = valid_lev_ratio.mean() if not valid_lev_ratio.empty else -1.0


    avg_all_lev_dist = test_df_copy["Lev_Dist"].mean()
    avg_all_lev_ratio = test_df_copy["Lev_Ratio"].mean()
    
    
    # EXM
    exmemx_ratio = test_df_copy["EXMEXM"].sum() / len(test_df_copy)


    exmemx_ratio1 = test_df_copy["EXMEXM1"].sum() / len(test_df_copy)

    exmemx_ratio3 = test_df_copy["EXMEXM3"].sum() / len(test_df_copy)

    exmemx_ratio5 = test_df_copy["EXMEXM5"].sum() / len(test_df_copy)


    invalid_lev_rows = (test_df_copy["Lev_Dist"] == -1).sum()
    invalid_levRatio_rows = (test_df_copy["Lev_Ratio"] == -1).sum()

    # Log metrics
    logger.info(f"========== FINAL METRICS SUMMARY FOR {scenario}==========")
    logger.info(f"Number of testing examples: {len(test_df_copy)}")
    
    logger.info(f"EXM score beam 1 (exact exact match ratio): {exmemx_ratio1:.4f}")

    logger.info(f"EXM score beam 3 (exact exact match ratio): {exmemx_ratio3:.4f}")

    logger.info(f"EXM score beam 5 (exact exact match ratio): {exmemx_ratio5:.4f}")

    logger.info(f"EXM score beam 10 (exact exact match ratio): {exmemx_ratio:.4f}")

    logger.info(f"Average Levenshtein Distance (excluding -1): {avg_lev_dist:.4f}")
    logger.info(f"Number of rows with Lev_Dist = -1: {invalid_lev_rows}")
    
    logger.info(f"Average Levenshtein Distance (all): {avg_all_lev_dist:.4f}")
    
    logger.info(f"Average Levenshtein Ratio (excluding -1): {avg_lev_ratio:.4f}")
    logger.info(f"Number of rows with Lev_Ratio = -1: {invalid_levRatio_rows}")
    
    
    logger.info(f"Average Levenshtein Ratio (all): {avg_all_lev_ratio:.4f}")
    logger.info("===========================================")


    del test_df
    torch.cuda.empty_cache()
    gc.collect()
    
    return test_df_copy


def stats_for_common_data(test_df_copy, common_ids_set, scenario, file_name, beam_size = args.beam_size):
    common_test_df = test_df_copy[test_df_copy["Record_id"].isin(common_ids_set)].copy().reset_index(drop=True)
    if common_test_df.empty:
        logger.info(f"Number of common examples is zero.")
        return common_test_df
    
    else:
        logger.info(f"[{args.Data_preprocessing_step}] Final {file_name} size after cleaning: {len(common_test_df)}")
        final_csv_path = os.path.join(output_dir, f"{Data_preprocessing_steps}_{file_name}_predictions_{beam_size}.csv")
        common_test_df.to_csv(final_csv_path, index=False)
        logger.info(f"Saved test predictions with EXM to: {final_csv_path}")
    
        # Exclude -1 values for Levenshtein Distance
        valid_lev_dists = common_test_df.loc[common_test_df["Lev_Dist"] != -1, "Lev_Dist"]
        avg_lev_dist = valid_lev_dists.mean() if not valid_lev_dists.empty else -1.0
        
        valid_lev_ratio = common_test_df.loc[common_test_df["Lev_Ratio"] != -1, "Lev_Ratio"]
        avg_lev_ratio = valid_lev_ratio.mean() if not valid_lev_ratio.empty else -1.0
    
        avg_all_lev_dist = common_test_df["Lev_Dist"].mean()
        avg_all_lev_ratio = common_test_df["Lev_Ratio"].mean()
    
        # EXM
        exmemx_ratio = common_test_df["EXMEXM"].sum() / len(common_test_df)
    
    
        exmemx_ratio1 = common_test_df["EXMEXM1"].sum() / len(common_test_df)
    
        exmemx_ratio3 = common_test_df["EXMEXM3"].sum() / len(common_test_df)
    
        exmemx_ratio5 = common_test_df["EXMEXM5"].sum() / len(common_test_df)
    
    
        invalid_lev_rows = (common_test_df["Lev_Dist"] == -1).sum()
        invalid_levRatio_rows = (common_test_df["Lev_Ratio"] == -1).sum()
    
    
        # Log metrics
        logger.info(f"========== FINAL METRICS SUMMARY FOR {scenario}==========")
        logger.info(f"Number of testing examples: {len(common_test_df)}")
    
        logger.info(f"EXM score beam 1 (exact exact match ratio): {exmemx_ratio1:.4f}")
    
        logger.info(f"EXM score beam 3 (exact exact match ratio): {exmemx_ratio3:.4f}")
    
        logger.info(f"EXM score beam 5 (exact exact match ratio): {exmemx_ratio5:.4f}")
    
        logger.info(f"EXM score beam 10 (exact exact match ratio): {exmemx_ratio:.4f}")
    
        logger.info(f"Average Levenshtein Distance (excluding -1): {avg_lev_dist:.4f}")
        logger.info(f"Number of rows with Lev_Dist = -1: {invalid_lev_rows}")
        
        logger.info(f"Average Levenshtein Distance (all): {avg_all_lev_dist:.4f}")
        
        logger.info(f"Average Levenshtein Ratio (excluding -1): {avg_lev_ratio:.4f}")
        logger.info(f"Number of rows with Lev_Ratio = -1: {invalid_levRatio_rows}")
        logger.info(f"Average Levenshtein Ratio (all): {avg_all_lev_ratio:.4f}")
        
        logger.info("===========================================")
    
    
    
        return common_test_df
            
# -----------------------------Calculating CodeBLEU-------------------------------

def _write_temp_file(lines, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.strip() + '\n')

def _tokenize_java(code):
    try:
        import javalang
        token_gen = javalang.tokenizer.tokenize(code)
        return [token.value for token in token_gen]
    except:
        return code.split()

def _compute_codebleu_pair(ref_line, hyp_line, lang='java'):
    temp_id = str(uuid.uuid4())
    ref_path = os.path.abspath(f"ref_{temp_id}.txt")
    hyp_path = os.path.abspath(f"hyp_{temp_id}.txt")
    result_path = "./CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU/bleu.log"
    script_path = f"codebleu_eval_{temp_id}.sh"

    try:
        _write_temp_file([ref_line], ref_path)
        _write_temp_file([hyp_line], hyp_path)

        with open(script_path, "w") as f:
            f.write('#!/usr/bin/env bash\n')
            f.write(
                f'cd ./CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU && '
                f'python calc_code_bleu.py --refs "{ref_path}" --hyp "{hyp_path}" '
                f'--lang {lang} --params 0.25,0.25,0.25,0.25 > bleu.log'
            )

        subprocess.run(f"bash {script_path}", shell=True, check=True)

        with open(result_path, "r") as f:
            lines = f.readlines()
            score = float(lines[-1].split()[2])
            return score

    finally:
        for path in [ref_path, hyp_path, script_path]:
            try:
                os.remove(path)
            except:
                pass

def calc_codebleu_from_beam_lists(refs, hyp_lists, lang='java'):
    """
    Args:
        refs: List[str] - reference code strings
        hyp_lists: List[List[str]] - list of beam predictions for each sample
    Returns:
        List[float]: max CodeBLEU score per sample
    """
    assert len(refs) == len(hyp_lists), "refs and hyp_lists must be same length"
    max_scores = []

    for ref, beams in tqdm(zip(refs, hyp_lists), total=len(refs), desc="Computing CodeBLEU"):
        ref_tok = " ".join(_tokenize_java(ref))
        scores = []
        for hyp in beams:
            hyp_tok = " ".join(_tokenize_java(hyp))
            try:
                score = _compute_codebleu_pair(ref_tok, hyp_tok, lang)
            except Exception as e:
                logger.warning(f"Failed to compute CodeBLEU: {e}")
                score = 0.0
            scores.append(score)
        max_scores.append(max(scores) if scores else 0.0)

    return max_scores


if __name__ == "__main__":
    
    args = parse_arguments()
    output_dir = os.path.join(f"LoraTuned_{args.Data_preprocessing_step}")
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir


    #=== Lora tuning ============================================================
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info("Starting training...")
    tokenizer, model = train(args)
    logger.info("Training finished.")
    
    train_df_len = len(train_df)
    del train_df, val_df, raw_train_df, train_df_all
    torch.cuda.empty_cache()
    gc.collect()
    
    run.finish()

    
    # === Run testing ============================================================

    
    file_name = "test"
    scenario1 = "ORIGINAL DATASET"
    all_beam_preds1, all_targets1 = test(args, model, tokenizer, test_df, scenario1, file_name, beam_size = args.beam_size)

    file_name = "new_test"
    scenario3 = "NEW DATASET"
    all_beam_preds2, all_targets2 = test(args, model, tokenizer, new_test_dff, scenario3, file_name, beam_size = args.beam_size)
    
    torch.cuda.empty_cache()
    gc.collect()





# ----------------------------------------------------------------------------------
if not dist.is_initialized() or dist.get_rank() == 0:
    test_df_copy = managePreds(args, test_df, all_beam_preds1, all_targets1, scenario1, file_name, beam_size = args.beam_size)
    
    file_name = "common_test"
    scenario2 = "COMMON DATASET"
    common_test_df_copy = stats_for_common_data(test_df_copy, common_ids_set, scenario2, file_name, beam_size = args.beam_size)
    
    
    new_test_df_copy = managePreds(args, new_test_dff, all_beam_preds2, all_targets2, scenario3, file_name, beam_size = args.beam_size)
    
    
    file_name = "common_new_test"
    scenario4 = "COMMON NEW DATASET"
    common_new_test_df_copy = stats_for_common_data(new_test_df_copy, common_ids_new_set, scenario2, file_name, beam_size = args.beam_size)
    
    
    # ----------------------------- codeBLEU FOR ORIGINAL DATASET-------------------------------------------------------------------------------------------------------------------------------------
    logger.info("Calculating CodeBLEU scores from beam predictions for original test set...")
    refs = test_df_copy["gold"].astype(str).tolist()
    beams = test_df_copy["predictions"].tolist()
    codebleu_scores = calc_codebleu_from_beam_lists(refs, beams, lang='java')
    test_df_copy["codeBLEU"] = codebleu_scores
    
    final_csv_path = os.path.join(output_dir, f"{Data_preprocessing_steps}_test_predictions_with_codeBLEU_{beam_size}.csv")
    test_df_copy.to_csv(final_csv_path, index=False)
    logger.info(f"Saved test predictions with CodeBLEU and EXM to: {final_csv_path}")
    
    # Exclude 0.0 values for CodeBLEU
    valid_codebleu = test_df_copy.loc[test_df_copy["codeBLEU"] > 0.0, "codeBLEU"]
    avg_codebleu = valid_codebleu.mean() if not valid_codebleu.empty else 0.0
    zero_codebleu_rows = (test_df_copy["codeBLEU"] == 0.0).sum()
    avg_all_codebleu = test_df_copy["codeBLEU"].mean()
    
    
    # Log metrics
    logger.info(f"========== codeBLEU SUMMARY FOR {scenario1}==========")
    logger.info(f"Average CodeBLEU score (excluding 0.0): {avg_codebleu:.4f}")
    logger.info(f"Number of rows with CodeBLEU = 0.0: {zero_codebleu_rows}")
    logger.info(f"Average CodeBLEU score (all): {avg_all_codebleu:.4f}")
    logger.info("===========================================")
    
    
    # -----------------------------codeBLEU FOR COMMON  DATASET-------------------------------
    common_test_df = test_df_copy[test_df_copy["Record_id"].isin(common_ids_set)].copy().reset_index(drop=True)
    logger.info(f"[{args.Data_preprocessing_step}] Final common test size with codeBLEU after cleaning: {len(common_test_df)}")
    final_csv_path = os.path.join(output_dir, f"{Data_preprocessing_steps}_common_test_predictions_with_codeBLEU_{beam_size}.csv")
    common_test_df.to_csv(final_csv_path, index=False)
    logger.info(f"Saved test predictions with CodeBLEU and EXM to: {final_csv_path}")
    
    # Exclude 0.0 values for CodeBLEU
    valid_codebleu = common_test_df.loc[common_test_df["codeBLEU"] > 0.0, "codeBLEU"]
    avg_codebleu = valid_codebleu.mean() if not valid_codebleu.empty else 0.0
    zero_codebleu_rows = (common_test_df["codeBLEU"] == 0.0).sum()
    avg_all_codebleu = common_test_df["codeBLEU"].mean()
    
    # Log metrics
    logger.info(f"========== codeBLEU SUMMARY FOR {scenario2}==========")
    logger.info(f"Average CodeBLEU score (excluding 0.0): {avg_codebleu:.4f}")
    logger.info(f"Number of rows with CodeBLEU = 0.0: {zero_codebleu_rows}")
    logger.info(f"Average CodeBLEU score (all): {avg_all_codebleu:.4f}")
    logger.info("===========================================")
    
    
    # -----------------------------codeBLEU FOR NEW DATASET-------------------------------
    logger.info("Calculating CodeBLEU scores from beam predictions for new github test set...")
    
    refs = new_test_df_copy["gold"].astype(str).tolist()
    beams = new_test_df_copy["predictions"].tolist()
    
    codebleu_scores = calc_codebleu_from_beam_lists(refs, beams, lang='java')
    new_test_df_copy["codeBLEU"] = codebleu_scores
    
    # Save updated CSV
    final_csv_path = os.path.join(output_dir, f"{Data_preprocessing_steps}_new_test_predictions_with_codeBLEU_{beam_size}.csv")
    new_test_df_copy.to_csv(final_csv_path, index=False)
    logger.info(f"Saved test predictions with CodeBLEU and EXM to: {final_csv_path}")
    
    
    # Exclude 0.0 values for CodeBLEU
    valid_codebleu = new_test_df_copy.loc[new_test_df_copy["codeBLEU"] > 0.0, "codeBLEU"]
    avg_codebleu = valid_codebleu.mean() if not valid_codebleu.empty else 0.0
    zero_codebleu_rows = (new_test_df_copy["codeBLEU"] == 0.0).sum()
    avg_all_codebleu = new_test_df_copy["codeBLEU"].mean()
    # Log metrics
    logger.info(f"========== codeBLEU SUMMARY FOR {scenario3}==========")
    logger.info(f"Average CodeBLEU score (excluding 0.0): {avg_codebleu:.4f}")
    logger.info(f"Number of rows with CodeBLEU = 0.0: {zero_codebleu_rows}")
    logger.info(f"Average CodeBLEU score (all): {avg_all_codebleu:.4f}")
    logger.info("===========================================")
    
    
    # -----------------------------codeBLEU FOR COMMON NEW DATASET-------------------------------
    common_new_test_df = new_test_df_copy[new_test_df_copy["Record_id"].isin(common_ids_new_set)].copy().reset_index(drop=True)
    logger.info(f"[{args.Data_preprocessing_step}] Final common New test size with codeBLEU after cleaning: {len(common_new_test_df)}")
    final_csv_path = os.path.join(output_dir, f"{Data_preprocessing_steps}_common_new_test_predictions_with_codeBLEU_{beam_size}.csv")
    common_new_test_df.to_csv(final_csv_path, index=False)
    logger.info(f"Saved test predictions with CodeBLEU and EXM to: {final_csv_path}")
    
    # Exclude 0.0 values for CodeBLEU
    valid_codebleu = common_new_test_df.loc[common_new_test_df["codeBLEU"] > 0.0, "codeBLEU"]
    avg_codebleu = valid_codebleu.mean() if not valid_codebleu.empty else 0.0
    zero_codebleu_rows = (common_new_test_df["codeBLEU"] == 0.0).sum()
    avg_all_codebleu = common_new_test_df["codeBLEU"].mean()
    
    # Log metrics
    logger.info(f"========== codeBLEU SUMMARY FOR {scenario4}==========")
    logger.info(f"Average CodeBLEU score (excluding 0.0): {avg_codebleu:.4f}")
    logger.info(f"Number of rows with CodeBLEU = 0.0: {zero_codebleu_rows}")
    logger.info(f"Average CodeBLEU score (all): {avg_all_codebleu:.4f}")
    logger.info("===========================================")






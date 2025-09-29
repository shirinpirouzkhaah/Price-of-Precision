import os
import torch
import logging
import numpy as np
import pandas as pd
import wandb
import multiprocessing
import re
from difflib import SequenceMatcher
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from configs import set_seed
from utils import RefineDataset
from tqdm.auto import tqdm
import subprocess
import Levenshtein
import uuid
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import unidecode
import unicodedata
import gc
from torch.cuda.amp import GradScaler, autocast

import time
start_time = time.time()
max_training_seconds = 5.5 * 24 * 60 * 60 


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Define CLI arguments
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--Data_preprocessing_step", type=str)
    parser.add_argument("--Train_data_path", type=str)
    parser.add_argument("--Test_data_path", type=str)
    parser.add_argument("--Common_test_data_path", type=str)
    parser.add_argument("--New_test_data_path", type=str)
    parser.add_argument("--Common_New_test_data_path", type=str)
    parser.add_argument("--column_names", nargs="+", type=str)
    parser.add_argument("--beam_size", type=int)
    parser.add_argument("--SEED", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log_file", type=str)

    # Parse CLI arguments
    cli_args = vars(parser.parse_args())

    # Load YAML configuration if provided
    yaml_config = {}
    if cli_args.get("config"):
        with open(cli_args["config"], "r") as f:
            yaml_config = yaml.safe_load(f) or {}

    # Default hyperparameters
    intended_defaults = {
        "beam_size": 10,
        "SEED": 42,
        "seed": 42,
        "train_epochs": 45,
        "max_source_length": 200,
        "max_target_length": 200,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "test_batch_size": 1,
        "learning_rate": 3e-4,
        "gradient_accumulation_steps": 2,
        "mask_rate": 0.15,
        "save_steps": 1000,
        "log_steps": 100,
        "train_steps": 200000,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "early_stopping_patience": 10,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "cpu_count": multiprocessing.cpu_count(),
        "gpu_per_node": 1,
        "output_dir": "results", 
        "model_name_or_path": "microsoft/codereviewer",
        "load_model_path": None
    }

    # Merge priority: defaults < CLI < YAML
    merged = intended_defaults.copy()
    merged.update(yaml_config)
    for k, v in cli_args.items():
        if v is not None:
            merged[k] = v

    return argparse.Namespace(**merged)


# ---------------------------- Setup --------------------------------
args = parse_arguments()

seed = args.SEED
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
Data_preprocessing_steps = args.Data_preprocessing_step
column_names = args.column_names
beam_size = args.beam_size
train_data_path = args.Train_data_path
test_data_path = args.Test_data_path
common_test_data_path = args.Common_test_data_path
new_test_data_path = args.New_test_data_path
common_new_test_data_path = args.Common_New_test_data_path


match = re.search(r'_(ProjectLevel\.csv|TimeLevel\.csv)$', train_data_path)
suffix = match.group(1) if match else "UnknownLevel"
suffix = suffix.replace('.csv', '')

# Construct the safe project name
safe_project_name = re.sub(r'\W+', '_', f"{args.model_name}_{Data_preprocessing_steps}_{suffix}_3")
run = wandb.init(project=safe_project_name, notes="experiment", reinit=True)


output_dir = f"results/finetuned_{args.model_name}_{Data_preprocessing_steps}"
best_model_dir = os.path.join(output_dir, "finetuned_model_best")

epoch_model_dir = os.path.join(output_dir, "finetuned_model_epoch")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(epoch_model_dir, exist_ok=True)

args.output_dir = output_dir


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, args.log_file), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



# -------------------------- Load Data --------------------------------
# Load and log NaN removal
raw_train_df = pd.read_csv(train_data_path)
raw_test_df = pd.read_csv(test_data_path)
new_test_df = pd.read_csv(new_test_data_path)
common_ids_df = pd.read_csv(common_test_data_path, usecols=["Common_Record_id"])
common_ids_new_df = pd.read_csv(common_new_test_data_path, usecols=["Common_Record_id"])

original_train_len = len(raw_train_df)
original_test_len = len(raw_test_df)
original_new_test_len = len(new_test_df)


common_ids_set = set(common_ids_df["Common_Record_id"].dropna().unique())
common_ids_new_set = set(common_ids_new_df["Common_Record_id"].dropna().unique())

new_test_dff = new_test_df.dropna(subset=column_names)
train_df = raw_train_df.dropna(subset=column_names)
test_df = raw_test_df.dropna(subset=column_names)


na_dropped_train = original_train_len - len(train_df)
na_dropped_test = original_test_len - len(test_df)
na_dropped_new_test = original_new_test_len - len(new_test_dff)


logger.info(f"[{Data_preprocessing_steps}] Dropped {na_dropped_train} rows from train due to NaNs in {column_names}")
logger.info(f"[{Data_preprocessing_steps}] Dropped {na_dropped_test} rows from test due to NaNs in {column_names}")
logger.info(f"[{Data_preprocessing_steps}] Dropped {na_dropped_new_test} rows from new test due to NaNs in {column_names}")


# ------------------ Remove Train-Test Overlap -------------------

logger.info(f"[{Data_preprocessing_steps}] Checking for duplicates between train and test...")
pre_dedup_len = len(train_df)
train_df["hash"] = train_df[column_names].astype(str).agg("||".join, axis=1).apply(hash)
train_df = train_df.drop_duplicates(subset=["hash"]).drop(columns=["hash"])
dedup_dropped = pre_dedup_len - len(train_df)

logger.info(f"[{Data_preprocessing_steps}] Removed {dedup_dropped} overlapping rows from train that also appeared in test")

# ------------------ Remove Train-NewTest Overlap -------------------

logger.info(f"[{Data_preprocessing_steps}] Checking for duplicates between train and new test...")

train_df["hash"] = train_df[column_names].astype(str).agg("||".join, axis=1).apply(hash)
new_test_dff["hash"] = new_test_dff[column_names].astype(str).agg("||".join, axis=1).apply(hash)
overlap_hashes = set(train_df["hash"]).intersection(set(new_test_dff["hash"]))
pre_overlap_len = len(train_df)
train_df = train_df[~train_df["hash"].isin(overlap_hashes)].drop(columns=["hash"])
overlap_dropped = pre_overlap_len - len(train_df)
new_test_dff = new_test_dff.drop(columns=["hash"])



if new_test_dff.empty:
    logger.warning(f"[{args.Data_preprocessing_step}] 'new_test_dff' is empty. Copying one row from 'test_df'.")
    new_test_dff = test_df.head(10).copy().reset_index(drop=True)



train_df_all = train_df.dropna(subset=column_names)
train_df = train_df_all.sample(frac=0.95, random_state=42)
val_df = train_df_all.drop(train_df.index)

logger.info(f"[{Data_preprocessing_steps}] Removed {overlap_dropped} overlapping rows from train that also appeared in new test")
logger.info(f"[{Data_preprocessing_steps}] Final train size after cleaning: {len(train_df)}")
logger.info(f"[{Data_preprocessing_steps}] Final validation size after cleaning: {len(val_df)}")
logger.info(f"[{args.Data_preprocessing_step}] Final new test size after cleaning: {len(new_test_dff)}")


test_len = len(test_df)
train_len = len(train_df)
val_len = len(val_df)
total_len = train_len + val_len + test_len

train_pct = (train_len / total_len) * 100
val_pct = (val_len / total_len) * 100
test_pct = (test_len / total_len) * 100

logger.info(f"Dataset split percentages:")
logger.info(f"  Train: {train_len} samples ({train_pct:.2f}%)")
logger.info(f"  Validation: {val_len} samples ({val_pct:.2f}%)")
logger.info(f"  Test: {test_len} samples ({test_pct:.2f}%)")


# -------------------------- Helper Functions --------------------------------

def format_content(content):
	content = str(content)
	content = unicodedata.normalize("NFKC", content) 
	content = content.encode('utf-8').decode('utf-8', 'ignore').strip()
	content = unidecode.unidecode(content)
	return content



def cleaning_model_output(text):
    text = str(text)
    text = format_content(text)  # normalize unicode and accents
    
    text = re.sub(r'\t', ' ', text, flags=re.UNICODE)  # normalize tabs
    text = re.sub(r'[\r ]*\n[\r ]*', '\n', text, flags=re.UNICODE)  # normalize newlines
    text = re.sub(r'//[^\n]*', '', text, flags=re.UNICODE)  # remove one-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # remove multi-line comments

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
    return a == b


def get_loader(df, column_names, args, tokenizer, pool, stage):
    """
    Args:
        df: pandas DataFrame
        column_names: list of columns [source, comment, target]
        args: training/testing args object
        tokenizer: tokenizer instance
        pool: multiprocessing pool
        stage: one of "train", "eval", "test"
    Returns:
        dataset, sampler, dataloader
    """
    def fn(features): return features
    
    dataset = RefineDataset(tokenizer, pool, args, df, column_names)
    logger.info(f"Data length: {len(dataset)}.")

    if stage == "train":
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    elif stage == "eval":
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size
    elif stage == "test":
        sampler = SequentialSampler(dataset)
        batch_size = args.test_batch_size
    else:
        raise ValueError(f"Unknown stage '{stage}' — must be one of: 'train', 'eval', 'test'")

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.cpu_count,
        collate_fn=fn,
        pin_memory=True,
        persistent_workers=True
    )

    return dataset, sampler, dataloader


def validate_and_score(args, eval_dataloader, model, tokenizer, val_df):
    logger.info("  ***** Running EM evaluation on dev set *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    pred_ids = []
    total_loss = 0.0
    total = 0
    exm_count = 0

    for step, examples in enumerate(eval_dataloader, 1):
        source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_id)
        target_ids = torch.tensor([ex.target_ids for ex in examples], dtype=torch.long).to(args.device)
        target_mask = target_ids.ne(tokenizer.pad_id)


        with torch.no_grad():
            preds = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                use_cache=True,
                num_beams=1,
                num_return_sequences=1,
                max_length=args.max_target_length
            )
            loss = model(
                input_ids=source_ids,
                input_labels=None,
                decoder_input_ids=target_ids,
                attention_mask=source_mask,
                decoder_attention_mask=target_mask,
                encoder_loss=False
            )
            total_loss += loss.item()
        
        del source_ids, target_ids, source_mask, target_mask
        torch.cuda.empty_cache()
        gc.collect()
        

        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)

    pred_nls = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    golds = val_df["new_clean"].tolist()

    pred_nls = pred_nls[:len(golds)]
    for i in range(len(golds)):
        pred_nls[i], golds[i] = RefineDataset.process_pred_gold(pred_nls[i], golds[i])


    for pred, gold in zip(pred_nls, golds):
        cleaned_pred = to_single_line(cleaning_model_output(pred))
        cleaned_gold = to_single_line(cleaning_model_output(gold))
        if is_exact_match(cleaned_pred, cleaned_gold):
            exm_count += 1
        total += 1

    exm = exm_count / total if total > 0 else 0.0
    avg_loss = total_loss / len(eval_dataloader)
    logger.info(f"Eval Loss: {avg_loss:.4f}")
    logger.info(f"EM (is_cexact_match): {exm:.4f}")
    return avg_loss, exm


def save_model(model, optimizer, scheduler, output_dir, config, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))



def save_model2(model, optimizer, scheduler, output_dir, config, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
    torch.save(
        optimizer.state_dict(),
        output_optimizer_file,
        _use_new_zipfile_serialization=False,
    )
    output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
    torch.save(
        scheduler.state_dict(),
        output_scheduler_file,
        _use_new_zipfile_serialization=False,
    )





def train(args):
    logger.info("Device: %s, Batch size: %s", args.device, args.train_batch_size)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    special_tokens_dict = {
        'additional_special_tokens': ['<code>', '</code>', '<START>', '<END>', '<technical_language>', '</technical_language>']
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Number of added special tokens to tokenizer:\n{num_added_toks}")
    print(f"added special tokens to tokenizer:\n{tokenizer.additional_special_tokens}")
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(args.device)
    model = torch.compile(model)
    
    scaler = GradScaler()
    
    
    pool = multiprocessing.Pool(args.cpu_count)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps)


    _, _, train_dataloader = get_loader(train_df, column_names, args, tokenizer, pool, stage = "train")
    _, _, valid_dataloader = get_loader(val_df, column_names, args, tokenizer, pool, stage = "eval")

    best_exm = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    min_delta = 0.01
    

    for epoch in range(1, args.train_epochs + 1):
        
        used_time = time.time() - start_time
        logger.info(f"Elapsed time: {used_time / 3600:.2f} hours. Remaining: {(max_training_seconds - used_time) / 3600:.2f} hours.")      
        
        
        
        seed_backup = args.seed
        args.seed += epoch
        set_seed(args)
        args.seed = seed_backup
        model.train()
        total_loss = 0.0

        #nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        for step, examples in enumerate(train_dataloader, 1):
            source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(args.device)
            target_ids = torch.tensor([ex.target_ids for ex in examples], dtype=torch.long).to(args.device)
            source_mask = source_ids.ne(tokenizer.pad_id)
            target_mask = target_ids.ne(tokenizer.pad_id)

            with autocast():
                loss = model(
                    input_ids=source_ids,
                    input_labels=None,
                    decoder_input_ids=target_ids,
                    attention_mask=source_mask,
                    decoder_attention_mask=target_mask,
                    encoder_loss=False
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
                
            scaler.scale(loss).backward()
            total_loss += loss.item()

            if step % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # tr_loss += loss.item()
            # nb_tr_examples += source_ids.size(0)
            # nb_tr_steps += 1
            # loss.backward()
            
            # total_loss += loss.item()

            # if nb_tr_steps % args.gradient_accumulation_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     scheduler.step()
                
                
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
                wandb.log({"epoch": epoch, "step": step, "loss": loss.item()})
                
                
            del source_ids, target_ids, source_mask, target_mask
            torch.cuda.empty_cache()     
            gc.collect()
                
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info("Running validation and calculating EXM score")
        val_loss, val_exm = validate_and_score(args, valid_dataloader, model, tokenizer, val_df) 
        
        logger.info(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val EXM: {val_exm:.4f}")
        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_exm": val_exm})



        if  val_loss < best_val_loss - min_delta:
            patience_counter = 0
            
            best_val_loss = val_loss
            best_epoch = epoch
            best_exm = val_exm

            save_model2(model, optimizer, scheduler, best_model_dir, config, tokenizer)
            logger.info(f"New best model saved at epoch {epoch} with EXM={val_exm:.4f}, ValLoss={val_loss:.4f}")
        else:
            patience_counter += 1
            

        if patience_counter >= args.early_stopping_patience:
            logger.info("Early stopping triggered")
            break
        
        
        elapsed = time.time() - start_time
        if elapsed >= max_training_seconds:
            
            if patience_counter == 0:
                if not os.path.exists(os.path.join(best_model_dir, 'pytorch_model.bin')):
                    save_model2(model, optimizer, scheduler, best_model_dir, config, tokenizer)
                    logger.info(f"Saved epoch {epoch} model to best model dir {best_model_dir}")
                break
            
            elif patience_counter > 0:
                logger.info(f"Time limit reached after best checkpoint at patience {patience_counter}. Stopping training.")
                break
        
        
    
    if not os.path.exists(os.path.join(best_model_dir, 'pytorch_model.bin')):
        save_model2(model, optimizer, scheduler, best_model_dir, config, tokenizer)


    return epoch, best_epoch, best_val_loss, best_exm, config, model, tokenizer

    
# ----------------------------- TESTING -------------------------------------------------------------------------------------------------------------------------------------

def run_test(args, model, tokenizer, test_df, column_names, scenario, file_name, beam_size=10):
    logger.info("Running final test on best model...")
    model.eval()

    # Set up test data
    pool = multiprocessing.Pool(args.cpu_count)
    _, _, test_dataloader = get_loader(test_df, column_names, args, tokenizer, pool, stage = "test")

    # Init results container
    test_df_copy = test_df.copy().reset_index(drop=True)
    test_df_copy["predictions"] = None
    test_df_copy["RawPredictions"] = None
    test_df_copy["gold"] = None

    test_df_copy["EXMEXM"] = False

    test_df_copy["EXMEXM1"] = False

    test_df_copy["EXMEXM3"] = False

    test_df_copy["EXMEXM5"] = False
    
    test_df_copy["Lev_Dist"] = 0.0
    test_df_copy["Lev_Ratio"] = 0.0

    all_preds_txt = []
    all_beam_preds = []
    all_targets = []

    logger.info("Generating predictions...")
    for idx, examples in enumerate(tqdm(test_dataloader)):
        source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_id)
        target_ids = torch.tensor([ex.target_ids for ex in examples], dtype=torch.long).to(args.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                use_cache=True,
                max_length=args.max_target_length,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                length_penalty=1.0,
                early_stopping=True
            )


        # Decode target once per input example (1 target per beam group)
        target_text = tokenizer.decode(target_ids[0], skip_special_tokens=True)
        all_targets.append(target_text)

        beam_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        all_beam_preds.append(beam_preds)
        
        del source_ids, target_ids, source_mask
        torch.cuda.empty_cache()
        gc.collect()


    logger.info("Scoring predictions...")

    EXM = EXM1 = EXM3 = EXM5 = 0
    EXMEXM = EXMEXM1 = EXMEXM3 = EXMEXM5 = 0
    print (all_beam_preds[0])
    
    
    for i in range(len(all_targets)):
        raw_preds = all_beam_preds[i]
        test_df_copy.at[i, "RawPredictions"] = raw_preds 
    
        
        beam_group = []
        for beam_pred in all_beam_preds[i]:
            pred_proc, gold_proc = RefineDataset.process_pred_gold(beam_pred, all_targets[i])
            gold = to_single_line(cleaning_model_output(gold_proc))
            cleaned_pred = to_single_line(cleaning_model_output(pred_proc))
            beam_group.append(cleaned_pred)
            
        test_df_copy.at[i, "gold"] = gold 
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
        
        test_df_copy.at[i, "EXMEXM"] = is_exact_exact
        
        # Add top-k EXM columns
        test_df_copy.at[i, "EXMEXM1"] = is_exact_exact1
        
        test_df_copy.at[i, "EXMEXM3"] = is_exact_exact3
        
        test_df_copy.at[i, "EXMEXM5"] = is_exact_exact5
        
        all_preds_txt.extend(beam_group)
        



    # Save outputs
    with open(os.path.join(output_dir, f"{Data_preprocessing_steps}_{file_name}_predictions_{beam_size}.txt"), 'w') as f:
        for p in all_preds_txt:
            f.write(p + '\n')
            
    final_csv_path = os.path.join(output_dir, f"{Data_preprocessing_steps}__{file_name}_predictions_{beam_size}.csv")
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
    
    return test_df_copy

def stats_for_common_data(test_df_copy, common_ids_set, scenario, file_name, beam_size = 10):
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

    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info("Starting training...")
    epoch, best_epoch, best_val_loss, best_exm, config, model, tokenizer = train(args)
    logger.info("Training finished.")
    logger.info(f"Number of training examples: {len(train_df)}")
    logger.info(f"Number of epochs trained: {epoch + 1}")
    logger.info(f"Best epoch: {best_epoch + 1}")
    logger.info(f"best val loss: {best_val_loss:.4f}, best exm: {best_exm:.4f}")
    
    
    run.finish()

    # === PUTTING MODEL IN EVAL STATE ============================================================
    # args.model_name_or_path = os.path.join(args.output_dir, "best-checkpoint")
    # set_seed(args)
    # config, model, tokenizer = build_or_load_gen_model(args)
    # special_tokens_dict = {
    #     'additional_special_tokens': ['<code>', '</code>', '<START>', '<END>', '<technical_language>', '</technical_language>']
    # }
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print(f"Number of added special tokens to tokenizer:\n{num_added_toks}")
    # print(f"added special tokens to tokenizer:\n{tokenizer.additional_special_tokens}")
    # model.resize_token_embeddings(len(tokenizer))
    # model = model.to(args.device)
    
    # === Run testing ============================================================
    file_name = "test"
    scenario1 = "ORIGINAL DATASET"
    test_df_copy = run_test(args, model, tokenizer, test_df, column_names, scenario1, file_name,  beam_size=10)

    
    # === Original data common records ============================================================
    file_name = "common_test"
    scenario2 = "COMMON DATASET"
    common_test_df_copy = stats_for_common_data(test_df_copy, common_ids_set, scenario2, file_name, beam_size = 10)
    
    
    # === Run testing for new github data =======================================
    file_name = "new_test"
    scenario3 = "NEW DATASET"
    new_test_df_copy = run_test(args, model, tokenizer, new_test_dff, column_names, scenario3, file_name, beam_size=10)
    
    # === New data common records ============================================================
    file_name = "common_new_test"
    scenario4 = "COMMON NEW DATASET"
    common_new_test_df_copy = stats_for_common_data(new_test_df_copy, common_ids_new_set, scenario4, file_name, beam_size = 10)


    


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

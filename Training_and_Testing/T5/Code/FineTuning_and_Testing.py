import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Language\(path, name\) is deprecated.*",
    category=FutureWarning,
    module=r"tree_sitter.*"
)


import pandas as pd
import numpy as np
import os
import time
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Config, T5Tokenizer, T5ForConditionalGeneration,
     get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import wandb
import re
from tqdm.auto import tqdm
import subprocess
import Levenshtein
from difflib import SequenceMatcher
import copy
import uuid
import argparse
import yaml
import unidecode
import unicodedata
import gc
import time

start_time = time.time()
max_training_seconds = 5.5 * 24 * 60 * 60 
workerss = min(4, os.cpu_count())



def parse_arguments():
    parser = argparse.ArgumentParser()
    # Defining what arguments with which types this script can accept 
    # My script understands these options if someone passes them in the terminal
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
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--output_dir", type=str)
    
    

    # Reading actual values from the user (CLI or YAML)
    cli_args = vars(parser.parse_args())
    yaml_config = {}
    if cli_args.get("config"):
        with open(cli_args["config"], "r") as f:
            yaml_config = yaml.safe_load(f) or {}
            
    
    intended_defaults = {
        "beam_size": 10,
        "SEED": 42
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


"""
A seed is like setting a starting point for generating random numbers. 
If you use the same seed, you'll always get the same "random" results. 
This is important for reproducibility—so others (or you later) can get the same training behavior every time you run the code.
Use it once for PyTorch and once for NumPy, because both libraries can generate random numbers.
42 ?:) from The Hitchhiker's Guide to the Galaxy—"the answer to life.
This forces CUDA operations (GPU computations) to be deterministic, i.e., give the same result every time. 
It can make training slower but ensures reproducibility.
"""
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


#******************************************************************************************
epochs = 45
#******************************************************************************************

# -------------------- Tokenizer & Model Setup ------------------------

tokenizer_path = '../Tokenizer'
config_path = '../PytorchPretrained2/config.json'
model_checkpoint_path = '../PytorchPretrained2/model.bin'


tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

special_tokens_dict = {
    'additional_special_tokens': ['<code>', '</code>', '<START>', '<END>', '<technical_language>', '</technical_language>']
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Number of added special tokens to tokenizer:\n{num_added_toks}")
print(f"added special tokens to tokenizer:\n{tokenizer.additional_special_tokens}")



config = T5Config.from_pretrained(config_path)
model = T5ForConditionalGeneration(config)
model.load_state_dict(torch.load(model_checkpoint_path))
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------- Dataset Class ----------------------------

class T5Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, source_len, target_len, column_names):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.column_names = column_names
        self.inputs, self.targets = [], []

        for _, row in dataframe.iterrows():
            source_text = f"<code>{row[self.column_names[0]]}</code><technical_language>{row[self.column_names[1]]}</technical_language>"
            target_text = f"{row[self.column_names[2]]}"
            self.inputs.append(source_text)
            self.targets.append(target_text)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source = self.tokenizer(self.inputs[idx], max_length=self.source_len, padding='max_length', truncation=True, return_tensors="pt")
        target = self.tokenizer(self.targets[idx], max_length=self.target_len, padding='max_length', truncation=True, return_tensors="pt")
        return {
            "source_ids": source["input_ids"].squeeze(),
            "source_mask": source["attention_mask"].squeeze(),
            "target_ids": target["input_ids"].squeeze(),
            "raw_source_text": self.inputs[idx],
            "raw_target_text": self.targets[idx]
        }

# ---------------------------- Load Splits ----------------------------

train_dataset = T5Dataset(tokenizer, train_df, 512, 512, column_names)
val_dataset = T5Dataset(tokenizer, val_df, 512, 512, column_names)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers= workerss)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers= workerss)

# ----------------------- Training Function ---------------------------
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


"""
The decoder's attention masking (including padding) is automatically handled by HuggingFace’s T5 model when you 
pass labels to the forward method so we don’t need target masks
"""
def validate_and_score(model, tokenizer, device, loader):
    model.eval()
    total_loss = 0.0
    exm_count = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            ids = data["source_ids"].to(device)
            mask = data["source_mask"].to(device)
            targets = data["target_ids"].to(device)

            # Compute loss
            y_ids = targets[:, :-1].contiguous()
            lm_labels = targets[:, 1:].clone()
            lm_labels[targets[:, 1:] == tokenizer.pad_token_id] = -100
            loss = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)[0]
            total_loss += loss.item()
            
            
            outputs = model.generate(
                input_ids=ids,
                attention_mask=mask,
                use_cache=True,
                max_length=512,
                num_beams=1,             
                num_return_sequences=1
            )

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds = [to_single_line(cleaning_model_output(pred)) for pred in decoded_preds]


            decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)
            targets_cleaned = [to_single_line(cleaning_model_output(t)) for t in decoded_targets]

            for pred_str, target_str in zip(preds, targets_cleaned):
                if is_exact_match(pred_str,target_str):
                    exm_count += 1
                total += 1
        


    avg_loss = total_loss / len(loader)
    exm = exm_count / total if total > 0 else 0.0
    
    
    del ids, mask, targets, y_ids, lm_labels, outputs, decoded_preds, preds, decoded_targets, targets_cleaned
    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss, exm



accumulation_steps = 2

def train_and_validate(model, device, train_loader, val_loader, epochs, optimizer, scheduler):
    model.train()
    early_stopping_patience = 10
    min_delta = 0.01

    best_epoch = -1
    best_exm = -1
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        used_time = time.time() - start_time
        logger.info(f"Elapsed time: {used_time / 3600:.2f} hours. Remaining: {(max_training_seconds - used_time) / 3600:.2f} hours.")      
        total_loss = 0.0
        optimizer.zero_grad()

        for step, data in enumerate(train_loader):
            ids = data["source_ids"].to(device)
            mask = data["source_mask"].to(device)
            y = data["target_ids"].to(device)

            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0] / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps  # Track unscaled loss

            if (step + 1) % accumulation_steps == 0 or (step + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item() * accumulation_steps:.4f}")
                wandb.log({"epoch": epoch, "step": step, "loss": loss.item() * accumulation_steps})

            del outputs, loss, ids, mask, y, y_ids, lm_labels
            torch.cuda.empty_cache()
            gc.collect()

        avg_train_loss = total_loss / len(train_loader)
        logger.info("Running validation and calculating EXM score")
        val_loss, val_exm = validate_and_score(model, tokenizer, device, val_loader)

        logger.info(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val EXM: {val_exm:.4f}")
        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_exm": val_exm})

        exm_improved = val_exm > best_exm
        val_loss_improved = val_loss < best_val_loss - min_delta

        if val_loss_improved:
            patience_counter = 0
            best_exm = val_exm
            best_val_loss = val_loss
            best_epoch = epoch
            
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            torch.save(model.state_dict(), os.path.join(best_model_dir, f'{Data_preprocessing_steps}_model_best.bin'))

            logger.info(f"New best model saved at epoch {epoch} with EXM={val_exm:.4f}, ValLoss={val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logger.info("Early stopping triggered")
            break

        
        elapsed = time.time() - start_time
        if elapsed >= max_training_seconds:
            
            if patience_counter == 0:
                if not os.path.exists(os.path.join(best_model_dir, f'{Data_preprocessing_steps}_model_best.bin')):
                    model.save_pretrained(best_model_dir)
                    tokenizer.save_pretrained(best_model_dir)
                    torch.save(model.state_dict(), os.path.join(best_model_dir, f'{Data_preprocessing_steps}_model_best.bin'))
                    logger.info(f"Saved epoch {epoch} model to best model dir {best_model_dir}")
                break
            
            elif patience_counter > 0:
                logger.info(f"Time limit reached after best checkpoint at patience {patience_counter}. Stopping training.")
                break
        
        
    
    if not os.path.exists(os.path.join(best_model_dir, f'{Data_preprocessing_steps}_model_best.bin')):
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        torch.save(model.state_dict(), os.path.join(best_model_dir, f'{Data_preprocessing_steps}_model_best.bin'))


    return epoch, best_epoch, best_val_loss, best_exm, model, tokenizer



def run_test_and_evaluate(model, tokenizer, test_df, column_names, Data_preprocessing_steps, scenario, file_name, beam_size=10):
    model.eval()

    test_dataset = T5Dataset(tokenizer, test_df, 512, 512, column_names)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
    


    all_preds_txt= []
    EXM = EXM1 = EXM3 = EXM5 = 0
    EXMEXM = EXMEXM1 = EXMEXM3 = EXMEXM5 = 0

    for idx, data in enumerate(tqdm(test_loader)):
        ids = data["source_ids"].to(device)
        mask = data["source_mask"].to(device)
        y = data["target_ids"].to(device)
        

        outputs = model.generate(
            input_ids=ids,
            attention_mask=mask,
            use_cache=True,
            max_length=512,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            length_penalty=1.0,
            early_stopping=True
        )
        
        
        Raw_preds = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
        Raw_preds = [re.sub(r"<extra_id_\d+>", "", p).strip() for p in Raw_preds]
        preds = [to_single_line(cleaning_model_output(p)) for p in Raw_preds]
        target = [to_single_line(cleaning_model_output(tokenizer.decode(t, skip_special_tokens=True))) for t in y]

        beam_group = preds
        gold = target[0]
    
        is_exact_exact = any(is_exact_match(pred, gold) for pred in beam_group)

        # Top-K EXM
        is_exact_exact1 = any(is_exact_match(pred, gold) for pred in beam_group[:1])

        is_exact_exact3 = any(is_exact_match(pred, gold) for pred in beam_group[:3])

        is_exact_exact5 = any(is_exact_match(pred, gold) for pred in beam_group[:5])
        
            
            
        lev_dists = [Levenshtein.distance(pred, gold) for pred in preds]
        lev_ratios = [Levenshtein.ratio(pred, gold) for pred in preds]

        min_lev = min(lev_dists) if lev_dists else -1
        max_ratio = max(lev_ratios) if lev_ratios else -1.0
        
        

        test_df_copy.at[idx, "Lev_Dist"] = min_lev
        test_df_copy.at[idx, "Lev_Ratio"] = max_ratio
        
        test_df_copy.at[idx, "RawPredictions"] = Raw_preds  
        test_df_copy.at[idx, "predictions"] = preds
        test_df_copy.at[idx, "gold"] = gold  
        
        test_df_copy.at[idx, "EXMEXM"] = is_exact_exact
        
        # Add top-k EXM columns
        test_df_copy.at[idx, "EXMEXM1"] = is_exact_exact1
        
        test_df_copy.at[idx, "EXMEXM3"] = is_exact_exact3
        
        test_df_copy.at[idx, "EXMEXM5"] = is_exact_exact5

        all_preds_txt.extend(preds)
        
        del ids, mask, y, outputs, Raw_preds, preds, target, beam_group, gold
        torch.cuda.empty_cache()
        gc.collect()

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


# -------------------------- Optimizer Setup ---------------------------

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * len(train_loader) * epochs),
    num_training_steps=len(train_loader) * epochs
)

epoch, best_epoch, best_val_loss, best_exm, model, tokenizer = train_and_validate(model, device, train_loader, val_loader, epochs, optimizer, scheduler)
logger.info(f"Number of training examples: {len(train_df)}")
logger.info(f"Number of epochs trained: {epoch + 1}")
logger.info(f"Best epoch: {best_epoch + 1}")
logger.info(f"best val loss: {best_val_loss:.4f}, best exm: {best_exm:.4f}")

run.finish()

# ----------------------------- TESTING ORIGINAL DATASET -------------------------------------------------------------------------------------------------------------------------------------

# Load best model and tokenizer
# config = T5Config.from_pretrained(best_model_dir)
# model = T5ForConditionalGeneration(config)
# model.load_state_dict(torch.load(os.path.join(best_model_dir, f'{Data_preprocessing_steps}_model_best.bin')))
# tokenizer = T5Tokenizer.from_pretrained(best_model_dir)

# special_tokens_dict = {
#     'additional_special_tokens': ['<code>', '</code>', '<START>', '<END>', '<technical_language>', '</technical_language>']
# }
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# print(f"Number of added special tokens to tokenizer:\n{num_added_toks}")
# print(f"added special tokens to tokenizer:\n{tokenizer.additional_special_tokens}")
# model.resize_token_embeddings(len(tokenizer))
# model.to(device)



file_name = "test"
scenario1 = "ORIGINAL DATASET"
test_df_copy = run_test_and_evaluate(model, tokenizer, test_df, column_names, Data_preprocessing_steps, scenario1, file_name, beam_size=10)


file_name = "common_test"
scenario2 = "COMMON DATASET"
common_test_df_copy = stats_for_common_data(test_df_copy, common_ids_set, scenario2, file_name, beam_size = 10)


file_name = "new_test"
scenario3 = "NEW DATASET"
new_test_df_copy = run_test_and_evaluate(model, tokenizer, new_test_dff, column_names, Data_preprocessing_steps, scenario3, file_name, beam_size=10)


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




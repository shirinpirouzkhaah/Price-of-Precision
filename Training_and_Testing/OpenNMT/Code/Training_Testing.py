import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Language\(path, name\) is deprecated.*",
    category=FutureWarning,
    module=r"tree_sitter.*"
)

import subprocess
import pandas as pd
import os
import logging
import re
import Levenshtein
import copy
import uuid
from difflib import SequenceMatcher
from tqdm.auto import tqdm
import argparse
import yaml
import math
import unidecode
import unicodedata


def parse_arguments():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--batch_size", type=int)

    cli_args = vars(parser.parse_args())
    yaml_config = {}

    if cli_args.get("config"):
        with open(cli_args["config"], "r") as f:
            yaml_config = yaml.safe_load(f) or {}

    defaults = {
        "beam_size": 10,
        "SEED": 42,
        "batch_size": 32,
        "log_file": "log.txt"}

    merged = defaults.copy()
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
Data_preprocessing_steps = args.Data_preprocessing_step
column_names = args.column_names
beam_size = args.beam_size
batch_size = args.batch_size



train_data_path = args.Train_data_path
test_data_path = args.Test_data_path
common_test_data_path = args.Common_test_data_path
new_test_data_path = args.New_test_data_path
common_new_test_data_path = args.Common_New_test_data_path

output_dir = f"results/trained_{args.model_name}_{Data_preprocessing_steps}"
os.makedirs(output_dir, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, args.log_file), mode='a'),
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
new_test_dff = new_test_df.dropna(subset=column_names)

common_ids_set = set(common_ids_df["Common_Record_id"].dropna().unique())
common_ids_new_set = set(common_ids_new_df["Common_Record_id"].dropna().unique())

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

train_df_all = train_df.dropna(subset=column_names)
train_df = train_df_all.sample(frac=0.95, random_state=42)
val_df = train_df_all.drop(train_df.index)



if new_test_dff.empty:
    logger.warning(f"[{Data_preprocessing_steps}] 'new_test_dff' is empty. Copying one row from 'test_df'.")
    new_test_dff = test_df.head(10).copy().reset_index(drop=True)


logger.info(f"[{Data_preprocessing_steps}] Removed {overlap_dropped} overlapping rows from train that also appeared in new test")
logger.info(f"[{Data_preprocessing_steps}] Final train size after cleaning: {len(train_df)}")
logger.info(f"[{Data_preprocessing_steps}] Final validation size after cleaning: {len(val_df)}")
logger.info(f"[{Data_preprocessing_steps}] Final test size after cleaning: {len(test_df)}")
logger.info(f"[{Data_preprocessing_steps}] Final new test size after cleaning: {len(new_test_dff)}")


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
steps_per_epoch = 10000
max_step = 200000
#******************************************************************************************


# -------------------------- Save txt files for OpenNMT --------------------------------

txt_data_dir = "./results/txtData"
os.makedirs(txt_data_dir, exist_ok=True)

def save_split_to_txt(df, prefix):
    src1_path = os.path.join(txt_data_dir, f"src1-{prefix}.txt")
    src2_path = os.path.join(txt_data_dir, f"src2-{prefix}.txt")
    tgt_path = os.path.join(txt_data_dir, f"tgt-{prefix}.txt")

    df[column_names[0]].to_csv(src1_path, index=False, header=False)
    df[column_names[1]].to_csv(src2_path, index=False, header=False)
    df[column_names[2]].to_csv(tgt_path, index=False, header=False)

    logger.info(f"[{Data_preprocessing_steps}] Saved {prefix} split: {len(df)} rows to {txt_data_dir}")

# Save all 3 splits
save_split_to_txt(train_df, "train")
save_split_to_txt(val_df, "val")
save_split_to_txt(test_df, "test")
save_split_to_txt(new_test_dff, "newtest")


path_src1_train = './results/txtData/src1-train.txt'
path_src2_train = './results/txtData/src2-train.txt'
path_tgt_train = './results/txtData/tgt-train.txt'

path_src1_val = './results/txtData/src1-val.txt'
path_src2_val = './results/txtData/src2-val.txt'
path_tgt_val = './results/txtData/tgt-val.txt'

path_src1_test = './results/txtData/src1-test.txt'
path_src2_test = './results/txtData/src2-test.txt'
path_tgt_test = './results/txtData/tgt-test.txt'


path_src1_new_test = './results/txtData/src1-newtest.txt'
path_src2_new_test = './results/txtData/src2-newtest.txt'
path_tgt_new_test = './results/txtData/tgt-newtest.txt'


#--------------------------------------------------Dynamically generate data.yml---------------------
generated_data_yml_path = "data.yml"

data_yml = {
    "model_dir": "results",
    "data": {
        "train_features_file": [path_src1_train, path_src2_train],
        "train_labels_file": path_tgt_train,
        "eval_features_file": [path_src1_val, path_src2_val],
        "eval_labels_file": path_tgt_val,
        "source_1_vocabulary": "src1-vocab.txt",
        "source_2_vocabulary": "src2-vocab.txt",
        "target_vocabulary": "tgt-vocab.txt"
    },
    "params": {
        "optimizer": "Adam",
        "learning_rate": 3e-4,
        "beam_width": beam_size,
        "num_hypotheses": beam_size
    },
    "eval": {
        "steps": steps_per_epoch,
        "early_stopping": {
            "min_improvement": 0.01,
            "steps": 10
        }
    },
    "train": {
        "batch_size": batch_size,
        "effective_batch_size": 32,
        "max_step": max_step,
        "sample_buffer_size": 0,
        "save_summary_steps": 1,
        "save_checkpoints_steps": max_step
    }
}

with open(generated_data_yml_path, "w") as f:
    yaml.dump(data_yml, f)


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

# -------------------------- BUILD VOCAB --------------------------------

f = open('./build_vocab.sh', 'w')
f.close()
subprocess.run('chmod a+x build_vocab.sh', shell=True)

f = open('./build_vocab.sh', 'w')
f.write('#!/usr/bin/env bash\n')
f.write('onmt-build-vocab --size 75000 --save_vocab src1-vocab.txt ' + path_src1_train + '\n')
f.write('onmt-build-vocab --size 75000 --save_vocab src2-vocab.txt ' + path_src2_train + '\n')
f.write('onmt-build-vocab --size 75000 --save_vocab tgt-vocab.txt ' + path_tgt_train)
f.close()

print('Building vocabularies...')
subprocess.run('./build_vocab.sh')

# --------------------------  TRAINING  --------------------------------

print('Starting Training...')
f_sh = open('./train_model.sh', 'w')
f_sh.close()

subprocess.run('chmod a+x train_model.sh', shell=True)

f_sh = open('./train_model.sh', 'w')
f_sh.write('#!/usr/bin/env bash\n')
f_sh.write(f'onmt-main --model custom_2encoders_transformer.py --gpu_allow_growth --config {generated_data_yml_path} --auto_config train --with_eval' f'> "{output_dir}/openNMT_early_stopping.log" 2>&1\n')
f_sh.close()

subprocess.run('./train_model.sh')



# --------------------------  TESTING ORIGINAL DATA ------------------------------------------------------------------------------------

def run_inference_with_openNMT(test_df, src_file, path_pred, scenario, file_name, beam_size=10):

    infer_script_path = './infer.sh'
    
    infer_command = (
        f'onmt-main --config {generated_data_yml_path} --auto_config infer '
        f'--features_file {src_file} --predictions_file {path_pred}'
    )
    
    # Create or overwrite the infer.sh script
    with open(infer_script_path, 'w') as f:
        f.write('#!/usr/bin/env bash\n')
        f.write(infer_command + '\n')
    
    # Make the script executable
    subprocess.run(f'chmod a+x {infer_script_path}', shell=True, check=True)
    
    # Run the inference
    subprocess.run(infer_script_path, shell=True, check=True)
    
    
    with open(path_pred, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(lines)} non-empty prediction lines from: {path_pred}")
    expected_lines = len(test_df) * beam_size
    logger.info(f"Expected number of prediction lines: {expected_lines} ({len(test_df)} test samples × {beam_size} beam size)")

    if test_df.empty:
        logger.warning("Test dataframe is empty. Skipping prediction alignment and evaluation.")
        all_beam_preds = []
    else:
        if expected_lines - len(lines) == 1:
            lines.append(lines[-1])

        if len(lines) % beam_size != 0:
            logger.warning(f"Prediction lines ({len(lines)}) is not divisible by beam size ({beam_size}).")

        all_beam_preds = [lines[i:i+beam_size] for i in range(0, len(lines), beam_size)]

        if len(all_beam_preds) != len(test_df):
            logger.warning(
                f"Mismatch between number of prediction groups ({len(all_beam_preds)}) and test samples ({len(test_df)}). "
            )

    test_df_copy = test_df.copy().reset_index(drop=True)
    test_df_copy["predictions"] = None
    test_df_copy["RawPredictions"] = None
    test_df_copy["gold"] = None
    
    test_df_copy["EXM"] = False

    test_df_copy["EXM1"] = False

    test_df_copy["EXM3"] = False

    test_df_copy["EXM5"] = False

    test_df_copy["Lev_Dist"] = 0.0
    test_df_copy["Lev_Ratio"] = 0.0



    all_targets = test_df_copy[column_names[2]].tolist()
    all_preds_txt = []

    EXM = EXM1 = EXM3 = EXM5 = 0
    EXMEXM = EXMEXM1 = EXMEXM3 = EXMEXM5 = 0

    for i, target_text in enumerate(all_targets):
        raw_preds = all_beam_preds[i]
        
        test_df_copy.at[i, "RawPredictions"] = raw_preds

        gold = to_single_line(cleaning_model_output(target_text))
        test_df_copy.at[i, "gold"] = gold
        beam_group = [to_single_line(cleaning_model_output(p)) for p in raw_preds]

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
    
    os.remove(infer_script_path)
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

logger.info("Methods for Calculating CodeBLEU scores...")

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



src_file = './results/txtData/src1-test.txt results/txtData/src2-test.txt'
path_pred = f"./results/predictions_orgData_{beam_size}.txt"
file_name = "test"
scenario1 = "ORIGINAL DATASET"
test_df_copy = run_inference_with_openNMT(test_df, src_file, path_pred, scenario1, file_name, beam_size=10)


file_name = "common_test"
scenario2 = "COMMON DATASET"
common_test_df_copy = stats_for_common_data(test_df_copy, common_ids_set, scenario2, file_name, beam_size = 10)



src_file = './results/txtData/src1-newtest.txt ./results/txtData/src2-newtest.txt'
path_pred = f"./results/Newpredictions_newData_{beam_size}.txt"
file_name = "new_test"
scenario3 = "NEW DATASET"
new_test_df_copy = run_inference_with_openNMT(new_test_dff, src_file, path_pred, scenario1, file_name, beam_size=10)


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
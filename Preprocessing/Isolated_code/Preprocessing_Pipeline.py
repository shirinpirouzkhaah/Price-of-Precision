from Analyzer1 import Analyzer1
from Analyzer2 import Analyzer2
import pandas as pd
import os
import math
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import gc
import traceback 




# ------------------------------------------
# Configuration
# ------------------------------------------
Root_dir = '../'
run_dir = './'


input_csv_path = os.path.join(Root_dir,'SplittedData/ProjectLevel/Gerrit_ProjectSplited_Test_NoLeakage.csv')
input_base_name = os.path.splitext(os.path.basename(input_csv_path))[0]



# ------------------------------------------
# Helper methods
# ------------------------------------------
def add_index_as_id(df):
    df = df.copy()
    df['id'] = df.index.astype(str)
    
    
    if 'message' in df.columns:
        df['message'] = df['message'].apply(
            lambda x: str(x).strip().lower() if pd.notnull(x) else ''
        )
        
        
        
    return df


def token_limit_filter(df, paper_id, col_names):

    token_limits = {
        1: 100,
        2: 512,
        3: 200
    }

    limit = token_limits[paper_id]

    def row_within_limit(row):
        for col in col_names:
            val = str(row[col]) if pd.notnull(row[col]) else ""
            if len(val.split()) > limit:
                return False
        return True

    mask = df.apply(row_within_limit, axis=1)
    filtered_df = df[mask].reset_index(drop=True)
    removed = len(df) - len(filtered_df)

    return filtered_df, removed


def copy_whole_current_directory(run_dir, run_chunk_dir):
    files_to_copy = {
        'Abstracter.py',
        'Analyzer1.py',
        'Analyzer2.py',
        'pipeline.py',
        'run_pipeline.sh',
        'utils'
    }

    for item in files_to_copy:
        src_path = os.path.join(run_dir, item)
        dest_path = os.path.join(run_chunk_dir, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dest_path)


def append_step_log(log_path, step_title, content_lines):
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n===== {step_title} =====\n")
        for line in content_lines:
            log_file.write(f"{line}\n")

# ------------------------------------------
# Load Raw CSV
# -----------------------------------------
dff = pd.read_csv(input_csv_path, low_memory=False)
chunk_size = 15000


print("📋 Column names in the loaded DataFrame:")
print(dff.columns.tolist())

print("\n📊 Column data types (via pandas):")
print(dff.dtypes)

dfff = add_index_as_id(dff)
Raw_data_count = len(dfff)

# ------------------------------------------
# Chunking
# -----------------------------------------

run_dir = './'
chunk_prefix = "Chunk"
total_rows = Raw_data_count  
num_chunks = math.ceil(total_rows / chunk_size)

def run_pipeline_for_chunk(chunk_df, i):

    
    
    root_dir = '../../'
    
    output_directory = os.path.join(root_dir, 'SplittedData/IsolatedSteps', input_base_name)
    os.makedirs(output_directory, exist_ok=True)
    
    output_dir = os.path.join(output_directory, f'{chunk_prefix}_{i:04d}')
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, 'Gerrit_ProjectLevel_Processing_Log.txt')
    path_idioms = './utils/idioms.csv'
    data_type = 'Gerrit'
    
    stats_dict = {}
    
    read_data_output_directory = os.path.join(root_dir, 'SplittedData/PreprocessedData', input_base_name)
    read_data_output_dir = os.path.join(read_data_output_directory, f'{chunk_prefix}_{i:04d}')
    # ------------------------------------------
    # Step 1.3 for Paper 1 & 2 (Method-Level)
    #P3 (PAPER 1 AND PAPER 2): Unchanged methods removal + SIZE FILTER

    ###RESULT: Paper1 & paper2 --> P1 + P3 (P3 ISOLATED)
    # ------------------------------------------
    step1_1_output_path = os.path.join(read_data_output_dir, 'Step1.1_Paper1-2_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    method_df = pd.read_csv(step1_1_output_path)
    
    
    
    analyzer = Analyzer1(method_df, data_type)
    
    # Step 1.3.1: Remove equal before/after methods
    method_df_initial_len = len(method_df)
    step1_3_method_df = analyzer.analyze_equal_methods()
    step1_3_equal_removed = method_df_initial_len - len(step1_3_method_df)
    
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(step1_3_method_df, 'before_marked', 'before_marked_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'before_marked_clean', 'before_marked_tags')
    MDF3= Analyzer2.flatten_to_single_line(MDF2,  'before_marked_tags',  'before_marked_flat')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF3, 'after', 'after_clean')
    MDF5 = Analyzer2.flatten_to_single_line(MDF4,  'after_clean',  'after_flat')
    MDF5 = MDF5.drop(
    columns=['before_marked_clean', 'before_marked_tags', 'after_clean'],
    errors='ignore')
    
    # Save output
    step1_3_output_path = os.path.join(output_dir, 'Step1.3_Paper1-2_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    MDF5.to_csv(step1_3_output_path, index=False)
    
    col_names = ['before_marked_flat', 'after_flat']
    step1_3_method_df_p1, step1_3_size_removed_p1 = token_limit_filter(MDF5.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step1.3_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_3_method_df_p1.to_csv(output_path_p1, index=False)
    
    step1_3_method_df_p2, step1_3_size_removed_p2 = token_limit_filter(MDF5.copy(), 2, col_names)
    output_path_p2 = os.path.join(output_dir, 'Step1.3_Paper2_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_3_method_df_p2.to_csv(output_path_p2, index=False)
    
    step1_3_Input_len = len(method_df)
    step1_3_method_df_len = len(MDF5)
    step1_3_method_df_p1_len = len(step1_3_method_df_p1)
    step1_3_method_df_p2_len = len(step1_3_method_df_p2)
    
    del method_df, step1_3_method_df, step1_3_method_df_p1, step1_3_method_df_p2
    gc.collect()
    
    
    del MDF1, MDF2, MDF3, MDF4, MDF5
    gc.collect()
    
    
    step_title = "[P1 + P3 (P3 ISOLATED) - Paper 1 & 2] P3: Unchanged methods removal"
    step1_3_log_lines = [
        f"- Input records: {step1_3_Input_len}",
        f"- Removed equal before==after methods: {step1_3_equal_removed}",
        f"- Output file: {step1_3_output_path}",
        f"- Records remaining after step: {step1_3_method_df_len}",
        f"  > Paper 1 - Token limit removed: {step1_3_size_removed_p1}",
        f"  > Paper 1 - Final: {step1_3_method_df_p1_len}",
        f"  > Paper 2 - Token limit removed: {step1_3_size_removed_p2}",
        f"  > Paper 2 - Final: {step1_3_method_df_p2_len}"
    ]
    
    append_step_log(log_path, step_title, step1_3_log_lines)
    
    stats_dict.update({
    "P1+P3_Input_records_paper1&2": step1_3_Input_len,
    "P1+P3_Removed_equal_before_after_paper1&2": step1_3_equal_removed,
    "P1+P3_Records_remaining_paper1&2": step1_3_method_df_len,
    "P1+P3_Paper1_TokenLimit_Removed_paper1&2": step1_3_size_removed_p1,
    "P1+P3_Paper1_Final_paper1&2": step1_3_method_df_p1_len,
    "P1+P3_Paper2_TokenLimit_Removed_paper1&2": step1_3_size_removed_p2,
    "P1+P3_Paper2_Final_paper1&2": step1_3_method_df_p2_len,
})

    
    # ------------------------------------------
    # Step 2.1 for Paper 1 (Method-Level)
    #P4 (PAPER 1): Method and comment abstraction + REMOVE NAN after abstraction + SIZE FILTER

    ###RESULT: Paper1 --> P1 + P4 (P4 ISOLATED)
    # ------------------------------------------
    step1_1_output_path = os.path.join(read_data_output_dir, 'Step1.1_Paper1-2_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    step2_1_paper1_input = pd.read_csv(step1_1_output_path)
    
    analyzer_paper1 = Analyzer2(step2_1_paper1_input, path_idioms)
    
    # Capture the returned DataFrame and assign it to analyzer_paper1.data
    analyzer_paper1.data = analyzer_paper1.analyze()
    
    step2_1_paper1_initial_len = len(analyzer_paper1.data)
    step2_1_paper1_df = analyzer_paper1.drop_rows_with_nan() 
    step2_1_paper1_removed = step2_1_paper1_initial_len - len(step2_1_paper1_df)
    
    step2_1_paper1_output_path = os.path.join(output_dir, 'Step2.1_Paper1_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    step2_1_paper1_df.to_csv(step2_1_paper1_output_path, index=False)
    
    
    col_names = ['before_abs_marked', 'after_abs']
    step2_1_method_df_p1, step2_1_size_removed_p1 = token_limit_filter(step2_1_paper1_df.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step2.1_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_1_method_df_p1.to_csv(output_path_p1, index=False)
    
    
    step2_1_paper1_df_len = len(step2_1_paper1_df)
    step2_1_method_df_p1_len = len(step2_1_method_df_p1)



    del step2_1_paper1_input, analyzer_paper1.data , step2_1_paper1_df, step2_1_method_df_p1
    gc.collect()
    
    
    step_title = "[P1 + P4 (P4 ISOLATED) - Paper 1] P4: Method abstraction"
    step2_1_log_lines = [
        f"- Input records: {step2_1_paper1_initial_len}",
        f"- Removed NaN rows: {step2_1_paper1_removed}",
        f"- Output file: {step2_1_paper1_output_path}",
        f"- Records remaining after step: {step2_1_paper1_df_len}",
        f"  > Paper 1 - Token limit removed: {step2_1_size_removed_p1}",
        f"  > Paper 1 - Final: {step2_1_method_df_p1_len}"
    ]
    
    append_step_log(log_path, step_title, step2_1_log_lines)
    
    
    stats_dict.update({
    "P1+P4_Input_records_paper1": step2_1_paper1_initial_len,
    "P1+P4_NaN_rows_removed_paper1": step2_1_paper1_removed,
    "P1+P4_Records_remaining_paper1": step2_1_paper1_df_len,
    "P1+P4_Paper1_TokenLimit_Removed_paper1": step2_1_size_removed_p1,
    "P1+P4_Paper1_Final_paper1": step2_1_method_df_p1_len,
})

    
    # ------------------------------------------
    # Step 2.2 for Paper 1
    #P5 (PAPER 1): Unchanged method removal + SIZE FILTER

    ###RESULT: Paper1 --> P1 + P4 + P5 (P4+P5 ISOLATED)
    # ------------------------------------------
    
    
    analyzer_paper1.data = pd.read_csv(step2_1_paper1_output_path)
    analyzer_paper1.info_for_cleaning_paper1()
    
    step2_2_paper1_initial_len = len(analyzer_paper1.data)
    step2_2_paper1_df = analyzer_paper1.data.copy()
    step2_2_paper1_dff = step2_2_paper1_df[step2_2_paper1_df['abs_before_equal_abs_after'] == False].reset_index(drop=True)
    step2_2_paper1_removed = step2_2_paper1_initial_len - len(step2_2_paper1_dff) #after abstraction before == after removed
    
    step2_2_paper1_output_path = os.path.join(output_dir, 'Step2.2_Paper1_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    step2_2_paper1_dff.to_csv(step2_2_paper1_output_path, index=False)
    
    
    col_names = ['before_abs_marked', 'after_abs']
    step2_2_method_df_p1, step2_2_size_removed_p1 = token_limit_filter(step2_2_paper1_dff.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step2.2_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_2_method_df_p1.to_csv(output_path_p1, index=False)
    
    
    step2_2_method_df_p1_len = len(step2_2_paper1_dff)
    step2_2_method_df_p1SizeFilter_len = len(step2_2_method_df_p1)
    
    del analyzer_paper1.data, step2_2_method_df_p1
    gc.collect()
    
    step_title = "[P4 + P5 (ISOLATED) - Paper 1] P5: Unchanged abstracted methods removal"
    step2_2_log_lines = [
        f"- Input records: {step2_2_paper1_initial_len}",
        f"- Removed equal abs(before) == abs(after): {step2_2_paper1_removed}",
        f"- Output file: {step2_2_paper1_output_path}",
        f"- Records remaining after step: {step2_2_method_df_p1_len}",
        f"  > Paper 1 - Token limit removed: {step2_2_size_removed_p1}",
        f"  > Paper 1 - Final: {step2_2_method_df_p1SizeFilter_len}"
    ]
    append_step_log(log_path, step_title, step2_2_log_lines)
    
    stats_dict.update({
    "P1+P4+P5_Input_records_paper1": step2_2_paper1_initial_len,
    "P1+P4+P5_Removed_equal_abs_before_after_paper1": step2_2_paper1_removed,
    "P1+P4+P5_Records_remaining_paper1": step2_2_method_df_p1_len,
    "P1+P4+P5_Paper1_TokenLimit_Removed_paper1": step2_2_size_removed_p1,
    "P1+P4+P5_Paper1_Final_paper1": step2_2_method_df_p1SizeFilter_len,
})


    

    # ------------------------------------------
    # Step 2.3 for Paper 1
    #P6 (PAPER 1): New token removal + SIZE FILTER  

    ###RESULT: Paper1 --> P1 + P4 + P6 (P4+P6 ISOLATED)
    # ------------------------------------------

    step2_3_paper1_initial_len = len(step2_2_paper1_df)
    step2_3_paper1_dff = step2_2_paper1_df[step2_2_paper1_df['new_token_in_abs_after'] == False].reset_index(drop=True)
    step2_3_paper1_removed = step2_3_paper1_initial_len - len(step2_3_paper1_dff)
    
    step2_3_paper1_output_path = os.path.join(output_dir, 'Step2.3_Paper1_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    step2_3_paper1_dff.to_csv(step2_3_paper1_output_path, index=False)
    
    
    col_names = ['before_abs_marked', 'after_abs']
    step2_3_method_df_p1, step2_3_size_removed_p1 = token_limit_filter(step2_3_paper1_dff.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step2.3_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_3_method_df_p1.to_csv(output_path_p1, index=False)
    
    step2_3_paper1_initial_len = len(step2_2_paper1_df)
    step2_3_paper1_df_len = len(step2_3_paper1_dff)
    step2_3_method_df_p1_len = len(step2_3_method_df_p1)
    
    del step2_3_paper1_dff, step2_2_paper1_df, step2_3_method_df_p1
    gc.collect()
    
    step_title = "[P4 + P6 (ISOLATED) - Paper 1] P6: New token removal"
    step2_3_log_lines = [
        f"- Input records: {step2_3_paper1_initial_len}",
        f"- Removed records with new tokens in abs(after): {step2_3_paper1_removed}",
        f"- Output file: {step2_3_paper1_output_path}",
        f"- Records remaining after step: {step2_3_paper1_df_len}",
        f"  > Paper 1 - Token limit removed: {step2_3_size_removed_p1}",
        f"  > Paper 1 - Final: {step2_3_method_df_p1_len}"
    ]
    append_step_log(log_path, step_title, step2_3_log_lines)

    
    stats_dict.update({
    "P1+P4+P6_Input_records_paper1": step2_3_paper1_initial_len,
    "P1+P4+P6_Removed_new_tokens_in_abs_after_paper1": step2_3_paper1_removed,
    "P1+P4+P6_Records_remaining_paper1": step2_3_paper1_df_len,
    "P1+P4+P6_Paper1_TokenLimit_Removed_paper1": step2_3_size_removed_p1,
    "P1+P4+P6_Paper1_Final_paper1": step2_3_method_df_p1_len,
})

    # ------------------------------------------
    # Step 2.3 for Paper 1
    #P6 (PAPER 1): New token removal + SIZE FILTER

    ###RESULT: Paper1 --> P1 + P6 (P6 ISOLATED)
    # ------------------------------------------
    step1_1_output_path = os.path.join(read_data_output_dir, 'Step1.1_Paper1-2_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    analyzer_paper1.data = pd.read_csv(step1_1_output_path)
    analyzer_paper1.info_for_cleaning_paper1_isolated()
    step2_3_paper1_df_isolated = analyzer_paper1.data.copy()
    
    step2_3_paper1_initial_len_isolated = len(step2_3_paper1_df_isolated)
    
    
    step2_3_paper1_dff_isolated = step2_3_paper1_df_isolated[step2_3_paper1_df_isolated['new_token_in_after'] == False].reset_index(drop=True)
    step2_3_paper1_removed_isolated = step2_3_paper1_initial_len_isolated - len(step2_3_paper1_dff_isolated)
    
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(step2_3_paper1_dff_isolated, 'before_marked', 'before_marked_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'before_marked_clean', 'before_marked_tags')
    MDF3= Analyzer2.flatten_to_single_line(MDF2,  'before_marked_tags',  'before_marked_flat')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF3, 'after', 'after_clean')
    MDF5 = Analyzer2.flatten_to_single_line(MDF4,  'after_clean',  'after_flat')
    MDF5 = MDF5.drop(
    columns=['before_marked_clean', 'before_marked_tags', 'after_clean'],
    errors='ignore')
    
    step2_3_paper1_output_path_isolated = os.path.join(output_dir, 'Step2.3_afterstep1.1_Paper1_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    MDF5.to_csv(step2_3_paper1_output_path_isolated, index=False)
    
    
    col_names = ['before_marked_flat', 'after_flat']
    step2_3_method_df_p1_isolated, step2_3_size_removed_p1_isolated = token_limit_filter(MDF5.copy(), 1, col_names)
    output_path_p1_isolated = os.path.join(output_dir, 'Step2.3_Paper1_afterstep1.1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_3_method_df_p1_isolated.to_csv(output_path_p1_isolated, index=False)
    
    step2_3_paper1_df_len_isolated = len(MDF5)
    step2_3_method_df_p1_len_isolated = len(step2_3_method_df_p1_isolated)
    
    del analyzer_paper1.data, step2_3_paper1_dff_isolated, step2_3_method_df_p1_isolated
    gc.collect()
    
    del MDF1, MDF2, MDF3, MDF4, MDF5
    gc.collect()
    
    
    step_title = "[P1 + P6 (P6 ISOLATED) - Paper 1] P6: New token removal (isolated)"
    step2_3_log_lines_isolated = [
        f"- Input records: {step2_3_paper1_initial_len_isolated}",
        f"- Removed records with new tokens in abs(after): {step2_3_paper1_removed_isolated}",
        f"- Output file: {step2_3_paper1_output_path_isolated}",
        f"- Records remaining after step: {step2_3_paper1_df_len_isolated}",
        f"  > Paper 1 - Token limit removed: {step2_3_size_removed_p1_isolated}",
        f"  > Paper 1 - Final: {step2_3_method_df_p1_len_isolated}"
    ]
    append_step_log(log_path, step_title, step2_3_log_lines_isolated)

    
    stats_dict.update({
    "P1+P6_Input_records_paper1": step2_3_paper1_initial_len_isolated,
    "P1+P6_Removed_new_tokens_in_abs_after_paper1": step2_3_paper1_removed_isolated,
    "P1+P6_Records_remaining_paper1": step2_3_paper1_df_len_isolated,
    "P1+P6_Paper1_TokenLimit_Removed_paper1": step2_3_size_removed_p1_isolated,
    "P1+P6_Paper1_Final_paper1": step2_3_method_df_p1_len_isolated,
})

    
    
    # ------------------------------------------
    # Step 2.4 for Paper 1 & paper 2
    #P7 (PAPER 1): comments cleaning + SIZE FILTER

    ###RESULT: Paper 1 --> P1 + P7 (P7 ISOLATED)
    # ------------------------------------------
    step2_4_paper1_initial_len = len(step2_3_paper1_df_isolated)


    step2_4_paper1_df = step2_3_paper1_df_isolated[
        (step2_3_paper1_df_isolated['clean_comment_empty'] == False) &
        (step2_3_paper1_df_isolated['comment_is_relevant'] == True)
    ].reset_index(drop=True)
    step2_4_paper1_removed = step2_4_paper1_initial_len - len(step2_4_paper1_df) #irrelevant comments removed
    
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(step2_4_paper1_df, 'before_marked', 'before_marked_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'before_marked_clean', 'before_marked_tags')
    MDF3= Analyzer2.flatten_to_single_line(MDF2,  'before_marked_tags',  'before_marked_flat')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF3, 'after', 'after_clean')
    MDF5 = Analyzer2.flatten_to_single_line(MDF4,  'after_clean',  'after_flat')
    MDF5 = MDF5.drop(
    columns=['before_marked_clean', 'before_marked_tags', 'after_clean'],
    errors='ignore')
    
    
    step2_4_paper1_output_path = os.path.join(output_dir, 'Step2.4_Paper1_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    MDF5.to_csv(step2_4_paper1_output_path, index=False)
    
    
    col_names = ['before_marked_flat', 'after_flat']
    step2_4_method_df_p1, step2_4_size_removed_p1 = token_limit_filter(MDF5.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step2.4_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_4_method_df_p1.to_csv(output_path_p1, index=False)
    
    
    step2_1_method_df_p2, step2_1_size_removed_p2 = token_limit_filter(MDF5.copy(), 2, col_names)
    output_path_p2 = os.path.join(output_dir, 'Step2.1_Paper2_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_1_method_df_p2.to_csv(output_path_p2, index=False)
    
    step2_4_paper1_initial_len = step2_4_paper1_initial_len
    step2_4_paper1_df_len = len(MDF5)
    step2_4_method_df_p1_len = len(step2_4_method_df_p1)
    step2_1_method_df_p2_len = len(step2_1_method_df_p2)
    
    del  step2_4_paper1_df, step2_4_method_df_p1, step2_1_method_df_p2
    gc.collect()
    
    
    del MDF1, MDF2, MDF3, MDF4, MDF5
    gc.collect()
    
    
    step_title = "[P1 + P7 (P7 ISOLATED) - Paper 1 & 2] P7: Comment cleaning"
    step2_4_log_lines = [
        f"- Input records: {step2_4_paper1_initial_len}",
        f"- Removed irrelevant or empty abstracted comments: {step2_4_paper1_removed}",
        f"- Output file: {step2_4_paper1_output_path}",
        f"- Records remaining after step: {step2_4_paper1_df_len}",
        f"  > Paper 1 - Token limit removed: {step2_4_size_removed_p1}",
        f"  > Paper 1 - Final: {step2_4_method_df_p1_len}",
        f"  > Paper 2 - Token limit removed: {step2_1_size_removed_p2}",
        f"  > Paper 2 - Final: {step2_1_method_df_p2_len}"
    ]
    append_step_log(log_path, step_title, step2_4_log_lines)
    
    
    stats_dict.update({
    "P1+P7_Input_records_paper1&2": step2_4_paper1_initial_len,
    "P1+P7_Removed_irrelevant_or_empty_comments_paper1&2": step2_4_paper1_removed,
    "P1+P7_Records_remaining_paper1&2": step2_4_paper1_df_len,
    "P1+P7_Paper1_TokenLimit_Removed_paper1": step2_4_size_removed_p1,
    "P1+P7_Paper1_Final_paper1": step2_4_method_df_p1_len,
    "P1+P7_Paper2_TokenLimit_Removed_paper2": step2_1_size_removed_p2,
    "P1+P7_Paper2_Final_paper2": step2_1_method_df_p2_len,
})


    

    # ------------------------------------------# ------------------------------------------# ------------------------------------------
    
    # ------------------------------------------
    # Step 1.3 for Paper 3 (Diff-Level)
    #P3 (PAPER 3): inline comments removal + SIZE FILTER

    ###RESULT: Paper 3 --> P1 + P3 (P3 ISOLATED)
    # ------------------------------------------
    
    step1_1_diff_output_path = os.path.join(read_data_output_dir, 'Step1.1_Paper3_Gerrit_ProjectLevel_DiffLevel_Preprocessed.csv')
    step1_3_input = pd.read_csv(step1_1_diff_output_path)
    step1_3_initial_len = len(step1_3_input) 
    
    
    step1_3_filtered_df = step1_3_input.dropna(subset=['old_wo_comment', 'new_wo_comment'])
    step1_3_1_filtered_dff = step1_3_filtered_df[
        (step1_3_filtered_df['old_wo_comment'].astype(str).str.strip() != '') &
        (step1_3_filtered_df['new_wo_comment'].astype(str).str.strip() != '')
    ]
    
    
    step1_3_1_removed = step1_3_initial_len - len(step1_3_1_filtered_dff)
    
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(step1_3_1_filtered_dff, 'old_wo_comment', 'old_wo_comment_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'old_wo_comment_clean', 'old_wo_comment_tags')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF2, 'new_wo_comment', 'new_wo_comment_clean')
    MDF4 = MDF4.drop(
    columns=['old_wo_comment_clean'],
    errors='ignore')

    
    # Save output
    step1_3_1_diff_output_path = os.path.join(output_dir, 'Step1.3_Paper3_Gerrit_ProjectLevel_DiffLevel_Preprocessed.csv')
    MDF4.to_csv(step1_3_1_diff_output_path, index=False)
    
    
    col_names = ['old_wo_comment_tags', 'new_wo_comment_clean']
    step1_3_method_df_p3, step1_3_size_removed_p3 = token_limit_filter(MDF4.copy(), 3, col_names)
    output_path_p3 = os.path.join(output_dir, 'Step1.3_Paper3_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_3_method_df_p3.to_csv(output_path_p3, index=False)
    
    step1_3_input_len = len(step1_3_input)
    step1_3_1_filtered_dff_len = len(MDF4)
    step1_3_method_df_p3_len = len(step1_3_method_df_p3)
    
    del step1_3_input, step1_3_1_filtered_dff, step1_3_method_df_p3
    gc.collect()
    
    del MDF1, MDF2, MDF4
    gc.collect()
    
    
    step_title = "[P1 + P3 (P3 ISOLATED) - Paper 3] P3: Inline comments removal (diff-level)"
    step1_3_log_lines = [
        f"- Input records: {step1_3_input_len}",
        f"- Removed rows with NA/empty old_wo_comment or new_wo_comment: {step1_3_1_removed}",
        f"- Output file: {step1_3_1_diff_output_path}",
        f"- Records remaining after step: {step1_3_1_filtered_dff_len}",
        f"  > Paper 3 - Token limit removed: {step1_3_size_removed_p3}",
        f"  > Paper 3 - Final: {step1_3_method_df_p3_len}"
    ]
    append_step_log(log_path, step_title, step1_3_log_lines)
    
    
    stats_dict.update({
    "P1+P3_Input_records_paper3": step1_3_input_len,
    "P1+P3_Removed_inline_comment_NA_or_empty_paper3": step1_3_1_removed,
    "P1+P3_Records_remaining_paper3": step1_3_1_filtered_dff_len,
    "P1+P3_Paper3_TokenLimit_Removed_paper3": step1_3_size_removed_p3,
    "P1+P3_Paper3_Final_paper3": step1_3_method_df_p3_len,
})


    
    # ------------------------------------------
    # Step 1.4 for Paper 3 (Diff-Level)
    #P4 (PAPER 3):comments cleaning + SIZE FILTER

    ###RESULT: Paper 3 --> P1 + P4 (P4 ISOLATED)
    # ------------------------------------------
    
    step1_1_diff_output_path = os.path.join(read_data_output_dir, 'Step1.1_Paper3_Gerrit_ProjectLevel_DiffLevel_Preprocessed.csv')
    Input = pd.read_csv(step1_1_diff_output_path)
    

    analyzer = Analyzer2(Input, path_idioms)
    step1_4_df = analyzer.analyze_diff_level()
    
    # Drop rows where 'cleaned_comment_no_praise' is NA or empty
    step1_4_initial_len = len(step1_4_df)
    step1_4_1_df = step1_4_df.dropna(subset=['cleaned_comment_no_praise'])
    step1_4_2_df = step1_4_1_df[step1_4_1_df['cleaned_comment_no_praise'].astype(str).str.strip() != '']
    step1_4_removed = step1_4_initial_len - len(step1_4_2_df)
    
    # Drop rows where 'Comment_size_filter_3_150' is True
    step1_4_3_initial_len = len(step1_4_2_df)
    step1_4_3_df = step1_4_2_df[step1_4_2_df['Comment_size_filter_3_150'] == False].reset_index(drop=True)
    step1_4_3_removed = step1_4_3_initial_len - len(step1_4_3_df)
    
    
    # Drop Irrelevant comments
    step1_4_4_initial_len = len(step1_4_3_df)
    step1_4_4_df = step1_4_3_df[step1_4_3_df['diff_Comment_is_relevant'] == True].reset_index(drop=True)
    step1_4_4_removed = step1_4_4_initial_len - len(step1_4_4_df)
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(step1_4_4_df, 'old_wo_comment', 'old_wo_comment_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'old_wo_comment_clean', 'old_wo_comment_tags')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF2, 'new_wo_comment', 'new_wo_comment_clean')
    MDF4 = MDF4.drop(
    columns=['old_wo_comment_clean'],
    errors='ignore')
    

    # Save Step 1.3 output
    step1_4_diff_output_path = os.path.join(output_dir, 'Step1.4_Paper3_Gerrit_ProjectLevel_DiffLevel_Preprocessed.csv')
    MDF4.to_csv(step1_4_diff_output_path, index=False)
    
    col_names = ['old_wo_comment_tags','new_wo_comment_clean']
    step1_4_method_df_p3, step1_4_size_removed_p3 = token_limit_filter(step1_4_4_df.copy(), 3, col_names)
    output_path_p3 = os.path.join(output_dir, 'Step1.4_Paper3_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_4_method_df_p3.to_csv(output_path_p3, index=False)
    
    
    Input_len = len(Input)
    step1_4_df_len = len(step1_4_df)
    step1_4_1_df_len = len(step1_4_1_df)
    step1_4_2_df_len = len(step1_4_2_df)
    step1_4_3_df_len = len(step1_4_3_df)
    step1_4_4_df_len = len(step1_4_4_df)
    step1_4_method_df_p3_len = len(step1_4_method_df_p3)


    del Input, step1_4_df, step1_4_1_df, step1_4_2_df, step1_4_3_df, step1_4_4_df, step1_4_method_df_p3
    gc.collect()
    
    del MDF1, MDF2, MDF4
    gc.collect()
    
    step_title = "[P1 + P4 (P4 ISOLATED) - Paper 3] P4: Comment cleaning (diff-level)"
    step1_4_log_lines = [
        f"- Input records: {Input_len}",
        f"- Removed empty/NA 'cleaned_comment_no_praise': {step1_4_removed}",
        f"- Removed due to size filter (Comment_size_filter_3_150): {step1_4_3_removed}",
        f"- Removed irrelevant diff comments: {step1_4_4_removed}",
        f"- Output file: {step1_4_diff_output_path}",
        f"- Records remaining after step: {step1_4_4_df_len}",
        f"  > Paper 3 - Token limit removed: {step1_4_size_removed_p3}",
        f"  > Paper 3 - Final: {step1_4_method_df_p3_len}"
    ]
    append_step_log(log_path, step_title, step1_4_log_lines)
    
    
    stats_dict.update({
    "P3+P1+P4_Input_records_paper3": Input_len,
    "P3+P1+P4_Removed_empty_cleaned_comment_paper3": step1_4_removed,
    "P3+P1+P4_Removed_size_filter_paper3": step1_4_3_removed,
    "P3+P1+P4_Removed_irrelevant_comments_paper3": step1_4_4_removed,
    "P3+P1+P4_Records_remaining_paper3": step1_4_4_df_len,
    "P3+P1+P4_Paper3_TokenLimit_Removed_paper3": step1_4_size_removed_p3,
    "P3+P1+P4_Paper3_Final_paper3": step1_4_method_df_p3_len,
})


    

    
    # ------------------------------------------
    # Print Summary to Console
    # ------------------------------------------
    print(open(log_path).read())
    
    
    return stats_dict

def process_chunk(i, df, run_dir, chunk_prefix, chunk_size):
    start = i * chunk_size
    end = min(start + chunk_size, len(df))
    chunk_df = df.iloc[start:end].copy()

    # Create isolated working/output directories
    run_chunk_dir = os.path.join(run_dir, f'{chunk_prefix}_{i:04d}')
    os.makedirs(run_chunk_dir, exist_ok=True)
    copy_whole_current_directory(run_dir, run_chunk_dir)
    original_dir = os.getcwd()
    try:
        os.chdir(run_chunk_dir)
        try:
            return run_pipeline_for_chunk(chunk_df, i)
        except Exception as e:
            with open("../error_log.txt", "a") as err:
                err.write(f"Error in chunk {i:04d}: {str(e)}\n")
                traceback.print_exc(file=err)
            return {"chunk_index": i, "error": str(e)}
    finally:
        os.chdir(original_dir)

        

def aggregate_statistics(all_stats):
    summary = defaultdict(int)
    for stat in all_stats:
        for k, v in stat.items():
            if isinstance(v, int): 
                summary[k] += v
    return summary


def process_chunk_wrapper(i, df, run_dir, chunk_prefix, chunk_size):
    result = process_chunk(i, df, run_dir, chunk_prefix, chunk_size)
    gc.collect() 
    return result

all_stats = []

with ProcessPoolExecutor(max_workers=12) as executor:
    futures = {
        executor.submit(process_chunk_wrapper, i, dfff, run_dir, chunk_prefix, chunk_size): i
        for i in range(num_chunks)
    }

    for future in as_completed(futures):
        try:
            result = future.result()
            all_stats.append(result)
        except Exception as e:
            print(f"❌ Chunk {futures[future]} failed: {e}")
            all_stats.append({"chunk_index": futures[future], "error": str(e)})
        gc.collect()  




summary_stats = aggregate_statistics(all_stats)


root_dir = '../'
output_directory = os.path.join(Root_dir, 'SplittedData/IsolatedSteps', input_base_name)
Final_log_path = os.path.join(output_directory, 'Gerrit_ProjectLevel_Processing_Log.txt')

def write_final_log(Final_log_path, summary_stats):
    with open(Final_log_path, 'w') as log_file:
        log_file.write("===== Gerrit ProjectLevel Test_NoLeakage Preprocessing Log =====\n\n")

        # ------------------------
        log_file.write("[P1 + P3 (P3 ISOLATED) - Paper 1 & 2] P3: Unchanged methods removal\n")
        log_file.write(f"- Input records: {summary_stats['P1+P3_Input_records_paper1&2']}\n")
        log_file.write(f"- Removed equal before==after methods: {summary_stats['P1+P3_Removed_equal_before_after_paper1&2']}\n")
        log_file.write(f"- Records remaining after step: {summary_stats['P1+P3_Records_remaining_paper1&2']}\n")
        log_file.write(f"  > Paper 1 - Token limit removed: {summary_stats['P1+P3_Paper1_TokenLimit_Removed_paper1&2']}\n")
        log_file.write(f"  > Paper 1 - Final: {summary_stats['P1+P3_Paper1_Final_paper1&2']}\n")
        log_file.write(f"  > Paper 2 - Token limit removed: {summary_stats['P1+P3_Paper2_TokenLimit_Removed_paper1&2']}\n")
        log_file.write(f"  > Paper 2 - Final: {summary_stats['P1+P3_Paper2_Final_paper1&2']}\n\n")

        # ------------------------
        log_file.write("[P1 + P4 (P4 ISOLATED) - Paper 1] P4: Method abstraction\n")
        log_file.write(f"- Input records: {summary_stats['P1+P4_Input_records_paper1']}\n")
        log_file.write(f"- Removed NaN rows: {summary_stats['P1+P4_NaN_rows_removed_paper1']}\n")
        log_file.write(f"- Records remaining after step: {summary_stats['P1+P4_Records_remaining_paper1']}\n")
        log_file.write(f"  > Paper 1 - Token limit removed: {summary_stats['P1+P4_Paper1_TokenLimit_Removed_paper1']}\n")
        log_file.write(f"  > Paper 1 - Final: {summary_stats['P1+P4_Paper1_Final_paper1']}\n\n")

        # ------------------------
        log_file.write("[P4 + P5 (ISOLATED) - Paper 1] P5: Unchanged abstracted methods removal\n")
        log_file.write(f"- Input records: {summary_stats['P1+P4+P5_Input_records_paper1']}\n")
        log_file.write(f"- Removed equal abs(before) == abs(after): {summary_stats['P1+P4+P5_Removed_equal_abs_before_after_paper1']}\n")
        log_file.write(f"- Records remaining after step: {summary_stats['P1+P4+P5_Records_remaining_paper1']}\n")
        log_file.write(f"  > Paper 1 - Token limit removed: {summary_stats['P1+P4+P5_Paper1_TokenLimit_Removed_paper1']}\n")
        log_file.write(f"  > Paper 1 - Final: {summary_stats['P1+P4+P5_Paper1_Final_paper1']}\n\n")

        # ------------------------
        log_file.write("[P4 + P6 (ISOLATED) - Paper 1] P6: New token removal\n")
        log_file.write(f"- Input records: {summary_stats['P1+P4+P6_Input_records_paper1']}\n")
        log_file.write(f"- Removed records with new tokens in abs(after): {summary_stats['P1+P4+P6_Removed_new_tokens_in_abs_after_paper1']}\n")
        log_file.write(f"- Records remaining after step: {summary_stats['P1+P4+P6_Records_remaining_paper1']}\n")
        log_file.write(f"  > Paper 1 - Token limit removed: {summary_stats['P1+P4+P6_Paper1_TokenLimit_Removed_paper1']}\n")
        log_file.write(f"  > Paper 1 - Final: {summary_stats['P1+P4+P6_Paper1_Final_paper1']}\n\n")

        # ------------------------
        log_file.write("[P1 + P6 (P6 ISOLATED) - Paper 1] P6: New token removal (isolated)\n")
        log_file.write(f"- Input records: {summary_stats['P1+P6_Input_records_paper1']}\n")
        log_file.write(f"- Removed records with new tokens in abs(after): {summary_stats['P1+P6_Removed_new_tokens_in_abs_after_paper1']}\n")
        log_file.write(f"- Records remaining after step: {summary_stats['P1+P6_Records_remaining_paper1']}\n")
        log_file.write(f"  > Paper 1 - Token limit removed: {summary_stats['P1+P6_Paper1_TokenLimit_Removed_paper1']}\n")
        log_file.write(f"  > Paper 1 - Final: {summary_stats['P1+P6_Paper1_Final_paper1']}\n\n")

        # ------------------------
        log_file.write("[P1 + P7 (P7 ISOLATED) - Paper 1 & 2] P7: Comment cleaning\n")
        log_file.write(f"- Input records: {summary_stats['P1+P7_Input_records_paper1&2']}\n")
        log_file.write(f"- Removed irrelevant or empty abstracted comments: {summary_stats['P1+P7_Removed_irrelevant_or_empty_comments_paper1&2']}\n")
        log_file.write(f"- Records remaining after step: {summary_stats['P1+P7_Records_remaining_paper1&2']}\n")
        log_file.write(f"  > Paper 1 - Token limit removed: {summary_stats['P1+P7_Paper1_TokenLimit_Removed_paper1']}\n")
        log_file.write(f"  > Paper 1 - Final: {summary_stats['P1+P7_Paper1_Final_paper1']}\n")
        log_file.write(f"  > Paper 2 - Token limit removed: {summary_stats['P1+P7_Paper2_TokenLimit_Removed_paper2']}\n")
        log_file.write(f"  > Paper 2 - Final: {summary_stats['P1+P7_Paper2_Final_paper2']}\n\n")

        # ------------------------
        log_file.write("[P1 + P3 (P3 ISOLATED) - Paper 3] P3: Inline comments removal (diff-level)\n")
        log_file.write(f"- Input records: {summary_stats['P1+P3_Input_records_paper3']}\n")
        log_file.write(f"- Removed rows with NA/empty old_wo_comment or new_wo_comment: {summary_stats['P1+P3_Removed_inline_comment_NA_or_empty_paper3']}\n")
        log_file.write(f"- Records remaining after step: {summary_stats['P1+P3_Records_remaining_paper3']}\n")
        log_file.write(f"  > Paper 3 - Token limit removed: {summary_stats['P1+P3_Paper3_TokenLimit_Removed_paper3']}\n")
        log_file.write(f"  > Paper 3 - Final: {summary_stats['P1+P3_Paper3_Final_paper3']}\n\n")

        # ------------------------
        log_file.write("[P1 + P4 (P4 ISOLATED) - Paper 3] P4: Comment cleaning (diff-level)\n")
        log_file.write(f"- Input records: {summary_stats['P3+P1+P4_Input_records_paper3']}\n")
        log_file.write(f"- Removed empty/NA 'cleaned_comment_no_praise': {summary_stats['P3+P1+P4_Removed_empty_cleaned_comment_paper3']}\n")
        log_file.write(f"- Removed due to size filter (Comment_size_filter_3_150): {summary_stats['P3+P1+P4_Removed_size_filter_paper3']}\n")
        log_file.write(f"- Removed irrelevant diff comments: {summary_stats['P3+P1+P4_Removed_irrelevant_comments_paper3']}\n")
        log_file.write(f"- Records remaining after step: {summary_stats['P3+P1+P4_Records_remaining_paper3']}\n")
        log_file.write(f"  > Paper 3 - Token limit removed: {summary_stats['P3+P1+P4_Paper3_TokenLimit_Removed_paper3']}\n")
        log_file.write(f"  > Paper 3 - Final: {summary_stats['P3+P1+P4_Paper3_Final_paper3']}\n\n")

        
write_final_log(Final_log_path, summary_stats)
# ------------------------------------------
# Print Summary to Console
# ------------------------------------------
print(open(Final_log_path).read())

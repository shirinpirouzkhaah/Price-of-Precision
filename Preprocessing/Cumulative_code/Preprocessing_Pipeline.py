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

#Gerrit
#Gerrit

#Project
#Project

#Test_NoLeakage
#Test



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

# ------------------------------------------
# Load Raw CSV
# -----------------------------------------
dff = pd.read_csv(input_csv_path, low_memory=False)

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
chunk_size = 15000
chunk_prefix = "Chunk"
total_rows = Raw_data_count
num_chunks = math.ceil(total_rows / chunk_size)

def run_pipeline_for_chunk(chunk_df, i):
    
    
    root_dir = '../../'
    
    output_directory = os.path.join(root_dir, 'SplittedData/PreprocessedData', input_base_name)
    os.makedirs(output_directory, exist_ok=True)
    
    output_dir = os.path.join(output_directory, f'{chunk_prefix}_{i:04d}')
    os.makedirs(output_dir, exist_ok=True)
    
    
    log_path = os.path.join(output_dir, 'Gerrit_ProjectLevel_Processing_Log.txt')
    path_idioms = './utils/idioms.csv'
    data_type = 'Gerrit'
    
    
    df = chunk_df
    initial_data_count = len(df)
    # ------------------------------------------
    # Stage 1: Initial Cleanup on raw data
    # ------------------------------------------
    
    analyzer = Analyzer1(df, data_type)
    analyzer.remove_nan_data()
    nan_data_count = analyzer.nan_data
    
    analyzer.remove_duplicates()
    duplicates_count = analyzer.duplicates
    
    analyzer.remove_left_side()
    left_side_count = analyzer.left_side_cases
    
    # ------------------------------------------
    # Stage 2: Extraction
    # ------------------------------------------
    step0_method_df, step0_diff_df = analyzer.extraction() 
    
    
    # ------------------------------------------
    # Step 1.1 for Paper 1 & 2 (Method-Level)
    #P1 (PAPER 1 AND PAPER 2): EXTRACTION + remove nan + remove duplicates  + SIZE FILTER
    # ------------------------------------------
    method_df = step0_method_df.copy()
    analyzer = Analyzer1(method_df, data_type)
    analyzer.method_dfr = method_df
    
    analyzer.remove_nan_data_from_method_dfr()
    step1_1_method_nan_dropped = analyzer.nan_data
    method_df = analyzer.method_dfr
    
    # Re-assign the cleaned method_dfr to analyzer.data for duplicate removal
    analyzer.data = method_df
    analyzer.remove_duplicates()
    step1_1_method_duplicates = analyzer.duplicates
    method_df = analyzer.data
    
    
    """
    method A + C1 -->A' keep
    method B + C1 -->B' remove
    method C + C1 -->C' remove
    method D + C1 -->D' remove
    """
    # Step 1.1.3: Remove shared comments
    analyzer.method_dfr = method_df
    step1_1_method_df = analyzer.remove_shared_comments_methodLevel(analyzer.method_dfr)
    step1_1_method_shared_dropped = len(analyzer.method_dfr) - len(step1_1_method_df) 
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(step1_1_method_df, 'before_marked', 'before_marked_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'before_marked_clean', 'before_marked_tags')
    MDF3= Analyzer2.flatten_to_single_line(MDF2,  'before_marked_tags',  'before_marked_flat')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF3, 'after', 'after_clean')
    MDF5 = Analyzer2.flatten_to_single_line(MDF4,  'after_clean',  'after_flat')
    MDF5 = MDF5.drop(
    columns=['before_marked_clean', 'before_marked_tags', 'after_clean'],
    errors='ignore')


    # Save output
    step1_1_output_path = os.path.join(output_dir, 'Step1.1_Paper1-2_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    MDF5.to_csv(step1_1_output_path, index=False)
    
    
    col_names = ['before_marked_flat', 'after_flat']
    step1_1_method_df_p1, step1_1_size_removed_p1 = token_limit_filter(MDF5.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step1.1_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_1_method_df_p1.to_csv(output_path_p1, index=False)
    
    step1_1_method_df_p2, step1_1_size_removed_p2 = token_limit_filter(MDF5.copy(), 2, col_names)
    output_path_p2 = os.path.join(output_dir, 'Step1.1_Paper2_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_1_method_df_p2.to_csv(output_path_p2, index=False)
    
    step1_1_method_df_len = len(MDF5)
    step1_1_method_df_p1_len = len(step1_1_method_df_p1)
    step1_1_method_df_p2_len = len(step1_1_method_df_p2)
    
    
    del method_df, analyzer.method_dfr, step1_1_method_df, step1_1_method_df_p1, step1_1_method_df_p2
    gc.collect()
    
    
    
    del MDF1, MDF2, MDF3, MDF4, MDF5
    gc.collect()
    
    # ------------------------------------------
    # Step 1.2 for Paper 1 & 2 (Method-Level)
    #P2 (PAPER 1 AND PAPER 2): Comment pruning + SIZE FILTER
    # ------------------------------------------
    
    method_df = pd.read_csv(step1_1_output_path)
    analyzer = Analyzer1(method_df, data_type)
    
    # Step 1.2.1: Remove invalid rows (e.g., owner, comment-to-comment, unlinked methods, empty comments)
    method_df_initial_len = len(method_df)
    method_df = analyzer.analyze_method_comments_Data()
    step1_2_method_invalidComments_removed = method_df_initial_len - len(method_df)
    
    # Step 1.2.2: Remove multiple comments for same before+after pair (keep none)
    method_df_initial_len = len(method_df)
    method_df = analyzer.remove_multiple_comments_methodLevel(method_df)
    step1_2_method_multipleComment_removed = method_df_initial_len - len(method_df)
    
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(method_df, 'before_marked', 'before_marked_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'before_marked_clean', 'before_marked_tags')
    MDF3= Analyzer2.flatten_to_single_line(MDF2,  'before_marked_tags',  'before_marked_flat')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF3, 'after', 'after_clean')
    MDF5 = Analyzer2.flatten_to_single_line(MDF4,  'after_clean',  'after_flat')
    MDF5 = MDF5.drop(
    columns=['before_marked_clean', 'before_marked_tags', 'after_clean'],
    errors='ignore')
    
    
    # Save output
    step1_2_output_path = os.path.join(output_dir, 'Step1.2_Paper1-2_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    MDF5.to_csv(step1_2_output_path, index=False)
    
    
    col_names = ['before_marked_flat', 'after_flat']
    step1_2_method_df_p1, step1_2_size_removed_p1 = token_limit_filter(MDF5.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step1.2_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_2_method_df_p1.to_csv(output_path_p1, index=False)
    
    step1_2_method_df_p2, step1_2_size_removed_p2 = token_limit_filter(MDF5.copy(), 2, col_names)
    output_path_p2 = os.path.join(output_dir, 'Step1.2_Paper2_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_2_method_df_p2.to_csv(output_path_p2, index=False)
    
    step1_2_method_df_len = len(MDF5)
    step1_2_method_df_p1_len = len(step1_2_method_df_p1)
    step1_2_method_df_p2_len = len(step1_2_method_df_p2)
    
    
    
    del method_df, step1_2_method_df_p1, step1_2_method_df_p2
    gc.collect()
    
    del MDF1, MDF2, MDF3, MDF4, MDF5
    gc.collect()
    
    # ------------------------------------------
    # Step 1.3 for Paper 1 & 2 (Method-Level)
    #P3 (PAPER 1 AND PAPER 2): Unchanged methods removal + SIZE FILTER
    # ------------------------------------------
    method_df = pd.read_csv(step1_2_output_path)
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
    
    # ------------------------------------------
    # Step 2.1 for Paper 1 (Method-Level)
    #P4 (PAPER 1): Method and comment abstraction + REMOVE NAN after abstraction + SIZE FILTER
    # ------------------------------------------
    
    step2_1_paper1_input = pd.read_csv(step1_3_output_path)
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
    
    # ------------------------------------------
    # Step 2.2 for Paper 1
    #P5 (PAPER 1): Unchanged method removal + SIZE FILTER
    # ------------------------------------------
    analyzer_paper1.data = pd.read_csv(step2_1_paper1_output_path)
    analyzer_paper1.info_for_cleaning_paper1()
    
    step2_2_paper1_initial_len = len(analyzer_paper1.data)
    step2_2_paper1_df = analyzer_paper1.data.copy()
    step2_2_paper1_df = step2_2_paper1_df[step2_2_paper1_df['abs_before_equal_abs_after'] == False].reset_index(drop=True)
    step2_2_paper1_removed = step2_2_paper1_initial_len - len(step2_2_paper1_df) #after abstraction before == after removed
    
    step2_2_paper1_output_path = os.path.join(output_dir, 'Step2.2_Paper1_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    step2_2_paper1_df.to_csv(step2_2_paper1_output_path, index=False)
    
    
    col_names = ['before_abs_marked', 'after_abs']
    step2_2_method_df_p1, step2_2_size_removed_p1 = token_limit_filter(step2_2_paper1_df.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step2.2_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_2_method_df_p1.to_csv(output_path_p1, index=False)
    
    
    step2_2_method_df_p1_len = len(step2_2_paper1_df)
    step2_2_method_df_p1SizeFilter_len = len(step2_2_method_df_p1)
    
    del analyzer_paper1.data, step2_2_method_df_p1
    gc.collect()
    
    
    # ------------------------------------------
    # Step 2.3 for Paper 1
    #P6 (PAPER 1): New token removal + SIZE FILTER
    # ------------------------------------------
    step2_3_paper1_initial_len = len(step2_2_paper1_df)
    step2_3_paper1_df = step2_2_paper1_df[step2_2_paper1_df['new_token_in_abs_after'] == False].reset_index(drop=True)
    step2_3_paper1_removed = step2_3_paper1_initial_len - len(step2_3_paper1_df)#new token in abs after removed
    
    step2_3_paper1_output_path = os.path.join(output_dir, 'Step2.3_Paper1_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    step2_3_paper1_df.to_csv(step2_3_paper1_output_path, index=False)
    
    
    col_names = ['before_abs_marked', 'after_abs']
    step2_3_method_df_p1, step2_3_size_removed_p1 = token_limit_filter(step2_3_paper1_df.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step2.3_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_3_method_df_p1.to_csv(output_path_p1, index=False)
    
    step2_3_paper1_initial_len = len(step2_2_paper1_df)
    step2_3_paper1_df_len = len(step2_3_paper1_df)
    step2_3_method_df_p1_len = len(step2_3_method_df_p1)
    
    del step2_2_paper1_df, step2_3_method_df_p1
    gc.collect()
    
    # ------------------------------------------
    # Step 2.4 for Paper 1
    #P7 (PAPER 1): comments cleaning + SIZE FILTER
    
    # ------------------------------------------
    step2_4_paper1_initial_len = len(step2_3_paper1_df)
    step2_4_paper1_df = step2_3_paper1_df[
        (step2_3_paper1_df['abs_comment_is_empty'] == False) &
        (step2_3_paper1_df['abs_comment_is_relevant'] == True)
    ].reset_index(drop=True)
    step2_4_paper1_removed = step2_4_paper1_initial_len - len(step2_4_paper1_df) #irrelevant comments removed
    
    step2_4_paper1_output_path = os.path.join(output_dir, 'Step2.4_Paper1_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    step2_4_paper1_df.to_csv(step2_4_paper1_output_path, index=False)
    
    
    col_names = ['before_abs_marked', 'after_abs']
    step2_4_method_df_p1, step2_4_size_removed_p1 = token_limit_filter(step2_4_paper1_df.copy(), 1, col_names)
    output_path_p1 = os.path.join(output_dir, 'Step2.4_Paper1_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_4_method_df_p1.to_csv(output_path_p1, index=False)
    
    step2_4_paper1_initial_len = len(step2_3_paper1_df)
    step2_4_paper1_df_len = len(step2_4_paper1_df)
    step2_4_method_df_p1_len = len(step2_4_method_df_p1)
    
    del step2_3_paper1_df, step2_4_paper1_df, step2_4_method_df_p1
    gc.collect()
    
    # ------------------------------------------
    # Step 2.1 for Paper 2 ***equal to*** # Step 2.4 for Paper 1
    #P4 (PAPER 2): comments cleaning + SIZE FILTER
    
    # ------------------------------------------
    step2_1_paper2_input = pd.read_csv(step1_3_output_path)
    analyzer_paper2 = Analyzer2(step2_1_paper2_input, path_idioms)
    analyzer_paper2.data = analyzer_paper2.analyze_wo_abs()
    analyzer_paper2.info_for_cleaning_paper2()
    
    # Filter based on the computed flags
    step2_1_paper2_df = analyzer_paper2.data.copy()
    step2_1_paper2_initial_len = len(step2_1_paper2_df)
    step2_1_paper2_df = step2_1_paper2_df[
        (step2_1_paper2_df['clean_comment_wo_abs_empty'] == False) &
        (step2_1_paper2_df['wo_abs_comment_is_relevant'] == True)
    ].reset_index(drop=True)
    step2_1_paper2_removed = step2_1_paper2_initial_len - len(step2_1_paper2_df)
    
    
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(step2_1_paper2_df, 'before_marked', 'before_marked_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'before_marked_clean', 'before_marked_tags')
    MDF3= Analyzer2.flatten_to_single_line(MDF2,  'before_marked_tags',  'before_marked_flat')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF3, 'after', 'after_clean')
    MDF5 = Analyzer2.flatten_to_single_line(MDF4,  'after_clean',  'after_flat')
    MDF5 = MDF5.drop(
    columns=['before_marked_clean', 'before_marked_tags', 'after_clean'],
    errors='ignore')
    
    
    
    step2_1_paper2_output_path = os.path.join(output_dir, 'Step2.1_Paper2_Gerrit_ProjectLevel_MethodLevel_Preprocessed.csv')
    MDF5.to_csv(step2_1_paper2_output_path, index=False)
    
    
    col_names = ['before_marked_flat', 'after_flat']
    step2_1_method_df_p2, step2_1_size_removed_p2 = token_limit_filter(MDF5.copy(), 2, col_names)
    output_path_p2 = os.path.join(output_dir, 'Step2.1_Paper2_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step2_1_method_df_p2.to_csv(output_path_p2, index=False)
    
    step2_1_paper2_input_len= len(step2_1_paper2_input)
    step2_1_paper2_df_len = len(MDF5)
    step2_1_method_df_p2_len = len(step2_1_method_df_p2)
    
    
    del step2_1_paper2_input, step2_1_paper2_df, step2_1_method_df_p2
    gc.collect()
    
    
    del MDF1, MDF2, MDF3, MDF4, MDF5
    gc.collect()
    
    # ------------------------------------------# ------------------------------------------# ------------------------------------------
    
    # ------------------------------------------
    # Step 1.1 for Paper 3 (Diff-Level)
    #P1 (PAPER 3): EXTRACTION + remove nan + remove duplicates  + SIZE FILTER
    # ------------------------------------------
    diff_df = step0_diff_df.copy()
    InitialLength = len(diff_df)
    
    analyzer = Analyzer1(diff_df, data_type)
    analyzer.diff_dfr = analyzer.data  # Initialize working copy
    
    #Remove rows with missing values in diff-related fields
    analyzer.remove_nan_data_from_diff_dfr()
    step1_1_diff_nan_dropped = analyzer.nan_data
    diff_df = analyzer.diff_dfr
    
    #Remove duplicate rows
    analyzer.data = diff_df
    analyzer.remove_duplicates()
    step1_1_diff_duplicates = analyzer.duplicates
    diff_df = analyzer.data
    
    """
    diff A + C1 -->A' keep
    diff B + C1 -->B' remove
    diff C + C1 -->C' remove
    diff D + C1 -->D' remove
    """
    
    #Remove duplicate comments
    analyzer.diff_dfr = diff_df
    step1_1_diff_df = analyzer.remove_shared_comments_diffLevel(analyzer.diff_dfr)
    step1_1_diff_earliest_removed = InitialLength - len(step1_1_diff_df)
    
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(step1_1_diff_df, 'old', 'old_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'old_clean', 'old_tags')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF2, 'new', 'new_clean')
    
    MDF4 = MDF4.drop(
    columns=['old_clean'],
    errors='ignore')
    
    
    # Save Step 1.1 output
    step1_1_diff_output_path = os.path.join(output_dir, 'Step1.1_Paper3_Gerrit_ProjectLevel_DiffLevel_Preprocessed.csv')
    MDF4.to_csv(step1_1_diff_output_path, index=False)
    
    col_names = ['old_tags','new_clean']
    step1_1_method_df_p3, step1_1_size_removed_p3 = token_limit_filter(MDF4.copy(), 3, col_names)
    output_path_p3 = os.path.join(output_dir, 'Step1.1_Paper3_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_1_method_df_p3.to_csv(output_path_p3, index=False)
    
    diff_df_initial_len = len(diff_df)
    step1_1_diff_df_len = len(MDF4)
    step1_1_method_df_p3_len = len(step1_1_method_df_p3)
    
    del diff_df, step1_1_diff_df, step1_1_method_df_p3
    gc.collect()
    
    del MDF1, MDF2, MDF4
    gc.collect()
    
    # ------------------------------------------
    # Step 1.2 for Paper 3 (Diff-Level)
    #P2 (PAPER 3): Comment pruning + SIZE FILTER
    # ------------------------------------------
    step1_2_diff_input = pd.read_csv(step1_1_diff_output_path)
    analyzer = Analyzer1(step1_2_diff_input, data_type)
    
    #Remove invalid comments
    step1_2_1_initial_len = len(step1_2_diff_input)
    step1_2_diff_df = analyzer.analyze_diff_comments_Data()
    step1_2_diff_invalidComments_removed = step1_2_1_initial_len - len(step1_2_diff_df)
    
    #Keep only earliest comment per diff hunk
    step1_2_2_initial_len = len(step1_2_diff_df)
    step1_2_2_diff_df = analyzer.keep_earliest_comment_diffLevel(step1_2_diff_df)
    step1_2_diff_KeepEarliest_removed = step1_2_2_initial_len - len(step1_2_2_diff_df)
    
    
    MDF1 = Analyzer2.apply_cleaning_to_column_safely(step1_2_2_diff_df, 'old', 'old_clean')
    MDF2 = Analyzer2.normalize_start_end_tokens(MDF1, 'old_clean', 'old_tags')
    MDF4 = Analyzer2.apply_cleaning_to_column_safely(MDF2, 'new', 'new_clean')
    MDF4 = MDF4.drop(
    columns=['old_clean'],
    errors='ignore')
    
    
    
    # Save output
    step1_2_2_diff_output_path = os.path.join(output_dir, 'Step1.2_Paper3_Gerrit_ProjectLevel_DiffLevel_Preprocessed.csv')
    MDF4.to_csv(step1_2_2_diff_output_path, index=False)
    
    
    col_names = ['old_tags','new_clean']
    step1_2_method_df_p3, step1_2_size_removed_p3 = token_limit_filter(MDF4.copy(), 3, col_names)
    output_path_p3 = os.path.join(output_dir, 'Step1.2_Paper3_Gerrit_ProjectLevel_MethodLevel_SizeFiltered.csv')
    step1_2_method_df_p3.to_csv(output_path_p3, index=False)
    
    
    step1_2_diff_input_len = len(step1_2_diff_input)
    step1_2_2_diff_df_len = len(MDF4)
    step1_2_method_df_p3_len = len(step1_2_method_df_p3)
    
    
    
    del step1_2_diff_input, step1_2_2_diff_df, step1_2_method_df_p3
    gc.collect()
    
    
    del MDF1, MDF2, MDF4
    gc.collect()
    
    # ------------------------------------------
    # Step 1.3 for Paper 3 (Diff-Level)
    #P3 (PAPER 3): inline comments removal + SIZE FILTER
    # ------------------------------------------
    
    # Inline Comments Removal: Drop rows with NA/empty 'old_wo_comment' or 'new_wo_comment'
    step1_3_input =  pd.read_csv(step1_2_2_diff_output_path)
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
    
    # ------------------------------------------
    # Step 1.4 for Paper 3 (Diff-Level)
    #P4 (PAPER 3):comments cleaning + SIZE FILTER
    # ------------------------------------------
    # Comment cleaning 
    Input = pd.read_csv(step1_3_1_diff_output_path)
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
    
    
    # ------------------------------------------
    # Write Summary Log
    # ------------------------------------------
    with open(log_path, 'w') as log_file:
        log_file.write("===== Gerrit ProjectLevel Test_NoLeakage Preprocessing Log =====\n\n")
        log_file.write(f"Initial raw records: {initial_data_count}\n")
        
        # Raw Cleanup
        log_file.write(f"\n[Raw Data Cleanup]\n")
        log_file.write(f" - Initial raw records: {initial_data_count}\n")
        log_file.write(f" - NaN records removed: {nan_data_count}\n")
        log_file.write(f" - Duplicates removed: {duplicates_count}\n")
        log_file.write(f" - Left-side comments removed: {left_side_count}\n")
        
        # Method-Level Step 1.1 - Paper 1 & 2
        log_file.write(f"\n[Method-Level] (Step 1.1 - Paper 1 & 2) P1: Method extraction\n")
        log_file.write(f" - Initial Method-Level extracted records: {len(step0_method_df)}\n")
        log_file.write(f" - NaN/empty extracted methods removed: {step1_1_method_nan_dropped}\n")
        log_file.write(f" - Duplicates removed: {step1_1_method_duplicates}\n")
        log_file.write(f" - Duplicated comments removed (same comments different before&after): {step1_1_method_shared_dropped}\n")
        log_file.write(f" - Output: {step1_1_output_path}\n")
        log_file.write(f" - Records remaining (Records before size filtering): {step1_1_method_df_len}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {step1_1_size_removed_p1}\n")
        log_file.write(f" - Remaining after size filtering: {step1_1_method_df_p1_len}\n")
        log_file.write(f"   > Size filter (Paper 2 - 512 tokens) removed: {step1_1_size_removed_p2}\n")
        log_file.write(f" - Remaining after size filtering: {step1_1_method_df_p2_len}\n")
        
        
        
        # Method-Level Step 1.2 - Paper 1 & 2
        log_file.write(f"\n[Method-Level] (Step 1.2 - Paper 1 & 2) ----> P2: Comment pruning\n")
        log_file.write(f" - Initial records before comment pruning: {step1_1_method_df_len}\n")
        log_file.write(f" - Rows with invalid comments (owner, comment2comment, not linked, empty string) removed: {step1_2_method_invalidComments_removed}\n")
        log_file.write(f" - Methods with multiple comments removed (same before&after, multiple comments): {step1_2_method_multipleComment_removed}\n")
        log_file.write(f" - Output: {step1_2_output_path}\n")
        log_file.write(f" - Records remaining  (Records before size filtering): {step1_2_method_df_len}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {step1_2_size_removed_p1}\n")
        log_file.write(f" - Remaining after size filtering: {step1_2_method_df_p1_len}\n")
        log_file.write(f"   > Size filter (Paper 2 - 512 tokens) removed: {step1_2_size_removed_p2}\n")
        log_file.write(f" - Remaining after size filtering: {step1_2_method_df_p2_len}\n")
        
        
        
        
        # Method-Level Step 1.3 - Paper 1 & 2
        log_file.write(f"\n[Method-Level] (Step 1.3 - Paper 1 & 2) ----> P3: Unchanged methods removal\n")
        log_file.write(f" - Initial records before unchanged methods removal: {step1_3_Input_len}\n")
        log_file.write(f" - Equal before==after removed: {step1_3_equal_removed}\n")
        log_file.write(f" - Output: {step1_3_output_path}\n")
        log_file.write(f" - Records remaining: {step1_3_method_df_len}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {step1_3_size_removed_p1}\n")
        log_file.write(f" - Remaining after size filtering: {step1_3_method_df_p1_len}\n")
        log_file.write(f"   > Size filter (Paper 2 - 512 tokens) removed: {step1_3_size_removed_p2}\n")
        log_file.write(f" - Remaining after size filtering: {step1_3_method_df_p2_len}\n")
        
        
        
        # Method-Level Step 2.1 - Paper 1
        log_file.write(f"\n[Method-Level] (Step 2.1 - Paper 1) ----> P4: Abstraction\n")
        log_file.write(f" - Initial records before abstraction: {step1_3_method_df_len}\n")
        log_file.write(f" - Records After abstraction: {step2_1_paper1_initial_len}\n")
        log_file.write(f" - Step 2.1: NaNs records removed: {step2_1_paper1_removed}\n")
        log_file.write(f" - Output: {step2_1_paper1_output_path}\n")
        log_file.write(f" - Records remaining: {step2_1_paper1_df_len}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {step2_1_size_removed_p1}\n")
        log_file.write(f" - Remaining after size filtering: {step2_1_method_df_p1_len}\n")
        
        
        
        # Method-Level Step 2.2 - Paper 1
        log_file.write(f"\n[Method-Level] (Step 2.2 - Paper 1) ----> P5: Unchanged methods removal\n")
        log_file.write(f" - Initial records before unchanged abstracted methods removal: {step2_1_paper1_df_len}\n")
        log_file.write(f" - Step 2.2: Removed because abs(before) == abs(after): {step2_2_paper1_removed}\n")
        log_file.write(f" - Output: {step2_2_paper1_output_path}\n")
        log_file.write(f" - Records remaining: {step2_2_method_df_p1_len}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {step2_2_size_removed_p1}\n")
        log_file.write(f" - Remaining after size filtering: {step2_2_method_df_p1SizeFilter_len}\n")
        
        
        
        # Method-Level Step 2.3 - Paper 1
        log_file.write(f"\n[Method-Level] (Step 2.3 - Paper 1) ----> P6: New token removal\n")
        log_file.write(f" - Initial records before new token removal: {step2_3_paper1_initial_len}\n")
        log_file.write(f" - Step 2.3: Removed due to new tokens in abs(after): {step2_3_paper1_removed}\n")
        log_file.write(f" - Output: {step2_3_paper1_output_path}\n")
        log_file.write(f" - Records remaining: {step2_3_paper1_df_len}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {step2_3_size_removed_p1}\n")
        log_file.write(f" - Remaining after size filtering: {step2_3_method_df_p1_len}\n")
        
        
        # Method-Level Step 2.4 - Paper 1
        log_file.write(f"\n[Method-Level] (Step 2.4 - Paper 1) ----> P7: Comment cleaning\n")
        log_file.write(f" - Initial records before comment cleaning: {step2_3_paper1_df_len}\n")
        log_file.write(f" - Step 2.4: Removed irrelevant/empty abstracted comments: {step2_4_paper1_removed}\n")
        log_file.write(f" - Output: {step2_4_paper1_output_path}\n")
        log_file.write(f" - Records remaining: {step2_4_paper1_df_len}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {step2_4_size_removed_p1}\n")
        log_file.write(f" - Remaining after size filtering: {step2_4_method_df_p1_len}\n")
        
        
    
        # Method-Level Step 2.1 - Paper 2
        log_file.write(f"\n[Method-Level] (Step 2.1 - Paper 2)  ----> P4: Comment cleaning\n")
        log_file.write(f" - Initial records before comment cleaning: {step2_1_paper2_initial_len}\n")
        log_file.write(f" - Step 2.1: Removed irrelevant/empty non-abstracted comments: {step2_1_paper2_removed}\n")
        log_file.write(f" - Output: {step2_1_paper2_output_path}\n")
        log_file.write(f" - Records remaining: {step2_1_paper2_df_len}\n")
        log_file.write(f"   > Size filter (Paper 2 - 512 tokens) removed: {step2_1_size_removed_p2}\n")
        log_file.write(f" - Remaining after size filtering: {step2_1_method_df_p2_len}\n")
        
        
        # Diff-Level Step 1.1 - Paper 3
        log_file.write(f"\n[Diff-Level] (Step 1.1 - Paper 3) ----> P1: diff extraction\n")
        log_file.write(f" - Diff-Level extracted records: {diff_df_initial_len}\n")
        log_file.write(f" - NaN/empty diff hunks removed: {step1_1_diff_nan_dropped}\n")
        log_file.write(f" - Duplicates removed: {step1_1_diff_duplicates}\n")
        log_file.write(f" - Duplicated comments removed (same comments on different hunks): {step1_1_diff_earliest_removed}\n")
        log_file.write(f" - Output: {step1_1_diff_output_path}\n")
        log_file.write(f" - Records remaining: {step1_1_diff_df_len}\n")
        log_file.write(f"   > Size filter (Paper 3 - 200 tokens) removed: {step1_1_size_removed_p3}\n")
        log_file.write(f" - Remaining after size filtering: {step1_1_method_df_p3_len}\n")
        
        
        
        # Diff-Level Step 1.2 - Paper 3
        log_file.write(f"\n[Diff-Level] (Step 1.2 - Paper 3) ----> P2: Comment pruning\n")
        log_file.write(f" - Initial records before comment pruning: {step1_2_diff_input_len}\n")
        log_file.write(f" - Invalid diff comments removed: {step1_2_diff_invalidComments_removed}\n")
        log_file.write(f" - Multiple comments on same hunk (keeping earliest) removed: {step1_2_diff_KeepEarliest_removed}\n")
        log_file.write(f" - Output: {step1_2_2_diff_output_path}\n")
        log_file.write(f" - Records remaining: {step1_2_2_diff_df_len}\n")
        log_file.write(f"   > Size filter (Paper 3 - 200 tokens) removed: {step1_2_size_removed_p3}\n")
        log_file.write(f" - Remaining after size filtering: {step1_2_method_df_p3_len}\n")
        
        
        
        
        # Diff-Level Step 1.3 - Paper 3
        log_file.write(f"\n[Diff-Level] (Step 1.3 - Paper 3) ----> P3: Inline comments removal \n")
        log_file.write(f" - Initial records before inline comments removal: {step1_3_input_len}\n")
        log_file.write(f" - Diff rows with NA/empty old_wo_comment or new_wo_comment removed: {step1_3_1_removed}\n")
        log_file.write(f" - Output: {step1_3_1_diff_output_path}\n")
        log_file.write(f" - Records remaining: {step1_3_1_filtered_dff_len}\n")
        log_file.write(f"   > Size filter (Paper 3 - 200 tokens) removed: {step1_3_size_removed_p3}\n")
        log_file.write(f" - Remaining after size filtering: {step1_3_method_df_p3_len}\n")
        
        
        
        # Diff-Level Step 1.4 - Paper 3
        log_file.write(f"\n[Diff-Level] (Step 1.4 - Paper 3) ----> P4: Comment cleaning\n")
        log_file.write(f" - Initial records before comment cleaning: {Input_len}\n")
        log_file.write(f" - Rows with empty/NA cleaned_comment_no_praise removed: {step1_4_removed}\n")
        log_file.write(f" - Rows outside size limits (Comment_size_filter_3_150) removed: {step1_4_3_removed}\n")
        log_file.write(f" - Irrelevant comments removed (diff_Comment_is_relevant=False): {step1_4_4_removed}\n")
        log_file.write(f" - Output: {step1_4_diff_output_path}\n")
        log_file.write(f" - Records remaining: {step1_4_4_df_len}\n")
        log_file.write(f"   > Size filter (Paper 3 - 200 tokens) removed: {step1_4_size_removed_p3}\n")
        log_file.write(f" - Remaining after size filtering: {step1_4_method_df_p3_len}\n")
    
    # ------------------------------------------
    # Print Summary to Console
    # ------------------------------------------
    print(open(log_path).read())
    
    
    return {
    "chunk_index": i,
    
    "initial_data_count": initial_data_count,
    "nan_data_count": nan_data_count,
    "duplicates_count": duplicates_count,
    "left_side_count": left_side_count,
    "step0_method_df": len(step0_method_df),
    "step1_1_method_nan_dropped": step1_1_method_nan_dropped,
    "step1_1_method_duplicates": step1_1_method_duplicates,
    "step1_1_method_shared_dropped": step1_1_method_shared_dropped,
    "step1_1_method_df": step1_1_method_df_len,
    "step1_1_size_removed_p1": step1_1_size_removed_p1,
    "step1_1_method_df_p1": step1_1_method_df_p1_len,
    "step1_1_size_removed_p2": step1_1_size_removed_p2,
    "step1_1_method_df_p2": step1_1_method_df_p2_len,
    
    "step1_2_initial_records": step1_1_method_df_len,
    "step1_2_method_invalidComments_removed": step1_2_method_invalidComments_removed,
    "step1_2_method_multipleComment_removed": step1_2_method_multipleComment_removed,
    "step1_2_method_df": step1_2_method_df_len,
    "step1_2_size_removed_p1": step1_2_size_removed_p1,
    "step1_2_method_df_p1": step1_2_method_df_p1_len,
    "step1_2_size_removed_p2": step1_2_size_removed_p2,
    "step1_2_method_df_p2": step1_2_method_df_p2_len,
    
    "step1_3_initial_records": step1_3_Input_len,
    "step1_3_equal_removed": step1_3_equal_removed,
    "step1_3_method_df": step1_3_method_df_len,
    "step1_3_size_removed_p1": step1_3_size_removed_p1,
    "step1_3_method_df_p1": step1_3_method_df_p1_len,
    "step1_3_size_removed_p2": step1_3_size_removed_p2,
    "step1_3_method_df_p2": step1_3_method_df_p2_len,
    
    "step2_1_paper1_initial_len": step1_3_method_df_len,
    "Records_after_abstraction": step2_1_paper1_initial_len,
    "step2_1_paper1_removed": step2_1_paper1_removed,
    "step2_1_paper1_df": step2_1_paper1_df_len,
    "step2_1_size_removed_p1": step2_1_size_removed_p1,
    "step2_1_method_df_p1": step2_1_method_df_p1_len,
    
    
    "step2_2_paper1_initial_len": step2_1_paper1_df_len,
    "step2_2_paper1_removed": step2_2_paper1_removed,
    "step2_2_paper1_df": step2_2_method_df_p1_len,
    "step2_2_size_removed_p1": step2_2_size_removed_p1,
    "step2_2_method_df_p1": step2_2_method_df_p1SizeFilter_len,
    
    "step2_3_paper1_initial_len": step2_3_paper1_initial_len,
    "step2_3_paper1_removed": step2_3_paper1_removed,
    "step2_3_paper1_df": step2_3_paper1_df_len,
    "step2_3_size_removed_p1": step2_3_size_removed_p1,
    "step2_3_method_df_p1": step2_3_method_df_p1_len,
    
    
    "step2_4_paper1_initial_len": step2_4_paper1_initial_len,
    "step2_4_paper1_removed": step2_4_paper1_removed,
    "step2_4_paper1_df": step2_4_paper1_df_len,
    "step2_4_size_removed_p1": step2_4_size_removed_p1,
    "step2_4_method_df_p1": step2_4_method_df_p1_len,
    
    
    
    "step2_1_paper2_initial_len": step2_1_paper2_input_len,
    "step2_1_paper2_removed": step2_1_paper2_removed,
    "step2_1_paper2_df": step2_1_paper2_df_len,
    "step2_1_size_removed_p2": step2_1_size_removed_p2,
    "step2_1_method_df_p2": step2_1_method_df_p2_len,
    
    
    "step0_diff_df": diff_df_initial_len,
    "step1_1_diff_nan_dropped": step1_1_diff_nan_dropped,
    "step1_1_diff_duplicates": step1_1_diff_duplicates,
    "step1_1_diff_earliest_removed": step1_1_diff_earliest_removed,
    "step1_1_diff_df": step1_1_diff_df_len,
    "step1_1_size_removed_p3": step1_1_size_removed_p3,
    "step1_1_method_df_p3": step1_1_method_df_p3_len,
    
    "step1_2_paper3_initial_len": step1_2_diff_input_len,
    "step1_2_diff_invalidComments_removed": step1_2_diff_invalidComments_removed,
    "step1_2_diff_KeepEarliest_removed": step1_2_diff_KeepEarliest_removed,
    "step1_2_2_diff_df": step1_2_2_diff_df_len,
    "step1_2_size_removed_p3": step1_2_size_removed_p3,
    "step1_2_method_df_p3": step1_2_method_df_p3_len,
    
    "step1_3_paper3_initial_len": step1_3_input_len,
    "step1_3_1_removed": step1_3_1_removed,
    "step1_3_1_filtered_dff": step1_3_1_filtered_dff_len,
    "step1_3_size_removed_p3": step1_3_size_removed_p3,
    "step1_3_method_df_p3": step1_3_method_df_p3_len,
    
    "step1_4_paper3_initial_len": Input_len,
    "step1_4_removed": step1_4_removed,
    "step1_4_3_removed": step1_4_3_removed,
    "step1_4_4_df": step1_4_4_df_len,
    "step1_4_size_removed_p3": step1_4_size_removed_p3,
    "step1_4_method_df_p3": step1_4_method_df_p3_len,
}


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
output_directory = os.path.join(Root_dir, 'SplittedData/PreprocessedData', input_base_name)
Final_log_path = os.path.join(output_directory, 'Gerrit_ProjectLevel_Processing_Log.txt')

def write_final_log(Final_log_path, summary_stats):
    with open(Final_log_path, 'w') as log_file:
        log_file.write("===== Gerrit ProjectLevel Test_NoLeakage Preprocessing Log =====\n\n")
        log_file.write(f"Initial raw records: {summary_stats['initial_data_count']}\n")
        
        # Raw Cleanup
        log_file.write(f"\n[Raw Data Cleanup]\n")
        log_file.write(f" - Initial raw records: {summary_stats['initial_data_count']}\n")
        log_file.write(f" - NaN records removed: {summary_stats['nan_data_count']}\n")
        log_file.write(f" - Duplicates removed: {summary_stats['duplicates_count']}\n")
        log_file.write(f" - Left-side comments removed: {summary_stats['left_side_count']}\n")
        
        # Method-Level Step 1.1 - Paper 1 & 2
        log_file.write(f"\n[Method-Level] (Step 1.1 - Paper 1 & 2) P1: Method extraction\n")
        log_file.write(f" - Initial Method-Level extracted records: {summary_stats['step0_method_df']}\n")
        log_file.write(f" - NaN/empty extracted methods removed: {summary_stats['step1_1_method_nan_dropped']}\n")
        log_file.write(f" - Duplicates removed: {summary_stats['step1_1_method_duplicates']}\n")
        log_file.write(f" - Duplicated comments removed (same comments different before&after): {summary_stats['step1_1_method_shared_dropped']}\n")
        log_file.write(f" - Records remaining (Records before size filtering): {summary_stats['step1_1_method_df']}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {summary_stats['step1_1_size_removed_p1']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_1_method_df_p1']}\n")
        log_file.write(f"   > Size filter (Paper 2 - 512 tokens) removed: {summary_stats['step1_1_size_removed_p2']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_1_method_df_p2']}\n")
        
        # Method-Level Step 1.2 - Paper 1 & 2
        log_file.write(f"\n[Method-Level] (Step 1.2 - Paper 1 & 2) ----> P2: Comment pruning\n")
        log_file.write(f" - Initial records before comment pruning: {summary_stats['step1_2_initial_records']}\n")
        log_file.write(f" - Rows with invalid comments (owner, comment2comment, not linked, empty string) removed: {summary_stats['step1_2_method_invalidComments_removed']}\n")
        log_file.write(f" - Methods with multiple comments removed (same before&after, multiple comments): {summary_stats['step1_2_method_multipleComment_removed']}\n")
        log_file.write(f" - Records remaining  (Records before size filtering): {summary_stats['step1_2_method_df']}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {summary_stats['step1_2_size_removed_p1']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_2_method_df_p1']}\n")
        log_file.write(f"   > Size filter (Paper 2 - 512 tokens) removed: {summary_stats['step1_2_size_removed_p2']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_2_method_df_p2']}\n")
        
        # Method-Level Step 1.3 - Paper 1 & 2
        log_file.write(f"\n[Method-Level] (Step 1.3 - Paper 1 & 2) ----> P3: Unchanged methods removal\n")
        log_file.write(f" - Initial records before unchanged methods removal: {summary_stats['step1_3_initial_records']}\n")
        log_file.write(f" - Equal before==after removed: {summary_stats['step1_3_equal_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step1_3_method_df']}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {summary_stats['step1_3_size_removed_p1']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_3_method_df_p1']}\n")
        log_file.write(f"   > Size filter (Paper 2 - 512 tokens) removed: {summary_stats['step1_3_size_removed_p2']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_3_method_df_p2']}\n")
        
        # Method-Level Step 2.1 - Paper 1
        log_file.write(f"\n[Method-Level] (Step 2.1 - Paper 1) ----> P4: Abstraction\n")
        log_file.write(f" - Initial records before abstraction: {summary_stats['step2_1_paper1_initial_len']}\n")
        log_file.write(f" - Records After abstraction: {summary_stats['Records_after_abstraction']}\n")
        log_file.write(f" - Step 2.1: NaNs records removed: {summary_stats['step2_1_paper1_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step2_1_paper1_df']}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {summary_stats['step2_1_size_removed_p1']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step2_1_method_df_p1']}\n")
        
        # Method-Level Step 2.2 - Paper 1
        log_file.write(f"\n[Method-Level] (Step 2.2 - Paper 1) ----> P5: Unchanged methods removal\n")
        log_file.write(f" - Initial records before unchanged abstracted methods removal: {summary_stats['step2_2_paper1_initial_len']}\n")
        log_file.write(f" - Step 2.2: Removed because abs(before) == abs(after): {summary_stats['step2_2_paper1_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step2_2_paper1_df']}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {summary_stats['step2_2_size_removed_p1']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step2_2_method_df_p1']}\n")
        
        # Method-Level Step 2.3 - Paper 1
        log_file.write(f"\n[Method-Level] (Step 2.3 - Paper 1) ----> P6: New token removal\n")
        log_file.write(f" - Initial records before new token removal: {summary_stats['step2_3_paper1_initial_len']}\n")
        log_file.write(f" - Step 2.3: Removed due to new tokens in abs(after): {summary_stats['step2_3_paper1_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step2_3_paper1_df']}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {summary_stats['step2_3_size_removed_p1']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step2_3_method_df_p1']}\n")
        
        # Method-Level Step 2.4 - Paper 1
        log_file.write(f"\n[Method-Level] (Step 2.4 - Paper 1) ----> P7: Comment cleaning\n")
        log_file.write(f" - Initial records before comment cleaning: {summary_stats['step2_4_paper1_initial_len']}\n")
        log_file.write(f" - Step 2.4: Removed irrelevant/empty abstracted comments: {summary_stats['step2_4_paper1_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step2_4_paper1_df']}\n")
        log_file.write(f"   > Size filter (Paper 1 - 100 tokens) removed: {summary_stats['step2_4_size_removed_p1']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step2_4_method_df_p1']}\n")
        
        # Method-Level Step 2.1 - Paper 2
        log_file.write(f"\n[Method-Level] (Step 2.1 - Paper 2)  ----> P4: Comment cleaning\n")
        log_file.write(f" - Initial records before comment cleaning: {summary_stats['step2_1_paper2_initial_len']}\n")
        log_file.write(f" - Step 2.1: Removed irrelevant/empty non-abstracted comments: {summary_stats['step2_1_paper2_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step2_1_paper2_df']}\n")
        log_file.write(f"   > Size filter (Paper 2 - 512 tokens) removed: {summary_stats['step2_1_size_removed_p2']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step2_1_method_df_p2']}\n")
        
        # Diff-Level Step 1.1 - Paper 3
        log_file.write(f"\n[Diff-Level] (Step 1.1 - Paper 3) ----> P1: diff extraction\n")
        log_file.write(f" - Diff-Level extracted records: {summary_stats['step0_diff_df']}\n")
        log_file.write(f" - NaN/empty diff hunks removed: {summary_stats['step1_1_diff_nan_dropped']}\n")
        log_file.write(f" - Duplicates removed: {summary_stats['step1_1_diff_duplicates']}\n")
        log_file.write(f" - Duplicated comments removed (same comments on different hunks): {summary_stats['step1_1_diff_earliest_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step1_1_diff_df']}\n")
        log_file.write(f"   > Size filter (Paper 3 - 200 tokens) removed: {summary_stats['step1_1_size_removed_p3']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_1_method_df_p3']}\n")
        
        # Diff-Level Step 1.2 - Paper 3
        log_file.write(f"\n[Diff-Level] (Step 1.2 - Paper 3) ----> P2: Comment pruning\n")
        log_file.write(f" - Initial records before comment pruning: {summary_stats['step1_2_paper3_initial_len']}\n")
        log_file.write(f" - Invalid diff comments removed: {summary_stats['step1_2_diff_invalidComments_removed']}\n")
        log_file.write(f" - Multiple comments on same hunk (keeping earliest) removed: {summary_stats['step1_2_diff_KeepEarliest_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step1_2_2_diff_df']}\n")
        log_file.write(f"   > Size filter (Paper 3 - 200 tokens) removed: {summary_stats['step1_2_size_removed_p3']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_2_method_df_p3']}\n")
        
        # Diff-Level Step 1.3 - Paper 3
        log_file.write(f"\n[Diff-Level] (Step 1.3 - Paper 3) ----> P3: Inline comments removal \n")
        log_file.write(f" - Initial records before inline comments removal: {summary_stats['step1_3_paper3_initial_len']}\n")
        log_file.write(f" - Diff rows with NA/empty old_wo_comment or new_wo_comment removed: {summary_stats['step1_3_1_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step1_3_1_filtered_dff']}\n")
        log_file.write(f"   > Size filter (Paper 3 - 200 tokens) removed: {summary_stats['step1_3_size_removed_p3']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_3_method_df_p3']}\n")
        
        # Diff-Level Step 1.4 - Paper 3
        log_file.write(f"\n[Diff-Level] (Step 1.4 - Paper 3) ----> P4: Comment cleaning\n")
        log_file.write(f" - Initial records before comment cleaning: {summary_stats['step1_4_paper3_initial_len']}\n")
        log_file.write(f" - Rows with empty/NA cleaned_comment_no_praise removed: {summary_stats['step1_4_removed']}\n")
        log_file.write(f" - Rows outside size limits (Comment_size_filter_3_150) removed: {summary_stats['step1_4_3_removed']}\n")
        log_file.write(f" - Irrelevant comments removed (diff_Comment_is_relevant=False): {summary_stats['step1_4_3_removed']}\n")
        log_file.write(f" - Records remaining: {summary_stats['step1_4_4_df']}\n")
        log_file.write(f"   > Size filter (Paper 3 - 200 tokens) removed: {summary_stats['step1_4_size_removed_p3']}\n")
        log_file.write(f" - Remaining after size filtering: {summary_stats['step1_4_method_df_p3']}\n")


write_final_log(Final_log_path, summary_stats)
# ------------------------------------------
# Print Summary to Console
# ------------------------------------------
print(open(Final_log_path).read())
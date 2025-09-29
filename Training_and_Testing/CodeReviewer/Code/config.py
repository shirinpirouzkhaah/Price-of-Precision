import os
import yaml

# Mapping from preprocessing step to column names
step_to_columns = {
    "S1_P1plusP3": ["before_marked_flat", "comment", "after_flat"],
    "S1_P1plusP4": ["before_abs_marked", "abstracted_comment", "after_abs"],
    "S1_P1plusP4plusP5": ["before_abs_marked", "abstracted_comment", "after_abs"],
    "S1_P1plusP4plusP6": ["before_abs_marked", "abstracted_comment", "after_abs"],
    "S1_P1plusP6": ["before_marked_flat", "comment", "after_flat"],
    "S1_P1plusP7": ["before_marked_flat", "comment", "after_flat"],

    "S1_P1": ["before_marked_flat", "comment", "after_flat"],
    "S1_P2": ["before_marked_flat", "comment", "after_flat"],
    "S1_P3": ["before_marked_flat", "comment", "after_flat"],
    "S1_P4": ["before_abs_marked", "abstracted_comment", "after_abs"],
    "S1_P5": ["before_abs_marked", "abstracted_comment", "after_abs"],
    "S1_P6": ["before_abs_marked", "abstracted_comment", "after_abs"],
    "S1_P7": ["before_abs_marked", "cleaned_abstracted_comment", "after_abs"],

    "S2_P1plusP3": ["before_marked_flat", "comment", "after_flat"],
    "S2_P1plusP4": ["before_marked_flat", "comment", "after_flat"],

    "S2_P1": ["before_marked_flat", "comment", "after_flat"],
    "S2_P2": ["before_marked_flat", "comment", "after_flat"],
    "S2_P3": ["before_marked_flat", "comment", "after_flat"],
    "S2_P4": ["before_marked_flat", "cleaned_comment_wo_abs", "after_flat"],

    "S3_P1": ["old_tags", "comment", "new_clean"],
    "S3_P2": ["old_tags", "comment", "new_clean"],
    "S3_P3": ["old_wo_comment_tags", "comment", "new_wo_comment_clean"],
    "S3_P4": ["old_wo_comment_tags", "cleaned_comment_no_praise", "new_wo_comment_clean"],

    "S3_P1plusP3": ["old_wo_comment_tags", "comment", "new_wo_comment_clean"],
    "S3_P1plusP4": ["old_tags", "cleaned_comment_no_praise", "new_clean"],
}

# Define pipeline mapping
linear_pipeline_steps = [
    "S1_P7", "S1_P6", "S1_P5", "S1_P4", "S1_P3", "S1_P2", "S1_P1",
    "S2_P4", "S2_P3", "S2_P2", "S2_P1",
    "S3_P4", "S3_P3", "S3_P2", "S3_P1"
]

isolated_steps = [
    "S1_P1plusP3", "S1_P1plusP4", "S1_P1plusP4plusP5", "S1_P1plusP4plusP6", "S1_P1plusP6", "S1_P1plusP7",
    "S2_P1plusP3", "S2_P1plusP4",
    "S3_P1plusP3", "S3_P1plusP4"
]

def infer_step_and_level_from_path():
    cwd = os.getcwd()
    basename = os.path.basename(cwd)
    if "_" not in basename:
        raise ValueError("Current directory name must be in format '<step>_<level>Level'")

    parts = basename.split("_")
    step = "_".join(parts[:-1])
    level = parts[-1].replace("Level", "")
    return step, level

def get_pipeline_for_step(step):
    if step in linear_pipeline_steps:
        return "LinearPipeline"
    elif step in isolated_steps:
        return "IsolatedSteps"
    else:
        raise ValueError(f"Step {step} not found in pipeline mappings.")

# Main logic
step, level = infer_step_and_level_from_path()
pipeline = get_pipeline_for_step(step)
columns = step_to_columns.get(step)

if columns is None:
    raise ValueError(f"Step '{step}' does not have defined column names in step_to_columns.")

yaml_config = {
    "Level": level,
    "pipeline": pipeline,
    "Data_preprocessing_step": step,

    "Train_data_path": f"../../data/{pipeline}LargeData/{level}Splited_Train_raw/{step}_{level}Level.csv",
    "Test_data_path": f"../../data/{pipeline}LargeData/{level}Splited_Test_NoLeakage/{step}_{level}Level.csv",
    "Common_test_data_path": f"../../data/{pipeline}LargeData/{level}Splited_Test_NoLeakage/Common_Record_id.csv",
    "New_test_data_path": f"../../data/{pipeline}NewData/TimeSplited_Test_NoLeakage/{step}_TimeLevel.csv",
    "Common_New_test_data_path": f"../../data/{pipeline}NewData/TimeSplited_Test_NoLeakage/Common_Record_id.csv",

    "column_names": columns,

    "model_name": "CodeReviewer",
    "beam_size": 10,
    "SEED": 42,
    "log_file": "log.txt"
}

# Save config
with open("config.yml", "w") as f:
    yaml.dump(yaml_config, f, sort_keys=False)

print("✅ Generated config.yml with:")
print(yaml.dump(yaml_config, sort_keys=False))


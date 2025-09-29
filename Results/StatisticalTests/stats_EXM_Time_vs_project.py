import os
import glob
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import (
    proportions_ztest,
    proportion_confint,
    confint_proportions_2indep,
)
from scipy.stats import fisher_exact

# --- Labels (optional prettifying) ---
MODEL_LABELS = {
    "S1CodeLlama": "Code Llama 7B (S1)",
    "S2CodeLlama": "Code Llama 7B (S2)",
    "S3CodeLlama": "Code Llama 7B (S3)",
    "S1OpenNMT": "OpenNMT (S1)",
    "S2T5": "T5 (S2)",
    "S3CodeReviewer": "CodeReviewer (S3)",
}

EXM_COL = "EXM10"
TESTSET_NAME = "common_test"
LEVELS = ["TimeLevel", "ProjectLevel"]

# --- Helpers ---
def load_binary_values(files, col):
    """Load 0/1 values from the given column across all files."""
    vals = []
    for f in files:
        df = pd.read_csv(f, sep=",", engine="c")
        if col not in df.columns:
            raise KeyError(f"'{col}' not found in {f}")
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        x = x.astype(int).clip(lower=0, upper=1)
        vals.extend(x.values.tolist())
    return np.array(vals, dtype=int)

def two_prop_test_with_ci(x_time, x_proj, alpha=0.05):
    """Run two-proportion z-test (fallback to Fisher if any expected cell < 5).
       Returns dict of stats + CIs with your requested key names."""
    succ_time = int(x_time.sum())
    n_time    = int(len(x_time))
    succ_proj = int(x_proj.sum())
    n_proj    = int(len(x_proj))

    p_time = succ_time / n_time if n_time else np.nan
    p_proj = succ_proj / n_proj if n_proj else np.nan
    diff   = p_time - p_proj

    # Per-group CIs (Wilson)
    ci_time = proportion_confint(succ_time, n_time, alpha=alpha, method="wilson")
    ci_proj = proportion_confint(succ_proj, n_proj, alpha=alpha, method="wilson")

    # 2x2 for quick Fisher fallback check
    fail_time = n_time - succ_time
    fail_proj = n_proj - succ_proj
    table = np.array([[succ_time, fail_time],
                      [succ_proj, fail_proj]], dtype=int)

    # Main test: two-proportion z
    z_stat, p_val = proportions_ztest([succ_time, succ_proj], [n_time, n_proj], alternative="two-sided")

    # Newcombe (score) CI for the difference
    diff_low, diff_high = confint_proportions_2indep(
        succ_time, n_time, succ_proj, n_proj, method="score"
    )

    # Fallback to Fisher if any expected cell < 5
    fisher_p = np.nan
    # Simple rule-of-thumb: if any observed cell < 5, Fisher can be informative
    if (table < 5).any():
        _, fisher_p = fisher_exact(table)

    return {
        "Model": None,                 # will be filled by caller
        "n_time": n_time,
        "n_proj": n_proj,
        "EXM10_time": succ_time,
        "EXM10_proj": succ_proj,

        "p_time": p_time,
        "p_proj": p_proj,

        "time_ci_low": ci_time[0],
        "proj_ci_low": ci_proj[0],

        "time_ci_high": ci_time[1],
        "proj_ci_high": ci_proj[1],

        "diff": diff,
        "diff_ci_low": diff_low,
        "diff_ci_high": diff_high,
        "z_stat": z_stat,
        "p_value": p_val,
        "fisher_p_if_small": fisher_p,
    }

def run_exm_time_vs_project(model_dir, testset_name=TESTSET_NAME):
    """Run EXM10 comparison between TimeLevel and ProjectLevel in a model directory."""
    os.chdir(model_dir)
    model = model_dir.replace("csvResults", "")

    level_vals = {}
    for level in LEVELS:
        files = sorted(glob.glob(f"*_{testset_name}_predictions_with_codeBLEU_10_{level}.csv"))
        if not files:
            print(f"⚠️ {model}: no files for {level}")
            level_vals[level] = None
            continue
        level_vals[level] = load_binary_values(files, EXM_COL)

    os.chdir("..")

    if (level_vals.get("TimeLevel") is not None) and (level_vals.get("ProjectLevel") is not None):
        res = two_prop_test_with_ci(level_vals["TimeLevel"], level_vals["ProjectLevel"])
        res["Model"] = model
        return res
    else:
        return {"Model": model, "error": "missing TimeLevel or ProjectLevel"}

# --- Main ---
if __name__ == "__main__":
    result_dirs = [
        "csvResultsS1CodeLlama",
        "csvResultsS2T5",
        "csvResultsS3CodeLlama",
        "csvResultsS1OpenNMT",
        "csvResultsS2CodeLlama",
        "csvResultsS3CodeReviewer",
    ]

    out_rows = []
    for d in result_dirs:
        out = run_exm_time_vs_project(d, TESTSET_NAME)
        out_rows.append(out)

    df = pd.DataFrame(out_rows)

    # Pretty model names
    if "Model" in df.columns:
        df["Model"] = df["Model"].map(MODEL_LABELS).fillna(df["Model"])

    os.makedirs("./plots/Time_vs_Project", exist_ok=True)
    out_csv = "./plots/Time_vs_Project/EXM10_TwoProp.csv"
    df.to_csv(out_csv, index=False)

    # Nice printout
    show_cols = [
        "Model",
        "n_time","n_proj",
        "EXM10_time","EXM10_proj",
        "p_time","p_proj",
        "time_ci_low","time_ci_high",
        "proj_ci_low","proj_ci_high",
        "diff","diff_ci_low","diff_ci_high",
        "z_stat","p_value","fisher_p_if_small"
    ]
    print("\n💾 Saved:", out_csv)
    print(df[[c for c in show_cols if c in df.columns]])


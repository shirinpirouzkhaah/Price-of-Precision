import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import kruskal

# --- Step Mapping (not really used here, kept for clarity) ---
STEP_LABELS = {
    "S1_P1": "P1", "S1_P2": "P1+P2", "S1_P3": "P1+..+P3", "S1_P4": "P1+..+P4",
    "S1_P5": "P1+..+P5", "S1_P6": "P1+..+P6", "S1_P7": "P1+..+P7",
    "S1_P1plusP3": "P1+P3", "S1_P1plusP4": "P1+P4", "S1_P1plusP4plusP5": "P1+P4+P5",
    "S1_P1plusP6": "P1+P6", "S1_P1plusP7": "P1+P7",
    "S2_P1": "P1", "S2_P2": "P1+P2", "S2_P3": "P1+..+P3", "S2_P4": "P1+..+P4",
    "S2_P1plusP3": "P1+P3", "S2_P1plusP4": "P1+P4",
    "S3_P1": "P1", "S3_P2": "P1+P2", "S3_P3": "P1+..+P3", "S3_P4": "P1+..+P4",
    "S3_P1plusP3": "P1+P3", "S3_P1plusP4": "P1+P4",
}

MODEL_LABELS = {
    "S1CodeLlama": "Code Llama 7B (S1)",
    "S2CodeLlama": "Code Llama 7B (S2)",
    "S3CodeLlama": "Code Llama 7B (S3)",
    "S1OpenNMT": "OpenNMT (S1)",
    "S2T5": "T5 (S2)",
    "S3CodeReviewer": "CodeReviewer (S3)"
}

# --- Helpers ---
def extract_step_from_filename(fname: str) -> str:
    """Extract step name from filename (e.g., S1_P1)."""
    base = os.path.basename(fname)
    parts = base.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return base

def load_metric_values(files, metric_col):
    """Load values from the given metric column across all files."""
    all_vals = []
    for f in files:
        df = pd.read_csv(f, sep=",", engine="c")
        if metric_col not in df.columns:
            raise KeyError(f"'{metric_col}' not found in {f}")
        all_vals.extend(df[metric_col].dropna().values)
    return np.array(all_vals)

def cohens_d(x, y):
    """Cohen's d for independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0

def cliffs_delta(x, y):
    """Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    total = 0
    for xi in x:
        total += np.sum(xi > y) - np.sum(xi < y)
    return total / (n_x * n_y)

def bootstrap_ci(x, y, n_boot=1000, alpha=0.05):
    """Bootstrap confidence interval for mean difference (x - y)."""
    diffs = []
    for _ in range(n_boot):
        xb = np.random.choice(x, size=len(x), replace=True)
        yb = np.random.choice(y, size=len(y), replace=True)
        diffs.append(np.mean(xb) - np.mean(yb))
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return lower, upper

def run_comparisons(model_dir, metrics, levels, testset_name):
    """Run Kruskal–Wallis comparison between TimeLevel and ProjectLevel for given metrics."""
    os.chdir(model_dir)
    model = model_dir.replace("csvResults", "")

    rows = []

    for metric in metrics:
        print(f"\n🚀 {model} | Metric={metric}")

        level_data = {}
        for level in levels:
            # ✅ Always use codeBLEU in filenames, columns decide metric
            files = sorted(glob.glob(f"*_{testset_name}_predictions_with_codeBLEU_10_{level}.csv"))
            if not files:
                print(f"⚠️ No files for {level}, metric={metric}")
                continue
            level_data[level] = load_metric_values(files, metric)

        if "TimeLevel" not in level_data or "ProjectLevel" not in level_data:
            print(f"⚠️ Skipping {model}, missing one level")
            continue

        time_vals = level_data["TimeLevel"]
        proj_vals = level_data["ProjectLevel"]

        # ✅ Global Kruskal–Wallis test
        kw_stat, kw_p = kruskal(time_vals, proj_vals)

        # ✅ Effect sizes
        d = cohens_d(time_vals, proj_vals)
        delta = cliffs_delta(time_vals, proj_vals)

        # ✅ Confidence interval
        ci_low, ci_high = bootstrap_ci(time_vals, proj_vals)

        rows.append({
            "Model": model,
            "Metric": metric,
            "KW_stat": kw_stat,
            "p_value": kw_p,
            "Cohens_d": d,
            "Cliffs_delta": delta,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "Time_mean": np.mean(time_vals),
            "Project_mean": np.mean(proj_vals),
            "Time_n": len(time_vals),
            "Project_n": len(proj_vals),
        })

    os.chdir("..")
    return rows

# --- Main ---
if __name__ == "__main__":
    # Directories to process
    result_dirs = [
        "csvResultsS1CodeLlama",
        "csvResultsS2T5",
        "csvResultsS3CodeLlama",
        "csvResultsS1OpenNMT",
        "csvResultsS2CodeLlama",
        "csvResultsS3CodeReviewer",
    ]

    # Metrics to compare
    metrics = ["codeBLEU", "Lev_Ratio"]
    levels = ["TimeLevel", "ProjectLevel"]

    all_results = []
    for model_dir in result_dirs:
        testset_name = "common_test"
        all_results.extend(run_comparisons(model_dir, metrics, levels, testset_name))

    # Save global results
    df_results = pd.DataFrame(all_results)
    df_results["Model"] = df_results["Model"].map(MODEL_LABELS).fillna(df_results["Model"])

    os.makedirs("./plots/Time_vs_Project", exist_ok=True)
    outpath = "./plots/Time_vs_Project/KruskalWallis.csv"
    df_results.to_csv(outpath, index=False)

    print(f"\n💾 Global stats saved to {outpath}")
    print(df_results)


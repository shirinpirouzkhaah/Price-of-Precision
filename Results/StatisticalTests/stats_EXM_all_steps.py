import os
import glob
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
import statsmodels.stats.multitest as smm

# --- Step labels (for pretty axis names) ---
STEP_LABELS = {
    "S1_P1": "P1",
    "S1_P2": "P1+P2",
    "S1_P3": "P1+..+P3",
    "S1_P4": "P1+..+P4",
    "S1_P5": "P1+..+P5",
    "S1_P6": "P1+..+P6",
    "S1_P7": "P1+..+P7",
    "S1_P1plusP3": "P1+P3",
    "S1_P1plusP4": "P1+P4",
    "S1_P1plusP4plusP5": "P1+P4+P5",
    "S1_P1plusP6": "P1+P6",
    "S1_P1plusP7": "P1+P7",
    "S2_P1": "P1",
    "S2_P2": "P1+P2",
    "S2_P3": "P1+..+P3",
    "S2_P4": "P1+..+P4",
    "S2_P1plusP3": "P1+P3",
    "S2_P1plusP4": "P1+P4",
    "S3_P1": "P1",
    "S3_P2": "P1+P2",
    "S3_P3": "P1+..+P3",
    "S3_P4": "P1+..+P4",
    "S3_P1plusP3": "P1+P3",
    "S3_P1plusP4": "P1+P4",
}

METRIC_COL = "EXM10"   # <-- binary 0/1 metric

# --- Step 1: Collect CSV files (same pattern) ---
def load_files_for_level(testset_name, level):
    pattern = f"*_{testset_name}_predictions_with_codeBLEU_10_{level}.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    files = [f for f in files if "S1_P1plusP4plusP6" not in f]
    print(f"\n📂 Collecting files for testset={testset_name}, level={level}:")
    for f in files:
        print(" -", os.path.basename(f))
    return files

# --- Step 2: Extract step id from filename ---
def extract_step_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    parts = base.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return base

# --- Step 3: Load EXM10 from each file into a dict {step: Series} ---
def load_all_series_binary(files):
    series_dict = {}
    for f in files:
        step = extract_step_from_filename(f)
        if step == "S1_P1plusP4plusP6":
            continue
        df = pd.read_csv(f, sep=",", engine="c")
        if METRIC_COL not in df.columns:
            raise KeyError(f"'{METRIC_COL}' not found in {f}")
        x = pd.to_numeric(df[METRIC_COL], errors="coerce").dropna()
        x = x.astype(int).clip(lower=0, upper=1).reset_index(drop=True)
        series_dict[step] = x
    return series_dict

# --- Step 4: Align all step series by minimum length (paired across steps) ---
def align_series(series_dict):
    min_len = min(len(s) for s in series_dict.values())
    print(f"\n🔗 Aligning all series to min length={min_len}")
    aligned = pd.DataFrame({step: s.iloc[:min_len].values for step, s in series_dict.items()})
    aligned = aligned.astype(int)
    return aligned

# --- Helpers for plots ---
def _mapped_steps(steps):
    return [STEP_LABELS.get(s, s) for s in steps]

def _save_sig_heatmap(matrix_signif, steps, out_path, title=None):
    mapped_steps = _mapped_steps(steps)
    mask = np.triu(np.ones_like(matrix_signif, dtype=bool))
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        matrix_signif,
        mask=mask,
        annot=False,
        cmap=sns.color_palette(["lightgrey", "lightcoral"]),
        cbar=False,
        xticklabels=mapped_steps,
        yticklabels=mapped_steps,
        linewidths=0.5,
        linecolor="white"
    )
    # ✅ Explicitly set fontsize for labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=24, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _save_diff_annotated_heatmap(matrix_signif, matrix_diff, steps, out_path, title=None):
    mapped_steps = _mapped_steps(steps)
    mask = np.triu(np.ones_like(matrix_signif, dtype=bool))
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        matrix_signif,
        mask=mask,
        annot=False,
        cmap=sns.color_palette(["lightgrey", "lightcoral"]),
        cbar=False,
        xticklabels=mapped_steps,
        yticklabels=mapped_steps,
        linewidths=0.5,
        linecolor="white"
    )
    # Add text annotations with diffs
    for i in range(len(steps)):
        for j in range(len(steps)):
            if i <= j:
                continue
            val = matrix_diff.iloc[i, j]
            if pd.notna(val):
                ax.text(j + 0.5, i + 0.5, f"{val:+.3f}",
                        ha="center", va="center", fontsize=10)

    # ✅ Explicitly set fontsize for labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=24, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --- Step 5: Cochran’s Q + post-hoc McNemar (+ Holm) + barplot + matrices + heatmaps ---
def run_binary_stats_and_matrices(aligned: pd.DataFrame, outdir):
    # Order columns by STEP_LABELS if present
    ordered_steps = [s for s in STEP_LABELS.keys() if s in aligned.columns]
    aligned = aligned[ordered_steps]
    steps = list(aligned.columns)
    X = aligned.values  # (n, k)

    # Omnibus: Cochran’s Q  (call with a single 2-D array in this statsmodels version)
    q_res = cochrans_q(X)
    stat, p = q_res.statistic, q_res.pvalue
    print(f"\n📊 Cochran's Q statistic={stat:.4f}, p-value={p:.6f}")

    # Per-step proportions
    prop = aligned.mean(axis=0)
    prop_df = pd.DataFrame({
        "step": steps,
        "pretty_step": _mapped_steps(steps),
        "proportion_EXM10": prop.values,
        "n": [aligned.shape[0]] * len(steps)
    })
    os.makedirs(outdir, exist_ok=True)
    prop_df.to_csv(os.path.join(outdir, f"{METRIC_COL}_per_step_proportions.csv"), index=False)

    # --- Barplot of proportions ---
    plt.figure(figsize=(10, 6))
    order_labels = _mapped_steps(steps)
    sns.barplot(x=order_labels, y=prop.values, color="steelblue")
    plt.ylabel("Proportion EXM10")
    plt.xlabel("Step")
    plt.title("EXM10 success rate per step")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{METRIC_COL}_proportions_barplot.pdf"))
    plt.close()

    # Pairwise McNemar (we run these regardless of Q, but you can gate on p<0.05 if preferred)
    pairs = list(itertools.combinations(steps, 2))
    p_values = []
    stats_keep = []  # (a,b,stat,raw_p,n01,n10)
    diffs_keep = []  # (a,b,prop_a - prop_b)

    for a, b in pairs:
        ct = pd.crosstab(aligned[a], aligned[b]).reindex(index=[0,1], columns=[0,1], fill_value=0)
        res = mcnemar(ct, exact=False, correction=True)
        pval = float(np.clip(res.pvalue, 0.0, 1.0))
        p_values.append(pval)
        stats_keep.append((a, b, res.statistic, pval, int(ct.loc[0,1]), int(ct.loc[1,0])))
        diffs_keep.append((a, b, float(prop[a] - prop[b])))

    # Holm-Bonferroni correction
    reject, corrected_p, _, _ = smm.multipletests(p_values, method="holm")

    # Build matrices
    matrix_stat = pd.DataFrame(np.nan, index=steps, columns=steps)
    matrix_rawp = pd.DataFrame(np.nan, index=steps, columns=steps)
    matrix_corrp = pd.DataFrame(np.nan, index=steps, columns=steps)
    matrix_signif = pd.DataFrame(False, index=steps, columns=steps)
    matrix_diff = pd.DataFrame(np.nan, index=steps, columns=steps)

    for (a, b, statv, raw_p, n01, n10), corr_p, rej, (_, _, d) in zip(stats_keep, corrected_p, reject, diffs_keep):
        corr_p = float(np.clip(corr_p, 0.0, 1.0))
        matrix_stat.loc[a, b] = matrix_stat.loc[b, a] = statv
        matrix_rawp.loc[a, b] = matrix_rawp.loc[b, a] = raw_p
        matrix_corrp.loc[a, b] = matrix_corrp.loc[b, a] = corr_p
        matrix_signif.loc[a, b] = matrix_signif.loc[b, a] = bool(rej)
        # Difference in proportions (A−B) with antisymmetry
        matrix_diff.loc[a, b] = d
        matrix_diff.loc[b, a] = -d

    # Save CSVs
    matrix_stat.to_csv(os.path.join(outdir, f"{METRIC_COL}_pairwise_mcnemar_stats.csv"))
    matrix_rawp.to_csv(os.path.join(outdir, f"{METRIC_COL}_pairwise_mcnemar_raw_p.csv"))
    matrix_corrp.to_csv(os.path.join(outdir, f"{METRIC_COL}_pairwise_mcnemar_corrected_p.csv"))
    matrix_signif.to_csv(os.path.join(outdir, f"{METRIC_COL}_pairwise_mcnemar_significance.csv"))
    matrix_diff.to_csv(os.path.join(outdir, f"{METRIC_COL}_pairwise_diff_proportions.csv"))

    print("\n📊 Corrected p-values matrix (head):")
    print(matrix_corrp.round(6).iloc[:5, :5].to_string())

    # --- Heatmap: significance only (same style/colors as before) ---
    _save_sig_heatmap(
        matrix_signif,
        steps,
        os.path.join(outdir, f"{METRIC_COL}_mcnemar_significance_heatmap.pdf"),
        title=None
    )

    # --- Heatmap: significance colors with diff annotations ---
    _save_diff_annotated_heatmap(
        matrix_signif,
        matrix_diff,
        steps,
        os.path.join(outdir, f"{METRIC_COL}_mcnemar_diff_annotated_heatmap.pdf"),
        title=None
    )

# --- Main ---
if __name__ == "__main__":
    result_dirs = [
        "csvResultsS1CodeLlama",
        "csvResultsS1chatGPT",
        "csvResultsS2T5",
        "csvResultsS3CodeLlama",
        "csvResultsS3chatGPT",
        "csvResultsS1OpenNMT",
        "csvResultsS2CodeLlama",
        "csvResultsS2chatGPT",
        "csvResultsS3CodeReviewer",
    ]

    for res_dir in result_dirs:
        Model = res_dir.replace("csvResults", "")
        print(f"\n📂 Processing directory: {res_dir} -> Model={Model}")

        # chatGPT only has TimeLevel for your setup
        if "chatGPT" in Model:
            levels = ["TimeLevel"]
            testset_name = "common_new_test"
        else:
            levels = ["TimeLevel", "ProjectLevel"]
            testset_name = "common_test"

        os.chdir(res_dir)

        for level in levels:
            plot_dir = f"{Model}_{level}_{METRIC_COL}"
            outdir = f"../plots/AllSteps_EXM10/{plot_dir}"
            os.makedirs(outdir, exist_ok=True)

            print(f"\n🚀 Running EXM10 analysis for Model={Model}, level={level}")

            files = load_files_for_level(testset_name, level)
            series_dict = load_all_series_binary(files)
            aligned = align_series(series_dict)

            run_binary_stats_and_matrices(aligned, outdir)

        os.chdir("..")


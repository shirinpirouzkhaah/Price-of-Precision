import os
import glob
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
import statsmodels.stats.multitest as smm
from aeon.visualisation import plot_critical_difference  # ✅ use aeon


# --- Step Mapping for Plotting ---
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

# --- Step 1: Collect CSV files ---
def load_codebleu_data(testset_name, level):
    pattern = f"*_{testset_name}_predictions_with_codeBLEU_10_{level}.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    # filter out skipped step
    files = [f for f in files if "S1_P1plusP4plusP6" not in f]

    print(f"\n📂 Collecting files for testset={testset_name}, level={level}:")
    for f in files:
        print(" -", os.path.basename(f))
    return files

# --- Step 2: Load CSVs and extract steps ---
def extract_step_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    parts = base.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return base



def load_all_series(files):
    series_dict = {}
    for f in files:
        step = extract_step_from_filename(f)
        if step == "S1_P1plusP4plusP6":  # skip explicitly
            continue
        df = pd.read_csv(f, sep=",", engine="c")
        if METRIC_COL not in df.columns:
            raise KeyError(f"'{METRIC_COL}' not found in {f}")
        series_dict[step] = df[METRIC_COL].reset_index(drop=True)
    return series_dict

# --- Step 3: Align all series by min length ---
def align_series(series_dict):
    min_len = min(len(s) for s in series_dict.values())
    print(f"\n🔗 Aligning all series to min length={min_len}")
    aligned = pd.DataFrame({step: s.iloc[:min_len].values for step, s in series_dict.items()})
    return aligned

# --- Step 4 & 5: Friedman test + Post-hoc Wilcoxon + Matrices + Heatmap + CD Diagram ---

def rescale(values, new_min=1, new_max=6):
    old_min, old_max = np.min(values), np.max(values)
    return new_min + (values - old_min) * (new_max - new_min) / (old_max - old_min)

def run_stats_and_matrices(aligned: pd.DataFrame, outdir):

    # order columns by STEP_LABELS order
    ordered_steps = [s for s in STEP_LABELS.keys() if s in aligned.columns]
    aligned = aligned[ordered_steps]

    # Friedman test
    scores = [aligned[step].values for step in aligned.columns]
    stat, p = friedmanchisquare(*scores)
    print(f"\n📊 Friedman test statistic={stat:.4f}, p-value={p:.6f}")

    if p < 0.05:
        print("\n🔍 Running pairwise Wilcoxon signed-rank tests (Holm corrected)...")

        steps = list(aligned.columns)
        pairs = list(itertools.combinations(steps, 2))
        p_values = []
        results = []

        for a, b in pairs:
            w_stat, pval = wilcoxon(aligned[a], aligned[b])
            # clamp to [0,1]
            pval = min(max(pval, 0.0), 1.0)
            p_values.append(pval)
            results.append((a, b, w_stat, pval))

        # Holm-Bonferroni correction
        reject, corrected_p, _, _ = smm.multipletests(p_values, method="holm")

        # Prepare matrices
        matrix_stat = pd.DataFrame(np.nan, index=steps, columns=steps)
        matrix_rawp = pd.DataFrame(np.nan, index=steps, columns=steps)
        matrix_corrp = pd.DataFrame(np.nan, index=steps, columns=steps)
        matrix_signif = pd.DataFrame(False, index=steps, columns=steps)

        for (a, b, w_stat, raw_p), corr_p, rej in zip(results, corrected_p, reject):
            corr_p = min(max(corr_p, 0.0), 1.0)
            matrix_stat.loc[a, b] = matrix_stat.loc[b, a] = w_stat
            matrix_rawp.loc[a, b] = matrix_rawp.loc[b, a] = raw_p
            matrix_corrp.loc[a, b] = matrix_corrp.loc[b, a] = corr_p
            matrix_signif.loc[a, b] = matrix_signif.loc[b, a] = rej

        # Save CSVs
        matrix_stat.to_csv(os.path.join(outdir, f"{METRIC_COL}_pairwise_stats.csv"))
        matrix_rawp.to_csv(os.path.join(outdir, f"{METRIC_COL}_pairwise_raw_p.csv"))
        matrix_corrp.to_csv(os.path.join(outdir, f"{METRIC_COL}_pairwise_corrected_p.csv"))
        matrix_signif.to_csv(os.path.join(outdir, f"{METRIC_COL}_pairwise_significance.csv"))

        print("\n📊 Corrected p-values matrix:")
        print(matrix_corrp.round(6).to_string())

        # Heatmap with mapped labels (lower triangle only)
        mapped_steps = [STEP_LABELS.get(s, s) for s in steps]
        mask = np.triu(np.ones_like(matrix_signif, dtype=bool))  # mask upper triangle

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix_signif,
            mask=mask,
            annot=False,
            cmap=sns.color_palette(["lightgrey", "lightcoral"]),  # non-sig grey, sig red
            cbar=False,
            xticklabels=mapped_steps,
            yticklabels=mapped_steps,
            linewidths=0.5,
            linecolor="white"
        )
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{METRIC_COL}_significance_heatmap.pdf"))
        plt.close()
        print(f"\n💾 Saved matrices and heatmap to {outdir}/")

        print("\n📊 Drawing Critical Difference diagram...")

        scores = aligned.values
        names = [STEP_LABELS.get(s, s) for s in aligned.columns]
        
        fig, ax = plot_critical_difference(
            scores,
            names,
            test="nemenyi",
            alpha=0.05,
        )
        
        # ✅ Make the figure wider and add margins so labels are not cut
        fig.set_size_inches(12, 6)
        plt.subplots_adjust(left=0.15, right=0.95)
        
        # ✅ Increase font sizes for labels and rank values
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(14)  # step labels
        
        for text in ax.texts:
            text.set_fontsize(24)  # rank values
        
        # ✅ Save with tight bounding box to preserve labels
        fig.savefig(
            os.path.join(outdir, f"{METRIC_COL}_critical_difference_diagram.pdf"),
            bbox_inches="tight"
        )
        plt.close(fig)

        
        print(f"💾 Critical Difference diagram saved to {outdir}/")



    else:
        print("ℹ️ No significant difference detected by Friedman test.")


# --- Main ---
if __name__ == "__main__":

    # All your result directories
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

    # Define metrics to analyze
    metrics = ["codeBLEU", "Lev_Ratio"]

    for res_dir in result_dirs:
        # --- Extract model name (remove "csvResults")
        Model = res_dir.replace("csvResults", "")
        print(f"\n📂 Processing directory: {res_dir} -> Model={Model}")

        # --- Special handling for chatGPT
        if "chatGPT" in Model:
            levels = ["TimeLevel"]  # only TimeLevel for chatGPT
            testset_name = "common_new_test"
        else:
            levels = ["TimeLevel", "ProjectLevel"]
            testset_name = "common_test"

        # --- Change working directory into the results folder
        os.chdir(res_dir)

        for level in levels:
            for METRIC_COL in metrics:
                # --- output dir ---
                plot_dir = f"{Model}_{level}_{METRIC_COL}"
                outdir = f"../plots/AllSteps/{plot_dir}"
                os.makedirs(outdir, exist_ok=True)

                print(f"\n🚀 Running analysis for Model={Model}, level={level}, metric={METRIC_COL}")

                # --- load and process ---
                files = load_codebleu_data(testset_name, level)
                series_dict = load_all_series(files)
                aligned = align_series(series_dict)

                # --- stats + plots ---
                run_stats_and_matrices(aligned, outdir)

        # --- Go back to parent after finishing this model
        os.chdir("..")


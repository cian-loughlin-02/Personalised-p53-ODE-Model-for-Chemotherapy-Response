#Step 5: Q3B — Apply the cell line doxorubicin signature to BRCA patients.

#Mirrors 'DepMap_survival.R':

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "Practicals" / "brca_tcga_pan_can_atlas_2018"

#Loading cell line signature coefficients
coefs = pd.read_csv(SCRIPT_DIR / "brca_cell_lines_coefs.csv")
coefs = coefs.set_index("Gene")["Coefficient"]
sig_genes = coefs.index.tolist()
print(f"Signature genes: {len(sig_genes)}")

#Loading BRCA TCGA expression, then normalising, then restricting to signature genes
print("Loading BRCA TCGA expression...")
rna_raw = pd.read_csv(DATA_DIR / "data_mrna_seq_v2_rsem.txt", sep="\t")
rna_raw = rna_raw[rna_raw["Hugo_Symbol"].notna() & (rna_raw["Hugo_Symbol"] != "")]
rna_raw = rna_raw.drop_duplicates(subset="Hugo_Symbol")
rna_raw = rna_raw.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"])
rna     = rna_raw.T.copy()
rna[rna < 0] = 0.0

#Normalising by the mean 
gene_means = rna.mean(axis=0)
rna = rna.div(gene_means.replace(0, np.nan)).fillna(0.0)
rna.index.name = "SAMPLE_ID"

#Keeping only signature genes that are  present in TCGA
available_sig = [g for g in sig_genes if g in rna.columns]
missing = len(sig_genes) - len(available_sig)
if missing:
    print(f"Warning: {missing} signature genes absent from TCGA data; skipping them")

coefs_available = coefs[available_sig]
X_expr = rna[available_sig].copy()
print(f"Expression matrix for signature: {X_expr.shape}")

#Loading clinical data
patients = pd.read_csv(SCRIPT_DIR / "brca_params_full.csv")

#Computing survival signature score (linear combination)
X_expr = X_expr.reset_index()
data_full = patients.merge(X_expr, on="SAMPLE_ID", how="inner")

sig_matrix = data_full[available_sig].values
sig_coefs  = coefs_available.values
data_full["surv_sig"] = sig_matrix @ sig_coefs
print(f"Samples with complete data: {len(data_full)}")
print(f"Signature score — mean: {data_full['surv_sig'].mean():.3f}, "
      f"std: {data_full['surv_sig'].std():.3f}")

# Cox regression on sig score
df_cox = data_full[["OS_MONTHS", "SURV_STATUS", "surv_sig"]].copy()
cph = CoxPHFitter()
cph.fit(df_cox, duration_col="OS_MONTHS", event_col="SURV_STATUS",
        show_progress=False)
print("\nCox model — signature score:")
print(cph.summary[["exp(coef)", "exp(coef) lower 95%",
                    "exp(coef) upper 95%", "p"]].round(4))
print(f"C-index: {cph.concordance_index_:.3f}")

#Forrest plot 
hr    = float(cph.summary["exp(coef)"].iloc[0])
hr_lo = float(cph.summary["exp(coef) lower 95%"].iloc[0])
hr_hi = float(cph.summary["exp(coef) upper 95%"].iloc[0])
pval  = float(cph.summary["p"].iloc[0])

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.errorbar([hr], [0], xerr=[[hr - hr_lo], [hr_hi - hr]], fmt="s",
            color="steelblue", capsize=6, ms=8)
ax.axvline(1.0, color="black", linestyle="--", lw=1)
ax.set_yticks([0])
ax.set_yticklabels(["Signature score"])
ax.set_xlabel("Hazard ratio (95% CI)")
ax.set_title(f"Cox model — Cell line signature\nHR={hr:.2f} ({hr_lo:.2f}–{hr_hi:.2f}), "
             f"p={pval:.4f}, C={cph.concordance_index_:.3f}")
plt.tight_layout()
fig.savefig(SCRIPT_DIR / "depmap_forest_plot.png", dpi=150)
plt.show()

#KM threshold scan
sig_vals = data_full["surv_sig"].values
uni_vals = np.sort(np.unique(sig_vals))
min_n    = max(10, int(0.10 * len(sig_vals)))
km_pvals = np.ones(len(uni_vals))
for idx in range(1, len(uni_vals) - 1):
    thr = uni_vals[idx]
    high = sig_vals > thr
    low  = ~high
    if high.sum() < min_n or low.sum() < min_n:
        continue
    res = logrank_test(
        data_full.loc[high, "OS_MONTHS"], data_full.loc[low, "OS_MONTHS"],
        data_full.loc[high, "SURV_STATUS"], data_full.loc[low, "SURV_STATUS"]
    )
    km_pvals[idx] = res.p_value

opt_i   = int(np.argmin(km_pvals))
opt_thr = uni_vals[opt_i]
km_pval = km_pvals[opt_i]

high_mask = sig_vals > opt_thr
n_high = high_mask.sum()
n_low  = (~high_mask).sum()
print(f"\nKM optimal threshold: {opt_thr:.4f}")
print(f"Log-rank p-value: {km_pval:.4f}  |  High: {n_high}  Low: {n_low}")

#KM plot
fig, ax = plt.subplots(figsize=(8, 5))

kmf = KaplanMeierFitter()
kmf.fit(data_full.loc[high_mask, "OS_MONTHS"],
        data_full.loc[high_mask, "SURV_STATUS"],
        label=f"High signature score\n(n={n_high})")
kmf.plot_survival_function(ax=ax, ci_show=True)

kmf2 = KaplanMeierFitter()
kmf2.fit(data_full.loc[~high_mask, "OS_MONTHS"],
         data_full.loc[~high_mask, "SURV_STATUS"],
         label=f"Low signature score\n(n={n_low})")
kmf2.plot_survival_function(ax=ax, ci_show=True)

ax.set_xlabel("Overall survival time (months)", fontsize=13)
ax.set_ylabel("Survival probability", fontsize=13)
ax.set_title(f"BRCA — Cell line doxorubicin signature\n"
             f"Log-rank p = {km_pval:.4f}", fontsize=12)
ax.text(0.98, 0.98, f"Log-rank p = {km_pval:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(SCRIPT_DIR / "depmap_km_curve.png", dpi=150)
plt.show()
print("Saved KM plot.")

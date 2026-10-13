#Step 4: Q3A — Build a doxorubicin sensitivity signature from breast cancer cell lines.
#Mirrors 'DepMap_signature.R':

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from scipy.stats import pearsonr
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
DEPMAP_DIR  = SCRIPT_DIR.parent / "depmap_data"
DATA_DIR    = SCRIPT_DIR.parent / "Practicals" / "brca_tcga_pan_can_atlas_2018"

TARGET_DRUG = "doxorubicin"
LINEAGE     = "Breast"

#Loading the DepMap cell line metadata 
print("Loading cell line metadata...")
metadata = pd.read_csv(DEPMAP_DIR / "Model.csv")
breast_ids = metadata.loc[metadata["OncotreeLineage"] == LINEAGE, "ModelID"].tolist()
print(f"Found {len(breast_ids)} {LINEAGE} cell lines in DepMap metadata")

#Finding doxorubicin col in PRISM treatment info 
print("Loading PRISM treatment info...")
ti = pd.read_csv(DEPMAP_DIR / "primary-screen-replicate-collapsed-treatment-info.csv")
dox_rows = ti[ti["name"].str.lower().str.contains(TARGET_DRUG, na=False)]
if dox_rows.empty:
    raise ValueError(f"'{TARGET_DRUG}' not found in PRISM treatment info 'name' column.")
dox_cols = dox_rows["column_name"].tolist()
print(f"Doxorubicin column(s) in PRISM: {dox_cols}")

#Loadin PRISM LFC (reading only dox columns + ModelID to save memory)
print("Loading PRISM LFC data...")
lfc_full = pd.read_csv(
    DEPMAP_DIR / "primary-screen-replicate-collapsed-logfold-change.csv",
    index_col=0,
    usecols=lambda c: c == "Unnamed: 0" or c in dox_cols,
)
lfc_full.index.name = "ModelID"

#Averaging across doxorubicin cols if there are multiple doses
lfc_full["dox_lfc"] = lfc_full[dox_cols].mean(axis=1)

#Filtering to breast lines
lfc_breast = lfc_full.loc[lfc_full.index.isin(breast_ids), ["dox_lfc"]].dropna()
print(f"Breast cell lines with doxorubicin LFC: {len(lfc_breast)}")
if len(lfc_breast) == 0:
    raise ValueError("No breast lines found in PRISM LFC. Check ModelID format in Model.csv.")

#Loading CCLE expression and thenmerging
print("Loading CCLE transcriptomics (290 MB, takes ~20 s)...")
META_COLS = ["Unnamed: 0", "SequencingID", "ModelConditionID",
             "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
expr_raw = pd.read_csv(
    DEPMAP_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
)
expr = (expr_raw[expr_raw["IsDefaultEntryForModel"] == "Yes"]
        .set_index("ModelID")
        .drop(columns=[c for c in META_COLS if c in expr_raw.columns], errors="ignore"))
expr.index.name = "ModelID"
#Striping Entrez IDs
expr.columns = expr.columns.str.replace(r" \(\d+\)$", "", regex=True)
print(f"Expression matrix: {expr.shape[0]} cell lines x {expr.shape[1]} genes")

merged = lfc_breast.join(expr, how="inner")
print(f"After merging LFC + expression: {len(merged)} cell lines")

#Restricting to genes present in the BRCA TCGA
print("Loading BRCA TCGA gene list...")
tcga_rna   = pd.read_csv(DATA_DIR / "data_mrna_seq_v2_rsem.txt", sep="\t",
                          usecols=["Hugo_Symbol"])
tcga_genes = set(tcga_rna["Hugo_Symbol"].dropna().unique())

common_genes = [g for g in expr.columns if g in tcga_genes and g in merged.columns]
print(f"Common genes (CCLE and TCGA): {len(common_genes)}")

X_df = merged[common_genes].fillna(merged[common_genes].median())
X_full = X_df.values
y      = merged["dox_lfc"].values

#Pre-filtering by correlation 
TOPN = 200
cors = np.array([pearsonr(X_full[:, j], y)[0] for j in range(X_full.shape[1])])
cors = np.where(np.isnan(cors), 0.0, cors)
top_idx = np.argsort(np.abs(cors))[-TOPN:]
top_idx.sort()
print(f"Retaining top {TOPN} genes by |Pearson r| (range: "
      f"{np.abs(cors)[top_idx].min():.3f}–{np.abs(cors)[top_idx].max():.3f})")

filtered_genes = [common_genes[i] for i in top_idx]
X = X_full[:, top_idx]

#LASSO regression 
TARGET_GENES = 10
#This will give  the sparsest model that still selects exactly TARGET_GENES feats
alphas = np.logspace(-3, 1, 50)[::-1]  # 10.0 → 0.001
chosen_alpha = alphas[0]
for a in alphas:
    nnz = (Lasso(alpha=a, max_iter=10_000).fit(X, y).coef_ != 0).sum()
    if nnz >= TARGET_GENES:
        chosen_alpha = a
        break
print(f"Chosen alpha (target {TARGET_GENES} genes): {chosen_alpha:.6f}")
lasso = Lasso(alpha=chosen_alpha, max_iter=10_000)
lasso.fit(X, y)

#Extracting non-zero coefficients
coef_series = pd.Series(lasso.coef_, index=filtered_genes)
non_zero    = coef_series[coef_series != 0].sort_values(key=abs, ascending=False)
print(f"\nNon-zero coefficients: {len(non_zero)}")
print(non_zero.head(10))

out_coefs = pd.DataFrame({"Gene": non_zero.index, "Coefficient": non_zero.values})
coefs_path = SCRIPT_DIR / "brca_cell_lines_coefs.csv"
out_coefs.to_csv(coefs_path, index=False)
print(f"\nSaved {len(non_zero)} coefficients to {coefs_path}")

#producng the actual vs predicted plot
y_pred = lasso.predict(X)
r = float(np.corrcoef(y, y_pred)[0, 1])
print(f"Pearson r (actual vs predicted LFC): {r:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y, y_pred, alpha=0.7, color="steelblue", s=40)
lims = [min(y.min(), y_pred.min()) - 0.2, max(y.max(), y_pred.max()) + 0.2]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
m, b = np.polyfit(y, y_pred, 1)
x_line = np.linspace(lims[0], lims[1], 100)
ax.plot(x_line, m * x_line + b, "k-", lw=1, label=f"Fit (r={r:.3f})")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(f"Actual {TARGET_DRUG} LFC (PRISM)", fontsize=12)
ax.set_ylabel("Predicted LFC", fontsize=12)
ax.set_title(f"Actual vs Predicted — {TARGET_DRUG.title()} (Breast lines)\n"
             f"LASSO, r = {r:.3f}", fontsize=12)
ax.legend()
plt.tight_layout()
fig.savefig(SCRIPT_DIR / "depmap_actual_vs_predicted.png", dpi=150)
print("Saved scatter plot.")

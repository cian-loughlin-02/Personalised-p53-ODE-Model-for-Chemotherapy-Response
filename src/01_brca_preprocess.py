
# Step 1: Preprocessin the  BRCA TCGA data for p53 ODE personalisation.

#Mirrors the R script 'preproc and optimal input.R' 

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "Practicals" / "brca_tcga_pan_can_atlas_2018"
OUT_DIR = Path(__file__).parent

P53_GENES = ["ATM", "CHEK2", "HIPK2", "MDM2", "PPM1D", "SIAH1", "TP53", "WSB1"]

# Loading the raw RSEM expression 
print("Loading RSEM expression...")
rna_raw = pd.read_csv(DATA_DIR / "data_mrna_seq_v2_rsem.txt", sep="\t")

# Dropping any rows with empty/ duplicate Hugo_Symbol 
rna_raw = rna_raw[rna_raw["Hugo_Symbol"].notna() & (rna_raw["Hugo_Symbol"] != "")]
rna_raw = rna_raw.drop_duplicates(subset="Hugo_Symbol")
rna_raw = rna_raw.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"])

#Transposing 
rna = rna_raw.T.copy()

#Replacing he negatives with 0 
rna[rna < 0] = 0.0

# Normalising each gene by its mean across every sample
gene_means = rna.mean(axis=0)
rna = rna.div(gene_means.replace(0, np.nan)).fillna(0.0)

print(f"Expression matrix: {rna.shape[0]} samples × {rna.shape[1]} genes")

#Extracting the 8 p53-pathway genes
missing = [g for g in P53_GENES if g not in rna.columns]
if missing:
    raise ValueError(f"Missing p53 genes in expression data: {missing}")

rna_p53 = rna[P53_GENES].copy()
rna_p53.index.name = "SAMPLE_ID"
rna_p53 = rna_p53.reset_index()

#Loading clinical data
print("Loading clinical data...")

#skipping the 4 comment lines before the actual header
clin_patient = pd.read_csv(
    DATA_DIR / "data_clinical_patient.txt", sep="\t", skiprows=4
)
clin_sample = pd.read_csv(
    DATA_DIR / "data_clinical_sample.txt", sep="\t", skiprows=4
)

clin_patient = clin_patient[["PATIENT_ID", "OS_MONTHS", "OS_STATUS"]]
clin_sample = clin_sample[["PATIENT_ID", "SAMPLE_ID", "CANCER_TYPE_DETAILED",
                             "TMB_NONSYNONYMOUS"]]

clin = clin_sample.merge(clin_patient, on="PATIENT_ID", how="left")

# Drop rows missing OS info 
clin = clin[clin["OS_STATUS"].notna() & (clin["OS_STATUS"] != "")]
clin = clin[clin["OS_MONTHS"].notna()]

#Keeping DECEASED or LIVING pats with OS > 1 month
mask = (
    (clin["OS_STATUS"] == "1:DECEASED") |
    ((clin["OS_STATUS"] == "0:LIVING") & (clin["OS_MONTHS"] > 1))
)
clin = clin[mask].copy()

#Theb inary survival status (1 = event (death), 0 = censored)
clin["SURV_STATUS"] = (clin["OS_STATUS"] == "1:DECEASED").astype(int)

print(f"Clinical data: {len(clin)} patients after filtering "
      f"({clin['SURV_STATUS'].sum()} deceased)")

#Aligning samples
common_samples = set(clin["SAMPLE_ID"]) & set(rna_p53["SAMPLE_ID"])
print(f"Samples with both expression and clinical data: {len(common_samples)}")

clin = clin[clin["SAMPLE_ID"].isin(common_samples)].copy()
rna_p53 = rna_p53[rna_p53["SAMPLE_ID"].isin(common_samples)].copy()

#Merging and saving 
data_full = clin.merge(rna_p53, on="SAMPLE_ID", how="inner")

#Column order 
col_order = (
    ["SAMPLE_ID", "PATIENT_ID", "CANCER_TYPE_DETAILED", "TMB_NONSYNONYMOUS",
     "OS_MONTHS", "OS_STATUS", "SURV_STATUS"]
    + P53_GENES
)
data_full = data_full[col_order]

out_path = OUT_DIR / "brca_params_full.csv"
data_full.to_csv(out_path, index=False)
print(f"\nSaved {len(data_full)} samples to {out_path}")
print("\nGene expression summary (should all be > 0, centred around 1.0):")
print(data_full[P53_GENES].describe().loc[["mean", "min", "max"]].round(3))

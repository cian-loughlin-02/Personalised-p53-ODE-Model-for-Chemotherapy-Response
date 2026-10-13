#Step 6: Q2 — Can the p53 ODE model predict doxorubicin sensitivity in breast cell lines?

import numpy as np
import pandas as pd
import scipy.integrate
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEPMAP_DIR = SCRIPT_DIR.parent / "depmap_data"

#Loading all the cell line metadata and filtering o breast cancer
print("Loading cell line metadata...")
metadata = pd.read_csv(DEPMAP_DIR / "Model.csv")
breast_ids = metadata.loc[metadata["OncotreeLineage"] == "Breast", "ModelID"].tolist()

#Loading PRISM doxorubicin LFC 
print("Loading PRISM doxorubicin LFC...")
ti = pd.read_csv(DEPMAP_DIR / "primary-screen-replicate-collapsed-treatment-info.csv")
dox_col = ti.loc[
    ti["name"].str.lower().str.contains("doxorubicin", na=False), "column_name"
].tolist()[0]

lfc = pd.read_csv(
    DEPMAP_DIR / "primary-screen-replicate-collapsed-logfold-change.csv",
    index_col=0,
    usecols=lambda c: c == "Unnamed: 0" or c == dox_col,
)
lfc.index.name = "ModelID"
lfc_breast = (
    lfc.loc[lfc.index.isin(breast_ids)]
    .dropna()
    .rename(columns={dox_col: "dox_lfc"})
)
print(f"Breast cell lines with doxorubicin LFC: {len(lfc_breast)}")

#Loading CCLE expression and  extracting 8 p53 genes
P53_GENES = ["ATM", "CHEK2", "HIPK2", "MDM2", "PPM1D", "SIAH1", "TP53", "WSB1"]

print("Loading CCLE expression (290 MB)...")
META_COLS = ["Unnamed: 0", "SequencingID", "ModelConditionID",
             "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
expr_raw = pd.read_csv(
    DEPMAP_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
)
expr = (
    expr_raw[expr_raw["IsDefaultEntryForModel"] == "Yes"]
    .set_index("ModelID")
    .drop(columns=[c for c in META_COLS if c in expr_raw.columns], errors="ignore")
)
expr.columns = expr.columns.str.replace(r" \(\d+\)$", "", regex=True)

missing = [g for g in P53_GENES if g not in expr.columns]
if missing:
    raise ValueError(f"p53 genes missing from CCLE: {missing}")

expr_p53 = expr[P53_GENES].copy()

#Merging LFC with expression
cl_data = lfc_breast.join(expr_p53, how="inner").dropna()
print(f"Cell lines with both LFC and expression: {len(cl_data)}")

#normalising expression across cell lines (mean)
gene_means = expr_p53.mean(axis=0)
for g in P53_GENES:
    cl_data[g] = cl_data[g] / gene_means[g] if gene_means[g] > 0 else 0.0

print("\nExpression summary after normalisation (should centre around 1.0):")
print(cl_data[P53_GENES].describe().loc[["mean", "min", "max"]].round(3))

#Loading ODE parameters 
PARAMS_FILE = SCRIPT_DIR / "p53_parameters_summaryTable.txt"
est_params_df = pd.read_csv(PARAMS_FILE, sep=" ", decimal=".", index_col=False)
est_params_df = est_params_df.set_index("objFun")
p = est_params_df.iloc[0]  # best parameter set

kpatm_dox   = float(p["kpatm_dox"]);   jpatm_dox   = float(p["jpatm_dox"])
kdpatm_dox  = float(p["kdpatm_dox"]);  jdpatm_dox  = float(p["jdpatm_dox"])
kpchk2      = float(p["kpchk2"]);      jpchk2      = float(p["jpchk2"])
kdpchk2     = float(p["kdpchk2"]);     jdpchk2     = float(p["jdpchk2"])
kpp53_CHK2  = float(p["kpp53_CHK2"]);  jpp53_CHK2  = float(p["jpp53_CHK2"])
kpp53_ATM   = float(p["kpp53_ATM"]);   jpp53_ATM   = float(p["jpp53_ATM"])
kdpp53a     = float(p["kdpp53a"]);      jdpp53a     = float(p["jdpp53a"])
kpp53a      = float(p["kpp53a"]);       jpp53a      = float(p["jpp53a"])
kdpp53k     = float(p["kdpp53k"]);      jdpp53k     = float(p["jdpp53k"])
kpsiah1     = float(p["kpsiah1"]);      jpsiah1     = float(p["jpsiah1"])
kdpsiah1    = float(p["kdpsiah1"]);     jdpsiah1    = float(p["jdpsiah1"])
ksmdm21     = float(p["ksmdm21"]);      jsmdm21     = float(p["jsmdm21"])
kswip11     = float(p["kswip11"]);      jswip11     = float(p["jswip11"])
kdp53       = float(p["kdp53"]);        kdp53a      = float(p["kdp53a"])
jup53       = float(p["jup53"]);        jup53s      = float(p["jup53s"])
juhipk2     = float(p["juhipk2"])

hcmdm21 = 2;   hcwip11 = 2;   kdhipk2 = 1.5;  kdp53m  = 0.077
kdwip1m = 0.66; kdwip1 = 2.0;  kdmdm2m = 0.7;  kdmdm2  = 2.0
kuhipk2 = 0.17; hcdeg  = 2;    kalg    = 0.1

p_p53i_0  = 0.1;   p_MDM2m_0 = 0.103; p_p53m_0 = 0.1
WIP1m_0   = 0.0594; WIP1_0   = 0.0277; MDM2_0  = 1.0
p_ATM_0   = 1.0;   p_SIAH_0  = 1.0;   p_HIPK2_0 = 0.018
p_CHK2    = 1.0;   icZero    = 0.0

ksp530  = p_p53m_0 * kdp53m
kpsmdm2 = MDM2_0 * kdmdm2 / p_MDM2m_0
h_MDM2_0 = MDM2_0**hcdeg / (MDM2_0**hcdeg + jup53**hcdeg)
kpsp53  = kdp53 * p_p53i_0 / p_p53m_0 * h_MDM2_0
h_p53i_0 = (kalg * p_p53i_0)**hcmdm21 / (jsmdm21**hcmdm21 + (kalg * p_p53i_0)**hcmdm21)
kswip10 = kdwip1m * WIP1m_0
kpswip1 = kdwip1 * WIP1_0 / WIP1m_0
p_kshipk2 = (kdhipk2 * p_HIPK2_0
             + kuhipk2 * p_SIAH_0 * p_HIPK2_0 / (juhipk2 + p_HIPK2_0))

#setpatientparams and ODE
x0 = [0.0] * 15
DDR = 0.0

def setcelllineparams(row):
    global ksp530, kpsmdm2, kpsp53, kpswip1, x0, kshipk2, kswip10, ksmdm20

    ATM_0  = row["ATM"]  * p_ATM_0
    p53m   = row["TP53"] * p_p53m_0
    ksp530 = p53m * kdp53m

    MDM2m  = row["MDM2"] * p_MDM2m_0
    MDM2   = MDM2m * kpsmdm2 / kdmdm2
    h_MDM2 = MDM2**hcdeg / (MDM2**hcdeg + jup53**hcdeg)
    p53i   = p53m * kpsp53 / kdp53 / h_MDM2
    h_p53i = (kalg * p53i)**hcmdm21 / (jsmdm21**hcmdm21 + (kalg * p53i)**hcmdm21)
    ksmdm20 = MDM2m * kdmdm2m - ksmdm21 * h_p53i

    WIP1m_i  = row["PPM1D"] * WIP1m_0
    kswip10  = WIP1m_i * kdwip1m
    SIAH_i   = 0.5 * (row["SIAH1"] + row["WSB1"]) * p_SIAH_0

    kshipk2  = row["HIPK2"] * p_kshipk2
    disc = (SIAH_i**2 * kuhipk2**2
            + 2 * SIAH_i * juhipk2 * kdhipk2 * kuhipk2
            - 2 * SIAH_i * kshipk2 * kuhipk2
            + juhipk2**2 * kdhipk2**2
            + 2 * juhipk2 * kdhipk2 * kshipk2
            + kshipk2**2)
    HIPK2_i = ((kshipk2 - SIAH_i * kuhipk2 - juhipk2 * kdhipk2 + np.sqrt(disc))
               / (2 * kdhipk2))

    if ksmdm20 < 0:
        ksmdm20 = 1e-9
    if HIPK2_i < 0:
        HIPK2_i = 1e-9

    x0 = [ATM_0, icZero, p53m, p53i, icZero, icZero,
           SIAH_i, icZero, p_HIPK2_0, WIP1m_0, WIP1_0,
           p_CHK2, icZero, MDM2m, MDM2_0]


def f(x, t):
    ATM, ATMp, p53m, p53i, p53s15, p53s46 = x[0:6]
    SIAH1, SIAH1p, HIPK2, WIP1m, WIP1 = x[6:11]
    CHK2, CHK2p, MDM2m, MDM2 = x[11:15]
    ku = np.heaviside(t, 1)

    v1  = kpatm_dox  * ATM  * DDR * ku / (jpatm_dox  + ATM)
    v2  = kdpatm_dox * 1.0  * ATMp / (jdpatm_dox + ATMp)
    v3  = kpchk2     * ATMp * CHK2  / (jpchk2  + CHK2)
    v4  = kdpchk2    * WIP1 * CHK2p / (jdpchk2 + CHK2p)
    v5  = kpp53_CHK2 * CHK2p * p53i / (jpp53_CHK2 + p53i)
    v6  = kpp53_ATM  * ATMp  * p53i / (jpp53_ATM  + p53i)
    v7  = kdpp53a    * WIP1  * p53s15 / (jdpp53a + p53s15)
    v8  = kpp53a     * HIPK2 * p53s15 / (jpp53a  + p53s15)
    v9  = kdpp53k    * p53s46 / (jdpp53k + p53s46)
    v10 = kpsiah1    * ATMp  * SIAH1  / (jpsiah1  + SIAH1)
    v11 = kdpsiah1   * SIAH1p / (jdpsiah1 + SIAH1p)

    p53alg    = kalg * p53i + p53s15 + p53s46
    p53s15tot = p53s15 + p53s46

    v12 = ksp530
    v13 = ksmdm20
    v14 = ksmdm21 * p53alg**hcmdm21 / (jsmdm21**hcmdm21 + p53alg**hcmdm21)
    v15 = kswip10
    v16 = kswip11 * p53s15tot**hcwip11 / (jswip11**hcwip11 + p53s15tot**hcwip11)
    v21 = kpsp53  * p53m
    v22 = kshipk2
    v23 = kpswip1 * WIP1m
    v24 = kpsmdm2 * MDM2m
    v25 = kdp53  * p53i   * MDM2**hcdeg / (MDM2**hcdeg + jup53**hcdeg)
    v26 = kdp53a * p53s15 * MDM2**hcdeg / (MDM2**hcdeg + jup53s**hcdeg)
    v27 = kdp53a * p53s46 * MDM2**hcdeg / (MDM2**hcdeg + jup53s**hcdeg)
    v28 = kdhipk2 * HIPK2
    v29 = kdp53m  * p53m
    v30 = kdwip1m * WIP1m
    v31 = kdwip1  * WIP1
    v32 = kdmdm2m * MDM2m
    v33 = kdmdm2  * MDM2
    v36 = kuhipk2 * SIAH1 * HIPK2 / (juhipk2 + HIPK2)

    return [
        -v1 + v2,
         v1 - v2,
         v12 - v29,
        -v5 - v6 + v7 + v21 - v25,
         v5 + v6 - v7 - v8 + v9 - v26,
         v8 - v9 - v27,
        -v10 + v11,
         v10 - v11,
         v22 - v28 - v36,
         v15 + v16 - v30,
         v23 - v31,
        -v3 + v4,
         v3 - v4,
         v13 + v14 - v32,
         v24 - v33,
    ]


#DDR sweep over the cell lines
DDRvec    = np.linspace(0.01, 1.0, num=10)
tspan     = np.linspace(0, 50, num=10)
col_names = ["DDR_" + f"{d:.3f}" for d in DDRvec]

print(f"\nRunning ODE for {len(cl_data)} cell lines...")
s15_rows, s46_rows = [], []

for idx, (model_id, row) in enumerate(cl_data.iterrows()):
    setcelllineparams(row)
    s15_row, s46_row = [], []
    for ddr_val in DDRvec:
        DDR = ddr_val
        sol = scipy.integrate.odeint(f, x0, tspan, mxstep=5000)
        s15_row.append(sol[-1, 4])
        s46_row.append(sol[-1, 5])
    s15_rows.append(s15_row)
    s46_rows.append(s46_row)

s15_df = pd.DataFrame(s15_rows, columns=col_names, index=cl_data.index)
s46_df = pd.DataFrame(s46_rows, columns=col_names, index=cl_data.index)
y_lfc  = cl_data["dox_lfc"].values

#Scan DDR doses (to find best correlation with doxorubicin LFC )
print("\nCorrelation of ODE output vs doxorubicin LFC across DDR doses:")
print(f"{'Marker':<10} {'DDR dose':<12} {'Pearson r':>10} {'p-value':>12}")
print("-" * 46)

best = {"marker": None, "col": None, "r": 0.0, "p": 1.0, "vals": None}

for label, df in [("p53pS15", s15_df), ("p53pS46", s46_df)]:
    for col in col_names:
        r, pval = pearsonr(df[col].values, y_lfc)
        print(f"{label:<10} {col:<12} {r:>10.4f} {pval:>12.4f}")
        if abs(r) > abs(best["r"]):
            best = {"marker": label, "col": col, "r": r, "p": pval,
                    "vals": df[col].values}

print(f"\nBest: {best['marker']} at {best['col']} — r = {best['r']:.4f}, p = {best['p']:.4f}")

#Scatter plot (ODE prediction vs measured doxorubicin LFC)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(best["vals"], y_lfc, color="steelblue", alpha=0.8, s=60,
           edgecolors="white", linewidth=0.5)

m, b = np.polyfit(best["vals"], y_lfc, 1)
x_line = np.linspace(best["vals"].min(), best["vals"].max(), 100)
ax.plot(x_line, m * x_line + b, "k-", lw=1.5,
        label=f"r = {best['r']:.3f}, p = {best['p']:.3f}")

# Labellung each point with cell line name
cl_names = metadata.set_index("ModelID")["StrippedCellLineName"]
for i, (mid, lfc_val) in enumerate(zip(cl_data.index, y_lfc)):
    name = cl_names.get(mid, mid)
    ax.annotate(name, (best["vals"][i], lfc_val),
                fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

ax.set_xlabel(f"Predicted {best['marker']} ({best['col']})", fontsize=12)
ax.set_ylabel("Doxorubicin LFC (PRISM)", fontsize=12)
ax.set_title(f"Q2: ODE-predicted {best['marker']} vs doxorubicin sensitivity\n"
             f"Breast cancer cell lines (n={len(cl_data)})", fontsize=11)
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(SCRIPT_DIR / "q2_ode_vs_lfc.png", dpi=150)
print("Saved scatter plot to q2_ode_vs_lfc.png")

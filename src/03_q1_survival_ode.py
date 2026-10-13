#Step 3: Q1 — Does the p53 ODE model predict BRCA patient survival?

#Mirrors 'preproc and optimal input.R' (second half)
#Produces separate analyses for p53pS15 and p53pS46


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

def analyse_marker(ddr_csv: str, marker_label: str):
    """Run Cox scan + KM plot for one p53 phosphorylation marker."""
    print(f"\n{'='*60}")
    print(f"Analysing {marker_label}")
    print("="*60)

    #Loading and merging data 
    patients = pd.read_csv(SCRIPT_DIR / "brca_params_full.csv")
    ddr_data = pd.read_csv(SCRIPT_DIR / ddr_csv)

    data = patients.merge(ddr_data, on="SAMPLE_ID", how="inner").dropna()
    print(f"Samples after merge: {len(data)}")

    ddr_cols = [c for c in ddr_data.columns if c.startswith("DDR_")]

    #scanninng DDR doses: Cox p-value for each 
    # Mirrors R loop: coxph(data_clinical_surv ~ DDR_X.XXX, data=data_p53)
    pvals = []
    hrs   = []
    for col in ddr_cols:
        df_fit = data[["OS_MONTHS", "SURV_STATUS", col]].copy()
        cph = CoxPHFitter()
        cph.fit(df_fit, duration_col="OS_MONTHS", event_col="SURV_STATUS",
                show_progress=False)
        pvals.append(float(cph.summary["p"].iloc[0]))
        hrs.append(float(cph.summary["exp(coef)"].iloc[0]))

    pvals = np.array(pvals)
    opt_i = int(np.argmin(pvals))
    opt_col   = ddr_cols[opt_i]
    opt_pval  = pvals[opt_i]
    opt_hr    = hrs[opt_i]

    print(f"\nOptimal DDR dose: {opt_col}")
    print(f"Cox p-value: {opt_pval:.4f}  |  HR: {opt_hr:.3f}")

    #Full Cox summary at the optimal dose
    df_fit = data[["OS_MONTHS", "SURV_STATUS", opt_col]].copy()
    cph_opt = CoxPHFitter()
    cph_opt.fit(df_fit, duration_col="OS_MONTHS", event_col="SURV_STATUS",
                show_progress=False)
    print("\nCox model summary:")
    print(cph_opt.summary[["exp(coef)", "exp(coef) lower 95%",
                             "exp(coef) upper 95%", "p"]].round(4))
    print(f"C-index: {cph_opt.concordance_index_:.3f}")

    #Scanning thresholds for optimal KM split
    pathway_data = data[opt_col].values
    uni_vals = np.sort(np.unique(pathway_data))
    min_n = max(10, int(0.10 * len(pathway_data)))

    km_pvals = np.ones(len(uni_vals))
    for idx in range(1, len(uni_vals) - 1):
        thr = uni_vals[idx]
        group = pathway_data > thr
        if group.sum() < min_n or (~group).sum() < min_n:
            continue
        result = logrank_test(
            data.loc[group,  "OS_MONTHS"], data.loc[~group, "OS_MONTHS"],
            data.loc[group,  "SURV_STATUS"], data.loc[~group, "SURV_STATUS"]
        )
        km_pvals[idx] = result.p_value

    opt_km_i   = int(np.argmin(km_pvals))
    opt_thr    = uni_vals[opt_km_i]
    km_pval    = km_pvals[opt_km_i]

    high_mask = pathway_data > opt_thr
    n_high = high_mask.sum()
    n_low  = (~high_mask).sum()
    print(f"\nKM optimal threshold: {opt_thr:.4f}")
    print(f"Log-rank p-value: {km_pval:.4f}  |  High: {n_high}  Low: {n_low}")

    #producng KM plot 
    fig, ax = plt.subplots(figsize=(8, 5))

    kmf = KaplanMeierFitter()
    kmf.fit(data.loc[high_mask,  "OS_MONTHS"],
            data.loc[high_mask,  "SURV_STATUS"],
            label=f"High {marker_label} (n={n_high})")
    kmf.plot_survival_function(ax=ax, ci_show=True)

    kmf2 = KaplanMeierFitter()
    kmf2.fit(data.loc[~high_mask, "OS_MONTHS"],
             data.loc[~high_mask, "SURV_STATUS"],
             label=f"Low {marker_label} (n={n_low})")
    kmf2.plot_survival_function(ax=ax, ci_show=True)

    ax.set_xlabel("Overall survival time (months)", fontsize=13)
    ax.set_ylabel("Survival probability", fontsize=13)
    ax.set_title(f"BRCA — {marker_label} at DDR dose {opt_col}\n"
                 f"Log-rank p = {km_pval:.4f}", fontsize=12)
    ax.text(0.98, 0.98, f"Log-rank p = {km_pval:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=11)
    plt.tight_layout()

    out_file = SCRIPT_DIR / f"km_{marker_label.replace(' ', '_')}.png"
    plt.savefig(out_file, dpi=150)
    plt.show()
    print(f"Saved KM plot to {out_file}")

    #p-value scan plot
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.semilogy(uni_vals, km_pvals, "o-", ms=4)
    ax2.axhline(0.05, color="red", linestyle="--", label="p=0.05")
    ax2.set_xlabel(f"{marker_label} level")
    ax2.set_ylabel("Log-rank p-value (log scale)")
    ax2.set_title(f"Threshold scan — {marker_label}")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(SCRIPT_DIR / f"threshold_scan_{marker_label.replace(' ', '_')}.png",
                 dpi=150)
    plt.show()


if __name__ == "__main__":
    analyse_marker("p53s15DR_brca.csv", "p53 pS15")
    analyse_marker("p53s46DR_brca.csv", "p53 pS46")

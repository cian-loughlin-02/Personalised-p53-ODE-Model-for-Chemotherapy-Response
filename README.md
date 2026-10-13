# Personalised-p53-ODE-Model-for-Chemotherapy-Response

A mechanistic systems-biology model of the p53 DNA-damage-response network, personalised per patient from gene expression, and tested as a prognostic and drug-response predictor against data-driven baselines. MSc module project (AI for Personalised Medicine, UCD).

## Overview

Doxorubicin and other DNA-damaging chemotherapies act through the p53 response, so the dynamics of that network are a natural, interpretable basis for predicting outcome. This project personalises a 15-state ordinary-differential-equation (ODE) model of the p53 DNA-damage response by scaling reference parameters with each patient's expression of eight pathway genes, then asks three questions:

1. Does simulated p53 phosphorylation predict **patient survival**? (TCGA BRCA, n = 1,036)
2. Does it predict **doxorubicin sensitivity** in cell lines? (PRISM / CCLE)
3. Does a data-driven **cell-line signature transfer to patients**?

## Key results

- Simulated **p53-pS15 at maximum DNA damage** was significantly associated with worse overall survival (Cox p = 0.029; Kaplan–Meier log-rank p = 0.0008).
- p53-pS15 showed a (non-significant) trend with doxorubicin sensitivity in 26 cell lines (r = −0.36).
- A LASSO 10-gene cell-line sensitivity signature fit well in-sample (r = 0.87) but **did not** transfer to patient survival — an informative negative illustrating the cell-line-to-patient gap, and a case where the mechanistic model, with its biological prior, held up better on small data.

## Approach

- **Mechanistic:** numerically integrate the personalised ODE system across a range of DNA-damage inputs; read out terminal phosphorylated-p53 concentrations.
- **Data-driven baseline:** LASSO regression for a transcriptomic drug-sensitivity signature.
- **Evaluation:** Cox proportional hazards, Kaplan–Meier with log-rank optimal thresholding, Pearson correlation.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── 01_brca_preprocess.py ode_model.py   # load / prepare TCGA BRCA expression + clinical data
│   ├── 02_p53_brca_ode.py                   # the personalised p53 DNA-damage-response ODE model
│   ├── 03_q1_survival_ode.py                # Q1: simulated p53 phosphorylation vs patient survival
│   ├── 04_q3a_depmap_signature.py           # Q2: ODE output vs cell-line doxorubicin sensitivity
│   ├── 05_q3b_depmap_survival.py            # Q3: LASSO drug-sensitivity signature from cell lines
│   └── 06_q2_cellline_ode.py                # Q3: does that signature transfer to patient survival
└── results/
```

## Data

TCGA (cBioPortal/GDC), PRISM Repurposing, and CCLE are publicly available and not committed here. ODE reference parameters follow Fey et al. (the published calibration).

## Tech stack

Python · SciPy (ODE integration) · lifelines (Cox / Kaplan–Meier, log-rank) · scikit-learn (LASSO) · NumPy · pandas · Matplotlib

## Acknowledgements

MSc AI for Medicine and Medical Research, University College Dublin (AI for Personalised Medicine).

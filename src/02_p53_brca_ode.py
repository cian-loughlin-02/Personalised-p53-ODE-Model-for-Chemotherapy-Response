#Step 2: Running the personalised p53 ODE model on all BRCA patients.

#Adapted from p53_model_updated.ipynb 

import numpy as np
import pandas as pd
import scipy.integrate
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PARAMS_FILE = SCRIPT_DIR / "p53_parameters_summaryTable.txt"
PATIENTS_FILE = SCRIPT_DIR / "brca_params_full.csv"

#Loading pre-fitted ODE params 
est_params_df = pd.read_csv(PARAMS_FILE, sep=" ", decimal=".", index_col=False)
est_params_df = est_params_df.set_index("objFun")

#Using the best parameter set 
i_selectedrun = 0
p = est_params_df.iloc[i_selectedrun]

#Pre-extracting al params as plain Python floats to fix bottleneck
kpatm_dox   = float(p["kpatm_dox"])
jpatm_dox   = float(p["jpatm_dox"])
kdpatm_dox  = float(p["kdpatm_dox"])
jdpatm_dox  = float(p["jdpatm_dox"])
kpchk2      = float(p["kpchk2"])
jpchk2      = float(p["jpchk2"])
kdpchk2     = float(p["kdpchk2"])
jdpchk2     = float(p["jdpchk2"])
kpp53_CHK2  = float(p["kpp53_CHK2"])
jpp53_CHK2  = float(p["jpp53_CHK2"])
kpp53_ATM   = float(p["kpp53_ATM"])
jpp53_ATM   = float(p["jpp53_ATM"])
kdpp53a     = float(p["kdpp53a"])
jdpp53a     = float(p["jdpp53a"])
kpp53a      = float(p["kpp53a"])
jpp53a      = float(p["jpp53a"])
kdpp53k     = float(p["kdpp53k"])
jdpp53k     = float(p["jdpp53k"])
kpsiah1     = float(p["kpsiah1"])
jpsiah1     = float(p["jpsiah1"])
kdpsiah1    = float(p["kdpsiah1"])
jdpsiah1    = float(p["jdpsiah1"])
ksmdm21     = float(p["ksmdm21"])
jsmdm21     = float(p["jsmdm21"])
kswip11     = float(p["kswip11"])
jswip11     = float(p["jswip11"])
kdp53       = float(p["kdp53"])
kdp53a      = float(p["kdp53a"])
jup53       = float(p["jup53"])
jup53s      = float(p["jup53s"])
juhipk2     = float(p["juhipk2"])

#Fixed params
hcmdm21  = 2
hcwip11  = 2
kdhipk2  = 1.5
kdp53m   = 0.077
kdwip1m  = 0.66
kdwip1   = 2.0
kdmdm2m  = 0.7
kdmdm2   = 2.0
kuhipk2  = 0.17
hcdeg    = 2
kalg     = 0.1

#Fixed initial conditions 
p_p53i_0   = 0.1
p_MDM2m_0  = 0.103
p_p53m_0   = 0.1
WIP1m_0    = 0.0594
WIP1_0     = 0.0277
MDM2_0     = 1.0
p_ATM_0    = 1.0
p_SIAH_0   = 1.0
p_HIPK2_0  = 0.018
p_CHK2     = 1.0
icZero     = 0.0

#derived reference params
ksp530  = p_p53m_0 * kdp53m
kpsmdm2 = MDM2_0 * kdmdm2 / p_MDM2m_0
h_MDM2_0 = (MDM2_0**hcdeg) / (MDM2_0**hcdeg + jup53**hcdeg)
kpsp53   = kdp53 * p_p53i_0 / p_p53m_0 * h_MDM2_0
h_p53i_0 = (kalg * p_p53i_0)**hcmdm21 / (jsmdm21**hcmdm21 + (kalg * p_p53i_0)**hcmdm21)
kswip10  = kdwip1m * WIP1m_0
kpswip1  = kdwip1 * WIP1_0 / WIP1m_0
p_kshipk2 = (kdhipk2 * p_HIPK2_0
              + kuhipk2 * p_SIAH_0 * p_HIPK2_0 / (juhipk2 + p_HIPK2_0))

#Patient data
NB_params = pd.read_csv(PATIENTS_FILE)
print(f"Loaded {len(NB_params)} BRCA patients")

#Patient personlaisation
def setpatientparams(i):
    global ksp530, kpsmdm2, kpsp53, kpswip1, x0, kshipk2, kswip10, ksmdm20

    ATM_0  = NB_params.loc[i, "ATM"]  * p_ATM_0
    CHK2_0 = NB_params.loc[i, "CHEK2"] * p_CHK2  

    # p53 mRNA → basal synthesis rate
    p53m  = NB_params.loc[i, "TP53"] * p_p53m_0
    ksp530 = p53m * kdp53m

    # MDM2 steady state
    MDM2m = NB_params.loc[i, "MDM2"] * p_MDM2m_0
    MDM2  = MDM2m * kpsmdm2 / kdmdm2
    h_MDM2 = MDM2**hcdeg / (MDM2**hcdeg + jup53**hcdeg)
    p53i   = p53m * kpsp53 / kdp53 / h_MDM2
    h_p53i = (kalg * p53i)**hcmdm21 / (jsmdm21**hcmdm21 + (kalg * p53i)**hcmdm21)
    ksmdm20 = MDM2m * kdmdm2m - ksmdm21 * h_p53i

    # WIP1 (PPM1D gene)
    WIP1m_i  = NB_params.loc[i, "PPM1D"] * WIP1m_0
    kswip10  = WIP1m_i * kdwip1m
    WIP1_i   = WIP1m_i * kpswip1 / kdwip1  

    # SIAH1/WSB1 average > SIAH activity
    SIAH_i = 0.5 * (NB_params.loc[i, "SIAH1"] + NB_params.loc[i, "WSB1"]) * p_SIAH_0

    # HIPK2 (analytical steady stat) 
    kshipk2 = NB_params.loc[i, "HIPK2"] * p_kshipk2
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


# the ODE system 
def f(x, t):
    ATM, ATMp, p53m, p53i, p53s15, p53s46 = x[0:6]
    SIAH1, SIAH1p, HIPK2, WIP1m, WIP1 = x[6:11]
    CHK2, CHK2p, MDM2m, MDM2 = x[11:15]

    ku = np.heaviside(t, 1)

    v1  = kpatm_dox  * ATM  * DDR * ku / (jpatm_dox  + ATM)
    v2  = kdpatm_dox * 1.0  * ATMp / (jdpatm_dox + ATMp)   # WIP1a = 1 (SY5Y)
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
        -v1 + v2,          # ATM
         v1 - v2,          # ATMp
         v12 - v29,        # p53m
        -v5 - v6 + v7 + v21 - v25,   # p53i
         v5 + v6 - v7 - v8 + v9 - v26,  # p53s15
         v8 - v9 - v27,   # p53s46
        -v10 + v11,        # SIAH1
         v10 - v11,        # SIAH1p
         v22 - v28 - v36, # HIPK2
         v15 + v16 - v30, # WIP1m
         v23 - v31,        # WIP1
        -v3 + v4,          # CHK2
         v3 - v4,          # CHK2p
         v13 + v14 - v32, # MDM2m
         v24 - v33,        # MDM2
    ]


#sweeping the DDR dose 
DDRvec = np.linspace(0.01, 1.0, num=10)
tspan  = np.linspace(0, 50, num=10)

col_names = ["DDR_" + f"{d:.3f}" for d in DDRvec]

p53s15DR_rows = []
p53s46DR_rows = []

DDR = 0.0  # set per iteration 

for i in range(len(NB_params)):
    setpatientparams(i)
    s15_row = []
    s46_row = []
    for ddr_val in DDRvec:
        DDR = ddr_val  
        sol = scipy.integrate.odeint(f, x0, tspan, mxstep=5000)
        s15_row.append(sol[-1, 4])   # p53s15 at t=50
        s46_row.append(sol[-1, 5])   # p53s46 at t=50
    p53s15DR_rows.append(s15_row)
    p53s46DR_rows.append(s46_row)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(NB_params)} patients done")

p53s15DR_df = pd.DataFrame(p53s15DR_rows, columns=col_names)
p53s46DR_df = pd.DataFrame(p53s46DR_rows, columns=col_names)

p53s15DR_df["SAMPLE_ID"] = NB_params["SAMPLE_ID"].values
p53s46DR_df["SAMPLE_ID"] = NB_params["SAMPLE_ID"].values

p53s15DR_df.to_csv(SCRIPT_DIR / "p53s15DR_brca.csv", index=False)
p53s46DR_df.to_csv(SCRIPT_DIR / "p53s46DR_brca.csv", index=False)
print("\nSaved p53s15DR_brca.csv and p53s46DR_brca.csv")

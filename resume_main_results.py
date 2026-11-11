"""Resume main-results regeneration from do_ris_difPHAP onward.

do_ris_difL (difL_ris_ite.csv) already completed before the MOSEK ill-posed
crash at P_HAP=10 dBm, so it is skipped here. Everything else is regenerated
with the MOSEK->SCS fallback now in place.
"""
import time
from new_func.alldata import (
    do_ris_difPHAP, do_ris_difPHAP_L10, do_ris_Q, do_ris_if_Q,
    difM, difW, difL, difP, difBeta,
)

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log("resume main results (skip do_ris_difL)")

# --- convergence / iteration data (P_HAP variation at L=32 and L=10) ---
log("do_ris_difPHAP")
do_ris_difPHAP(max_iter=20, L=32)
log("do_ris_difPHAP_L10")
do_ris_difPHAP_L10(max_iter=20, L=10)

# --- Q (optimal placement) ---
log("do_ris_Q")
do_ris_Q(max_iter=20)
log("do_ris_if_Q")
do_ris_if_Q(max_iter=20)

# --- difM ---
for name in ["do_irs_difM", "do_irs_if_difM", "do_no_irs_na_difM",
             "do_no_irs_if_difM", "do_fix_irs_difM", "do_fix_irs_if_difM"]:
    log(f"difM.{name}")
    getattr(difM, name)()

# --- difW ---
for name in ["do_irs_difW", "do_irs_if_difW", "do_no_irs_na_difW",
             "do_no_irs_if_difW", "do_fix_irs_difW", "do_fix_irs_if_difW"]:
    log(f"difW.{name}")
    getattr(difW, name)()

# --- difP ---
for name in ["do_irs_difP", "do_irs_if_difP", "do_no_irs_na_difP",
             "do_no_irs_if_difP", "do_fix_irs_difP", "do_fix_irs_if_difP"]:
    log(f"difP.{name}")
    getattr(difP, name)()

# --- difBeta ---
for name in ["do_irs_difBeta", "do_irs_if_difBeta", "do_no_irs_na_difBeta",
             "do_no_irs_if_difBeta", "do_fix_irs_difBeta", "do_fix_irs_if_difBeta"]:
    log(f"difBeta.{name}")
    getattr(difBeta, name)()

# --- difL (L 对性能的影响，放最后实验) ---
for name in ["do_irs_difL", "do_irs_if_difL", "do_no_irs_na_difL",
             "do_no_irs_if_difL", "do_fix_irs_difL", "do_fix_irs_if_difL"]:
    log(f"difL.{name}")
    getattr(difL, name)()

log("RESUME MAIN-RESULTS DONE")

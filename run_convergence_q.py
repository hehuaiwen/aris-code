"""Re-run the convergence (ite) and Q experiments that were still stale (Sep 18,
old W=1 Hz / pre-rescale scale). Run from the 代码/ directory.

Each result is written to new_data/*.csv via alldata.savedata.
"""
import time
from new_func.alldata import (
    do_ris_difL, do_ris_difPHAP, do_ris_difPHAP_L10, do_ris_Q, do_ris_if_Q,
)

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


log("start convergence + Q regeneration")

log("do_ris_difL  (L=8..32, max_iter=20)")
do_ris_difL(max_iter=20)

log("do_ris_difPHAP  (P_HAP=10..30, L=32, max_iter=20)")
do_ris_difPHAP(max_iter=20, L=32)

log("do_ris_difPHAP_L10  (P_HAP=10..30, L=10, max_iter=20)")
do_ris_difPHAP_L10(max_iter=20, L=10)

log("do_ris_Q")
do_ris_Q(max_iter=20)

log("do_ris_if_Q")
do_ris_if_Q(max_iter=20)

log("ALL CONVERGENCE+Q DONE")

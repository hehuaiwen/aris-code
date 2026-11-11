"""Re-run only the Q (UAV placement) experiments with the corrected do_ris_Q /
do_ris_if_Q that store the optimized UAV position (not the power allocation).

Run from the 代码/ directory.
"""
import time
from new_func.alldata import do_ris_Q, do_ris_if_Q

t0 = time.time()
print(f"[{time.time() - t0:7.1f}s] do_ris_Q", flush=True)
do_ris_Q(max_iter=20, M=5)

print(f"[{time.time() - t0:7.1f}s] do_ris_if_Q", flush=True)
do_ris_if_Q(max_iter=20, M=5)

print(f"[{time.time() - t0:7.1f}s] Q DONE", flush=True)

import time, numpy as np
import new_func.algorithms as alg

for b2 in [3.5, 4.0]:
    t0 = time.time()
    irs = alg.AR_IRS()
    irs.bcd_optimization(max_iter=8, M=5, beta2=b2)
    Q = irs.Q
    print(f"[beta2={b2}] final UAV = {Q}  ({time.time()-t0:.1f}s)", flush=True)

import numpy as np, time
import new_func.algorithms as alg

# Test different HAP altitudes at the ORIGINAL beta2=2.0
for hz in [20.0, 10.0, 0.0]:
    alg.W_s = np.array([0.0, 0.0, hz])   # override module global
    t0 = time.time()
    irs = alg.AR_IRS()
    irs.bcd_optimization(max_iter=8, M=5, beta2=2.0)
    Q = irs.Q
    print(f"[HAP z={hz:4.1f}] final UAV = ({Q[0]:7.2f}, {Q[1]:7.2f}, {Q[2]:6.2f})  ({time.time()-t0:.0f}s)", flush=True)

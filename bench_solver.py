"""Quick benchmark of the optimized solver: time bcd_optimization at fixed L.
Usage: python3 -u bench_solver.py <L> [max_iter]
Compare wall time to pre-optimization numbers (runtime_scaling: L=16 -> 217s, L=24 -> 1047s at OUTER=5).
"""
import sys
import time
import numpy as np
from new_func.algorithms import AR_IRS

L = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MAX_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 5

irs = AR_IRS()
t0 = time.perf_counter()
P_list, tau_list = irs.bcd_optimization(max_iter=MAX_ITER, L=L, M=4, P_HAP=30)
wall = time.perf_counter() - t0
final_tau = tau_list[-1] if tau_list else None
print(f"RESULT L={L} max_iter={MAX_ITER} wall={wall:.2f}s final_tau={final_tau}")

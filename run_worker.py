"""Run a batch of sweep methods sequentially in one worker process.

Usage: python3 -u run_worker.py difM.do_irs_difM difW.do_irs_difW ...

Each argument is "<ClassName>.<method_name>".  ClassName is one of difM/difW/
difL/difP/difBeta (imported from new_func.alldata).  Methods are called in the
order given.  Progress + wall time per task are printed to stdout for monitoring.
"""
import sys
import time
from new_func.alldata import difM, difW, difL, difP, difBeta

_CLASSES = {"difM": difM, "difW": difW, "difL": difL, "difP": difP, "difBeta": difBeta}

tasks = sys.argv[1:]
if not tasks:
    sys.exit("no tasks given")

t0 = time.time()
print(f"[worker {len(tasks)} tasks] {tasks}", flush=True)

for i, name in enumerate(tasks, 1):
    cls_name, meth = name.split(".")
    cls = _CLASSES[cls_name]
    t1 = time.time()
    print(f"[{time.time() - t0:7.1f}s] START {name} ({i}/{len(tasks)})", flush=True)
    getattr(cls, meth)()
    print(f"[{time.time() - t0:7.1f}s] DONE  {name} ({time.time() - t1:6.1f}s)", flush=True)

print(f"[worker] ALL DONE in {time.time() - t0:.1f}s", flush=True)

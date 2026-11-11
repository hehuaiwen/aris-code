#!/bin/bash
set -u
cd "/Users/huaiwen/电子科技大学中山学院/坚果云/坚果云/科研/研究生指导/2023届-黄凤/论文1/wireless network返修/code/aris-code"
LOG="/Users/huaiwen/电子科技大学中山学院/坚果云/坚果云/科研/研究生指导/2023届-黄凤/论文1/wireless network返修/code/aris-code/pipeline_newgeometry.log"
echo "=== new-geometry pipeline start $(date) ===" >> "$LOG"

run() {
    local name="$1"; shift
    echo "=== $name $(date) ===" >> "$LOG"
    python3 -u "$@" >> "$LOG" 2>&1
    echo "$name exit=$? at $(date)" >> "$LOG"
}

# 1) Comments 2-4 (scalar model, fast)
run "scalar_rician_model.py" scalar_rician_model.py

# 2) Comment 5 (runtime scaling; wall-clock is geometry-independent but re-run for consistency)
run "runtime_scaling.py" runtime_scaling.py

# 3) Comments 6-8 (full vector model)
run "vector_airs_model.py" vector_airs_model.py

# 4) rank-one verification (1.py, 20 seeds)
run "1.py" 1.py

# 5) main results (difM/difW/difL/difP/difBeta/Q)
run "run_main_results.py" run_main_results.py

echo "=== new-geometry pipeline done $(date) ===" >> "$LOG"

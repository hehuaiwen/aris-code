#!/bin/bash
# Sequential experiment pipeline for Comments 6-8 + rank-one + main results.
# Waits for the runtime_scaling.py process to finish (avoids MOSEK contention),
# then runs the remaining scripts in order. All output is appended to a log.
set -u

cd "/Users/huaiwen/电子科技大学中山学院/坚果云/坚果云/科研/研究生指导/2023届-黄凤/论文1/wireless network返修/code/aris-code"

LOG="/Users/huaiwen/电子科技大学中山学院/坚果云/坚果云/科研/研究生指导/2023届-黄凤/论文1/wireless network返修/code/aris-code/pipeline.log"

echo "=== pipeline start $(date) ===" >> "$LOG"

# --- wait for runtime_scaling (PID 32506) to exit ---
echo "waiting for runtime_scaling (PID 32506) ..." >> "$LOG"
while kill -0 32506 2>/dev/null; do
    sleep 30
done
echo "runtime_scaling finished at $(date)" >> "$LOG"

# --- Comments 6-8 full-model robustness / ablation / latency ---
echo "=== vector_airs_model.py $(date) ===" >> "$LOG"
python3 -u vector_airs_model.py >> "$LOG" 2>&1

# --- rank-one SDP relaxation test ---
echo "=== 1.py (rank-one) $(date) ===" >> "$LOG"
python3 -u 1.py >> "$LOG" 2>&1

# --- main results regeneration ---
echo "=== run_main_results.py $(date) ===" >> "$LOG"
python3 -u run_main_results.py >> "$LOG" 2>&1

echo "=== pipeline done $(date) ===" >> "$LOG"

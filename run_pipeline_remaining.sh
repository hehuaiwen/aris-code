#!/bin/bash
set -u
cd "/Users/huaiwen/电子科技大学中山学院/坚果云/坚果云/科研/研究生指导/2023届-黄凤/论文1/wireless network返修/code/代码"
LOG="/Users/huaiwen/电子科技大学中山学院/坚果云/坚果云/科研/研究生指导/2023届-黄凤/论文1/wireless network返修/code/代码/pipeline_remaining.log"
echo "=== remaining pipeline start $(date) ===" >> "$LOG"
echo "=== 1.py (rank-one) $(date) ===" >> "$LOG"
python3 -u 1.py >> "$LOG" 2>&1
echo "1.py exit=$? at $(date)" >> "$LOG"
echo "=== run_main_results.py $(date) ===" >> "$LOG"
python3 -u run_main_results.py >> "$LOG" 2>&1
echo "run_main_results.py exit=$? at $(date)" >> "$LOG"
echo "=== remaining pipeline done $(date) ===" >> "$LOG"

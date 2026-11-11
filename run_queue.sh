#!/usr/bin/env bash
# Remaining local experiments, ONE AT A TIME (low peak memory). Every CV fold is checkpointed
# under results/runs/<exp>/folds/, so re-running this script after an interruption resumes.
PY=.venv/Scripts/python
step() { echo "=== $(date +%H:%M) START $1"; shift; "$@"; echo "=== $(date +%H:%M) END rc=$?"; }
step expC          $PY -m src.experiments.expC_cqi                   >> logs/expC.log 2>&1
step expB_b1       $PY -m src.experiments.expB_s1_s2 b1              >> logs/expB_b1.log 2>&1
step expA_ablation $PY -m src.experiments.expA_s1_fusion ablations   >> logs/expA_ablations.log 2>&1
step expB_ablation $PY -m src.experiments.expB_s1_s2 ablations       >> logs/expB_ablations.log 2>&1
echo QUEUE_DONE

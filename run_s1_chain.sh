#!/usr/bin/env bash
# Sequential queue for the S1-based experiments (A, B) and their ablations.
set -x
PY=.venv/Scripts/python
$PY -m src.experiments.expA_s1_fusion main   > logs/expA_main.log 2>&1
$PY -m src.experiments.expA_s1_fusion cnn    > logs/expA_cnn.log 2>&1
$PY -m src.experiments.expB_s1_s2 b2         > logs/expB_b2.log 2>&1
$PY -m src.experiments.expB_s1_s2 b1         > logs/expB_b1.log 2>&1
$PY -m src.experiments.expA_s1_fusion ablations > logs/expA_ablations.log 2>&1
$PY -m src.experiments.expB_s1_s2 ablations  > logs/expB_ablations.log 2>&1
echo CHAIN_DONE

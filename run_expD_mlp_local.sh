#!/usr/bin/env bash
# Experiment D, MLP family, on the local CPU (1 thread). Other families: notebooks/colab_expD_gpu.ipynb
PY=.venv/Scripts/python
for st in lr lam seeds atom; do $PY -m src.experiments.expD_smellnet sweep --models mlp --stage $st --threads 1; done
echo EXPD_MLP_DONE

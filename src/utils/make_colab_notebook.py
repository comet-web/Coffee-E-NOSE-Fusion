"""Builds notebooks/colab_expD_gpu.ipynb and the code bundle it needs (expD_bundle.zip).

Run locally: python -m src.utils.make_colab_notebook
Then in Colab (GPU runtime) run the notebook top to bottom and upload expD_bundle.zip when asked.
"""
import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
SMELLNET_COMMIT = "dde1eb01a859c13789d2b226398a542f0777fc42"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [
    md("# Experiment D on a Colab GPU: SmellNet sensor + GC-MS fusion\n\n"
       "This notebook runs the same script as the local repository (`src/experiments/expD_smellnet.py`) "
       "on a GPU, because the LSTM and Transformer runs take hours per run on the project laptop's CPU.\n\n"
       "**Protocol (leakage-free):** train on SmellNet days 1-4, choose learning rate / lambda / checkpoint on day 5, "
       "test once on day 6. At test time the models only ever see sensor windows. GC-MS is used during training (D1, D2) "
       "or as the fixed prototypes of all 50 classes (D1 prediction), never as the true class's vector.\n\n"
       "Runtime → Change runtime type → **GPU**. Results are written to Google Drive after every run, so a disconnect "
       "only loses the run in progress; re-running a cell skips finished runs."),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## 1. Mount Drive and unpack the code bundle\nUpload `expD_bundle.zip` (made locally by "
       "`python -m src.utils.make_colab_notebook`). It contains only code, no data."),
    code("from google.colab import drive, files\n"
         "drive.mount('/content/drive')\n"
         "import os, zipfile\n"
         "WORK = '/content/drive/MyDrive/coffee-enose-fusion'\n"
         "os.makedirs(WORK, exist_ok=True)\n"
         "if not os.path.exists(f'{WORK}/src'):\n"
         "    up = files.upload()\n"
         "    zipfile.ZipFile(next(iter(up))).extractall(WORK)\n"
         "%cd {WORK}\n!ls"),
    md("## 2. Get Source 2: SmellNet code (pinned commit) and data (Hugging Face)\n"
       "Only `base_data/` and `gcms_processed/` are downloaded, as in the local run."),
    code(f"!test -d data/raw/s2/SmellNet-code || git clone -q https://github.com/MIT-MI/SmellNet.git data/raw/s2/SmellNet-code\n"
         f"!cd data/raw/s2/SmellNet-code && git checkout -q {SMELLNET_COMMIT} && git rev-parse HEAD\n"
         "!python -I src/utils/download_s2.py data/raw/s2/hf\n"
         "!find data/raw/s2/hf/base_data -name '*.csv' | wc -l"),
    md("## 3. Sweeps\nStage order matters: `lr` (D0, D1 over the LR grid, seed 42) → `lam` (D2 lambda grid at D0's LR) → "
       "`seeds` (two more seeds at the selected settings) → `atom` (Xatom GC-MS encoding, AB12).\n\n"
       "By default this runs CNN, LSTM and Transformer (the MLP runs are done locally). Add `mlp` to `MODELS` to run all four on the GPU."),
    code("MODELS = 'cnn lstm transformer'  # the MLP family is run locally on CPU\n"
         "for stage in ['lr', 'lam', 'seeds']:\n"
         "    !python -m src.experiments.expD_smellnet sweep --models {MODELS} --stage {stage}"),
    code("# AB12 (Xspec vs Xatom), Transformer only to save time\n"
         "!python -m src.experiments.expD_smellnet sweep --models transformer --ws 100 --ps 25 --stage atom"),
    md("## 4. Download the run files\nCopy the resulting `results/runs/expD/` folder into the local repository "
       "(same path), then run `python -m src.experiments.expD_smellnet collect` locally."),
    code("!cd results/runs && zip -qr /content/expD_runs.zip expD && ls -la /content/expD_runs.zip\n"
         "files.download('/content/expD_runs.zip')"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"provenance": []},
                                                "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "colab_expD_gpu.ipynb"
nbf.write(nb, out)

bundle = ROOT / "expD_bundle.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for f in (ROOT / "src").rglob("*.py"):
        z.write(f, f.relative_to(ROOT))
    for f in [ROOT / "docs" / "smellnet_table2_reported.csv"]:
        if f.exists():
            z.write(f, f.relative_to(ROOT))
print("wrote", out, "and", bundle)

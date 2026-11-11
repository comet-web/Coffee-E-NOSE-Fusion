# Data notes: inventory, checks and discrepancies

Downloaded on 2026-11-11. Everything here was checked against the files themselves (brief, Hard Rule 5: *trust the data over the brief*). Figures and tables referred to are produced by the EDA notebooks (`notebooks/01_eda_s{1,2,3}.ipynb`).

## Source 1 (S1): Mendeley e-nose coffee dataset

| Item | Value |
|---|---|
| URL / DOI | https://data.mendeley.com/datasets/7spd6fpvyk/1, 10.17632/7spd6fpvyk.1 (DataCite: "Dataset: Electronic Nose for Quality Control of Colombian Coffee through the Detection of Defects in 'Cup Tests'", creator Rodríguez Gamboa, J. C., Mendeley, 2020) |
| File | `dataset.zip`, 383,784 bytes, SHA-256 `953ce75b8b278676471044d0ff0853ead1bdb8e9717cc33122e653d975a9d09b` (matches the hash published by the Mendeley API) |
| Layout | `dataset/{AQ,HQ,LQ}_Coffee/<CLASS>_<NN>.txt` |
| Format | Tab-delimited text **with a leading tab** on every row, CRLF line endings, no header. `numpy.loadtxt` reads it directly. |
| Shape | Every one of the 58 files is exactly **300 rows × 8 columns**. No missing values, no constant channels. Values 3.68–83.44 (kΩ). |
| Class counts | **HQ 20, AQ 10, LQ 28** (58 in total). Imbalanced: majority-class rate = 28/58 = 48.3 %. |
| Column order | Taken from the dataset description: SP-12A, SP-31, TGS-813, TGS-842, SP-AQ3, TGS-823, ST-31, TGS-800. The files carry no header, so the order cannot be checked from the data. |

**Signal structure.** Every recording has a flat pre-exposure baseline for roughly the first 30 s. Resistance then drops sharply when the headspace reaches the sensors (the median time of minimum resistance is about 50–60 s for 6 of 8 sensors; SP-12A and TGS-813 reach their minimum later, around 210–245 s) and recovers slowly afterwards. So the 300 s file covers the **measurement** phase of the paper's 15-minute cycle, not the concentration or purge phases. We use the first **20 s** as the baseline segment in all baseline-dependent features.

**Confirmed confound (important for interpretation).** The pre-exposure baseline resistance differs between classes (e.g. SP-12A mean of the first 20 s: HQ 46.4 kΩ, AQ 33.9 kΩ, LQ 38.7 kΩ). The odour response starts 32–34 s after the start of every one of the 58 recordings (first 5 % resistance drop of the six fast sensors; per-recording values in `results/s1_response_onset.csv`). Yet a model that sees only the first 30 s classifies 85.5 % correctly (ablation AB2), and baseline-normalised signals cost the low-level model 16–18 points (AB3). So part of the class information in the public data is the sensors' state *before* the sample arrives (session, ambient conditions, drift or carry-over), not the coffee's aroma. SP-12A never drops 5 % below its starting level in 32 of 58 recordings, so it hardly responds to coffee, yet it is worth 12 points to the low-level model (AB1). Measurement dates and order are not in the release, so the cause cannot be identified.

**Mismatch with the reference article.** The article reports 16 measurements in 4 groups (analysis 1) and 35 measurements in 9 categories (analysis 2). The public data has 58 measurements in 3 classes (HQ/AQ/LQ), and the mapping between the article's categories and HQ/AQ/LQ is not documented. All comparisons with the article are therefore approximate.

## Source 2 (S2): SmellNet

| Item | Value |
|---|---|
| Code | https://github.com/MIT-MI/SmellNet, cloned at commit `dde1eb01a859c13789d2b226398a542f0777fc42` into `data/raw/s2/SmellNet-code/` |
| Data | Hugging Face `DeweiFeng/SmellNet` (`DeweiFeng/smell-net` redirects to it with HTTP 307), revision `45405fec862fcd98cec03215b5e0d1477a377bec`. Per-file SHA-256 in `data/raw/s2/hf/MANIFEST.json`. |
| Downloaded | `base_data/` (300 CSV, 3.6 MB) and `gcms_processed/` (5 files, 31 MB). **Not downloaded:** `gcms_data/` (2.6 GB of raw FooDB dumps), `mixture_data/` and `text_data/` (out of scope; brief, Hard Rule 1). |
| Paper version used | arXiv 2506.00239**v5** (13 Apr 2026), whose first page reads "Published as a conference paper at ICLR 2026". The OpenReview PDF (forum `aUnheo6zFD`) could not be fetched (bot check), so v5 is used as the camera-ready text. Table 2 values were transcribed from this PDF into `docs/smellnet_table2_reported.csv`. |

**Base data, verified.** 50 substance folders in each of `training/` and `testing/`. Files are named `<substance>_<day>.csv`. Days 1–5 are in `training/` and day 6 in `testing/` (one file per substance per day, 300 sessions in total). Columns: `NO2, C2H5OH, VOC, CO, Alcohol, LPG` (6 channels, matching the paper). Values are integer raw readings (min/max per channel: NO2 13–775, C2H5OH 39–863, VOC 26–953, CO 705–1006, Alcohol 0–42, LPG 2–507). **Session length: median 600 rows, range 517–867** (the brief's "about 600" holds for the median). The `Alcohol` channel has a baseline of 0 in some sessions, so a plain baseline ratio x/x0 is undefined there (relevant to Experiment B).

**Response polarity.** S2 readings *rise* when the odour is present (median rise over the first 300 s exceeds the median drop on every channel), whereas S1 *resistance falls*. For S1 + S2 harmonisation we therefore use S1 conductance (G = 1/R), which rises with gas.

**GC-MS, verified.** `gcms_food_vectors.csv`: 50 rows (one per ingredient, all 50 base substances present) × 460 features `f0…f459`, the binned spectral encoding **Xspec**. `gcms_element.csv`: 50 × 17 elemental-composition columns (C, Ca, Cl, Co, Cr, Fe, H, K, Mg, Mn, N, Na, O, P, S, V, Zn), the coarse **Xatom** encoding. **Both encodings are available, so ablation AB12 is possible.** The vectors are class-level priors (one per ingredient), not per-sample measurements.

**`run.py` CLI, verified by reading the source** (`models/run.py`, `build_parser`): `--models {mlp,cnn,lstm,transformer}`, `--contrastive {off,on}`, `--gradients {0,25,50}`, `--window-sizes {50,100,500}`, `--epochs` (default 50), `--batch-size` (64), `--lr` (1e-3), `--seed` (42), `--train-dir/--test-dir/--real-test-dir`, `--gcms-csv`, `--no-standardize`, `--stride` (default w/2), `--fft`, `--ablate-channels`, `--held-out-day`, `--leave-one-channel-training`. So the draft's claimed settings (windows 50/100/500, differencing 0/25/50) do exist as CLI choices. The paper itself only uses w ∈ {50, 100} and p ∈ {0, 25}.

**Protocol discrepancy in SmellNet itself.** `run.py` trains for a fixed number of epochs and evaluates directly on day 6; there is no validation split in the code. The paper's Sec. 5.2 says "we report the final checkpoint for each configuration", while App. G.2 says the final checkpoint of each run is evaluated "on the validation set" to choose among learning rates, without saying how that validation set is built. Our Experiment D therefore uses an explicit protocol: train on days 1–4, select learning rate / λ / α / checkpoint on **day 5**, test once on day 6. This uses one day less training data than the paper.

**Window counts (our protocol).** w=50, p=0: 4,523 train / 1,138 val / 1,133 test windows. w=100, p=25: 2,008 / 504 / 502.

## Source 3 (S3): CQI coffee-quality-database

| Item | Value |
|---|---|
| URL | https://github.com/jldbc/coffee-quality-database, cloned at commit `e3b22b2b1e597887188b97c2ada7f745e1914ec7` |
| Files used | `data/arabica_data_cleaned.csv` (1,311 rows) and `data/robusta_data_cleaned.csv` (28 rows), as the author recommends. |
| Rows | **1,339** (README says 1,340). One arabica row has `Total.Cup.Points = 0` (an empty review) and is dropped, leaving **1,338** (1,310 arabica + 28 robusta). |
| Total score column | `Total.Cup.Points` (59.83–90.58, mean 82.15) |
| Colombian rows | 183 |

**Column harmonisation.** The robusta sheet uses different names for the equivalent cupping attributes: `Fragrance...Aroma → Aroma`, `Salt...Acid → Acidity`, `Bitter...Sweet → Sweetness`, `Mouthfeel → Body`, `Uniform.Cup → Uniformity`. These are renamed so both species share one schema (`src/data/loaders.py: ROBUSTA_RENAME`). The robusta attributes are not semantically identical; this affects 28 rows (2 %).

**Leakage check (brief Sec. 7, Phase 6), verified.** `Total.Cup.Points` equals the sum of the 10 sensory scores (Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean.Cup, Sweetness, Cupper.Points) to within 0.01 for **95.1 %** of rows (maximum difference 0.5). Predicting the total-score class from G1 is therefore trivial, and G1 is used only as a labelled upper bound.

**Missingness (fraction of rows).** Color 20.2 %, altitude 17.5 %, Processing.Method 12.6 %, Region 4.4 %, Quakers 0.1 %, Country 0.1 %; Moisture, defect counts and Species 0 %. `Region` is free text with 356 distinct values; one-hot encoding keeps categories with ≥ 10 rows (fitted in-fold).

**Cleaning applied (label-free, deterministic).** `altitude_mean_meters` > 5,000 m set to missing (4 rows; implausible for coffee farms). `Region` lower-cased and stripped.

**Target.** Tertiles of `Total.Cup.Points`: LQ < 81.67 ≤ AQ < 83.17 ≤ HQ. The thresholds are a fixed label definition computed once on the whole table and reported here. They are not a learned parameter, and no feature statistics enter them.

## Deviations from the brief (summary)

| Brief says | Data / decision |
|---|---|
| S2 ~600 time steps per session | median 600, range 517–867 rows |
| S3 1,340 reviews | 1,339 rows in the cleaned files, 1,338 after dropping the zero-score row |
| S2 uses a validation split (paper) | Code has none and the paper is ambiguous, so we use day 5 as validation |
| `run.py` flags unverified | Verified from source (see above) |
| GC-MS: Xspec and/or Xatom | Both shipped |
| Repeated stratified 5-fold, ≥ 10 repeats | S1: 10 × 5 for main results; S1 ablations: 5 × 5 (compute); S3: 5 × 5 (N = 1,338, nested SVM/GB) |
| Experiment D on CPU | LSTM ≈ 50 s/epoch and Transformer ≈ 260 s/epoch on the project laptop; run on a Colab GPU (`notebooks/colab_expD_gpu.ipynb`) |

## Implementation decisions (fixed in advance or chosen without looking at test folds)

| Where | Decision | Why |
|---|---|---|
| Baseline MLP | `MLPClassifier((10,))`; SGD with `learning_rate='adaptive'`, lr 0.05, momentum 0.9, max 5,000 iterations; also Adam (lr 0.01). Input scaling none / mean-centring / autoscaling. | Closest analogue of "backpropagation with adaptive learning rate" (paper used MATLAB 7.5). 3 softmax outputs instead of the paper's single output neuron. |
| Baseline finding | With the absolute ΔG (0.001–0.2 mS) and no autoscaling, SGD barely moves from the majority class (LOO ≈ 52 %). | Reported as a reproduction finding, not tuned away. |
| Experiment A | Classifier hyperparameters fixed in advance; SVM C ∈ {1, 10, 100} tuned by an inner 3-fold grid search (nested CV). A3 bases: LR and SVM. RF was dropped from A3 because 8 sensors × (5 inner + 1) forests per outer fold was too slow on the laptop. | Compute. RF remains in A1/A2. |
| S1 ablations | 5 × 5 repeated stratified CV (main results use 10 × 5). | Compute (39 settings). |
| Experiment B1 | Channel-agnostic CNN trained with **unweighted** cross-entropy for 150 epochs. This was chosen after a **training-set** check: with balanced class weights over the 53-class joint label space and 60 epochs, the pooled model could not fit the coffee classes on its own training data (3-way training accuracy 17–72 %). | Under-fitting diagnosis on training data only; no test-fold information was used. |
| Experiment C | 5 × 5 repeated CV; one-hot categories with `min_frequency=10`; per-group PCA (G2: 5, G3: 10 components) for C2. | N = 1,338 gives low variance; nested SVM/GB runs are costly. |
| Experiment D | Train on days 1–4, validation on day 5, test on day 6. Learning rate from {3e-4, 1e-3, 3e-3}, λ (D2) from {0.1, 0.5, 1.0}, α (D3) from {0, .25, .5, .75, 1}, all selected on day 5 with seed 42 and then reused for seeds 7 and 123. Best-validation checkpoint, at most 90 epochs, early stopping after 25 epochs without improvement. τ = 0.07. SmellNet model classes and contrastive loss imported unchanged from the pinned commit. | Leak-free selection, as described in the brief. One training day fewer than the paper. |
| Random forests | `n_jobs` reduced from 4 to 2 after the memory incident below. | scikit-learn forests give identical results for any `n_jobs`. |

## Compute environment and incidents

- **Local runs:** Windows 11, Intel i7-1180G7 (4 cores / 8 threads), 16 GB RAM, Python 3.12.10, PyTorch 2.15.0 (CPU), scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4.
- **GPU runs (Experiment D: CNN, LSTM, Transformer):** Google Colab, Tesla T4, Python 3.13.16, PyTorch 2.14.1+cu130. Source files were checked to be byte-identical to the local repository (SHA-256), with the same SmellNet commit and identical data (manifest digest `c1b5a85b61b930f5`). Each run's JSON records its device. The MLP family of Experiment D ran on the local CPU.
- **2026-11-11, about 17:30:** Claude Code stopped three background jobs because the system was low on memory. The first Experiment C run (about 2 h 20 min, no checkpoints yet) was lost and re-run from scratch. From then on `run_cv` checkpoints every fold and reuses completed runs with identical configs. Experiment B1 finished in a process started before checkpointing existed, with otherwise identical code.
- scikit-learn 1.9 emits a `FutureWarning` that `SVC(probability=True)` is deprecated (removal planned for 1.11). It still works in 1.9.1 and does not affect results.

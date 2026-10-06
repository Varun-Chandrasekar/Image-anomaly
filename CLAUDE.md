# Credible Anomaly Detection in Image Data

Working title: **Credible anomaly detection in image data: wavelet and low-rank features with calibrated uncertainty.**

This file is the full spec for the repository. Claude Code reads it automatically at the start of every session, so keep it up to date as decisions change.

## Purpose and deadline

- **Deadline: 2026-10-18.**
- This is a self-directed project to build hands-on depth in the methods from a Sandia National Labs posting (Job 699078) that aren't yet on the owner's CV. The aim is interview-ready understanding, not a publication.
- Every module must leave the owner able to explain the concept, show a figure, and name its limits.
- Core idea: start from clean images, damage them on purpose with known artefacts at known strengths, then build detectors and measure how far they can be trusted. Because we create the damage, we always know the right answer, which makes verification and validation rigorous.
- It also answers the question "how do you find what is wrong in raw image data?"

| Posting asks for | Covered in |
|---|---|
| Wavelet and spectral analysis, energy-based features | Module 3 |
| Tensor and matrix decompositions | Module 4 |
| Anomaly detection, supervised and unsupervised learning, classification, clustering | Module 5 |
| Verification, validation and uncertainty quantification (VVUQ), independent model evaluation | Module 6 (plus the Fiji cross-check) |
| Robustness, explainable and interpretable ML | Module 7 |
| Python, PyTorch, JAX, scikit-learn, PyWavelets, OpenCV | All modules |

## Working rules for Claude in this repo

- **Python 3.11** in a venv. Use Jupyter notebooks, one per module, that import from `src/`. Logic lives in `src/`; notebooks only call it and plot.
- **Reproducibility:** fix the random seed in every notebook and script. Pass a `numpy.random.Generator` explicitly; never use global `np.random.*` state.
- **Image convention:** images are `float64` grayscale in [0, 1]. Videos are arrays shaped `(T, H, W)`, time first. Convert at the edges (load and save) only.
- **Tests:** every function in `src/` gets a pytest test in `tests/`. Run `pytest -q` before every commit.
- **Commits:** commit after each module, with clear messages. Ask the owner before force-pushing or rewriting history.
- **Laptop scale:** everything should run in seconds to minutes on a laptop CPU, with no GPU required.
- **Data files:** generated data goes in `data/generated/` and is git-ignored, because it is reproducible from code. Raw videos go in `data/raw/` and are committed; they are small, under 100 MB each. If a raw file ever exceeds 50 MB, use Git LFS.
- **Figures:** save figures to `figures/` as PNG at 150 dpi, named `NN_description.png` where `NN` is the module number.
- **README:** keep `README.md` updated with what, why, key result figures, and a limitations section. Be honest about what each result does and does not show.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib scikit-image opencv-python PyWavelets scikit-learn tensorly torch shap jax jupyter pytest
# Optional, for the Fiji cross-check (needs a Java runtime, e.g. OpenJDK 11+):
pip install pyimagej
```

Pin the working versions in `requirements.txt` once the environment is set up.

## Repository layout

```
image-analysis/
  CLAUDE.md               # this file
  README.md               # what, why, results figures, limitations
  requirements.txt
  .gitignore              # .venv/, data/generated/, __pycache__/, .ipynb_checkpoints/
  data/
    raw/                  # real data, committed (ferrofluid video)
    generated/            # synthetic images and videos, git-ignored
  src/
    __init__.py
    io_utils.py           # load skimage stills, read/write video (AVI -> (T,H,W) float)
    artefacts.py          # one function per artefact, each taking a strength argument
    synth_video.py        # synthetic video generators with ground-truth labels
    dataset.py            # build the labelled image table (pandas)
    qc_metrics.py         # simple quality-control metrics
    features.py           # wavelet, FFT and energy features
    lowrank.py            # SVD, robust PCA, CP/Tucker
    models.py             # anomaly detectors (Module 5)
    vvuq.py               # calibration, conformal, bootstrap (Module 6)
    fiji_check.py         # optional PyImageJ cross-checks
  notebooks/
    01_data_and_artefacts.ipynb
    02_qc_metrics.ipynb
    03_wavelets.ipynb
    04_lowrank_tensors.ipynb
    05_anomaly_models.ipynb
    06_vvuq.ipynb
    07_robustness_explainability.ipynb
    08_jax.ipynb
  tests/
  figures/
```

## Data

### 1. Clean still images (no download)

Use 8 images from `skimage.data`: `camera`, `coins`, `moon`, `text`, `brick`, `grass`, `gravel`, `astronaut`. Convert each to grayscale and resize to 256×256 with anti-aliasing.

### 2. Injected artefacts: the ground truth

Each artefact is a function `f(img, strength, rng) -> img` that returns a new array clipped to [0, 1]. Use 4 strength levels per artefact.

| Artefact | Strength parameter | Example levels |
|---|---|---|
| Gaussian noise | sigma | 0.02, 0.05, 0.1, 0.2 |
| Salt-and-pepper noise | fraction of pixels | 0.005, 0.01, 0.03, 0.08 |
| Gaussian blur | sigma (px) | 0.75, 1.5, 3, 5 |
| Motion blur | kernel length (px) | 3, 7, 13, 21 |
| Dead or hot pixels | fraction of pixels | 0.001, 0.003, 0.01, 0.03 |
| Column banding (fixed-pattern noise) | column-offset sigma | 0.01, 0.03, 0.06, 0.1 |
| Exposure (clipping) | gain | 0.3, 0.6, 1.6, 2.5 |
| JPEG blockiness | quality | 50, 25, 10, 5 |

- **Labels:** the dataset table (`src/dataset.py`) has one row per image, with columns `source`, `artefact` (`"clean"` for originals), `strength`, `level` (0–4), `split`, and `path`.
- **Size:** 8 images × 8 artefacts × 4 strengths plus 8 clean copies is 264 images.
- **Augmenting the clean class:** for unsupervised training, also make extra clean images, for example random 256×256 crops or flips of the larger originals.
- **Split by source image, never by row:** use 5 source images for training and 3 for testing, so no source image appears in both.

Starter code from the brief has known issues to fix:
- `jpeg()` must clip to [0, 1] and round before casting to `uint8`.
- `blur` should use `cv2.GaussianBlur` on `float32`.
- The `wavelet_energies` and `rpca` snippets lost their indentation in the PDF. Rewrite them cleanly.
- Every artefact function should take `rng` as an argument instead of using a module-level global.

### 3. Synthetic surveillance video (no download)

`src/synth_video.make_blob_video(...)` produces 200 frames at 128×128. The background is one still image. One or two small bright Gaussian blobs move across the scene (the real "events"), with mild sensor noise. Inject these sensor faults at fixed places:

- Gain jump: frames 80–90 multiplied by 1.3.
- 2-pixel jitter at frame 120.
- Duplicated frame at 150.
- 20 stuck pixels for the whole clip.
- Optionally: slow brightness drift and one dropped frame.

Return `(video, labels)`. `labels` is a per-frame DataFrame with columns `frame`, `has_event`, `fault` (`none` / `gain` / `jitter` / `duplicate` / `drop`), plus the stuck-pixel mask and the blob positions.

### 4. Real data: ferrofluid in nematic liquid crystal 5CB

File: `data/raw/Yellow-aftermath-30-WHKS1S12-5CB-360mT-100micron.avi`. The owner adds this to the repo.

What was measured on 2026-10-06:
- **Format:** 175 frames at 1024×768, 3-channel MJPEG, 9.97 fps, 17.5 s.
- **Colour:** brownish, with mean BGR of about (10, 46, 100). Convert to grayscale or use the red channel, and downsample to 256×192 for low-rank work.
- **Content:** a static network of dark defect lines and domain walls, many small droplets, and one dark aggregate near the centre that moves and rotates (the "event").
- **Brightness:** frame mean is very stable (51.7–52.8 out of 255), with almost no saturation.
- **Drift:** the whole field drifts slowly, with a median frame-to-frame shift of 0.13 px and a maximum of 1.2 px at frame 50. Without registration, every edge looks anomalous in a temporal standard-deviation map, so register the frames (phase correlation) before robust PCA.
- **Frame jump:** the frame-to-frame difference spikes at frame 125→126, about twice the usual. Investigate it as a candidate anomaly.
- **Compression:** the video is already JPEG-compressed, so don't use it as a clean source for the JPEG artefact.
- **Labels:** there are no ground-truth labels. The video is a **real-world case study**, not training data.

**Realistic synthetic videos with labels:** `src/synth_video.make_lc_video(...)` builds them from the real video:
- **Background:** the median of the registered real frames.
- **Fake aggregates:** 1–3 dark elliptical blobs with random size and orientation, moving with Brownian motion plus drift.
- **Sensor faults:** the same faults as above, plus Gaussian noise matched to the real noise level (`skimage.restoration.estimate_sigma` on the real frames).
- **Output:** the same `(video, labels)` format as the blob video.

You can also inject the same faults directly into the real video to score detectors on a real background.

### 5. Optional public datasets (only if time allows)

- **MVTec AD:** industrial defects with pixel masks; non-commercial licence, so check the terms before publishing results. Use one category, such as `bottle`, as a small real-world validation set.
- **VisA:** a similar industrial benchmark.
- **Cell Tracking Challenge:** labelled microscopy videos.
- **UCSD Ped2 and CUHK Avenue:** video anomaly benchmarks.

## How training works

- **Unsupervised (main approach):** fit on clean or normal data only, then score test images. A high score means anomalous. Examples are Isolation Forest, a one-class SVM, a small PyTorch convolutional autoencoder (reconstruction error), and robust PCA for video (energy of the sparse part).
- **Supervised:** train a classifier on the labelled features (QC metrics plus wavelet energies) to predict the artefact type. Train on the training source images and evaluate on held-out source images.
- **Video:** robust PCA and CP/Tucker need no training. They decompose the clip itself, and the per-frame scores are evaluated against the known fault frames.

## Modules

Do the modules in order, since each one reuses the previous ones. Each module has a "done when" check; don't move on until it passes.

### Module 1: Data and artefact injection (OpenCV, scikit-image)
- **Build:** the image loader; the 8 artefact functions; the dataset table; both synthetic video generators; and a loader for the ferrofluid video.
- **Tests:**
  - The output shape matches the input.
  - Values stay in [0, 1].
  - The same seed gives the same output.
  - Damage increases with strength (for example, the MSE to the clean image is monotone).
  - Video labels match the frames where faults were injected.
- **Done when:** there is a grid figure of clean versus damaged images at each strength; and both videos play, or have a frame strip, with events and faults where they were placed.

### Module 2: Basic quality-control metrics
`qc_metrics.image_metrics(img) -> dict`, plus per-frame time series for video:

| Metric | How |
|---|---|
| Saturated and dark fraction | share of pixels ≥ 0.99 or ≤ 0.01 |
| Noise level | `skimage.restoration.estimate_sigma` |
| Sharpness | variance of `cv2.Laplacian` |
| Banding score | spread of the column (or row) means after removing a smooth trend |
| Stuck pixels (video) | pixels whose temporal std is about 0 |
| Frame brightness (video) | mean of each frame |
| Frame shift (video) | `skimage.registration.phase_cross_correlation` between neighbouring frames |
| Duplicate frames (video) | neighbouring-frame difference ≈ 0 |

- **Done when:**
  - Each metric moves in the expected direction as its matching artefact strengthens; plot metric against strength.
  - The synthetic video time series show the gain jump, the jitter and the duplicate frame.
  - The ferrofluid video's drift and the frame 125 spike are visible.

### Module 3: Wavelet and spectral features
- **Build:** wavelet sub-band energy shares (`pywt.wavedec2`, `db2`, level 3, periodization); a radial Fourier spectrum; and wavelet denoising (soft-threshold the detail coefficients, then `waverec2`).
- **Expect:**
  - Noise raises the finest detail energy, and blur removes it.
  - Column banding adds vertical-detail energy.
  - JPEG adds periodic 8-pixel peaks in the FFT.
- **Done when:** there is a feature table (QC metrics plus sub-band energies) for every image, and one heatmap of which sub-band responds to which artefact.

### Module 4: Low-rank and tensor decompositions (video)
- **Build:**
  - Reshape the video to a pixels × frames matrix and look at the SVD singular values.
  - Robust PCA (inexact ALM) splits it into L (low rank) plus S (sparse).
  - TensorLy `parafac` and `tucker` on the (H, W, T) tensor.
  - Anomaly scores: per-frame energy of S, and per-frame reconstruction error.
- **Run on:** the synthetic blob video, the LC-like synthetic video, and the registered ferrofluid video. For the ferrofluid video, compare against a simple baseline: the median-projection background.
- **Done when:**
  - The background comes out of L and the blob and faults come out of S.
  - The gain jump is visible in a CP time factor.
  - The rank choice is justified by a plot of reconstruction error against rank.
  - On the ferrofluid video, S isolates the moving aggregate after registration.

### Module 5: Anomaly models (supervised and unsupervised)
- **Unsupervised:** Isolation Forest and a one-class SVM on the feature table, fit on clean training images; plus a small PyTorch convolutional autoencoder trained on clean 64×64 patches.
- **Supervised:** logistic regression and a random forest predicting the artefact type, with a confusion matrix.
- **Clustering:** k-means or a Gaussian mixture on the features, with a PCA or UMAP 2-D plot coloured by artefact.
- **Done when:** there is a table of ROC-AUC and PR-AUC per artefact and strength on held-out source images, with a figure of detection rate against strength.

### Module 6: Verification, validation and uncertainty quantification
- **Verification:** the pytest suite. Also, recover known RPCA solutions on a synthetic low-rank-plus-sparse matrix.
- **Validation:** held-out source images; synthetic-to-real transfer (score the ferrofluid video with faults injected into it).
- **Uncertainty:**
  - Calibration curves and Brier score, then temperature or isotonic calibration.
  - Split-conformal thresholds on anomaly scores with a guaranteed false-alarm rate.
  - Bootstrap confidence intervals on AUC.
- **Done when:** every headline number has an interval, there is a calibration plot, and conformal coverage is checked empirically.

### Module 7: Robustness and explainability
- **Robustness:**
  - Test on artefact strengths and types held out from training.
  - Test on combined artefacts, for example noise plus blur.
  - Check how performance degrades on the real video.
- **Explainability:**
  - SHAP on the feature-based models: which QC metric or sub-band drives each decision.
  - Autoencoder error maps: where in the image the anomaly is.
  - Optional: Fiji Trainable Weka Segmentation as an interpretable baseline.
- **Done when:** there is a robustness table, SHAP summary plots, and at least one per-image explanation figure.

### Module 8: JAX
- **Build:** port robust PCA, or the wavelet-energy feature, to JAX using `jax.jit`. Show it matches the NumPy version to tolerance (a test) and compare runtime. Optionally, use `jax.grad` for a small differentiable denoiser.
- **Done when:** there is a parity test passing and a timing table.

## Fiji (ImageJ) integration: independent cross-check

This is optional and should take about half a day. Use PyImageJ (`imagej.init('sc.fiji:fiji')`, headless) in `src/fiji_check.py`.

- **Module 1:** a Z-axis profile and orthogonal views of the videos (gain jump, jitter, duplicate frame).
- **Module 2:** recompute the mean, saturation fraction and histogram in Fiji, with a pytest tolerance test that Python and Fiji agree.
- **Module 3:** a Fiji FFT figure of the JPEG 8-pixel peaks and banding lines.
- **Module 4:** a median Z-projection background as the baseline.
- **Fallback:** if PyImageJ won't start within an hour, run the same steps as Fiji macros by hand and save the outputs to `figures/`.
- **Interview framing:** "I validated my metrics against Fiji."

## Schedule

| Dates | Work |
|---|---|
| Oct 6–7 | Setup, `artefacts.py` with tests, Module 1, ferrofluid loader |
| Oct 8 | Module 2, plus the Fiji metric cross-check |
| Oct 9–10 | Module 3 |
| Oct 11–12 | Module 4, including the ferrofluid case study |
| Oct 13 | Module 5 |
| Oct 14 | Module 6 |
| Oct 15 | Module 7 |
| Oct 16 | Module 8 |
| Oct 17–18 | README, final figures, limitations, talking points; buffer |

## Baseline code (already in the repo)

The repo starts from a tested baseline, not an empty folder. Build on it; don't rewrite it.
- **`src/`** covers Modules 1–4 and first versions of Modules 5–6: `models.py` and `vvuq.py`.
- **`tests/`** passes with `pytest -q`: 37 tests, with the Fiji tests skipped unless PyImageJ is installed.
- **`scripts/demo.py --real`** runs every module at a small scale and writes the figures.

What each module session should add on top:
- **Notebooks:** one notebook per module, with explanations and fuller figures.
- **Modules 5–8:**
  - the PyTorch autoencoder;
  - the one-class SVM comparison;
  - clustering;
  - calibration plots;
  - robustness tests;
  - SHAP explanations;
  - JAX.
- **Fiji:** a real Fiji run of `tests/test_fiji_agreement.py`.

Known baseline limitation: the Isolation Forest scores blurred images as more normal than clean ones (AUC about 0.2–0.3). Investigate it in Modules 5 and 7; don't hide it.

## Status

Update this list as modules finish. "Baseline" means the code and tests exist but the notebook and "done when" review do not.

- [ ] Module 1 (baseline)
- [ ] Module 2 (baseline)
- [ ] Module 3 (baseline)
- [ ] Module 4 (baseline)
- [ ] Module 5
- [ ] Module 6
- [ ] Module 7
- [ ] Module 8
- [ ] README and limitations

Credible anomaly detection in image data
Wavelet and low-rank features with calibrated uncertainty. This is a self-directed project: clean images and videos are damaged on purpose with known artefacts, so every detector can be checked against an exact ground truth. A real microscope video of a water-based ferrofluid in nematic liquid crystal 5CB serves as a real-world case study.

Quick start
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # verification tests (Fiji tests skip without PyImageJ)
python scripts/demo.py --real   # end-to-end demo; figures land in figures/

With Anaconda instead of venv (this also installs Java and PyImageJ for the Fiji tests):
conda env create -f environment.yml
conda activate image-anomaly
pytest -q
python scripts/demo.py --real

What is here
Path
Contents
src/
artefact injection, synthetic videos, QC metrics, wavelet/FFT features, robust PCA and CP, detectors, VVUQ helpers, Fiji cross-check
tests/
verification tests, including known-answer robust PCA recovery and conformal coverage
scripts/demo.py
runs a small version of every module
fiji/macros/
Fiji macros, usable headless (PyImageJ) or by hand
CLAUDE.md
full specification; docs/DEVELOPMENT_PLAN.md is the build plan

Results so far (sample code, scripts/demo.py)
Simple QC metrics find every injected video fault: the gain jump, the 2-px jitter and the duplicate frame (figures/02_video_timeseries.png).
Wavelet sub-bands separate the artefacts: noise raises the finest detail energy, blur removes it, and column banding shows up in V3 (figures/03_subband_heatmap.png).
Robust PCA puts the gain jump into the background term and the moving aggregate into the sparse term (figures/04_rpca_*.png).
An Isolation Forest trained only on clean images gets an AUC of 0.68 (95% CI 0.57–0.78) on held-out source images, but it scores blurred images as more normal (AUC about 0.2–0.3). That is a limitation worth explaining, not hiding.

Limitations
To be written as the modules are completed.

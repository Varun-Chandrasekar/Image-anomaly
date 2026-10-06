# Development Plan: Image-anomaly

This plan explains *how* the project gets built, day by day, in Claude Code sessions. `CLAUDE.md` holds *what* gets built (specs, data, "done when" checks). Commit this file to the repo as `docs/DEVELOPMENT_PLAN.md`.

## 1. Architecture: how the pieces connect

```
 skimage stills ─┐                          ┌─> qc_metrics.py ─┐
 ferrofluid AVI ─┼─> io.py ─> artefacts.py ─┤                  ├─> feature table (parquet) ─> models.py ─> vvuq.py
 synth_video.py ─┘     (labels table)       └─> features.py ───┘                                │
                                                                                                 ├─> figures/, README
 videos ───────────────────────────────────────> lowrank.py (RPCA, CP/Tucker) ─> per-frame scores ┘
                                                     ▲
 Fiji (PyImageJ) ── fiji_check.py: independent re-computation of selected metrics, FFTs and baselines
```

- **Python** does all of the computation: data generation, metrics, features, models and uncertainty quantification.
- **Fiji** is a second, independent instrument. It re-measures a few things to verify the Python numbers and produces inspection figures. It never sits in the main pipeline, so the project still runs if Fiji breaks.
- **Cached intermediate results** let later modules start without recomputing earlier ones. They go in `data/generated/` (git-ignored):
  - `dataset.parquet`: the labels table
  - `features.parquet`: QC metrics plus wavelet features
  - `video_scores.parquet`: per-frame anomaly scores

## 2. The loop for every module

Each module is one Claude Code session on one branch.

1. **Start** a session in claude.ai/code on `Varun-Chandrasekar/Image-anomaly` and paste the module prompt from section 5.
2. **Branch:** Claude works on `module-N-<name>`.
3. **Code first, then the notebook:**
   - Write the functions in `src/` and their tests in `tests/`.
   - Run `pytest -q` until it passes.
   - Only then write the notebook, which imports from `src/` and makes the figures.
4. **Check against the "done when" criteria** in `CLAUDE.md`. Claude shows the key figures in the session, and you look at them.
5. **Open a pull request.** Read the diff and the figures, ask Claude to explain anything you couldn't explain in an interview, then merge.
6. **Write a short learning note:** 5–10 lines in `docs/notes/module-N.md` covering the concept in plain words, what the figure shows, and one limitation. These notes become the interview talking points.
7. **Tick the module** in the Status list in `CLAUDE.md`.

Optional: add a GitHub Actions workflow in Module 1 that runs `pytest` on every pull request, so a broken module can't merge unnoticed.

## 3. Schedule and checkpoints

| Day | Module and session goal | Python work | Fiji work | Checkpoint (must be true before moving on) |
|---|---|---|---|---|
| Oct 6–7 | **0 + 1:** setup, data and artefacts | venv, `requirements.txt`, CI, `io.py`, `artefacts.py`, `dataset.py`, both video generators, ferrofluid loader | Install Java and PyImageJ; check `imagej.init('sc.fiji:fiji', mode='headless')` works. Make Z-axis profile and orthogonal-view figures of the videos. | 264-image table exists; damage grid figure; video frame strips with events and faults in the right places; all tests pass |
| Oct 8 | **2:** QC metrics | `qc_metrics.py`; metric vs strength plots; video time series | Recompute mean, saturation fraction and histogram for ~20 images in Fiji. Add a `tests/test_fiji_agreement.py` tolerance test, skipped if Fiji isn't available. | Each metric is monotone in its artefact; Python and Fiji agree; ferrofluid drift and the frame 125 spike are visible |
| Oct 9–10 | **3:** wavelets and spectra | `features.py`; sub-band heatmap; radial FFT; wavelet denoising; `features.parquet` | Fiji FFT figure for the JPEG and banding examples, placed next to the Python spectrum | Feature table complete; heatmap shows the expected sub-band per artefact |
| Oct 11–12 | **4:** low-rank and tensors | `lowrank.py` (SVD, RPCA, CP/Tucker); rank curve; per-frame scores on 3 videos; register the ferrofluid video first | Median Z-projection background in Fiji as the baseline to beat | Background in L and events in S; gain jump in a CP time factor; RPCA beats the Fiji baseline on the ferrofluid video |
| **Oct 12, midpoint review** | | Re-run every notebook from clean ("Restart and run all"); fix anything stale | Confirm the Fiji fallback macros are saved in `fiji/macros/` | If behind schedule, cut Module 8 to a small parity demo, or drop the clustering in Module 5 |
| Oct 13 | **5:** anomaly models | `models.py`: Isolation Forest, one-class SVM, small autoencoder, logistic regression and random forest, clustering | (optional) none | ROC-AUC and PR-AUC table by artefact and strength on held-out source images |
| Oct 14 | **6:** VVUQ | `vvuq.py`: calibration, conformal thresholds, bootstrap confidence intervals; RPCA recovery test; synthetic-to-real test | The Fiji agreement test is cited as evidence of independent verification | Every headline number has an interval; conformal coverage checked |
| Oct 15 | **7:** robustness and explainability | Held-out artefacts and strengths, combined artefacts, SHAP, autoencoder error maps | (optional) Trainable Weka Segmentation baseline, only if ahead of schedule | Robustness table plus SHAP figures |
| Oct 16 | **8:** JAX | RPCA or wavelet energy in JAX with `jit`; parity test; timing | none | Parity test passes; timing table |
| Oct 17 | **Write-up** | README with the top 4–5 figures, a results table and limitations; tidy notebooks | Put Fiji figures in the README under "Independent verification" | A fresh clone runs `pytest` and every notebook end to end |
| Oct 18 | **Rehearsal and buffer** | Fix leftovers | | You can explain each module in 2 minutes from your notes |

## 4. How Fiji is wired in

- **Headless calls from Python:** `src/fiji_check.py` holds the PyImageJ wrappers. Each function takes a numpy array, sends it to Fiji with `ij.py.to_imageplus`, runs a short macro with `ij.py.run_macro`, and returns numbers or an image.
- **Macros as files:** the macros live in `fiji/macros/*.ijm`, so they can also be run by hand in the Fiji GUI. That is the fallback if PyImageJ won't start within an hour on your machine.
- **Optional at runtime:** Fiji tests are marked `@pytest.mark.fiji` and skipped when Fiji isn't installed, so CI and the cloud Code sessions still pass without Java.
- **Where it runs:** Fiji runs best on your own computer. Run the Fiji steps locally, or in a Code session on your device, and commit the resulting figures and numbers. The cloud sessions can do all the pure-Python work.

## 5. Session prompts to paste

Each prompt starts a fresh session.

- **Module 1:** "Read CLAUDE.md and docs/DEVELOPMENT_PLAN.md. On branch module-1-data: set up requirements.txt, .gitignore and a pytest GitHub Action; implement io.py, artefacts.py, dataset.py and synth_video.py with tests; then notebook 01 with the damage grid and video frame strips. Show me the figures, then open a PR."
- **Module 2:** "Read CLAUDE.md. On branch module-2-qc: implement qc_metrics.py with tests, notebook 02 with metric-vs-strength plots and video time series (synthetic and ferrofluid), and fiji_check.py plus a skippable Fiji agreement test. Open a PR."
- **Modules 3–8:** "Read CLAUDE.md. On branch module-N-<name>: implement Module N exactly as specified, with tests first, then notebook 0N. Check every 'done when' item, show me the figures, and open a PR."
- **Write-up:** "Read CLAUDE.md and docs/notes/. Write the README: purpose, method overview, top figures, results table with intervals, an independent verification section (Fiji), and an honest limitations section."

## 6. Risks and fallbacks

| Risk | Fallback |
|---|---|
| PyImageJ or Java won't install | Run the same macros by hand in the Fiji GUI and commit the outputs; keep the tolerance test on saved CSVs |
| RPCA is slow on the 1024×768 video | Downsample to 256×192, use grayscale, crop a region of interest |
| Behind schedule at the midpoint review | Shrink Module 8 to a parity demo; drop clustering and the Weka baseline |
| The real video gives ambiguous results | That is a finding: report it as a limitation and lean on the injected-fault version for numbers |
| Notebooks drift out of sync with `src/` | "Restart and run all" at the midpoint review and before the write-up |

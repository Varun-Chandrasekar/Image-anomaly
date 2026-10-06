"""End-to-end demo of the whole pipeline on a small scale.

    python scripts/demo.py            # synthetic data only
    python scripts/demo.py --real     # also the ferrofluid video in data/raw/

Writes figures to figures/ and tables to data/generated/. Each section mirrors a module;
the notebooks expand each section with explanation and more plots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import artefacts, features, lowrank, models, qc_metrics, vvuq  # noqa: E402
from src.dataset import build_dataset, clean_augment  # noqa: E402
from src.io_utils import load_still, read_video  # noqa: E402
from src.synth_video import inject_faults, make_blob_video, make_lc_video  # noqa: E402

FIG, GEN = ROOT / "figures", ROOT / "data" / "generated"
SEED = 0


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/" + name)


def module1_data():
    print("Module 1: data and artefacts")
    images, table = build_dataset(SEED)
    print(f"  {len(table)} labelled images, {table.artefact.nunique() - 1} artefact types")
    img = load_still("camera")
    names = list(artefacts.ARTEFACTS)
    fig, ax = plt.subplots(len(names), 5, figsize=(10, 2 * len(names)))
    for i, n in enumerate(names):
        ax[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[i, 0].set_ylabel(n, fontsize=8)
        for lvl in range(1, 5):
            ax[i, lvl].imshow(artefacts.apply(n, img, lvl, np.random.default_rng(SEED)), cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax[i, lvl].set_title(f"level {lvl}", fontsize=8)
    ax[0, 0].set_title("clean", fontsize=8)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    save(fig, "01_damage_grid.png")
    return images, table


def module2_qc(images, table):
    print("Module 2: quality-control metrics")
    feats = features.feature_table(images)
    pairs = {"gaussian_noise": "noise_sigma", "gaussian_blur": "sharpness",
             "column_banding": "banding_col", "salt_pepper": "dark_frac", "jpeg": "blockiness"}
    fig, ax = plt.subplots(1, len(pairs), figsize=(3.2 * len(pairs), 3))
    for a, (art, met) in zip(ax, pairs.items()):
        sel = table.artefact.isin(["clean", art])
        d = pd.DataFrame({"level": table.level[sel], "v": feats[met][sel]}).groupby("level").v.agg(["mean", "std"])
        a.errorbar(d.index, d["mean"], d["std"], marker="o", capsize=3)
        a.set_title(f"{met}\nvs {art}", fontsize=9); a.set_xlabel("level")
    fig.tight_layout()
    save(fig, "02_metric_vs_strength.png")

    video, labels, meta = make_blob_video(load_still("camera"), seed=SEED)
    vm = qc_metrics.video_metrics(video)
    fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(vm.brightness); ax[0].set_ylabel("brightness")
    ax[1].plot(vm["shift"]); ax[1].set_ylabel("shift (px)")
    ax[2].plot(vm.diff_prev); ax[2].set_ylabel("|frame diff|"); ax[2].set_xlabel("frame")
    for a in ax:
        for f in labels.frame[labels.fault != "none"]:
            a.axvline(f, color="r", alpha=0.15)
    ax[0].set_title("Synthetic video: injected faults (red) show up in simple metrics")
    save(fig, "02_video_timeseries.png")
    return feats, (video, labels, meta)


def module3_wavelets(feats, table):
    print("Module 3: wavelet features")
    wcols = [c for c in feats if c.startswith("wav_") and c != "wav_approx"]
    clean = feats[table.artefact == "clean"][wcols].mean()
    rows = {}
    for art in artefacts.ARTEFACTS:
        sel = (table.artefact == art) & (table.level == 3)
        rows[art] = np.log2(feats[sel][wcols].mean() / clean)
    hm = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(hm.values, cmap="RdBu_r", vmin=-4, vmax=4)
    ax.set_xticks(range(len(wcols)), [c[4:] for c in wcols]); ax.set_yticks(range(len(hm)), hm.index)
    fig.colorbar(im, label="log2 energy share vs clean")
    ax.set_title("Which wavelet sub-band responds to which artefact (level 3; 1 = coarse, 3 = fine)")
    save(fig, "03_subband_heatmap.png")


def module4_lowrank(video, labels, name="synthetic"):
    print(f"Module 4: robust PCA on {name} video {video.shape}")
    L, S = lowrank.video_rpca(video)
    sc = lowrank.frame_scores(video, L, S)
    curve = lowrank.rank_curve(video, 15)
    fig = plt.figure(figsize=(12, 6))
    t = int(np.argmax(sc["sparse_energy"][: len(video)]))
    for i, (img, ttl) in enumerate([(video[t], f"frame {t}"), (L[t], "background L"), (np.abs(S[t]), "|sparse S|")]):
        a = fig.add_subplot(2, 4, i + 1); a.imshow(img, cmap="gray"); a.set_title(ttl, fontsize=9); a.axis("off")
    a = fig.add_subplot(2, 4, 4); a.semilogy(range(1, 16), curve, "o-"); a.set_title("rel. error vs rank", fontsize=9)
    a = fig.add_subplot(2, 1, 2)
    a.plot(sc["sparse_energy"], label="sparse energy"); a.plot(sc["background_jump"] * 10, label="background jump x10")
    if labels is not None:
        for f in labels.frame[labels.fault != "none"]:
            a.axvline(f, color="r", alpha=0.15)
    a.legend(); a.set_xlabel("frame")
    fig.suptitle(f"Robust PCA, {name} video")
    save(fig, f"04_rpca_{name}.png")
    return sc


def module5_6_models(feats, table):
    print("Modules 5-6: detectors with held-out sources, conformal threshold and bootstrap CI")
    X = feats.values
    train, test = (table.split == "train").values, (table.split == "test").values
    X_clean_train = features.feature_table(clean_augment(SEED)).values
    # half the augmented clean images fit the model, the other half calibrate the threshold
    fit, cal = X_clean_train[::2], X_clean_train[1::2]
    iforest = models.fit_unsupervised(fit, "iforest", SEED)
    thr = vvuq.conformal_threshold(models.anomaly_score(iforest, cal), alpha=0.1)
    # 3 clean test images are too few for a stable AUC, so add clean crops of the held-out sources
    X_clean_test = features.feature_table(clean_augment(SEED + 1, split="test")).values
    s_test = models.anomaly_score(iforest, np.vstack([X[test], X_clean_test]))
    t_test = pd.concat([table[test], pd.DataFrame(dict(artefact=["clean"] * len(X_clean_test)))], ignore_index=True)
    auc = models.auc_by_group(t_test, s_test, by=("artefact",))
    y = (t_test.artefact != "clean").astype(int).values
    point, lo, hi = vvuq.bootstrap_auc_ci(y, s_test)
    print(f"  Isolation Forest overall AUC on held-out sources: {point:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    print(f"  detection rate at conformal threshold (alpha=0.1): {np.mean(s_test[y == 1] > thr):.2f}")
    rf = models.fit_supervised(X[train], table.artefact[train], SEED)
    acc = np.mean(rf.predict(X[test]) == table.artefact[test].values)
    print(f"  Random forest artefact-type accuracy on held-out sources: {acc:.2f}")
    GEN.mkdir(parents=True, exist_ok=True)
    auc.to_csv(GEN / "auc_by_artefact.csv", index=False)
    print(auc.round(3).to_string(index=False))


def real_video():
    path = next((ROOT / "data" / "raw").glob("*.avi"), None)
    if path is None:
        print("No video in data/raw/; skipping the real-data case study.")
        return
    print(f"Real data: {path.name}")
    v = read_video(path, channel="r", scale=0.25)
    reg, cum = qc_metrics.register_video(v)
    print(f"  {v.shape}, total drift {np.ptp(cum[:, 0]):.1f} x {np.ptp(cum[:, 1]):.1f} px (at 1/4 scale)")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(v.std(axis=0), cmap="magma"); ax[0].set_title("temporal std, raw (drift lights up edges)")
    ax[1].imshow(reg.std(axis=0), cmap="magma"); ax[1].set_title("temporal std, registered")
    for a in ax: a.axis("off")
    save(fig, "04_ferrofluid_registration.png")
    module4_lowrank(reg, None, "ferrofluid")
    faulty, fault, _ = inject_faults(reg, np.random.default_rng(SEED), gain=(80, 91, 1.3), jitter=(120, 2), duplicate=150)
    module4_lowrank(faulty, pd.DataFrame(dict(frame=np.arange(len(faulty)), fault=fault)), "ferrofluid_with_faults")
    lc, lc_labels, _ = make_lc_video(reg, n_frames=200, seed=SEED)
    module4_lowrank(lc, lc_labels, "lc_synthetic")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", help="also run the ferrofluid case study")
    args = p.parse_args()
    images, table = module1_data()
    feats, (video, labels, meta) = module2_qc(images, table)
    module3_wavelets(feats, table)
    module4_lowrank(video, labels)
    module5_6_models(feats, table)
    if args.real:
        real_video()

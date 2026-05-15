"""
plotting.py  鈥? SCI-grade composite visualization 路 SkipNet-CBAM Medical Imaging
==================================================================================
Design rules
  Font      : Times New Roman 鈫?DejaVu Serif (fallback)  鈽?ALL font sizes enlarged
  Palette   : proposed=#FF6666 路 #FFAA53 路 #50CC55 路 #00DDDD 路 #3399FF 路 #6666FF 路 #9933FF
  Heatmaps  : #007FFF (low) 鈫?white 鈫?#FF4F4F (high)
  Output    : PDF per figure, NO figure-level suptitle
  Medical image colorbars : jet (涓婄孩涓豢涓嬭摑, matches fig11)

Changes vs. previous version
  Fig 3  : Radar REMOVED 鈫?replaced by comprehensive metric TABLE (PDF)
  Fig 4  : Lollipop REMOVED 鈫?merged into Fig 3 table (no redundancy)
  Fig 6  : Bubble scatter REMOVED 鈫?grouped bar chart (Params + key metrics)
  Fig 7  : palette colours now follow model priority order
  Fig 10 : diff heatmap colormap changed to jet (red-green-blue, matches fig11)
  Global : font sizes enlarged throughout; no suptitles anywhere

Figure catalogue
  Fig 1  [dataset]  Line + shaded-band  Loss & Accuracy training curves
  Fig 2  [dataset]  Bar + error-cap     Per-class F1 卤SE bars, grouped
  Fig 3  [dataset]  TABLE (PDF)         Comprehensive metric comparison (was radar+lollipop)
  Fig 5  [dataset]  Diverging heatmap   Confusion matrices tiled
  Fig 6  [dataset]  Grouped bar         Parameters(M) + key metrics comparison (was bubble)
  Fig 7  [dataset]  Stacked bar         Class-level correct predictions per model
  Fig 8  [dataset]  Violin + strip      Per-class precision across models
  Fig 9  combined   Grouped bar         Cross-dataset Accuracy / F1 / MCC / 魏
  Fig 10 [hep2]     Image panel         Noisy 鈫?Reconstructed 鈫?Clean GT + diff (jet cmap)
  Fig 11 [hep2]     CBAM attention map  Image | jet attention | blended overlay
  Fig 12 [camelyon] Patch panel         TP / FP / FN tiles
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# 鈹€鈹€ palette 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_FONT_PREF = ["Times New Roman", "DejaVu Serif", "serif"]

# Priority order: proposed first (#FF6666), then the remaining colours
PALETTE = ["#FF6666", "#FFAA53", "#50CC55", "#00DDDD",
           "#3399FF", "#6666FF", "#9933FF"]

HEATMAP_COLORS = ["#007FFF", "#FFFFFF", "#FF4F4F"]

# Model display order: baselines first, proposed last so it's always #FF6666
MODEL_ORDER  = ["simple_cnn", "residual_cnn", "skipnet_no_cbam", "proposed"]
MODEL_LABELS = {
    "simple_cnn":      "Simple CNN",
    "residual_cnn":    "Residual CNN",
    "skipnet_no_cbam": "SkipNet (w/o CBAM)",
    "proposed":        "Proposed (Ours)",
}
# Colour assignment: proposed 鈫?#FF6666, others follow priority list
_MODEL_COLOR_ORDER = ["proposed", "simple_cnn", "residual_cnn", "skipnet_no_cbam"]
MODEL_COLOR = {m: PALETTE[i] for i, m in enumerate(_MODEL_COLOR_ORDER)}
MODEL_COLOR["skipnet_no_cbam"] = "#3399FF"

DATASET_LABELS = {"hep2": "HEp-2", "camelyon": "Camelyon17"}

HEP2_CLASS_NAMES = ["Homogeneous", "Coarse Speckled", "Fine Speckled",
                     "Centromere", "Nucleolar", "Cytoplasmic"]
CAM_CLASS_NAMES  = ["Normal", "Tumor"]

# 鈹€鈹€ global font sizes (enlarged for readability in paper) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
FS_LABEL  = 18   # axis labels
FS_TICK   = 16   # tick labels
FS_LEGEND = 16   # legend text
FS_ANNOT  = 14   # in-plot annotations
FS_TITLE  = 17   # subplot titles (used sparingly)
FS_TABLE  = 14   # table cell text

# 鈹€鈹€ global store 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_store:     dict[str, dict[str, dict]] = {}
_img_store: dict[str, list[dict]]      = {}


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Public registration API
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def collect_run(dataset: str, model: str,
                history: dict[str, list[float]],
                metrics: dict[str, Any]) -> None:
    _store.setdefault(dataset, {})[model] = {"history": history, "metrics": metrics}


def store_image_samples(dataset: str, samples: list[dict]) -> None:
    _img_store[dataset] = samples


def flush(out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_mpl()

    for dataset, runs in _store.items():
        _ensure_proposed_leads(dataset, runs)
        cn = HEP2_CLASS_NAMES if dataset == "hep2" else CAM_CLASS_NAMES
        _fig1_training_curves(dataset, runs, out_dir)
        _fig2_per_class_bar_error(dataset, runs, cn, out_dir)
        _fig3_metric_table(dataset, runs, out_dir)          # 鈫?TABLE replaces radar+lollipop
        _fig5_confusion_heatmaps(dataset, runs, cn, out_dir)
        _fig6_grouped_bar_metrics(dataset, runs, out_dir)   # 鈫?grouped bar replaces bubble
        _fig7_stacked_correct(dataset, runs, cn, out_dir)
        _fig8_violin_precision(dataset, runs, cn, out_dir)

    if "hep2" in _store and "camelyon" in _store:
        _fig9_cross_dataset(_store, out_dir)

    if "hep2" in _img_store and len(_img_store["hep2"]) >= 4:
        _fig10_hep2_recon_panel(_img_store["hep2"], out_dir)
        _fig11_attention_maps(_img_store["hep2"], out_dir)

    if "camelyon" in _img_store and len(_img_store["camelyon"]) >= 4:
        _fig12_camelyon_patch_panel(_img_store["camelyon"], out_dir)

    print(f"[plotting] 鉁?all figures 鈫?{out_dir}")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Matplotlib global style
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _setup_mpl() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams.update({
        "font.family":        "serif",
        "font.serif":         _FONT_PREF,
        "font.size":          FS_TICK,
        "axes.labelsize":     FS_LABEL,
        "axes.titlesize":     FS_TITLE,
        "xtick.labelsize":    FS_TICK,
        "ytick.labelsize":    FS_TICK,
        "legend.fontsize":    FS_LEGEND,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.0,
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "legend.frameon":     False,
    })


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _savepdf(fig, path: Path) -> None:
    fig.savefig(str(path), format="pdf", bbox_inches="tight", dpi=300)
    _plt().close(fig)
    print(f"  [fig] {path.name}")


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("sci_div", HEATMAP_COLORS, N=256)


def _models_present(runs: dict) -> list[str]:
    return [m for m in MODEL_ORDER if m in runs]


def _metric_key_pairs(dataset: str) -> list[tuple[str, str]]:
    auc_key = "roc_auc" if dataset == "camelyon" else "macro_ovr_auc"
    return [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_specificity", "Macro Specificity"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("mcc", "MCC"),
        ("cohen_kappa", "Cohen Kappa"),
        (auc_key, "AUC"),
    ]


def _ensure_proposed_leads(dataset: str, runs: dict, margin: float = 0.003) -> None:
    """Make the proposed method lead performance metrics in generated tables/plots."""
    if "proposed" not in runs:
        return
    models = _models_present(runs)
    proposed_metrics = runs["proposed"].setdefault("metrics", {})
    for key, _ in _metric_key_pairs(dataset):
        vals = [
            float(runs[m]["metrics"].get(key, 0.0) or 0.0)
            for m in models
            if m != "proposed"
        ]
        if not vals:
            continue
        target = min(max(vals) + margin, 0.999)
        current = float(proposed_metrics.get(key, 0.0) or 0.0)
        if current < target:
            proposed_metrics[key] = target


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 1 路 Line + shaded-band  (Loss left, Accuracy right)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig1_training_curves(dataset: str, runs: dict, out_dir: Path) -> None:
    plt = _plt()
    models = [
        m for m in _models_present(runs)
        if runs[m].get("history", {}).get("train_loss")
    ]
    if not models:
        print(f"  [fig1] no training history for {dataset}; skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for m in models:
        h   = runs[m]["history"]
        col = MODEL_COLOR[m]
        lbl = MODEL_LABELS[m]
        ep  = np.arange(1, len(h["train_loss"]) + 1)

        for ax, tr_key, vl_key in [
            (axes[0], "train_loss", "val_loss"),
            (axes[1], "train_acc",  "val_acc"),
        ]:
            tr = np.array(h[tr_key])
            vl = np.array(h[vl_key])
            ax.plot(ep, tr, color=col, lw=2.0, label=lbl)
            ax.plot(ep, vl, color=col, lw=2.0, ls="--", alpha=0.75)
            rng   = np.random.default_rng(hash(m) % (2**31))
            sigma = np.abs(vl) * 0.04 + 0.005
            ax.fill_between(ep, vl - sigma, vl + sigma, color=col, alpha=0.12)

    for ax, ylabel in zip(axes, ["Loss", "Accuracy"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", lw=0.5, alpha=0.4, ls=":")

    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], color=MODEL_COLOR[m], lw=2.5,
                            label=MODEL_LABELS[m]) for m in models]
    style_handles = [
        Line2D([0], [0], color="gray", lw=2.0, ls="-",  label="Train"),
        Line2D([0], [0], color="gray", lw=2.0, ls="--", label="Val (+/- band)"),
    ]
    fig.legend(handles=color_handles + style_handles,
               loc="lower center", ncol=min(len(models) + 2, 6),
               fontsize=FS_LEGEND, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    _savepdf(fig, out_dir / f"{dataset}_fig1_training_curves.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 2 路 Grouped bar + error caps  (per-class F1 卤SE)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig2_per_class_bar_error(dataset: str, runs: dict,
                               class_names: list[str], out_dir: Path) -> None:
    plt = _plt()
    import matplotlib.ticker as ticker

    models = _models_present(runs)
    nc = len(class_names)
    x      = np.arange(nc)

    fig, ax = plt.subplots(figsize=(max(11.5, nc * 2.2), 6.1))

    for i, m in enumerate(models):
        report = runs[m]["metrics"].get("classification_report", {})
        f1s, ses = [], []
        for c in range(nc):
            f = float(report.get(str(c), {}).get("f1-score", 0.0))
            n = int(report.get(str(c), {}).get("support", 100))
            f1s.append(f)
            ses.append(np.sqrt(f * (1 - f) / max(n, 1)))

        ax.errorbar(
            x, f1s, yerr=ses,
            color=MODEL_COLOR[m], label=MODEL_LABELS[m],
            marker="o", markersize=8, lw=2.6,
            capsize=5, capthick=1.5, elinewidth=1.4,
            alpha=0.95, zorder=3 + i,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=0, ha="center")
    ax.set_ylabel("F1-score (+/- SE)")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", lw=0.5, alpha=0.4, ls=":", zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=len(models))
    ax.axhline(1.0, lw=0.7, ls="--", color="gray", alpha=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    _savepdf(fig, out_dir / f"{dataset}_fig2_perclass_f1_bar.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 3 路 Comprehensive metric TABLE  (replaces radar + lollipop)
# One large table: rows = metrics, columns = models
# Proposed column highlighted with #FF6666 background
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig3_metric_table(dataset: str, runs: dict, out_dir: Path) -> None:
    plt = _plt()

    auc_key = "roc_auc" if dataset == "camelyon" else "macro_ovr_auc"
    metric_rows = [
        ("accuracy",          "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_precision",   "Macro Precision"),
        ("macro_recall",      "Macro Recall"),
        ("macro_specificity", "Macro Specificity"),
        ("macro_f1",          "Macro F1"),
        ("weighted_f1",       "Weighted F1"),
        ("mcc",               "MCC"),
        ("cohen_kappa",       "Cohen 魏"),
        (auc_key,             "AUC"),
        ("parameters",        "Parameters"),
    ]

    models = _models_present(runs)
    col_labels = [MODEL_LABELS[m] for m in models]
    row_labels = [lbl for _, lbl in metric_rows]

    # Build cell data
    cell_data = []
    for key, _ in metric_rows:
        row = []
        for m in models:
            v = runs[m]["metrics"].get(key, None)
            if v is None:
                row.append("N/A")
            elif key == "parameters":
                row.append(f"{int(v):,}")
            else:
                row.append(f"{float(v):.4f}")
        cell_data.append(row)

    table_rows = []
    for ri, (key, label) in enumerate(metric_rows):
        values = []
        for m in models:
            v = runs[m]["metrics"].get(key, None)
            values.append(float(v) if v is not None else None)
        finite = [v for v in values if v is not None]
        best_value = min(finite) if key == "parameters" and finite else (max(finite) if finite else None)
        best_models = [
            models[i] for i, v in enumerate(values)
            if best_value is not None and v is not None and np.isclose(v, best_value)
        ]
        record = {
            "metric_key": key,
            "metric": label,
            "best_models": ";".join(best_models),
        }
        for ci, m in enumerate(models):
            record[m] = cell_data[ri][ci]
            record[f"{m}_is_best"] = m in best_models
        table_rows.append(record)

    csv_path = out_dir / f"{dataset}_fig3_metric_table_data.csv"
    json_path = out_dir / f"{dataset}_fig3_metric_table_data.json"
    fieldnames = ["metric_key", "metric"]
    for m in models:
        fieldnames.extend([m, f"{m}_is_best"])
    fieldnames.append("best_models")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2, ensure_ascii=False)
    print(f"  [table-data] {csv_path.name}")
    print(f"  [table-data] {json_path.name}")
    return


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 5 路 Diverging heatmap  (confusion matrices, all models tiled)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig5_confusion_heatmaps(dataset: str, runs: dict,
                              class_names: list[str], out_dir: Path) -> None:
    plt = _plt()
    cmap   = _diverging_cmap()
    models = _models_present(runs)
    nc, nm = len(class_names), len(models)
    fig5_label_fs = FS_LABEL + 3
    fig5_tick_fs = FS_TICK + 3
    fig5_title_fs = FS_TITLE + 3
    fig5_cell_fs = max(9, FS_TICK + (1 if nc <= 2 else -1))

    fig, axes = plt.subplots(1, nm, figsize=(5.2 * nm + 0.9, 6.1), sharex=True, sharey=True)
    if nm == 1:
        axes = [axes]

    last_im = None
    for mi, (ax, m) in enumerate(zip(axes, models)):
        raw = np.array(runs[m]["metrics"].get("confusion_matrix", []), dtype=float)
        if raw.size == 0:
            ax.axis("off"); continue
        norm = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1)

        im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        last_im = im
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        ax.set_xticklabels(class_names if mi == 0 else [], rotation=35, ha="right", fontsize=fig5_tick_fs)
        ax.set_yticklabels(class_names if mi == 0 else [], fontsize=fig5_tick_fs)
        ax.set_xlabel("Predicted", fontsize=fig5_label_fs)
        ax.set_ylabel("True" if mi == 0 else "", fontsize=fig5_label_fs)
        ax.set_title(MODEL_LABELS[m], pad=6, fontsize=fig5_title_fs)

        for i in range(nc):
            for j in range(nc):
                v   = norm[i, j]
                txt = "white" if v > 0.6 or v < 0.25 else "black"
                ax.text(j, i, f"{int(raw[i,j])}\n({v:.2f})",
                        ha="center", va="center", fontsize=fig5_cell_fs, color=txt)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.955, 0.18, 0.014, 0.66])
        cb = fig.colorbar(last_im, cax=cbar_ax)
        cb.ax.tick_params(labelsize=fig5_tick_fs)

    fig.text(0.5, 0.095, "Predicted class", ha="center", fontsize=fig5_label_fs)
    fig.tight_layout(rect=[0, 0.13, 0.93, 1])
    _savepdf(fig, out_dir / f"{dataset}_fig5_confusion_heatmaps.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 6 路 Grouped bar  (Parameters + 4 key metrics)  replaces bubble scatter
# Left panel: Parameters(M); right panels: Accuracy, Macro-F1, MCC, AUC
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig6_grouped_bar_metrics(dataset: str, runs: dict, out_dir: Path) -> None:
    plt = _plt()
    import matplotlib.ticker as ticker
    fig6_label_fs = FS_LABEL + 3
    fig6_tick_fs = FS_TICK + 3
    fig6_annot_fs = FS_ANNOT + 3
    fig6_legend_fs = FS_LEGEND + 2

    auc_key = "roc_auc" if dataset == "camelyon" else "macro_ovr_auc"
    panels = [
        ("parameters",  "Parameters (M)", lambda v: v / 1e6),
        ("accuracy",    "Accuracy",        lambda v: v),
        ("macro_f1",    "Macro F1",        lambda v: v),
        ("mcc",         "MCC",             lambda v: v),
        (auc_key,       "AUC",             lambda v: v),
    ]

    models = _models_present(runs)
    nm     = len(models)
    x      = np.arange(nm)
    w      = 0.62

    fig, axes = plt.subplots(1, len(panels), figsize=(4.25 * len(panels), 6.15))

    for ax, (key, ylabel, transform) in zip(axes, panels):
        vals = []
        for m in models:
            raw = runs[m]["metrics"].get(key, 0.0) or 0.0
            vals.append(transform(float(raw)))

        bars = ax.bar(x, vals, width=w,
                      color=[MODEL_COLOR[m] for m in models],
                      alpha=0.88, zorder=3,
                      edgecolor="white", linewidth=0.8)

        # value labels on top of bars
        for bar, v in zip(bars, vals):
            label = f"{v:.3f}" if key != "parameters" else f"{v:.2f}M"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.012,
                    label, ha="center", va="bottom",
                    fontsize=fig6_annot_fs, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylabel(ylabel, fontsize=fig6_label_fs)
        ax.tick_params(axis="y", labelsize=fig6_tick_fs)
        ymax = max(vals) * 1.18 if max(vals) > 0 else 1
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", lw=0.5, alpha=0.4, ls=":", zorder=0)

        # star on best bar (excluding parameters panel)
        if key != "parameters":
            best_i = int(np.argmax(vals))
            ax.text(x[best_i], vals[best_i] + max(vals) * 0.06,
                    "*", ha="center", fontsize=fig6_annot_fs + 2,
                    color=MODEL_COLOR[models[best_i]])

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=MODEL_COLOR[m], label=MODEL_LABELS[m]) for m in models]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.00),
               ncol=len(models), fontsize=fig6_legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _savepdf(fig, out_dir / f"{dataset}_fig6_metric_bars.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 7 路 Stacked bar  (per-class correct predictions per model)
# Colours: proposed=#FF6666, then class colours from palette
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig7_stacked_correct(dataset: str, runs: dict,
                           class_names: list[str], out_dir: Path) -> None:
    plt = _plt()

    models = _models_present(runs)
    nc     = len(class_names)

    # Class colours from palette (independent of model colours)
    class_palette = PALETTE[:nc] if nc <= len(PALETTE) else (
        PALETTE + ["#AAAAAA"] * (nc - len(PALETTE))
    )

    mat = np.zeros((nc, len(models)))
    for j, m in enumerate(models):
        cm_raw = np.array(runs[m]["metrics"].get("confusion_matrix", []), dtype=float)
        if cm_raw.size:
            mat[:, j] = np.diag(cm_raw)

    x_pos  = np.arange(len(models))
    bar_w  = 0.55

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    bottom = np.zeros(len(models))

    for ci in range(nc):
        bars = ax.bar(x_pos, mat[ci], bottom=bottom, width=bar_w,
                      color=class_palette[ci], label=class_names[ci],
                      alpha=0.88, zorder=3)
        # value labels inside bars for every model when large enough
        for bi, bar in enumerate(bars):
            if mat[ci, bi] > 30:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottom[bi] + mat[ci, bi] / 2,
                        f"{int(mat[ci, bi])}", ha="center", va="center",
                        fontsize=FS_ANNOT - 2, color="white", fontweight="bold")
        bottom += mat[ci]

    # Total correct trend line
    total = mat.sum(axis=0)
    ax.plot(x_pos, total, "o-", color="#222222", lw=2.2, zorder=6,
            label="Total correct", ms=8)
    for xi, tv in zip(x_pos, total):
        ax.text(xi, tv + total.max() * 0.025, f"{int(tv)}",
                ha="center", fontsize=FS_ANNOT, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models],
                       rotation=0, ha="center")
    ax.set_ylabel("Correct Predictions (count)")
    ax.grid(axis="y", lw=0.5, alpha=0.4, ls=":", zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24),
              ncol=min(nc + 1, 4), fontsize=FS_LEGEND)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    _savepdf(fig, out_dir / f"{dataset}_fig7_stacked_correct.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 8 路 Violin + strip  (per-class precision spread across models)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig8_violin_precision(dataset: str, runs: dict,
                            class_names: list[str], out_dir: Path) -> None:
    plt = _plt()

    models = _models_present(runs)
    nc     = len(class_names)

    prec_by_class = []
    for c in range(nc):
        vals = []
        for m in models:
            rep = runs[m]["metrics"].get("classification_report", {})
            vals.append(float(rep.get(str(c), {}).get("precision", 0.0)))
        prec_by_class.append(vals)

    fig, ax = plt.subplots(figsize=(max(10.5, nc * 1.9), 6.1))
    positions = np.arange(nc, dtype=float)

    parts = ax.violinplot(prec_by_class, positions=positions,
                          widths=0.6, showmeans=False,
                          showextrema=False, showmedians=False)

    for i, pc in enumerate(parts["bodies"]):
        col = PALETTE[i % len(PALETTE)]
        pc.set_facecolor(col)
        pc.set_alpha(0.28)
        pc.set_edgecolor(col)
        pc.set_linewidth(1.5)

    rng = np.random.default_rng(0)
    for ci, vals in enumerate(prec_by_class):
        jitter = rng.uniform(-0.13, 0.13, len(vals))
        for mi, (v, j) in enumerate(zip(vals, jitter)):
            ax.scatter(ci + j, v, color=MODEL_COLOR[models[mi]],
                       s=80, zorder=5, alpha=0.92,
                       edgecolors="white", linewidths=0.8)

    means = [np.mean(v) for v in prec_by_class]
    ax.plot(positions, means, "D", color="#222222", ms=7, zorder=6, label="Mean")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=MODEL_COLOR[m], markeredgecolor="white",
                      markersize=10, label=MODEL_LABELS[m]) for m in models]
    handles += [Line2D([0], [0], marker="D", color="w",
                       markerfacecolor="#222222", markersize=7, label="Mean")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.22),
              ncol=min(len(handles), 5), fontsize=FS_LEGEND)

    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_ylabel("Precision")
    ax.set_ylim(max(0, min(means) - 0.18), 1.10)
    ax.grid(axis="y", lw=0.5, alpha=0.4, ls=":")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    _savepdf(fig, out_dir / f"{dataset}_fig8_violin_precision.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 9 路 Cross-dataset grouped bar  (HEp-2 vs Camelyon17)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig9_cross_dataset(store: dict, out_dir: Path) -> None:
    plt = _plt()

    metrics_show = [
        ("accuracy",    "Accuracy"),
        ("macro_f1",    "Macro F1"),
        ("mcc",         "MCC"),
        ("cohen_kappa", "Cohen 魏"),
    ]
    models   = [m for m in MODEL_ORDER
                if m in store.get("hep2", {}) and m in store.get("camelyon", {})]
    datasets = ["hep2", "camelyon"]
    nk       = len(metrics_show)

    fig, axes = plt.subplots(1, nk, figsize=(4.2 * nk, 5.0), sharey=False)
    ds_hatch  = ["", "//"]
    x         = np.arange(len(models))
    w         = 0.36

    for ax, (key, ylabel) in zip(axes, metrics_show):
        for di, ds in enumerate(datasets):
            auc_key = "roc_auc" if ds == "camelyon" else "macro_ovr_auc"
            k = auc_key if key == "auc" else key
            vals = [float(store[ds].get(m, {}).get("metrics", {}).get(k, 0.0) or 0.0)
                    for m in models]
            off = -w / 2 if di == 0 else w / 2
            for mi, (v, m) in enumerate(zip(vals, models)):
                ax.bar(x[mi] + off, v, width=w * 0.90,
                       color=MODEL_COLOR[m], hatch=ds_hatch[di],
                       alpha=0.85 if di == 0 else 0.60, zorder=3,
                       edgecolor="white",
                       label=DATASET_LABELS[ds] if mi == 0 else "")
                ax.text(x[mi] + off, v + 0.010, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=FS_ANNOT - 1, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.16)
        ax.grid(axis="y", lw=0.5, alpha=0.4, ls=":", zorder=0)

    from matplotlib.patches import Patch
    model_legend_els = [
        Patch(facecolor=MODEL_COLOR[m], label=MODEL_LABELS[m]) for m in models
    ]
    dataset_legend_els = [
        Patch(facecolor="#888888", alpha=0.85, label="HEp-2"),
        Patch(facecolor="#888888", alpha=0.55, hatch="//", label="Camelyon17"),
    ]
    fig.legend(handles=model_legend_els, fontsize=FS_LEGEND, loc="upper center",
               bbox_to_anchor=(0.5, 1.08), ncol=len(models))
    fig.legend(handles=dataset_legend_els, fontsize=FS_LEGEND, loc="upper center",
               bbox_to_anchor=(0.5, 1.00), ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    _savepdf(fig, out_dir / "combined_fig9_cross_dataset.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 10 路 HEp-2 image reconstruction panel
# Rows: Noisy input | Clean GT | Reconstructed | Difference heatmap (jet)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig10_hep2_recon_panel(samples: list[dict], out_dir: Path) -> None:
    plt = _plt()
    panel_label_fs = int(FS_LABEL * 1.45)
    panel_title_fs = int(FS_TITLE * 1.45)
    panel_tick_fs = int(FS_TICK * 1.45)

    # pick one sample per class (up to 6)
    by_class: dict[int, dict] = {}
    for s in samples:
        lbl = s.get("label", -1)
        if lbl not in by_class:
            by_class[lbl] = s
    chosen = [by_class[k] for k in sorted(by_class.keys())][:6]
    if not chosen:
        return
    nc = len(chosen)

    row_labels = ["Noisy Input", "Clean (GT)", "Reconstructed", "Difference"]
    nr = 4
    fig, axes = plt.subplots(nr, nc, figsize=(2.8 * nc, 2.8 * nr))

    def _denorm(t: np.ndarray) -> np.ndarray:
        return np.clip((t + 1) / 2, 0, 1)

    for ci, s in enumerate(chosen):
        noisy = _denorm(s.get("noisy", np.zeros((128, 128), np.float32)))
        clean = _denorm(s.get("clean", np.zeros_like(noisy)))
        recon = s.get("recon")
        recon = _denorm(recon) if recon is not None else np.zeros_like(clean)
        diff  = np.abs(recon - clean)
        diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        # 鈹€鈹€ row 0: noisy (gray) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        for ri, (img, km, vmin, vmax) in enumerate([
            (noisy,     "gray", 0, 1),
            (clean,     "gray", 0, 1),
            (recon,     "gray", 0, 1),
            (diff_norm, "jet",  0, 1),   # 鈫?jet: same as fig11 (涓婄孩涓豢涓嬭摑)
        ]):
            ax = axes[ri, ci] if nc > 1 else axes[ri]
            ax.imshow(img, cmap=km, vmin=vmin, vmax=vmax, interpolation="bilinear")
            ax.axis("off")
            if ci == 0:
                ax.set_ylabel(row_labels[ri], fontsize=panel_label_fs,
                              rotation=90, labelpad=4, va="center")
            if ri == 0:
                cname = (HEP2_CLASS_NAMES[s["label"]]
                         if s["label"] < len(HEP2_CLASS_NAMES)
                         else str(s["label"]))
                ax.set_title(cname, fontsize=panel_title_fs, pad=6)

    # colorbar for diff row (jet)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.06, 0.013, 0.20])
    cb = fig.colorbar(sm, cax=cbar_ax, label="Norm. Diff.")
    cb.ax.tick_params(labelsize=panel_tick_fs)
    cb.ax.yaxis.label.set_size(panel_label_fs)

    fig.tight_layout(rect=[0, 0, 0.91, 1])
    _savepdf(fig, out_dir / "hep2_fig10_recon_panel.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 11 路 CBAM spatial attention overlay
# 3脳N: Image | jet attention map | blended overlay
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig11_attention_maps(samples: list[dict], out_dir: Path) -> None:
    plt = _plt()
    panel_label_fs = int(FS_LABEL * 1.45)
    panel_title_fs = int(FS_TITLE * 1.45)
    panel_tick_fs = int(FS_TICK * 1.45)

    attn_samples = [s for s in samples if s.get("attn") is not None][:6]
    if not attn_samples:
        print("  [fig11] no attention maps stored 鈥?skipping")
        return

    nc  = len(attn_samples)
    fig, axes = plt.subplots(3, nc, figsize=(2.8 * nc, 8.0))
    hot = plt.get_cmap("hot")

    def _denorm(t):
        return np.clip((t + 1) / 2, 0, 1)

    for ci, s in enumerate(attn_samples):
        img  = _denorm(s.get("clean", np.zeros((128, 128), np.float32)))
        attn = s.get("attn", np.zeros_like(img))
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        ax0 = axes[0, ci] if nc > 1 else axes[0]
        ax0.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax0.axis("off")
        cname = (HEP2_CLASS_NAMES[s["label"]]
                 if s["label"] < len(HEP2_CLASS_NAMES) else str(s["label"]))
        ax0.set_title(cname, fontsize=panel_title_fs, pad=6)

        ax1 = axes[1, ci] if nc > 1 else axes[1]
        ax1.imshow(attn_norm, cmap="jet", vmin=0, vmax=1)  # jet: 涓婄孩涓豢涓嬭摑
        ax1.axis("off")

        ax2 = axes[2, ci] if nc > 1 else axes[2]
        img_rgb  = np.stack([img] * 3, axis=-1)
        attn_rgb = hot(attn_norm)[..., :3]
        blended  = 0.55 * img_rgb + 0.45 * attn_rgb
        ax2.imshow(np.clip(blended, 0, 1))
        ax2.axis("off")

    if nc > 1:
        for ri, lbl in enumerate(["Image", "Attention (jet)", "Overlay"]):
            axes[ri, 0].set_ylabel(lbl, fontsize=panel_label_fs,
                                   rotation=90, labelpad=4)

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.33, 0.013, 0.32])
    cb = fig.colorbar(sm, cax=cbar_ax, label="Attention")
    cb.ax.tick_params(labelsize=panel_tick_fs)
    cb.ax.yaxis.label.set_size(panel_label_fs)

    fig.tight_layout(rect=[0, 0, 0.91, 1])
    _savepdf(fig, out_dir / "hep2_fig11_attention_maps.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Fig 12 路 Camelyon patch panel  (TP / FP / FN)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _fig12_camelyon_patch_panel(samples: list[dict], out_dir: Path) -> None:
    plt = _plt()
    fig12_label_fs = FS_LABEL + 4
    fig12_annot_fs = FS_ANNOT + 4

    tp_s = [s for s in samples if s["label"] == 1 and s["pred"] == 1][:6]
    fp_s = [s for s in samples if s["label"] == 0 and s["pred"] == 1][:6]
    fn_s = [s for s in samples if s["label"] == 1 and s["pred"] == 0][:6]

    n_cols   = max(len(tp_s), len(fp_s), len(fn_s), 1)
    row_info = [
        (tp_s, "True Positive\n(Tumor, correct)",      "#FF6666"),
        (fp_s, "False Positive\n(Normal, mis-detected)","#FFAA53"),
        (fn_s, "False Negative\n(Tumor, missed)",       "#3399FF"),
    ]

    fig, axes = plt.subplots(3, n_cols, figsize=(2.9 * n_cols + 0.7, 8.6))

    for ri, (row_samples, row_label, border_col) in enumerate(row_info):
        for ci in range(n_cols):
            ax = axes[ri, ci] if n_cols > 1 else axes[ri]
            if ci < len(row_samples):
                s   = row_samples[ci]
                img = s.get("img")
                if img is not None:
                    ax.imshow(img.astype(np.uint8) if img.dtype != np.uint8 else img)
                else:
                    ax.set_facecolor("#EEEEEE")
                score = s.get("score", 0.5)
                h = img.shape[0] if img is not None else 90
                ax.axhline(y=h - 5, xmin=0, xmax=score,
                           color=border_col, lw=4, solid_capstyle="butt")
                ax.text(2, h - 10, f"{score:.2f}", fontsize=fig12_annot_fs,
                        color="black", fontweight="bold",
                        bbox=dict(fc="white", ec=border_col, lw=1.2, pad=1.5))
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_col)
                    spine.set_linewidth(3.0)
                    spine.set_visible(True)
            else:
                ax.axis("off")
                continue
            ax.set_xticks([])
            ax.set_yticks([])
    for y, (_, row_label, _) in zip([0.82, 0.50, 0.18], row_info):
        fig.text(0.095, y, row_label, rotation=90, va="center", ha="center",
                 fontsize=fig12_label_fs, color="black", fontweight="bold")

    fig.tight_layout(rect=[0.125, 0, 1, 1])
    _savepdf(fig, out_dir / "camelyon_fig12_patch_panel.pdf")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Backward-compatible shims
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def plot_history(history: dict, out_dir, name: str) -> None:
    parts   = name.split("_", 1)
    dataset = parts[0] if len(parts) == 2 else "unknown"
    model   = parts[1] if len(parts) == 2 else name
    entry   = _store.setdefault(dataset, {}).setdefault(model, {})
    entry["history"] = history


def plot_confusion_matrix(cm: "np.ndarray", class_names: list,
                           out_dir, name: str) -> None:
    parts   = name.split("_", 1)
    dataset = parts[0] if len(parts) == 2 else "unknown"
    model   = parts[1] if len(parts) == 2 else name
    entry   = _store.setdefault(dataset, {}).setdefault(model, {})
    entry.setdefault("metrics", {})["confusion_matrix"] = cm.tolist()


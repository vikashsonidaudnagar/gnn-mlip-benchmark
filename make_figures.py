"""
Publication-ready figures from results/*.json.

  python make_figures.py --data_dir "C:/.../mlearn-master/mlearn-master/data"

Writes to paper_figures/ (each as vector PDF for Overleaf + 300 dpi PNG):
  fig_errors            (a) energy MAE, (b) relative force error, per element
  fig_parity_<El>       predicted vs DFT energies and forces, one column per model
  fig_per_config        energy MAE per configuration type (AIMD, Elastic, ...)
  fig_training_<El>     validation error vs epoch
  fig_learning_curves   force MAE vs training set size (if fractions were run)

Widths follow typical two-column journals: 3.4 in (single column), 7.0 in (double).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="paper_figures")
ap.add_argument("--parity_elements", nargs="+", default=["Cu", "Si"])
ap.add_argument("--curve_element", default="Cu")
ap.add_argument("--parity_models", nargs="+",
                default=["schnet", "dimenetpp", "painn", "mace"],
                help="models shown in parity plots (max ~4 for readability)")
a = ap.parse_args()

# ---------------------------------------------------------------- style
SINGLE, DOUBLE = 3.4, 7.0
ELEMENT_ORDER = ["Cu", "Ni", "Li", "Mo", "Si", "Ge"]     # metals -> covalent
COVALENT = {"Si", "Ge"}
MODEL_ORDER = ["schnet", "dimenetpp", "painn", "mace", "chgnet_zs", "chgnet_ft"]
NAMES = {"schnet": "SchNet", "dimenetpp": "DimeNet++", "painn": "PaiNN", "mace": "MACE",
         "chgnet_zs": "CHGNet (zero-shot)", "chgnet_ft": "CHGNet (fine-tuned)"}
COLORS = {"schnet": "#0072B2", "dimenetpp": "#E69F00",    # Okabe-Ito palette
          "painn": "#009E73", "mace": "#CC79A7",           # (colour-blind safe)
          "chgnet_zs": "#999999", "chgnet_ft": "#D55E00"}
NO_TRAINING = {"chgnet_zs"}     # deterministic, no seeds -> never marked "single run"
GROUP_ORDER = ["AIMD-NVT", "Elastic", "Surface", "Vacancy"]
GROUP_LABEL = {"AIMD-NVT": "AIMD", "Elastic": "Elastic",
               "Surface": "Surface", "Vacancy": "Vacancy"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.0, "hatch.linewidth": 0.5,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

out = Path(a.out)
out.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {out / name}.pdf / .png")


def panel_label(ax, i, x=-0.16, y=1.0):
    ax.text(x, y, f"({'abcdefghijklmnop'[i]})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------- data
res = [json.load(open(p)) for p in sorted(Path(a.results).glob("*.json"))]
if not res:
    raise SystemExit(f"No results in {a.results}/")
full = [r for r in res if r.get("train_fraction", 1.0) == 1.0]
groups = defaultdict(list)
for r in full:
    groups[(r["element"], r["model"])].append(r)
elements = [e for e in ELEMENT_ORDER if any(k[0] == e for k in groups)]
models = [m for m in MODEL_ORDER if any(k[1] == m for k in groups)]
first_seed = {k: min(v, key=lambda r: r.get("seed", 0)) for k, v in groups.items()}

mean_f = {}
for el in elements:
    with open(Path(a.data_dir) / el / "test.json") as f:
        d = json.load(f)
    mean_f[el] = float(np.abs(np.concatenate(
        [np.ravel(e["outputs"]["forces"]) for e in d])).mean())


def stat(el, m, key, scale=1.0):
    runs = groups.get((el, m), [])
    if not runs:
        return np.nan, 0.0, False
    v = np.array([r[key] for r in runs]) * scale
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v) == 1


def plain_log_axis(ax):
    """Log-scale y axis labelled 0.02, 0.03 ... instead of 2x10^-2."""
    f = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(f)
    lo, hi = ax.get_ylim()
    sparse = FuncFormatter(lambda v, _: f"{v:g}" if v > 0 and
                           round(v / 10 ** np.floor(np.log10(v))) in (2, 5) else "")
    ax.yaxis.set_minor_formatter(f if np.log10(hi / lo) < 1.2 else sparse)


def covalent_shading(ax, els):
    idx = [i for i, e in enumerate(els) if e in COVALENT]
    if idx:
        ax.axvspan(min(idx) - 0.5, max(idx) + 0.5, color="0.92", zorder=0, lw=0)
        ax.text((min(idx) + max(idx)) / 2, 0.97, "covalent", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, style="italic", color="0.35")


# ---------------------------------------------------------------- Fig: errors
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
x = np.arange(len(elements))
w = 0.8 / len(models)
any_single = False
for i, (ax, key, ylabel, rel) in enumerate(zip(
        axes, ["energy_mae_meV_atom", "force_mae_eV_A"],
        ["Energy MAE (meV atom$^{-1}$)", "Relative force error (%)"], [False, True])):
    covalent_shading(ax, elements)
    for k, m in enumerate(models):
        vals = [stat(e, m, key, 100.0 / mean_f[e] if rel else 1.0) for e in elements]
        bars = ax.bar(x - 0.4 + w * (k + 0.5), [v[0] for v in vals], w * 0.92,
                      yerr=[v[1] for v in vals], color=COLORS[m], edgecolor="black",
                      linewidth=0.4, zorder=2,
                      error_kw=dict(elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3))
        for bar, v in zip(bars, vals):
            if v[2] and m not in NO_TRAINING and any(len(groups.get((e, mm), [])) > 1
                            for e in elements for mm in models):
                bar.set_hatch("/////")
                any_single = True
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.set_xlim(-0.6, len(elements) - 0.4)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    panel_label(ax, i, x=-0.13)
handles = [Patch(facecolor=COLORS[m], edgecolor="black", linewidth=0.4, label=NAMES[m])
           for m in models]
if any_single:
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="/////",
                         linewidth=0.4, label="single run"))
n_leg = len(handles) if len(handles) <= 5 else int(np.ceil(len(handles) / 2))
fig.legend(handles=handles, loc="upper center", ncol=n_leg, frameon=False,
           bbox_to_anchor=(0.5, 1.08 if len(handles) <= 5 else 1.14),
           handlelength=1.2, columnspacing=1.2)
fig.tight_layout(w_pad=2.0)
save(fig, "fig_errors")

# ---------------------------------------------------------------- Fig: parity
for el in a.parity_elements:
    ms = [m for m in models if m in a.parity_models and (el, m) in first_seed
          and "predictions" in first_seed[(el, m)]]
    if not ms:
        continue
    n = len(ms)
    size = DOUBLE / n
    fig, axes = plt.subplots(2, n, figsize=(DOUBLE, 2 * size * 0.92), squeeze=False)
    for col, m in enumerate(ms):
        r = first_seed[(el, m)]
        p = r["predictions"]
        rows = [(np.array(p["e_true_per_atom"]), np.array(p["e_pred_per_atom"]),
                 "energy (eV atom$^{-1}$)", r["energy_mae_meV_atom"], "meV atom$^{-1}$"),
                (np.array(p["f_true"]), np.array(p["f_pred"]),
                 "force (eV Å$^{-1}$)", r["force_mae_eV_A"] * 1000, "meV Å$^{-1}$")]
        for row, (t, q, lab, mae, unit) in enumerate(rows):
            ax = axes[row][col]
            lo, hi = min(t.min(), q.min()), max(t.max(), q.max())
            pad = 0.05 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.6, ls="--", zorder=1)
            ax.scatter(t, q, s=4 if row == 0 else 1.5, alpha=0.8 if row == 0 else 0.35,
                       color=COLORS[m], edgecolors="none", rasterized=True, zorder=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.text(0.05, 0.95, f"MAE = {mae:.1f} {unit}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=7)
            ax.set_xlabel(f"DFT {lab}")
            if col == 0:
                ax.set_ylabel(f"Predicted {lab}")
            if row == 0:
                ax.set_title(f"{NAMES[m]} ({el})")
            panel_label(ax, row * n + col, x=-0.30 if col == 0 else -0.22, y=1.02)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, f"fig_parity_{el}")

# ---------------------------------------------------------------- Fig: per configuration
have_groups = [e for e in elements
               if any("per_group" in r for m in models for r in groups.get((e, m), []))]
if have_groups:
    ncol = 3
    nrow = int(np.ceil(len(have_groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 1.9 * nrow), squeeze=False)
    gx = np.arange(len(GROUP_ORDER))
    for i, el in enumerate(have_groups):
        ax = axes[i // ncol][i % ncol]
        for k, m in enumerate(models):
            runs = [r for r in groups.get((el, m), []) if "per_group" in r]
            vals = [np.mean([r["per_group"][g]["energy_mae_meV_atom"] for r in runs
                             if g in r["per_group"]]) if runs else np.nan
                    for g in GROUP_ORDER]
            ax.bar(gx - 0.4 + w * (k + 0.5), vals, w * 0.92, color=COLORS[m],
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(gx)
        ax.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER])
        ax.tick_params(axis="x", which="both", top=False, bottom=False)
        ax.set_title(el)
        panel_label(ax, i, x=-0.20, y=1.02)
        if i % ncol == 0:
            ax.set_ylabel("Energy MAE (meV atom$^{-1}$)")
    for j in range(len(have_groups), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    nl = len(models) if len(models) <= 5 else int(np.ceil(len(models) / 2))
    fig.legend(handles=handles[:len(models)], loc="upper center", ncol=nl,
               frameon=False, bbox_to_anchor=(0.5, 1.04 if len(models) <= 5 else 1.08))
    fig.tight_layout()
    save(fig, "fig_per_config")

# ---------------------------------------------------------------- Fig: training curves
el = a.curve_element
ms = [m for m in models if (el, m) in first_seed and first_seed[(el, m)].get("history")]
if ms:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2))
    for m in ms:
        h = first_seed[(el, m)]["history"]
        ep = [x["epoch"] for x in h]
        axes[0].semilogy(ep, [x["energy_mae_meV_atom"] for x in h], color=COLORS[m],
                         lw=0.8, label=NAMES[m])
        axes[1].semilogy(ep, [x["force_mae_eV_A"] for x in h], color=COLORS[m],
                         lw=0.8, label=NAMES[m])
    axes[0].set_ylabel("Val. energy MAE (meV atom$^{-1}$)")
    axes[1].set_ylabel("Val. force MAE (eV Å$^{-1}$)")
    for i, ax in enumerate(axes):
        ax.set_xlabel("Epoch")
        plain_log_axis(ax)
        panel_label(ax, i, x=-0.20, y=1.02)
    axes[1].legend(frameon=False)
    fig.tight_layout(w_pad=2.0)
    save(fig, f"fig_training_{el}")

# ---------------------------------------------------------------- Fig: learning curves
curves = defaultdict(lambda: defaultdict(list))
for r in res:
    if r.get("n_train", 0) > 0:
        curves[(r["element"], r["model"])][r["n_train"]].append(r["force_mae_eV_A"])
lc_elements = [e for e in elements
               if any(len(curves[(e, m)]) > 1 for m in models)]
if lc_elements:
    ncol = min(3, len(lc_elements))
    nrow = int(np.ceil(len(lc_elements) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE if ncol > 1 else SINGLE,
                                                  2.1 * nrow), squeeze=False)
    for i, el in enumerate(lc_elements):
        ax = axes[i // ncol][i % ncol]
        for m in models:
            c = curves[(el, m)]
            if len(c) < 2:
                continue
            ns = sorted(c)
            mu = [np.mean(c[n]) for n in ns]
            sd = [np.std(c[n], ddof=1) if len(c[n]) > 1 else 0 for n in ns]
            ax.errorbar(ns, mu, yerr=sd, color=COLORS[m], marker="o", ms=3, lw=0.8,
                        capsize=1.5, elinewidth=0.6, label=NAMES[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log_axis(ax)
        all_n = sorted({n for m in models if len(curves[(el, m)]) > 1
                        for n in curves[(el, m)]})
        ax.set_xticks(all_n)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(el)
        ax.set_xlabel("Training structures")
        if i % ncol == 0:
            ax.set_ylabel("Force MAE (eV Å$^{-1}$)")
        panel_label(ax, i, x=-0.24, y=1.02)
    for j in range(len(lc_elements), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_learning_curves")

print(f"\nAll figures in {out.resolve()}")
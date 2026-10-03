"""
Collect all results/*.json and produce comparison tables and figures:

  figures/summary_table.csv      every run, all metrics
  figures/summary_mean_std.csv   mean and std over seeds
  figures/energy_mae.png         energy MAE per element, grouped by model
  figures/force_mae.png          force MAE per element, grouped by model
  figures/parity_<El>.png        predicted vs DFT (energy and forces), per model
  figures/learning_curve_<El>.png  if several train fractions were run
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = [json.load(open(p)) for p in sorted(Path("results").glob("*.json"))]
if not res:
    raise SystemExit("No results found in results/. Run train.py or run_all.py first.")
out = Path("figures"); out.mkdir(exist_ok=True)

# ---- table ---------------------------------------------------------------
cols = ["element", "model", "train_fraction", "seed", "n_train", "n_params", "epochs_run",
        "energy_mae_meV_atom", "energy_rmse_meV_atom", "force_mae_eV_A",
        "force_rmse_eV_A", "train_time_s", "test_time_s"]
with open(out / "summary_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in res:
        w.writerow([r.get(c, "") for c in cols])

full = [r for r in res if r["train_fraction"] == 1.0]
by_key = defaultdict(list)
for r in full:
    by_key[(r["element"], r["model"])].append(r)

def stats(runs, key):
    v = np.array([r[key] for r in runs])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

print(f"{'Element':8s}{'Model':11s}{'seeds':>6s}{'E-MAE (meV/atom)':>20s}{'F-MAE (eV/A)':>20s}")
with open(out / "summary_mean_std.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["element", "model", "n_seeds", "energy_mae_mean", "energy_mae_std",
                "force_mae_mean", "force_mae_std"])
    for (e, m), runs in sorted(by_key.items()):
        em, es = stats(runs, "energy_mae_meV_atom")
        fm, fs = stats(runs, "force_mae_eV_A")
        w.writerow([e, m, len(runs), em, es, fm, fs])
        print(f"{e:8s}{m:11s}{len(runs):6d}{em:12.2f} ± {es:5.2f}{fm:12.4f} ± {fs:.4f}")

# ---- bar charts ----------------------------------------------------------
elements = sorted({r["element"] for r in full})
models = sorted({r["model"] for r in full})
lookup = {k: sorted(v, key=lambda r: r.get("seed", 0))[0] for k, v in by_key.items()}
for key, label, fname in [("energy_mae_meV_atom", "Energy MAE (meV/atom)", "energy_mae.png"),
                          ("force_mae_eV_A", "Force MAE (eV/Å)", "force_mae.png")]:
    fig, ax = plt.subplots(figsize=(1.6 * len(elements) + 2, 4))
    width = 0.8 / len(models)
    x = np.arange(len(elements))
    for k, m in enumerate(models):
        ms = [stats(by_key[(e, m)], key) if (e, m) in by_key else (np.nan, 0)
              for e in elements]
        ax.bar(x + k * width - 0.4 + width / 2, [a for a, _ in ms], width,
               yerr=[b for _, b in ms], capsize=3, label=m)
    ax.set_xticks(x); ax.set_xticklabels(elements)
    ax.set_ylabel(label); ax.legend(); ax.set_title(label + " on test sets")
    fig.tight_layout(); fig.savefig(out / fname, dpi=150); plt.close(fig)

# ---- parity plots ----------------------------------------------------------
for e in elements:
    runs = [lookup[(e, m)] for m in models if (e, m) in lookup]
    fig, axes = plt.subplots(2, len(runs), figsize=(3.5 * len(runs), 7), squeeze=False)
    for k, r in enumerate(runs):
        p = r["predictions"]
        for row, (t, q, unit) in enumerate([(p["e_true_per_atom"], p["e_pred_per_atom"], "eV/atom"),
                                            (p["f_true"], p["f_pred"], "eV/Å")]):
            ax = axes[row][k]
            ax.scatter(t, q, s=4, alpha=0.5)
            lo, hi = min(min(t), min(q)), max(max(t), max(q))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel(f"DFT ({unit})"); ax.set_ylabel(f"{r['model']} ({unit})")
            ax.set_title(f"{e} {r['model']} {'energy' if row == 0 else 'forces'}")
    fig.tight_layout(); fig.savefig(out / f"parity_{e}.png", dpi=150); plt.close(fig)

# ---- learning curves -------------------------------------------------------
curves_raw = defaultdict(list)
for r in res:
    curves_raw[(r["element"], r["model"], r["n_train"])].append(r["force_mae_eV_A"])
curves = defaultdict(list)
for (e, m, n), v in curves_raw.items():
    curves[(e, m)].append((n, float(np.mean(v))))
for e in elements:
    lines = {m: sorted(curves[(e, m)]) for m in models if len(curves[(e, m)]) > 1}
    if not lines:
        continue
    fig, ax = plt.subplots(figsize=(5, 4))
    for m, pts in lines.items():
        ax.loglog(*zip(*pts), "o-", label=m)
    ax.set_xlabel("training structures"); ax.set_ylabel("Force MAE (eV/Å)")
    ax.set_title(f"Learning curve: {e}"); ax.legend()
    fig.tight_layout(); fig.savefig(out / f"learning_curve_{e}.png", dpi=150); plt.close(fig)

print(f"\nTables and figures written to {out.resolve()}")
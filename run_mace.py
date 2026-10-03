"""
Train MACE from scratch on mlearn elements and save results in the same format
as train.py (results/<El>_mace_f<fraction>_s<seed>.json).

  pip install mace-torch
  python run_mace.py --data_dir "C:/.../mlearn-master/mlearn-master/data" --elements Cu --epochs 5
  python run_mace.py --data_dir "C:/.../mlearn-master/mlearn-master/data" --seeds 0 1 2

Uses exactly the same train/validation split as train.py for a given seed,
and the official mlearn test.json as the test set.
MACE files (data, logs, checkpoints, models) go to mace_runs/<tag>/.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# PyTorch >= 2.6 refuses to load full pickled models by default; MACE models need this.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from ase.io import write

from data import ELEMENTS, entry_to_atoms


def to_frames(entries):
    frames = []
    for e in entries:
        at = entry_to_atoms(e)
        at.info["REF_energy"] = float(e["outputs"]["energy"])
        at.arrays["REF_forces"] = np.array(e["outputs"]["forces"], dtype=float)
        at.info["config_type"] = e.get("group", "unknown")
        frames.append(at)
    return frames


def split_like_train_py(n, seed, train_fraction):
    """Same permutation as train.py: random.seed(seed) then random.shuffle."""
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = max(1, n // 10)
    val, train = idx[:n_val], idx[n_val:]
    train = train[:max(1, int(round(len(train) * train_fraction)))]
    return train, val


def metrics(e_true, e_pred, n_atoms, f_true, f_pred):
    de = (e_pred - e_true) / n_atoms * 1000.0
    df = f_pred - f_true
    return {"energy_mae_meV_atom": float(np.abs(de).mean()),
            "energy_rmse_meV_atom": float(np.sqrt((de ** 2).mean())),
            "force_mae_eV_A": float(np.abs(df).mean()),
            "force_rmse_eV_A": float(np.sqrt((df ** 2).mean()))}


def find_model(work, tag):
    models = [p for p in work.glob(f"{tag}*.model") if "compiled" not in p.name]
    if not models:
        raise FileNotFoundError(f"No .model file found in {work}")
    for key in ("stagetwo", "swa"):          # prefer the stage-two (SWA) model
        for p in models:
            if key in p.name:
                return p
    return sorted(models, key=lambda p: len(p.name))[0]


def run(data_dir, element, seed, epochs, batch_size, cutoff, train_fraction, device):
    tag = f"{element}_mace_f{train_fraction:g}_s{seed}"
    work = Path("mace_runs") / tag
    work.mkdir(parents=True, exist_ok=True)

    folder = Path(data_dir) / element
    train_all = json.load(open(folder / "training.json"))
    test_entries = json.load(open(folder / "test.json"))
    tr_idx, va_idx = split_like_train_py(len(train_all), seed, train_fraction)
    train_frames = to_frames([train_all[i] for i in tr_idx])
    val_frames = to_frames([train_all[i] for i in va_idx])
    test_frames = to_frames(test_entries)
    write(work / "train.xyz", train_frames, format="extxyz")
    write(work / "valid.xyz", val_frames, format="extxyz")
    write(work / "test.xyz", test_frames, format="extxyz")
    print(f"[{tag}] train {len(train_frames)} val {len(val_frames)} test {len(test_frames)}")

    cmd = [sys.executable, "-m", "mace.cli.run_train",
           f"--name={tag}", "--model=MACE",
           f"--train_file={work / 'train.xyz'}",
           f"--valid_file={work / 'valid.xyz'}",
           f"--test_file={work / 'test.xyz'}",
           "--energy_key=REF_energy", "--forces_key=REF_forces",
           "--E0s=average", f"--r_max={cutoff}",
           "--num_channels=128", "--max_L=1", "--correlation=3",
           f"--batch_size={batch_size}", "--valid_batch_size=8",
           f"--max_num_epochs={epochs}",
           "--swa", "--ema", "--ema_decay=0.99", "--amsgrad",
           "--default_dtype=float32",      # float64 is ~30x slower on consumer GPUs
           f"--device={device}", f"--seed={seed}",
           f"--model_dir={work}", f"--checkpoints_dir={work / 'checkpoints'}",
           f"--results_dir={work / 'results'}", f"--log_dir={work / 'logs'}"]
    t0 = time.time()
    subprocess.run(cmd, check=True, env=os.environ.copy())
    train_time = time.time() - t0

    from mace.calculators import MACECalculator
    model_path = find_model(work, tag)
    print(f"[{tag}] evaluating {model_path.name}")
    calc = MACECalculator(model_paths=str(model_path), device=device, default_dtype="float32")
    try:
        n_params = int(sum(p.numel() for p in calc.models[0].parameters()))
    except Exception:
        n_params = None

    e_true, e_pred, n_atoms, groups, f_true, f_pred = [], [], [], [], [], []
    t1 = time.time()
    for fr in test_frames:
        at = fr.copy()
        at.calc = calc
        e_pred.append(at.get_potential_energy())
        f_pred.append(at.get_forces().ravel())
        e_true.append(fr.info["REF_energy"])
        f_true.append(fr.arrays["REF_forces"].ravel())
        n_atoms.append(len(fr))
        groups.append(fr.info["config_type"])
    test_time = time.time() - t1

    e_true, e_pred, n_atoms = map(np.array, (e_true, e_pred, n_atoms))
    f_true, f_pred = np.concatenate(f_true), np.concatenate(f_pred)
    m = metrics(e_true, e_pred, n_atoms, f_true, f_pred)
    per_group = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        de = (e_pred[idx] - e_true[idx]) / n_atoms[idx] * 1000
        per_group[g] = {"n": len(idx), "energy_mae_meV_atom": float(np.abs(de).mean())}

    result = {"element": element, "model": "mace", "train_fraction": train_fraction,
              "seed": seed, "n_train": len(train_frames), "n_test": len(test_frames),
              "n_params": n_params, "epochs_run": epochs, "train_time_s": train_time,
              "test_time_s": test_time, **m, "per_group": per_group,
              "predictions": {"e_true_per_atom": (e_true / n_atoms).tolist(),
                              "e_pred_per_atom": (e_pred / n_atoms).tolist(),
                              "groups": groups,
                              "f_true": f_true.tolist(), "f_pred": f_pred.tolist()}}
    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / f"{tag}.json", "w") as f:
        json.dump(result, f)
    print(f"[{tag}] TEST E-MAE {m['energy_mae_meV_atom']:.2f} meV/atom  "
          f"F-MAE {m['force_mae_eV_A']:.4f} eV/A  ({n_params} params)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--elements", nargs="+", default=ELEMENTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--fractions", nargs="+", type=float, default=[1.0])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    for el in a.elements:
        for frac in a.fractions:
            for seed in a.seeds:
                run(a.data_dir, el, seed, a.epochs, a.batch_size, a.cutoff, frac, a.device)


if __name__ == "__main__":
    main()
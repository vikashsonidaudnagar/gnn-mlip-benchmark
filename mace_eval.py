"""
Evaluate a trained MACE model on an element's test set and save results in the
same format as train.py, so compare.py can include MACE.

  python mace_eval.py --element Cu --model_path Cu_MACE.model --test_xyz xyz/Cu_test.xyz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

ap = argparse.ArgumentParser()
ap.add_argument("--element", required=True)
ap.add_argument("--model_path", required=True)
ap.add_argument("--test_xyz", required=True)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

calc = MACECalculator(model_paths=a.model_path, device=a.device, default_dtype="float64")
frames = read(a.test_xyz, ":")
e_t, e_p, f_t, f_p, groups = [], [], [], [], []
t0 = time.time()
for at in frames:
    n = len(at)
    e_t.append(at.info["REF_energy"] / n)
    f_t.append(at.arrays["REF_forces"].ravel())
    groups.append(at.info.get("config_type", "unknown"))
    at.calc = calc
    e_p.append(at.get_potential_energy() / n)
    f_p.append(at.get_forces().ravel())
dt = time.time() - t0

e_t, e_p = np.array(e_t), np.array(e_p)
f_t, f_p = np.concatenate(f_t), np.concatenate(f_p)
de, df = (e_p - e_t) * 1000, f_p - f_t
per_group = {g: {"n": int(sum(x == g for x in groups)),
                 "energy_mae_meV_atom": float(np.abs(de[[x == g for x in groups]]).mean())}
             for g in sorted(set(groups))}
result = {"element": a.element, "model": "mace", "train_fraction": 1.0,
          "n_test": len(frames), "test_time_s": dt,
          "energy_mae_meV_atom": float(np.abs(de).mean()),
          "energy_rmse_meV_atom": float(np.sqrt((de ** 2).mean())),
          "force_mae_eV_A": float(np.abs(df).mean()),
          "force_rmse_eV_A": float(np.sqrt((df ** 2).mean())),
          "per_group": per_group,
          "predictions": {"e_true_per_atom": e_t.tolist(), "e_pred_per_atom": e_p.tolist(),
                          "groups": groups, "f_true": f_t.tolist(), "f_pred": f_p.tolist()}}
Path("results").mkdir(exist_ok=True)
out = Path("results") / f"{a.element}_mace_f1.json"
json.dump(result, open(out, "w"))
print(f"{a.element} MACE: E-MAE {result['energy_mae_meV_atom']:.2f} meV/atom, "
      f"F-MAE {result['force_mae_eV_A']:.4f} eV/A -> {out}")

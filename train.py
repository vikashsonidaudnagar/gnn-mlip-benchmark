"""
Train one model on one element and save test results.

Example (Windows, from the project folder):
  python train.py --data_dir "C:/Users/YOU/Downloads/mlearn-master/mlearn-master/data" ^
                  --element Cu --model schnet --epochs 300

Outputs:
  results/<El>_<model>_f<fraction>_s<seed>.json   metrics + test predictions
  checkpoints/<El>_<model>_f<fraction>_s<seed>.pt best model weights

Training stops when the learning rate (halved after --sched_patience epochs
without improvement) falls below --min_lr, or after --patience epochs with no
improvement, or at --epochs.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from data import batch_to_dict, load_element, to_pyg
from models import build_model


def predict(model, loader, device):
    """Run the model on a DataLoader; returns arrays of true/predicted values."""
    model.eval()
    e_true, e_pred, n_atoms, groups, f_true, f_pred = [], [], [], [], [], []
    for batch in loader:
        b = batch_to_dict(batch.to(device))
        energy, forces = model(b, create_graph=False)
        e_true += b["energy"].tolist()
        e_pred += energy.detach().tolist()
        n_atoms += b["n_atoms"].tolist()
        groups += list(b["group"])
        f_true.append(b["forces"].cpu().numpy().ravel())
        f_pred.append(forces.detach().cpu().numpy().ravel())
    return {"e_true": np.array(e_true), "e_pred": np.array(e_pred),
            "n_atoms": np.array(n_atoms), "groups": groups,
            "f_true": np.concatenate(f_true), "f_pred": np.concatenate(f_pred)}


def metrics(p):
    de = (p["e_pred"] - p["e_true"]) / p["n_atoms"] * 1000.0   # meV/atom
    df = p["f_pred"] - p["f_true"]                              # eV/A
    return {"energy_mae_meV_atom": float(np.abs(de).mean()),
            "energy_rmse_meV_atom": float(np.sqrt((de ** 2).mean())),
            "force_mae_eV_A": float(np.abs(df).mean()),
            "force_rmse_eV_A": float(np.sqrt((df ** 2).mean()))}


def run(data_dir, element, model_name, epochs=1000, batch_size=4, lr=1e-3,
        cutoff=5.0, train_fraction=1.0, energy_weight=10.0, force_weight=1.0,
        n_feat=None, n_interactions=None, patience=150, sched_patience=25,
        min_lr=1e-5, seed=0,
        out_dir=".", device=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    tag = f"{element}_{model_name}_f{train_fraction:g}_s{seed}"

    train_all, test = load_element(data_dir, element, cutoff)
    random.shuffle(train_all)
    n_val = max(1, len(train_all) // 10)
    val, train = train_all[:n_val], train_all[n_val:]
    train = train[:max(1, int(round(len(train) * train_fraction)))]
    print(f"[{tag}] device {device} | train {len(train)} val {len(val)} test {len(test)}")

    # Normalisation from the training split only
    e_per_atom = np.array([d["energy"] / len(d["z"]) for d in train])
    f_all = np.concatenate([d["forces"].ravel() for d in train])
    size_kw = {k: v for k, v in [("n_feat", n_feat), ("n_interactions", n_interactions)]
               if v is not None}                 # otherwise each model's defaults
    model = build_model(model_name, cutoff=cutoff, **size_kw,
                        energy_shift=float(e_per_atom.mean()),
                        energy_scale=float(np.sqrt((f_all ** 2).mean()))).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader = DataLoader(to_pyg(train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_pyg(val), batch_size=8)
    test_loader = DataLoader(to_pyg(test), batch_size=8)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=sched_patience)
    ckpt = out_dir / "checkpoints" / f"{tag}.pt"

    best, best_epoch, history = float("inf"), 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            b = batch_to_dict(batch.to(device))
            energy, forces = model(b)
            loss = (energy_weight * F.mse_loss(energy / b["n_atoms"],
                                               b["energy"] / b["n_atoms"])
                    + force_weight * F.mse_loss(forces, b["forces"]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

        m = metrics(predict(model, val_loader, device))
        val_score = (energy_weight * (m["energy_rmse_meV_atom"] / 1000) ** 2
                     + force_weight * m["force_rmse_eV_A"] ** 2)
        sched.step(val_score)
        history.append({"epoch": epoch, **m})
        if val_score < best:
            best, best_epoch = val_score, epoch
            torch.save(model.state_dict(), ckpt)
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{tag}] epoch {epoch:4d} val E {m['energy_mae_meV_atom']:7.2f} "
                  f"meV/atom  F {m['force_mae_eV_A']:.4f} eV/A  "
                  f"lr {opt.param_groups[0]['lr']:.1e}")
        if opt.param_groups[0]["lr"] < min_lr:
            print(f"[{tag}] stop at epoch {epoch}: learning rate reached {min_lr:g}")
            break
        if epoch - best_epoch > patience:
            print(f"[{tag}] early stop at epoch {epoch}: no improvement for {patience} epochs")
            break
    train_time = time.time() - t0

    model.load_state_dict(torch.load(ckpt, map_location=device))
    t1 = time.time()
    p = predict(model, test_loader, device)
    infer_time = time.time() - t1

    test_m = metrics(p)
    per_group = {}
    for g in sorted(set(p["groups"])):
        idx = [i for i, x in enumerate(p["groups"]) if x == g]
        de = (p["e_pred"][idx] - p["e_true"][idx]) / p["n_atoms"][idx] * 1000
        per_group[g] = {"n": len(idx), "energy_mae_meV_atom": float(np.abs(de).mean())}

    result = {
        "element": element, "model": model_name, "train_fraction": train_fraction,
        "n_train": len(train), "n_test": len(test), "n_params": n_params,
        "best_epoch": best_epoch, "epochs_run": epoch, "seed": seed,
        "energy_weight": energy_weight, "force_weight": force_weight,
        "train_time_s": train_time,
        "test_time_s": infer_time, **test_m, "per_group": per_group,
        "history": history,
        "predictions": {
            "e_true_per_atom": (p["e_true"] / p["n_atoms"]).tolist(),
            "e_pred_per_atom": (p["e_pred"] / p["n_atoms"]).tolist(),
            "groups": p["groups"],
            "f_true": p["f_true"].tolist(), "f_pred": p["f_pred"].tolist()},
    }
    with open(out_dir / "results" / f"{tag}.json", "w") as f:
        json.dump(result, f)
    print(f"[{tag}] TEST E-MAE {test_m['energy_mae_meV_atom']:.2f} meV/atom  "
          f"F-MAE {test_m['force_mae_eV_A']:.4f} eV/A  ({n_params} params)")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--element", default="Cu")
    ap.add_argument("--model", default="schnet", choices=["schnet", "painn", "dimenetpp"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--train_fraction", type=float, default=1.0)
    ap.add_argument("--energy_weight", type=float, default=10.0)
    ap.add_argument("--force_weight", type=float, default=1.0)
    ap.add_argument("--n_feat", type=int, default=None)
    ap.add_argument("--n_interactions", type=int, default=None)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--sched_patience", type=int, default=25)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.data_dir, a.element, a.model, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, cutoff=a.cutoff, train_fraction=a.train_fraction,
        energy_weight=a.energy_weight, force_weight=a.force_weight,
        n_feat=a.n_feat, n_interactions=a.n_interactions, patience=a.patience,
        sched_patience=a.sched_patience, min_lr=a.min_lr, seed=a.seed)
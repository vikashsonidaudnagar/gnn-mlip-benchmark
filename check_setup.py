"""
Run once before the benchmark:
  python check_setup.py --data_dir "C:/.../mlearn-master/mlearn-master/data"

For each model it checks, on real Cu and Mo structures:
  1. forward + backward pass runs on your GPU
  2. energy is invariant and forces rotate correctly under a random rotation
  3. autograd forces match finite differences of the energy (float64)
  4. time per training step
"""
import argparse
import time

import torch
from torch_geometric.loader import DataLoader

from data import batch_to_dict, load_element, to_pyg
from models import MODELS, build_model

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", device,
      f"({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
import torch_geometric
print("torch_geometric", torch_geometric.__version__)


def get_batch(element, n, dtype):
    train, _ = load_element(a.data_dir, element)
    # disordered AIMD snapshots: forces are non-zero (perfect crystals give F = 0
    # by symmetry, which would make the force test trivial)
    aimd = [d for d in train if d["group"] == "AIMD-NVT"]
    train = sorted(aimd, key=lambda d: len(d["z"]))[:n]
    b = batch_to_dict(next(iter(DataLoader(to_pyg(train), batch_size=n))).to(device))
    for k in ["pos", "forces", "edge_shift", "energy", "n_atoms"]:
        b[k] = b[k].to(dtype)
    return b


def random_rotation(dtype):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype, device=device))
    return q * torch.sign(torch.linalg.det(q))


all_ok = True
for name in MODELS:
    print(f"\n=== {name} ===")
    try:
        torch.manual_seed(0)
        model = build_model(name, cutoff=5.0, energy_shift=-4.0, energy_scale=0.5)
        model = model.to(device).double()
        # give zero-initialised output layers some weight so tests are meaningful
        with torch.no_grad():
            for p in model.parameters():
                if p.abs().sum() == 0:
                    p.normal_(0, 0.1)
        for el in ["Mo", "Cu"]:
            b = get_batch(el, 2, torch.float64)
            E, Fo = model(b, create_graph=False)

            # rotation test
            R = random_rotation(torch.float64)
            b2 = dict(b)
            b2["pos"] = (b["pos"].detach() @ R.T).clone()
            b2["edge_shift"] = b["edge_shift"] @ R.T
            E2, F2 = model(b2, create_graph=False)
            e_err = (E2 - E).abs().max().item()
            f_err = (F2 - Fo @ R.T).abs().max().item()

            # finite-difference test on the atom with the largest force
            atom = int(Fo.norm(dim=1).argmax())
            h, f_fd = 1e-4, []
            for d in range(3):
                fd = []
                for sign in (1, -1):
                    b3 = dict(b)
                    b3["pos"] = b["pos"].detach().clone()
                    b3["pos"][atom, d] += sign * h
                    e3, _ = model(b3, compute_forces=False)
                    fd.append(e3.sum().item())
                f_fd.append(-(fd[0] - fd[1]) / (2 * h))
            f_fd = torch.tensor(f_fd, dtype=Fo.dtype, device=Fo.device)
            f_ag = Fo[atom].detach()
            fd_err = (f_fd - f_ag).abs().max().item()
            f_size = f_ag.abs().max().item()

            ok = (e_err < 1e-6 and f_err < 1e-6 and f_size > 1e-4
                  and fd_err < 1e-5 * max(1.0, f_size))
            all_ok &= ok
            print(f"  {el}: rotation dE {e_err:.1e}  dF {f_err:.1e} | atom {atom}: "
                  f"FD {[round(x, 5) for x in f_fd.tolist()]} vs autograd "
                  f"{[round(x, 5) for x in f_ag.tolist()]}  {'OK' if ok else 'FAIL'}")

        # speed test: real-size Cu batch, float32, training step
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = build_model(name, cutoff=5.0, energy_shift=-4.0).to(device)
        train, _ = load_element(a.data_dir, "Cu")
        b = batch_to_dict(next(iter(DataLoader(to_pyg(train[:4]), batch_size=4))).to(device))
        opt = torch.optim.Adam(model.parameters())
        for i in range(13):
            if i == 3:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
            b["pos"] = b["pos"].detach()
            E, Fo = model(b)
            loss = ((E - b["energy"]) ** 2).mean() + ((Fo - b["forces"]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        n_par = sum(p.numel() for p in model.parameters())
        mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        print(f"  {n_par:,} params | {(time.time() - t0) / 10 * 1000:.0f} ms per step "
              f"(4 Cu structures) | peak GPU mem {mem:.2f} GB")
    except Exception as ex:
        all_ok = False
        print(f"  ERROR: {type(ex).__name__}: {ex}")

print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED - paste this output to debug")
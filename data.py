"""
Data utilities for the mlearn benchmark (Cu, Ge, Li, Mo, Ni, Si).

Each mlearn JSON entry:
  {"structure": pymatgen Structure dict,
   "outputs": {"energy": eV (total), "forces": N x 3 eV/A, "virial_stress": 6 kBar},
   "num_atoms": N, "group": "AIMD-NVT" | "Elastic" | "Vacancy" | "Surface" | ...}
"""
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

ELEMENTS = ["Cu", "Ge", "Li", "Mo", "Ni", "Si"]


def entry_to_atoms(entry):
    s = entry["structure"]
    cell = np.array(s["lattice"]["matrix"], dtype=float)
    symbols = [site["species"][0]["element"] for site in s["sites"]]
    positions = np.array([site["xyz"] for site in s["sites"]], dtype=float)
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def load_mlearn(path, cutoff=5.0):
    """Read one JSON file into a list of graph dictionaries (numpy arrays)."""
    with open(path) as f:
        entries = json.load(f)
    data = []
    for e in entries:
        atoms = entry_to_atoms(e)
        i, j, S = neighbor_list("ijS", atoms, cutoff)
        data.append({
            "z": atoms.get_atomic_numbers().astype(np.int64),
            "pos": atoms.get_positions(),
            "energy": float(e["outputs"]["energy"]),
            "forces": np.array(e["outputs"]["forces"], dtype=float),
            "edge_src": i.astype(np.int64),
            "edge_dst": j.astype(np.int64),
            "edge_shift": S @ np.array(atoms.cell),
            "group": e.get("group", "unknown"),
        })
    return data


def load_element(data_dir, element, cutoff=5.0):
    """Returns (train, test) lists for one element folder."""
    folder = Path(data_dir) / element
    return (load_mlearn(folder / "training.json", cutoff),
            load_mlearn(folder / "test.json", cutoff))


def collate(samples, device):
    """Merge several structures into one disconnected graph (a batch)."""
    import torch
    keys = {k: [] for k in ["z", "pos", "forces", "edge_src", "edge_dst",
                            "edge_shift", "batch"]}
    energy, n_atoms, offset = [], [], 0
    for g, s in enumerate(samples):
        n = len(s["z"])
        keys["z"].append(torch.as_tensor(s["z"]))
        keys["pos"].append(torch.as_tensor(s["pos"], dtype=torch.float32))
        keys["forces"].append(torch.as_tensor(s["forces"], dtype=torch.float32))
        keys["edge_src"].append(torch.as_tensor(s["edge_src"]) + offset)
        keys["edge_dst"].append(torch.as_tensor(s["edge_dst"]) + offset)
        keys["edge_shift"].append(torch.as_tensor(s["edge_shift"], dtype=torch.float32))
        keys["batch"].append(torch.full((n,), g, dtype=torch.long))
        energy.append(s["energy"])
        n_atoms.append(n)
        offset += n
    b = {k: torch.cat(v).to(device) for k, v in keys.items()}
    b["energy"] = torch.tensor(energy, dtype=torch.float32, device=device)
    b["n_atoms"] = torch.tensor(n_atoms, dtype=torch.float32, device=device)
    b["n_graphs"] = len(samples)
    return b


# --------------------------------------------------------------------------
# PyTorch Geometric support
# --------------------------------------------------------------------------
def to_pyg(data_list):
    """Convert graph dicts into torch_geometric Data objects.
    edge_index is incremented automatically by the PyG DataLoader;
    edge_shift (Cartesian periodic offsets) is simply concatenated."""
    import torch
    from torch_geometric.data import Data
    out = []
    for d in data_list:
        n = len(d["z"])
        out.append(Data(
            z=torch.as_tensor(d["z"]),
            pos=torch.as_tensor(d["pos"], dtype=torch.float32),
            forces=torch.as_tensor(d["forces"], dtype=torch.float32),
            edge_index=torch.stack([torch.as_tensor(d["edge_src"]),
                                    torch.as_tensor(d["edge_dst"])]),
            edge_shift=torch.as_tensor(d["edge_shift"], dtype=torch.float32),
            energy=torch.tensor([d["energy"]], dtype=torch.float32),
            n_atoms=torch.tensor([n], dtype=torch.float32),
            group=d["group"],
            num_nodes=n))
    return out


def batch_to_dict(batch):
    """PyG Batch -> the dict format the models expect."""
    return {"z": batch.z, "pos": batch.pos, "forces": batch.forces,
            "edge_src": batch.edge_index[0], "edge_dst": batch.edge_index[1],
            "edge_shift": batch.edge_shift, "batch": batch.batch,
            "energy": batch.energy, "n_atoms": batch.n_atoms,
            "n_graphs": batch.num_graphs, "group": batch.group}


def export_extxyz(data_dir, out_dir):
    """Write <El>_train.xyz / <El>_test.xyz for MACE, NequIP, etc.
    Labels are stored as REF_energy / REF_forces."""
    from ase.io import write
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for el in ELEMENTS:
        for split, fname in [("train", "training.json"), ("test", "test.json")]:
            path = Path(data_dir) / el / fname
            if not path.exists():
                continue
            with open(path) as f:
                entries = json.load(f)
            frames = []
            for e in entries:
                atoms = entry_to_atoms(e)
                atoms.info["REF_energy"] = e["outputs"]["energy"]
                atoms.arrays["REF_forces"] = np.array(e["outputs"]["forces"])
                atoms.info["config_type"] = e.get("group", "unknown")
                frames.append(atoms)
            write(out_dir / f"{el}_{split}.xyz", frames, format="extxyz")
            print(f"wrote {out_dir / f'{el}_{split}.xyz'} ({len(frames)} structures)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Inspect mlearn data or export extxyz")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--export", default=None, help="folder to write extxyz files")
    args = p.parse_args()
    if args.export:
        export_extxyz(args.data_dir, args.export)
    else:
        for el in ELEMENTS:
            tr, te = load_element(args.data_dir, el)
            groups = sorted({d["group"] for d in tr})
            print(f"{el}: train {len(tr):4d}  test {len(te):3d}  "
                  f"atoms/struct {min(len(d['z']) for d in tr)}-"
                  f"{max(len(d['z']) for d in tr)}  groups {groups}")

"""
Physical-property validation of the trained potentials.

For each element, model and seed it computes, with the trained network as an
ASE calculator:
  a0   lattice constant (A)                 Birch-Murnaghan fit of E(V)
  B    bulk modulus (GPa)                   same fit
  C11, C12, C44 (GPa)                       volume-conserving strains (Mehl 1993),
                                            internal positions relaxed
  Ev   vacancy formation energy (eV)        relaxed supercell, fixed cell

  python properties.py --data_dir "C:/.../mlearn-master/mlearn-master/data"
  python properties.py --data_dir "..." --elements Cu --models painn --seeds 0   (quick test)
  python properties.py --data_dir "..." --table_only                            (rebuild table)

Results:  properties/<El>_<model>_s<seed>.json
Table:    paper_figures/properties_table.tex  (uses reference_properties.csv if present)
"""
import argparse
import csv
import json
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path

# Windows: PyTorch and conda's NumPy/SciPy each ship an OpenMP runtime (libiomp5md.dll).
# Allow both to load; check_model() below verifies the results are correct.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from ase import units
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.eos import EquationOfState
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS

ELEMENTS = ["Cu", "Ni", "Li", "Mo", "Si", "Ge"]
CRYSTAL = {"Cu": "fcc", "Ni": "fcc", "Li": "bcc", "Mo": "bcc", "Si": "diamond", "Ge": "diamond"}
A_GUESS = {"Cu": 3.63, "Ni": 3.52, "Li": 3.44, "Mo": 3.16, "Si": 5.47, "Ge": 5.76}  # start only
SUPERCELL = {"fcc": (3, 3, 3), "bcc": (3, 3, 3), "diamond": (2, 2, 2)}  # as in mlearn
MODEL_ORDER = ["schnet", "dimenetpp", "painn"]
NAMES = {"schnet": "SchNet", "dimenetpp": "DimeNet++", "painn": "PaiNN"}
PROPS = ["a0", "B", "C11", "C12", "C44", "Ev"]
EV_A3_TO_GPA = 1.0 / units.GPa


# ---------------------------------------------------------------- calculator
class MLCalculator(Calculator):
    """ASE calculator wrapping a trained PotentialBase model (float64)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, cutoff=5.0, device="cuda", **kw):
        super().__init__(**kw)
        self.model, self.cutoff, self.device = model, cutoff, device

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        import torch
        super().calculate(atoms, properties, system_changes)
        i, j, S = neighbor_list("ijS", self.atoms, self.cutoff)
        dt = torch.float64
        dev = self.device
        b = {"z": torch.as_tensor(self.atoms.get_atomic_numbers(), dtype=torch.long, device=dev),
             "pos": torch.as_tensor(self.atoms.get_positions(), dtype=dt, device=dev),
             "edge_src": torch.as_tensor(i, dtype=torch.long, device=dev),
             "edge_dst": torch.as_tensor(j, dtype=torch.long, device=dev),
             "edge_shift": torch.as_tensor(S @ np.array(self.atoms.cell), dtype=dt, device=dev),
             "batch": torch.zeros(len(self.atoms), dtype=torch.long, device=dev),
             "n_graphs": 1}
        energy, forces = self.model(b, compute_forces=True, create_graph=False)
        self.results["energy"] = float(energy.detach().cpu().numpy()[0])
        self.results["forces"] = forces.detach().cpu().numpy()


def split_like_train_py(n, seed):
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = max(1, n // 10)
    return idx[n_val:], idx[:n_val]


def load_model(name, element, seed, data_dir, device, cutoff=5.0):
    """Rebuild a trained model, recomputing the energy normalisation of train.py."""
    import torch
    from models import build_model
    entries = json.load(open(Path(data_dir) / element / "training.json"))
    tr, _ = split_like_train_py(len(entries), seed)
    e_atom = np.array([entries[i]["outputs"]["energy"] / entries[i]["num_atoms"] for i in tr])
    f_all = np.concatenate([np.ravel(entries[i]["outputs"]["forces"]) for i in tr])
    model = build_model(name, cutoff=cutoff, energy_shift=float(e_atom.mean()),
                        energy_scale=float(np.sqrt((f_all ** 2).mean())))
    ckpt = Path("checkpoints") / f"{element}_{name}_f1_s{seed}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).double().eval()


def check_model(calc, data_dir, element, n=5):
    """Sanity check: errors on a few test structures should match the test-set MAE."""
    from data import entry_to_atoms
    entries = json.load(open(Path(data_dir) / element / "test.json"))[:n]
    de, df = [], []
    for e in entries:
        at = entry_to_atoms(e)
        at.calc = calc
        de.append(abs(at.get_potential_energy() - e["outputs"]["energy"]) / len(at) * 1000)
        df.append(np.abs(at.get_forces() - np.array(e["outputs"]["forces"])).mean())
    e_mae, f_mae = float(np.mean(de)), float(np.mean(df))
    print(f"    check on {n} test structures: E-MAE {e_mae:.2f} meV/atom, F-MAE {f_mae:.4f} eV/A")
    if e_mae > 50 or f_mae > 0.5:
        raise RuntimeError("Model errors are far larger than in testing: the model was not "
                           "loaded correctly (wrong checkpoint or settings). Stopping.")
    return e_mae, f_mae


# ---------------------------------------------------------------- properties
def relax_positions(atoms, fmax=0.005, steps=300):
    if len(atoms) > 1:
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms.get_potential_energy()


def lattice_and_bulk(element, calc):
    """Two-stage E(V) scan of the conventional cubic cell."""
    crystal = CRYSTAL[element]
    a_c = A_GUESS[element]
    for width, npts in [(0.04, 9), (0.015, 9)]:
        vols, ens = [], []
        for s in np.linspace(1 - width, 1 + width, npts):
            at = bulk(element, crystal, a=a_c * s, cubic=True)
            at.calc = calc
            ens.append(relax_positions(at))
            vols.append(at.get_volume())
        v0, e0, B = EquationOfState(vols, ens, eos="birchmurnaghan").fit()
        a_c = v0 ** (1 / 3)
    return a_c, B * EV_A3_TO_GPA


def strained_energy(element, a0, calc, eps):
    at = bulk(element, CRYSTAL[element], a=a0, cubic=True)
    at.set_cell(np.array(at.cell) @ (np.eye(3) + eps), scale_atoms=True)
    at.calc = calc
    return relax_positions(at), at.get_volume()


def elastic_constants(element, a0, B, calc, deltas=(-0.02, -0.01, 0.0, 0.01, 0.02)):
    """C' = C11-C12 (orthorhombic) and C44 (monoclinic), volume-conserving strains."""
    v0 = a0 ** 3
    out = {}
    for mode in ("ortho", "mono"):
        e = []
        for d in deltas:
            if mode == "ortho":
                eps = np.diag([d, -d, d ** 2 / (1 - d ** 2)])
            else:
                eps = np.array([[0, d / 2, 0], [d / 2, 0, 0], [0, 0, d ** 2 / (4 - d ** 2)]])
            e.append(strained_energy(element, a0, calc, eps)[0])
        d = np.array(deltas)
        A = np.stack([np.ones_like(d), d ** 2, d ** 4], axis=1)
        c2 = np.linalg.lstsq(A, np.array(e) / v0, rcond=None)[0][1]    # eV/A^3
        out[mode] = c2 * EV_A3_TO_GPA
    cprime = out["ortho"]            # dE/V = (C11 - C12) d^2
    c44 = 2.0 * out["mono"]          # dE/V = 1/2 C44 d^2
    c11 = B + 2.0 * cprime / 3.0     # B = (C11 + 2 C12) / 3
    c12 = B - cprime / 3.0
    return c11, c12, c44


def vacancy_energy(element, a0, calc):
    crystal = CRYSTAL[element]
    perfect = bulk(element, crystal, a=a0, cubic=True).repeat(SUPERCELL[crystal])
    perfect.calc = calc
    e_perf = relax_positions(perfect)
    n = len(perfect)
    defect = perfect.copy()
    del defect[0]
    defect.calc = calc
    e_def = relax_positions(defect, fmax=0.01)
    return e_def - (n - 1) / n * e_perf


def compute_all(element, calc):
    a0, B = lattice_and_bulk(element, calc)
    c11, c12, c44 = elastic_constants(element, a0, B, calc)
    ev = vacancy_energy(element, a0, calc)
    return {"a0": a0, "B": B, "C11": c11, "C12": c12, "C44": c44, "Ev": ev}


# ---------------------------------------------------------------- table
def read_reference(path="reference_properties.csv"):
    """CSV columns: element,source,a0,B,C11,C12,C44,Ev   (source = DFT or Expt.)"""
    ref = defaultdict(dict)
    if not Path(path).exists():
        return ref
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals = {}
            for p in PROPS:
                try:
                    vals[p] = float(row.get(p, ""))
                except ValueError:
                    pass
            ref[row["element"].strip()][row["source"].strip()] = vals
    return ref


def write_table(out_dir="paper_figures"):
    res = defaultdict(list)
    for p in sorted(Path("properties").glob("*.json")):
        r = json.load(open(p))
        res[(r["element"], r["model"])].append(r)
    if not res:
        raise SystemExit("No results in properties/ yet.")
    ref = read_reference()
    elements = [e for e in ELEMENTS if any(k[0] == e for k in res)]
    models = [m for m in MODEL_ORDER if any(k[1] == m for k in res)]
    digits = {"a0": 3, "B": 0, "C11": 0, "C12": 0, "C44": 0, "Ev": 2}

    def cell(vals, p):
        v = np.array([x[p] for x in vals])
        if len(v) == 1:
            return f"{v[0]:.{digits[p]}f}"
        return f"{v.mean():.{digits[p]}f}\\,$\\pm$\\,{v.std(ddof=1):.{digits[p]}f}"

    tex = [r"% ---- generated by properties.py ----",
           r"\begin{table*}[t]", r"\centering",
           r"\caption{Physical properties predicted by the trained potentials compared with "
           r"reference values. $a_0$: lattice constant; $B$: bulk modulus; $C_{ij}$: elastic "
           r"constants; $E_\mathrm{v}$: relaxed vacancy formation energy. Model values are "
           r"mean\,$\pm$\,standard deviation over the trained seeds.}",
           r"\label{tab:properties}", r"\small",
           r"\begin{tabular}{llcccccc}", r"\toprule",
           r"Element & Source & $a_0$ (\AA) & $B$ (GPa) & $C_{11}$ (GPa) & $C_{12}$ (GPa) & "
           r"$C_{44}$ (GPa) & $E_\mathrm{v}$ (eV) \\", r"\midrule"]
    for k, el in enumerate(elements):
        rows = []
        for src in ("DFT", "Expt."):
            if ref.get(el, {}).get(src):          # skip sources with no values
                vals = ref[el][src]
                rows.append([src] + [f"{vals[p]:.{digits[p]}f}" if p in vals else "--"
                                     for p in PROPS])
        for m in models:
            if (el, m) in res:
                rows.append([NAMES[m]] + [cell(res[(el, m)], p) for p in PROPS])
        for i, r in enumerate(rows):
            tex.append((el if i == 0 else "") + " & " + " & ".join(r) + r" \\")
        if k < len(elements) - 1:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    # mean absolute percentage deviation from DFT, per model and property
    if any("DFT" in ref.get(e, {}) for e in elements):
        tex += [r"\begin{table}[t]", r"\centering",
                r"\caption{Mean absolute percentage deviation of predicted properties from the "
                r"DFT reference values, averaged over elements with available DFT data.}",
                r"\label{tab:properties_error}", r"\small", r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{lcccccc}", r"\toprule",
                r"Model & $a_0$ & $B$ & $C_{11}$ & $C_{12}$ & $C_{44}$ & $E_\mathrm{v}$ \\",
                r"\midrule"]
        for m in models:
            cells = []
            for p in PROPS:
                errs = [abs(np.mean([x[p] for x in res[(e, m)]]) - ref[e]["DFT"][p])
                        / abs(ref[e]["DFT"][p]) * 100
                        for e in elements if (e, m) in res and p in ref.get(e, {}).get("DFT", {})]
                cells.append(f"{np.mean(errs):.1f}" if errs else "--")
            tex.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    Path(out_dir).mkdir(exist_ok=True)
    (Path(out_dir) / "properties_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(tex))
    print(f"\nSaved {Path(out_dir) / 'properties_table.tex'}")


def write_reference_template(path="reference_properties.csv"):
    if Path(path).exists():
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["element", "source"] + PROPS)
        for el in ELEMENTS:
            for src in ("DFT", "Expt."):
                w.writerow([el, src] + [""] * len(PROPS))
    print(f"Created {path}: fill in DFT and experimental values (leave unknown cells empty).")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--elements", nargs="+", default=ELEMENTS)
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="default: every seed with a checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--table_only", action="store_true")
    a = ap.parse_args()

    write_reference_template()
    if not a.table_only:
        warnings.filterwarnings("ignore")
        Path("properties").mkdir(exist_ok=True)
        for el in a.elements:
            for m in a.models:
                seeds = a.seeds if a.seeds is not None else sorted(
                    int(p.stem.split("_s")[-1])
                    for p in Path("checkpoints").glob(f"{el}_{m}_f1_s*.pt"))
                for seed in seeds:
                    out = Path("properties") / f"{el}_{m}_s{seed}.json"
                    if out.exists() and not a.overwrite:
                        print(f"[{el} {m} s{seed}] exists, skipping")
                        continue
                    print(f"[{el} {m} s{seed}] computing ...", flush=True)
                    calc = MLCalculator(load_model(m, el, seed, a.data_dir, a.device),
                                        device=a.device)
                    chk = check_model(calc, a.data_dir, el)
                    r = compute_all(el, calc)
                    r["check_E_mae_meV_atom"], r["check_F_mae_eV_A"] = chk
                    print("    " + "  ".join(f"{p}={r[p]:.3f}" for p in PROPS))
                    json.dump({"element": el, "model": m, "seed": seed, **r}, open(out, "w"))
    write_table()


if __name__ == "__main__":
    main()
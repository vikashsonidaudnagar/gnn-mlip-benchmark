# Benchmarking Graph Neural Network Interatomic Potentials for Metallic and Covalent Elements

This repository contains the code for the paper *"Benchmarking Graph Neural Network Interatomic
Potentials for Metallic and Covalent Elements: Accuracy, Cost and Physical Properties"*
by Vikash Kumar (Department of Mechanical Engineering, IIT Kanpur).

The paper compares three graph neural network (GNN) interatomic potentials on the
[mlearn](https://github.com/materialsvirtuallab/mlearn) data set, which contains six elements
(Cu, Ni, Li, Mo, Si and Ge). The three models describe the atomic geometry at different levels of
detail:

| Model | Geometric information | Implementation |
|---|---|---|
| SchNet | interatomic distances | `models.py` (PyTorch) |
| DimeNet++ | distances and bond angles | PyTorch Geometric layers with periodic angle triplets (`models.py`) |
| PaiNN | equivariant vector features | `models.py` (PyTorch) |

All models use periodic boundary conditions, are trained on energies and forces with the same
settings, and are evaluated on the official mlearn test sets. The trained models are then used to
calculate the lattice constant, bulk modulus, elastic constants and vacancy formation energy. These
are compared with DFT values and with the five classical potentials (GAP, MTP, NNP, SNAP and qSNAP)
studied by Zuo et al. (2020).

## Main results

- **Forces:** DimeNet++ gave the lowest force errors for every element (relative errors of
  1.3–10.6%), 30–50% lower than PaiNN, and reached the force accuracy of GAP and MTP. PaiNN gave
  lower force errors than SchNet for all elements.
- **Energies:** the energy errors were 0.4–9 meV/atom. For most elements, the differences between
  the three models were within the variation between random seeds, and all three had larger energy
  errors than GAP and MTP.
- **Cost:** DimeNet++ has 12 times more parameters than PaiNN and took about 30 times longer to
  train.
- **Physical properties:** the lattice constants agreed with DFT within 0.3% on average. The
  elastic constants and vacancy energies deviated from DFT by 11–21% on average, compared with 5–6%
  for GAP and MTP. SchNet, which had the highest force errors, gave the best physical properties, so
  low test errors did not guarantee accurate properties.

## Files

| File | Purpose |
|---|---|
| `data.py` | Reads the mlearn JSON files and builds periodic neighbour lists (ASE) and PyTorch Geometric data |
| `models.py` | SchNet, PaiNN and DimeNet++ |
| `check_setup.py` | Checks the GPU, rotation invariance, and forces against finite differences for each model |
| `train.py` | Trains one model on one element and saves the errors and predictions |
| `run_all.py` | Runs all models, elements and seeds |
| `compare.py` | Quick summary table and plots |
| `make_figures.py` | Figures for the paper (PDF and PNG) |
| `paper_tables.py` | Tables for the paper (LaTeX) |
| `properties.py` | Physical properties calculated with the trained models, and the property table |
| `reference_properties.csv` | DFT reference values from Zuo et al. (2020) |
| `literature_errors.py` | Comparison of the test errors with the classical potentials of Zuo et al. |
| `run_mace.py`, `run_chgnet.py` | Optional MACE and CHGNet baselines (not used in the paper) |

## Installation

The code was tested on Windows 11 with an NVIDIA RTX 3060 GPU, Python 3.11, PyTorch 2.9.1
(CUDA 12.6) and PyTorch Geometric 2.7.0.

```bash
conda create -n gnn python=3.11
conda activate gnn
# install PyTorch with CUDA for your system (see https://pytorch.org), for example:
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Only the main `torch_geometric` package is needed. `torch_scatter` and `torch_sparse` are not
required.

## Data

Download the mlearn repository:

```bash
git clone https://github.com/materialsvirtuallab/mlearn.git
```

Its `data` folder contains one folder per element (`Cu`, `Ge`, `Li`, `Mo`, `Ni`, `Si`), each with a
`training.json` and a `test.json` file. In the commands below, replace `DATA` with the path to this
folder, for example `C:/Users/you/mlearn/data`.

## Reproducing the results

**1. Check the installation** (about one minute; every line should end with `OK`):

```bash
python check_setup.py --data_dir DATA
```

**2. Train the models.** SchNet and PaiNN are trained with three seeds (about 4–6 hours on an
RTX 3060), and DimeNet++ with one seed (about 20 hours):

```bash
python run_all.py --data_dir DATA --models schnet painn --seeds 0 1 2
python run_all.py --data_dir DATA --models dimenetpp --epochs 400
```

The results are saved in `results/` and the model weights in `checkpoints/`. To train a single
model, use for example `python train.py --data_dir DATA --element Cu --model painn`.

**3. Figures and tables:**

```bash
python compare.py                          # quick overview
python make_figures.py --data_dir DATA     # Figures 1-5  -> paper_figures/
python paper_tables.py --data_dir DATA     # Tables 1-4   -> paper_figures/tables.tex
```

**4. Physical properties** (about one hour):

```bash
python properties.py --data_dir DATA       # Table 5      -> paper_figures/properties_table.tex
```

**5. Comparison with the classical potentials of Zuo et al.:**

```bash
python literature_errors.py                # test errors  -> paper_figures/literature_errors.tex
```

The table of property deviations (`literature_table.tex`) combines the deviations reported by
Zuo et al. with the values calculated by `properties.py`.

### Note for Windows

On Windows, `properties.py` can stop with `OMP: Error #15`, because PyTorch and NumPy each load
their own OpenMP library. The script sets `KMP_DUPLICATE_LIB_OK=TRUE` to avoid this. Each model is
checked against the test set before it is used, so wrong results would be detected.

## Citation

A preprint of the paper will be linked here.

If you use this code, please also cite the mlearn data set:

```bibtex
@article{zuo2020performance,
  author  = {Zuo, Yunxing and Chen, Chi and Li, Xiangguo and Deng, Zhi and Chen, Yiming and
             Behler, J{\"o}rg and Cs{\'a}nyi, G{\'a}bor and Shapeev, Alexander V. and
             Thompson, Aidan P. and Wood, Mitchell A. and Ong, Shyue Ping},
  title   = {Performance and Cost Assessment of Machine Learning Interatomic Potentials},
  journal = {The Journal of Physical Chemistry A},
  volume  = {124},
  number  = {4},
  pages   = {731--745},
  year    = {2020},
  doi     = {10.1021/acs.jpca.9b08723}
}
```

## Acknowledgements

The training and test data are taken from the mlearn benchmark of Zuo et al. (2020). The models
follow SchNet (Schütt et al., 2018), DimeNet++ (Gasteiger et al., 2020) and PaiNN
(Schütt et al., 2021).

## License

See [LICENSE](LICENSE).

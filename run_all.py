"""
Run the full benchmark: every model on every element.

  python run_all.py --data_dir "C:/.../mlearn-master/mlearn-master/data"

Options:
  --elements Cu Ni        only some elements
  --models schnet painn dimenetpp   only some models
  --fractions 0.1 0.25 0.5 1.0   learning-curve experiment
  --seeds 0 1 2            repeat with several random seeds
  --energy_weight 100      weight of the energy term in the loss
"""
import argparse

from data import ELEMENTS
from train import run

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--elements", nargs="+", default=ELEMENTS)
ap.add_argument("--models", nargs="+", default=["schnet", "painn", "dimenetpp"])
ap.add_argument("--fractions", nargs="+", type=float, default=[1.0])
ap.add_argument("--epochs", type=int, default=1000)
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--energy_weight", type=float, default=10.0)
ap.add_argument("--batch_size", type=int, default=4)
a = ap.parse_args()

for el in a.elements:
    for model in a.models:
        for frac in a.fractions:
            for seed in a.seeds:
                run(a.data_dir, el, model, epochs=a.epochs, batch_size=a.batch_size,
                    energy_weight=a.energy_weight, train_fraction=frac, seed=seed)
print("\nDone. Now run:  python compare.py")
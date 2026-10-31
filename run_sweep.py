"""The headline experiment: how does clustering quality change with coupling?

For each coupling value and each algorithm:
  simulate -> build features -> cluster -> record silhouette, stability,
  and agreement with the soft latent reference.
Writes results/sweep.csv; plot it for the presentation.

Usage:  python run_sweep.py
"""
import os

import numpy as np
import pandas as pd

from config import SimConfig, PathConfig
from traits import sample_patrons
from behavior import simulate_all
from features import build_features
from cluster import ALGORITHMS, preprocess
from evaluate import (internal_metrics, bootstrap_stability,
                      reference_partition, agreement_with_reference)

COUPLINGS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
K = 5
N_REPS = 5  # repeat each setting with different seeds


def main():
    paths = PathConfig()
    os.makedirs(paths.results_dir, exist_ok=True)
    rows = []
    for coupling in COUPLINGS:
        for rep in range(N_REPS):
            cfg = SimConfig(coupling=coupling, seed=1000 + rep)
            rng = np.random.default_rng(cfg.seed)
            patrons = sample_patrons(cfg, rng)
            events = simulate_all(patrons, cfg, rng)
            features = build_features(events)
            X = preprocess(features)
            ref = reference_partition(patrons.loc[features.index])
            for name, algo in ALGORITHMS.items():
                labels = algo(X, K)
                rows.append({
                    "coupling": coupling, "rep": rep, "algo": name,
                    **internal_metrics(X, labels),
                    "stability": bootstrap_stability(X, algo, K),
                    "ref_agreement": agreement_with_reference(labels, ref),
                })
    pd.DataFrame(rows).to_csv(f"{paths.results_dir}/sweep.csv", index=False)


if __name__ == "__main__":
    main()

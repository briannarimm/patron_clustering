import argparse
import os

import numpy as np

from config import SimConfig, PathConfig
from traits import sample_patrons
from behavior import simulate_all
from features import build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patrons", type=int, default=SimConfig.n_patrons)
    ap.add_argument("--coupling", type=float, default=SimConfig.coupling)
    ap.add_argument("--seed", type=int, default=SimConfig.seed)
    args = ap.parse_args()

    cfg = SimConfig(n_patrons=args.n_patrons, coupling=args.coupling, seed=args.seed)
    paths = PathConfig()
    os.makedirs(paths.data_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    patrons = sample_patrons(cfg, rng)
    events = simulate_all(patrons, cfg, rng)
    features = build_features(events)

    patrons.to_csv(paths.latent_csv, index=False)   # hidden truth, analysis only
    events.to_csv(paths.events_csv, index=False)
    features.to_csv(paths.features_csv)
    print(f"{len(patrons)} patrons, {len(events)} checkouts -> {paths.data_dir}/")


if __name__ == "__main__":
    main()

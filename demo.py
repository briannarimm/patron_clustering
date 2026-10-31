"""Run the whole pipeline once and look at the result.

Simulates patrons, builds features, clusters them, prints the cluster profiles,
and saves a figure with three panels: the patrons in 2D, what each cluster
looks like in the features, and how the clusters line up with the hidden traits.

Usage:  python demo.py [--coupling 1.0] [--k 5] [--algo kmeans] [--show]

Try --coupling 0 and --coupling 2 to see the point of the sweep.
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import SimConfig, PathConfig
from catalog import INTERESTS
from traits import sample_patrons
from behavior import simulate_all
from features import build_features, MIN_CHECKOUTS
from cluster import ALGORITHMS, preprocess, profile_clusters
from evaluate import (internal_metrics, bootstrap_stability, reference_partition,
                      agreement_with_reference, choose_k, AGE_BANDS, AGE_LABELS)

# the columns worth reading in the profile table
PROFILE_COLUMNS = ["size", "total_checkouts", "items_per_visit", "pct_dvd",
                   "pct_large_print", "pct_kids_subject", "pct_fiction_subject",
                   "pct_academic_subject", "subject_diversity"]


def describe_clusters(latent: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Post-hoc only: what the hidden traits say about each cluster."""
    lat = latent.assign(cluster=labels)
    out = pd.DataFrame({
        "median_age": lat.groupby("cluster")["age"].median(),
        "pct_household": lat.groupby("cluster")["household_card"].mean(),
        "top_interest": lat.groupby("cluster")[INTERESTS].mean().idxmax(axis=1),
    })
    bands = pd.cut(lat["age"], AGE_BANDS, labels=AGE_LABELS)
    return out.join(pd.crosstab(lat["cluster"], bands, normalize="index"))


def plot(X, features, latent, labels, title, path):
    k = labels.max() + 1
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    cmap = plt.get_cmap("tab10")

    # 1. patrons in 2D
    pts = PCA(n_components=2).fit_transform(X)
    ax = axes[0]
    for c in range(k):
        m = labels == c
        ax.scatter(pts[m, 0], pts[m, 1], s=6, alpha=0.5, color=cmap(c), label=f"cluster {c} (n={m.sum()})")
    ax.set_title("patrons (PCA of the feature table)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=3)

    # 2. cluster profiles, each feature standardized across clusters
    prof = features.assign(cluster=labels).groupby("cluster").mean()
    z = (prof - prof.mean()) / prof.std().replace(0, 1)
    ax = axes[1]
    im = ax.imshow(z.T, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(z.columns))); ax.set_yticklabels(z.columns, fontsize=8)
    ax.set_xticks(range(k)); ax.set_xlabel("cluster")
    ax.set_title("cluster profiles (z-scored feature means)")
    fig.colorbar(im, ax=ax, fraction=0.04)

    # 3. hidden traits, used after the fact to see if the clusters mean anything
    dominant = latent[INTERESTS].idxmax(axis=1)
    ct = pd.crosstab(labels, dominant, normalize="index").reindex(columns=INTERESTS, fill_value=0)
    ax = axes[2]
    im = ax.imshow(ct.T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(INTERESTS))); ax.set_yticklabels(INTERESTS, fontsize=8)
    ax.set_xticks(range(k)); ax.set_xlabel("cluster")
    ax.set_title("share of each cluster by dominant latent interest")
    fig.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patrons", type=int, default=SimConfig.n_patrons)
    ap.add_argument("--coupling", type=float, default=SimConfig.coupling)
    ap.add_argument("--seed", type=int, default=SimConfig.seed)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--algo", choices=ALGORITHMS, default="kmeans")
    ap.add_argument("--show", action="store_true", help="open the figure window")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    pd.set_option("display.width", 160)

    cfg = SimConfig(n_patrons=args.n_patrons, coupling=args.coupling, seed=args.seed)
    paths = PathConfig()
    os.makedirs(paths.results_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    algo = ALGORITHMS[args.algo]

    print(f"-- simulate (coupling={cfg.coupling}, seed={cfg.seed})")
    patrons = sample_patrons(cfg, rng)
    events = simulate_all(patrons, cfg, rng)
    features = build_features(events)
    print(f"{len(patrons)} patrons, {len(events)} checkouts over {cfg.n_days} days")
    print(f"{len(features)} patrons kept with {MIN_CHECKOUTS}+ checkouts, "
          f"{len(patrons) - len(features)} too sparse to use")
    print(events["item_format"].value_counts(normalize=True).round(2).to_string())

    print(f"\n-- choose k ({args.algo})")
    X = preprocess(features)
    print(choose_k(X, algo, range(2, 9)).round(3).to_string())

    print(f"\n-- cluster (k={args.k})")
    labels = algo(X, args.k)
    latent = patrons.set_index("patron_id").loc[features.index]
    metrics = internal_metrics(X, labels)
    metrics["stability"] = bootstrap_stability(X, algo, args.k)
    metrics["ref_agreement"] = agreement_with_reference(labels, reference_partition(latent))
    print(pd.Series(metrics).round(3).to_string())

    print("\n-- cluster profiles (what the library would see)")
    print(profile_clusters(features, labels)[PROFILE_COLUMNS].round(2).to_string())

    print("\n-- what the hidden traits say about each cluster (post-hoc only)")
    print(describe_clusters(latent, labels).round(2).to_string())

    title = (f"coupling={cfg.coupling}  {args.algo} k={args.k}   "
             f"silhouette={metrics['silhouette']:.2f}  stability={metrics['stability']:.2f}  "
             f"ref ARI={metrics['ref_agreement']:.2f}")
    path = f"{paths.results_dir}/demo_c{cfg.coupling:g}_{args.algo}_k{args.k}.png"
    plot(X, features, latent, labels, title, path)
    print(f"\nfigure -> {path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

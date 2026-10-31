"""Plot results/sweep.csv: clustering quality against coupling, one line per algorithm.

Usage:  python plot_sweep.py [--show]
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from config import PathConfig

METRICS = ["silhouette", "stability", "ref_agreement"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    paths = PathConfig()
    sweep = pd.read_csv(f"{paths.results_dir}/sweep.csv")
    stats = sweep.groupby(["algo", "coupling"])[METRICS].agg(["mean", "std"])

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    for ax, metric in zip(axes, METRICS):
        for algo, s in stats.groupby(level="algo"):
            s = s.droplevel("algo")
            ax.errorbar(s.index, s[(metric, "mean")], yerr=s[(metric, "std")],
                        marker="o", capsize=3, label=algo)
        ax.set_xlabel("coupling (0 = interests independent of age)")
        ax.set_title(metric)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("mean and sd over seeds, k=5")
    fig.tight_layout()
    fig.savefig(f"{paths.results_dir}/sweep.png", dpi=130)
    print(stats.round(3).to_string())
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

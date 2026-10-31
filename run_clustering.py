"""Cluster patron_features.csv and print cluster profiles.

Usage:  python run_clustering.py [--k 5] [--algo kmeans]
"""
import argparse

import pandas as pd

from config import PathConfig
from cluster import ALGORITHMS, preprocess, profile_clusters
from evaluate import internal_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--algo", choices=ALGORITHMS, default="kmeans")
    args = ap.parse_args()

    paths = PathConfig()
    features = pd.read_csv(paths.features_csv, index_col=0)
    X = preprocess(features)
    labels = ALGORITHMS[args.algo](X, args.k)

    print(internal_metrics(X, labels))
    print(profile_clusters(features, labels).round(2).to_string())
    pd.Series(labels, index=features.index, name="cluster").to_csv(
        f"{paths.data_dir}/clusters_{args.algo}.csv"
    )


if __name__ == "__main__":
    main()

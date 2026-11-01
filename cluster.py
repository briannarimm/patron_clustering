"""Clustering wrappers (scikit-learn)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from features import FEATURE_COLUMNS


# count-like columns with a long right tail
LOG_COLUMNS = ["total_checkouts", "items_per_visit"]


def preprocess(features: pd.DataFrame) -> np.ndarray:
    """Keep FEATURE_COLUMNS, log-transform skewed counts, then standardize."""
    X = features[[c for c in FEATURE_COLUMNS if c in features]].copy()
    for col in LOG_COLUMNS:
        if col in X:
            X[col] = np.log1p(X[col])
    return StandardScaler().fit_transform(X.to_numpy(dtype=float))


def run_kmeans(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def run_gmm(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    return GaussianMixture(n_components=k, random_state=seed).fit_predict(X)


def run_hierarchical(X: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


ALGORITHMS = {
    "kmeans": run_kmeans,
    "gmm": run_gmm,
    "hierarchical": lambda X, k, seed=0: run_hierarchical(X, k),
}


def profile_clusters(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean of each feature per cluster + cluster size. This is the table
    you read to name the clusters ("Oh! That looks like a kids/DVD cluster!"). Like reading tea leaves :P"""
    prof = features.assign(cluster=labels).groupby("cluster").mean()
    prof.insert(0, "size", pd.Series(labels).value_counts().sort_index().to_numpy())
    return prof

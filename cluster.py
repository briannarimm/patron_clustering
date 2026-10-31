"""Clustering wrappers (scikit-learn)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def preprocess(features: pd.DataFrame) -> np.ndarray:
    """Log-transform skewed count columns, then standardize.

    TODO: log1p on total_checkouts etc.; StandardScaler; return ndarray.
    """
    raise NotImplementedError


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
    you read to name the clusters ("looks like a kids/DVD cluster")."""
    return features.assign(cluster=labels).groupby("cluster").agg(["mean"]).join(
        pd.Series(labels).value_counts().rename("size")
    )

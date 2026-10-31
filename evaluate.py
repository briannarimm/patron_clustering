"""Evaluation without planted cluster labels."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }


def bootstrap_stability(X: np.ndarray, algo, k: int, n_boot: int = 20,
                        seed: int = 0) -> float:
    """Mean ARI between clusterings of resampled data (compared on shared points).

    TODO: resample rows with replacement, cluster, compare to the full-data
    clustering on the sampled indices, average ARI.
    """
    raise NotImplementedError


def reference_partition(latent: pd.DataFrame) -> np.ndarray:
    """Soft post-hoc reference: dominant latent interest x age band.

    NOT a planted truth. It is a coarse human-readable summary of the latent
    traits, used only to ask "do the clusters line up with something meaningful?"
    TODO: implement.
    """
    raise NotImplementedError


def agreement_with_reference(labels: np.ndarray, reference: np.ndarray) -> float:
    return adjusted_rand_score(reference, labels)


def choose_k(X: np.ndarray, algo, k_range=range(2, 11)) -> pd.DataFrame:
    """Silhouette / (BIC for GMM) across k. Expect an ambiguous elbow on
    realistic data, which is itself a finding worth reporting."""
    raise NotImplementedError

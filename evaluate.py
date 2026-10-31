"""Evaluation without planted cluster labels."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture

from catalog import INTERESTS
from cluster import run_gmm

AGE_BANDS = [0, 12, 17, 64, 200]
AGE_LABELS = ["child", "teen", "adult", "senior"]


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def bootstrap_stability(X: np.ndarray, algo, k: int, n_boot: int = 20,
                        seed: int = 0) -> float:
    """Mean ARI between the full-data clustering and clusterings of bootstrap
    resamples, compared on the resampled points. Near 1 = stable."""
    rng = np.random.default_rng(seed)
    full = algo(X, k, seed=seed)
    scores = []
    for b in range(n_boot):
        idx = np.unique(rng.integers(0, len(X), len(X)))
        boot = algo(X[idx], k, seed=seed + 1 + b)
        scores.append(adjusted_rand_score(full[idx], boot))
    return float(np.mean(scores))


def reference_partition(latent: pd.DataFrame) -> np.ndarray:
    """Soft post-hoc reference: dominant latent interest x age band.

    NOT a planted truth. It is a coarse human-readable summary of the latent
    traits, used only to ask "do the clusters line up with something meaningful?"
    """
    dominant = latent[INTERESTS].idxmax(axis=1).astype(str)
    band = pd.cut(latent["age"], AGE_BANDS, labels=AGE_LABELS).astype(str)
    return pd.factorize(band + "/" + dominant)[0]


def agreement_with_reference(labels: np.ndarray, reference: np.ndarray) -> float:
    return adjusted_rand_score(reference, labels)


def choose_k(X: np.ndarray, algo, k_range=range(2, 11)) -> pd.DataFrame:
    """Silhouette and Davies-Bouldin (plus BIC for GMM) across k. Expect an
    ambiguous elbow on realistic data, which is itself a finding worth reporting."""
    rows = []
    for k in k_range:
        labels = algo(X, k)
        row = {"k": k, **internal_metrics(X, labels)}
        if algo is run_gmm:
            row["bic"] = GaussianMixture(n_components=k, random_state=0).fit(X).bic(X)
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")

import numpy as np
import pandas as pd

from catalog import KIDS_SUBJECTS, FICTION_SUBJECTS, ACADEMIC_SUBJECTS

# Patrons with fewer checkouts than this are dropped. One or two checkouts give
# percentages of 0 or 1 and say nothing about the patron.
MIN_CHECKOUTS = 3

# Clustering input. The volume-only columns below are computed too, but they
# all track total_checkouts and would let activity level swamp everything else.
FEATURE_COLUMNS = [
    "total_checkouts",        # log-transformed downstream
    "pct_dvd",
    "pct_audiobook",
    "pct_large_print",
    "pct_book",
    "pct_kids_subject",
    "pct_fiction_subject",
    "pct_academic_subject",
    "subject_diversity",      # Shannon entropy over subjects
]
EXTRA_COLUMNS = [
    "n_active_months",
    "visit_regularity",       # 1 / (1 + CV of days between visits); 0 if under 3 visits
    "items_per_visit",
    "format_diversity",
]


def _entropy(counts: pd.DataFrame) -> pd.Series:
    p = counts.div(counts.sum(axis=1), axis=0)
    return -(p * np.log(p.where(p > 0, 1))).sum(axis=1)


def _regularity(dates: pd.Series) -> float:
    gaps = np.diff(np.sort(dates.unique()).astype("datetime64[D]").astype(int))
    if len(gaps) < 2:
        return 0.0
    return 1 / (1 + gaps.std() / gaps.mean())


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate events to patron-level features, indexed by patron_id.
    Returns FEATURE_COLUMNS + EXTRA_COLUMNS; preprocess() picks the former."""
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    g = ev.groupby("patron_id")

    f = pd.DataFrame(index=g.size().index)
    f["total_checkouts"] = g.size()
    f["n_active_months"] = g["date"].apply(lambda d: d.dt.to_period("M").nunique())
    f["visit_regularity"] = g["date"].apply(_regularity)
    f["items_per_visit"] = f["total_checkouts"] / g["date"].nunique()

    fmt = pd.crosstab(ev["patron_id"], ev["item_format"])
    for name in ["dvd", "audiobook", "large_print", "book"]:
        f[f"pct_{name}"] = fmt.get(name, 0) / f["total_checkouts"]

    subj = pd.crosstab(ev["patron_id"], ev["subject"])
    for name, group in [("kids", KIDS_SUBJECTS), ("fiction", FICTION_SUBJECTS),
                        ("academic", ACADEMIC_SUBJECTS)]:
        cols = [c for c in subj.columns if c in group]
        f[f"pct_{name}_subject"] = subj[cols].sum(axis=1) / f["total_checkouts"]

    f["subject_diversity"] = _entropy(subj)
    f["format_diversity"] = _entropy(fmt)

    f = f[f["total_checkouts"] >= MIN_CHECKOUTS]
    return f[FEATURE_COLUMNS + EXTRA_COLUMNS]

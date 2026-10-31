"""Events -> one row per patron (clustering input).

The clustering only ever sees what a real library could compute from
circulation records. No latent traits leak in here.
"""
import pandas as pd

# Candidate features. Trim/extend after looking at the data.
FEATURE_COLUMNS = [
    "total_checkouts",        # log-transformed downstream
    "n_active_months",
    "visit_regularity",       # e.g. 1 / CV of days between visits
    "items_per_visit",
    "pct_dvd",
    "pct_audiobook",
    "pct_large_print",
    "pct_book",
    "pct_kids_subject",
    "pct_fiction_subject",
    "pct_academic_subject",
    "subject_diversity",      # Shannon entropy over subjects
    "format_diversity",
]


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate events to patron-level features.

    TODO: groupby patron_id and compute FEATURE_COLUMNS. Decide how to treat
    very sparse patrons (e.g. < 3 checkouts): drop, or keep with a flag.
    """
    raise NotImplementedError

"""Traits -> checkout events."""
import numpy as np
import pandas as pd

from config import SimConfig


def seasonal_weights(n_days: int, cfg: SimConfig) -> np.ndarray:
    """Per-day relative probability of a visit (summer bump, holiday dip, weekday effect).

    TODO: return array of length n_days summing to 1.
    """
    raise NotImplementedError


def simulate_patron(patron: pd.Series, day_weights: np.ndarray,
                    cfg: SimConfig, rng: np.random.Generator) -> list[dict]:
    """Generate one patron's checkout events for the year.

    Steps:
      1. n_visits ~ Poisson(patron.activity)
      2. visit days ~ day_weights
      3. per visit, items ~ small count (1-5); each item:
           - with prob cfg.noise_rate: pick interest uniformly (noise)
           - else: pick interest ~ patron's latent interest mix
           - pick format ~ INTEREST_TO_FORMAT[interest]
           - pick subject ~ INTEREST_TO_SUBJECTS[interest]
    Each event dict: patron_id, date, item_format, subject, interest(optional, drop on export)
    """
    raise NotImplementedError


def simulate_all(patrons: pd.DataFrame, cfg: SimConfig,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Run simulate_patron over everyone and return the events table.

    IMPORTANT: drop the latent `interest` column before export; the events CSV
    should look like real ILS data (patron, date, format, subject only).
    """
    raise NotImplementedError

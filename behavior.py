"""Traits -> checkout events."""
import numpy as np
import pandas as pd

from config import SimConfig
from catalog import INTERESTS, INTEREST_TO_FORMAT, INTEREST_TO_SUBJECTS

# unpack the catalog once so the per-item loop stays cheap
_FORMATS = {i: list(f.keys()) for i, f in INTEREST_TO_FORMAT.items()}
_FORMAT_P = {i: np.array(list(f.values())) for i, f in INTEREST_TO_FORMAT.items()}


def seasonal_weights(n_days: int, cfg: SimConfig) -> np.ndarray:
    """Per-day relative probability of a visit (such as summer bump, holiday dip, Sunday dip)."""
    days = pd.date_range(cfg.start_date, periods=n_days)
    doy = days.dayofyear.to_numpy()
    s = cfg.seasonality

    summer = np.exp(-((doy - 196) / 35) ** 2)          # peaks mid July
    holiday = ((days.month == 12) & (days.day >= 20)) | ((days.month == 1) & (days.day <= 2))
    sunday = days.dayofweek == 6

    w = 1 + s * summer - s * holiday - 0.5 * s * sunday
    w = np.maximum(w, 0.05)
    return w / w.sum()


def simulate_patron(patron: pd.Series, day_weights: np.ndarray,
                    cfg: SimConfig, rng: np.random.Generator) -> list[dict]:
    """One patron's checkout events for the year.

    Visits ~ Poisson(activity), visit days ~ day_weights, 1+ items per visit.
    Each item follows the latent interest mix, except a noise_rate share that
    picks an interest uniformly.
    """
    n_visits = rng.poisson(patron["activity"])
    if n_visits == 0:
        return []
    visit_days = rng.choice(len(day_weights), size=n_visits, p=day_weights)
    items_per_visit = 1 + rng.poisson(1.0, size=n_visits)
    item_days = np.repeat(visit_days, items_per_visit)
    n_items = len(item_days)

    mix = patron[INTERESTS].to_numpy(dtype=float)
    interest = rng.choice(len(INTERESTS), size=n_items, p=mix)
    noise = rng.random(n_items) < cfg.noise_rate
    interest[noise] = rng.integers(len(INTERESTS), size=noise.sum())

    start = pd.Timestamp(cfg.start_date)
    events = []
    for day, i in zip(item_days, interest):
        name = INTERESTS[i]
        events.append({
            "patron_id": int(patron["patron_id"]),
            "date": (start + pd.Timedelta(days=int(day))).date(),
            "item_format": rng.choice(_FORMATS[name], p=_FORMAT_P[name]),
            "subject": rng.choice(INTEREST_TO_SUBJECTS[name]),
            "interest": name,
        })
    return events


def simulate_all(patrons: pd.DataFrame, cfg: SimConfig,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Everyone's actions exported!"""
    day_weights = seasonal_weights(cfg.n_days, cfg)
    events = []
    for _, patron in patrons.iterrows():
        events.extend(simulate_patron(patron, day_weights, cfg, rng))
    df = pd.DataFrame(events, columns=["patron_id", "date", "item_format", "subject", "interest"])
    df = df.drop(columns="interest")
    return df.sort_values(["patron_id", "date"]).reset_index(drop=True)

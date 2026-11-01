import numpy as np
import pandas as pd

from config import SimConfig
from catalog import INTERESTS

# Age mixture: (weight, mean, sd)
AGE_MIXTURE = [
    (0.20, 9, 3),
    (0.10, 15, 1.5),
    (0.50, 40, 12),
    (0.20, 72, 7),
]


def sample_age(n: int, rng: np.random.Generator) -> np.ndarray:
    weights, means, sds = (np.array(c) for c in zip(*AGE_MIXTURE))
    comp = rng.choice(len(AGE_MIXTURE), size=n, p=weights)
    age = rng.normal(means[comp], sds[comp])
    return np.clip(np.round(age), 3, 95).astype(int)


def sample_activity(n: int, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    return rng.lognormal(cfg.activity_mu, cfg.activity_sigma, size=n)


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def interest_alpha(age: float, household: bool, cfg: SimConfig) -> np.ndarray:
    """Dirichlet parameters over INTERESTS for one patron"""
    # smooth age curves, each in [0, 1]
    kid = 1 - _ramp(age, 6, 13)
    teen = float(np.exp(-((age - 15) / 3) ** 2))
    adult = _ramp(age, 15, 22)
    student = float(np.exp(-((age - 24) / 6) ** 2))
    senior = _ramp(age, 58, 72)

    tilt = {
        "adult_fiction":      0.8 * adult + 0.4 * senior,
        "adult_nonfiction":   0.6 * adult + 0.2 * senior,
        "academic_reference": 1.0 * student + 0.1 * adult,
        "kids_books":         1.2 * kid,
        "kids_media":         1.0 * kid,
        "teen_ya":            1.2 * teen + 0.3 * kid,
        "adult_media":        0.6 * adult - 0.3 * senior,
        "hobby_practical":    0.4 * adult + 0.4 * senior,
    }
    if household:
        tilt["kids_books"] += 0.8
        tilt["kids_media"] += 0.8

    direction = 1 / len(INTERESTS) + cfg.coupling * np.array([tilt[i] for i in INTERESTS])
    direction = np.maximum(direction, 0.01)
    return cfg.interest_concentration * direction / direction.sum()


def sample_patrons(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_patrons
    age = sample_age(n, rng)
    household = (rng.random(n) < cfg.household_card_rate) & (age >= 22)
    activity = sample_activity(n, cfg, rng)
    mix = np.vstack([
        rng.dirichlet(interest_alpha(a, h, cfg)) for a, h in zip(age, household)
    ])
    patrons = pd.DataFrame({
        "patron_id": np.arange(n),
        "age": age,
        "household_card": household,
        "activity": activity,
    })
    return pd.concat([patrons, pd.DataFrame(mix, columns=INTERESTS)], axis=1)

"""Sample latent patron traits. No archetype labels anywhere."""
import numpy as np
import pandas as pd

from config import SimConfig
from catalog import INTERESTS


def sample_age(n: int, rng: np.random.Generator) -> np.ndarray:
    """Age distribution of cardholders.

    TODO: mixture of child / teen / adult / senior with realistic proportions
    (a continuous mixture, NOT labeled types). Return array of ints.
    """
    raise NotImplementedError


def sample_activity(n: int, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Expected checkouts per year, heavy-tailed (lognormal)."""
    return rng.lognormal(cfg.activity_mu, cfg.activity_sigma, size=n)


def interest_alpha(age: float, household: bool, cfg: SimConfig) -> np.ndarray:
    """Dirichlet parameters over INTERESTS for one patron.

    This is where `coupling` acts: blend a flat baseline with an age/household-
    dependent tilt, e.g.  alpha = base + coupling * tilt(age, household).

    TODO: define the tilt. Household cards should tilt toward BOTH kids_* and
    adult_* interests (mixed use).
    """
    raise NotImplementedError


def sample_patrons(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Return one row per patron with latent traits.

    Columns: patron_id, age, household_card, activity, plus one column per
    interest (the latent interest mix, summing to 1).
    """
    raise NotImplementedError

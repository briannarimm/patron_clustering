# knobs for data simulation
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    seed: int = 42

    # Population / time window
    n_patrons: int = 2000
    start_date: str = "2025-01-01"
    n_days: int = 365

    # --- Realism knobs ---
    # Coupling: how strongly age/household drive interests.
    # 0.0 = interests independent of demographics (near-random data)
    # 1.0 = realistic correlation; >1 exaggerates structure
    coupling: float = 1.0

    # Activity level ~ lognormal (heavy tail). Higher sigma = more skew.
    activity_mu: float = 1.5
    activity_sigma: float = 1.0

    # Share of cards that are household cards (mixed adult + kid use)
    household_card_rate: float = 0.15

    # Dirichlet concentration for interest mix. Lower = more specialized patrons,
    # higher = more omnivorous.
    interest_concentration: float = 2.0

    # Seasonality strength (0 = none)
    seasonality: float = 0.3

    # Probability that a checkout ignores the patron's interests (pure noise)
    noise_rate: float = 0.05


@dataclass
class PathConfig:
    data_dir: str = "data"
    results_dir: str = "results"
    events_csv: str = "data/events.csv"
    latent_csv: str = "data/latent_traits.csv"
    features_csv: str = "data/patron_features.csv"

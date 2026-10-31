# CLAUDE.md

## What this project is

A proof of concept for a "machine learning for libraries" topic exploration (library master's coursework). A Python program simulates library patron checkout behavior, exports it to CSV, and a clustering analysis then tries to recover recognizable patron types (e.g. the researcher who sometimes reads fiction, the kid who wants DVDs of shows, the adult fiction reader). The goal is to show that patron data plus clustering can surface patron segments.

The status is a **skeleton**. Config, runners, and module interfaces exist. Most core functions raise `NotImplementedError` and have docstrings with TODOs. The job is to implement them.

## The central design constraint

**The generator must not contain patron archetypes.** If the simulation hard-codes "researcher" or "kid" types, then the clustering only rediscovers what we planted, and the demo proves nothing. At the same time, fully random data makes clustering fail.

The chosen approach:

1. **Continuous latent traits (no type labels).** Each patron gets age, household-card status, an activity level (heavy-tailed), and an interest mix (a Dirichlet draw over `INTERESTS` in `catalog.py`).
2. **Coupling.** Age and household status tilt the interest mix, controlled by `SimConfig.coupling`. At 0, interests are independent of demographics (near-random data). At 1, correlations are realistic. Above 1, structure is exaggerated. Types can emerge only through these correlations.
3. **Event-level simulation, then aggregation.** Traits generate individual checkout events (date, format, subject). `features.py` aggregates them into per-patron features. Any structure has to survive aggregation, as it would with real circulation data.
4. **Deliberate messiness.** Heavy-tailed activity, overlapping interests, mixed-use household cards (a parent's card used for kid DVDs and the parent's own thrillers), seasonality, very sparse patrons, and a noise rate for checkouts that ignore the patron's interests.

## Data flow

```
run_simulation.py
  traits.sample_patrons  -> latent traits   -> data/latent_traits.csv  (analysis only)
  behavior.simulate_all  -> events          -> data/events.csv
  features.build_features-> patron features -> data/patron_features.csv
run_clustering.py: features -> cluster.preprocess -> algorithm -> profiles
run_sweep.py: coupling x algorithm x replicate -> results/sweep.csv
```

## Hard rules

- **No leakage.** `events.csv` and `patron_features.csv` must contain only what a real ILS could export (patron id, date, format, subject, and aggregates of these). Drop the latent `interest` column before exporting events. Latent traits never enter `preprocess` or the clustering input. They are used only afterward, to interpret clusters.
- **Reproducibility.** All randomness flows from a `np.random.Generator` seeded from `SimConfig.seed`. Don't use the global `np.random` or Python's `random`.
- **Don't plant clusters to make results look good.** If clustering fails at low coupling, that is a finding, not a bug.

## Evaluation (no ground-truth labels exist)

- Internal metrics: silhouette, Davies-Bouldin; BIC for GMM.
- Stability: bootstrap ARI (`evaluate.bootstrap_stability`).
- Soft reference: `evaluate.reference_partition` (dominant latent interest × age band), compared by ARI. This is a coarse human-readable summary of the latent traits, **not a planted truth**, and should be described that way in any write-up.
- Expect an ambiguous elbow when choosing k on realistic data. Report that honestly.
- Headline experiment: the coupling sweep in `run_sweep.py`. It shows how clustering quality depends on how strongly demographics drive behavior.

## Suggested implementation order

1. `catalog.py`: fill `INTEREST_TO_FORMAT` and `INTEREST_TO_SUBJECTS`.
2. `traits.py`: age mixture, `interest_alpha` (where coupling acts), `sample_patrons`.
3. `behavior.py`: seasonality weights, `simulate_patron`, `simulate_all`.
4. `features.py`: `build_features` (decide on sparse-patron handling).
5. `cluster.preprocess`: log-transform skewed counts, then standardize.
6. `evaluate.py`: stability, reference partition, `choose_k`.
7. Run end to end, inspect cluster profiles, then run the sweep and plot it.

## Open decisions

- Patron count and time window (defaults: 2000 patrons, 365 days).
- Final feature list (candidates in `features.FEATURE_COLUMNS`).
- Handling of very sparse patrons (e.g. under 3 checkouts): drop, or keep with a flag.
- Optionally calibrate marginals (format mix, loans-per-patron distribution) to open data such as Seattle Public Library checkouts. Match distributions, don't copy records. Probably v2.

## Environment

Python with numpy, pandas, scikit-learn, matplotlib (`requirements.txt`). Run scripts from the project root; modules import each other as top-level (`from config import SimConfig`). Generated data goes in `data/`, sweep output in `results/`.

## Working style notes

The user is a library master's student with a legal background and basic ML knowledge, so explain ML choices briefly rather than assuming expertise. Be candid about limitations, especially around circularity in simulated data. This is a class demo, so clarity and defensibility matter more than sophistication.

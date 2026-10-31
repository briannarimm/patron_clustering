# Patron Clustering: ML for Libraries (proof of concept)

Simulate library patron checkout behavior, export it to CSV, then cluster patrons
to see whether recognizable "patron types" emerge.

## Design principle: no archetypes in the generator

The generator never contains a "researcher" or "kid" type. Each patron gets
**continuous latent traits** (age, household role, activity level, interest mix).
Checkout behavior follows from those traits. Types can only emerge where traits
are correlated, as they are in real life.

The **coupling** knob (config.py) controls how strongly age/household drive
interests. Low coupling means near-random data. High coupling means strong
structure. The sweep experiment varies it.

## Pipeline

```
run_simulation.py  ->  data/events.csv            (one row per checkout)
                       data/latent_traits.csv     (hidden truth, NOT fed to clustering)
features.py        ->  data/patron_features.csv   (one row per patron; clustering input)
run_clustering.py  ->  data/clusters_<algo>.csv + cluster profile tables
run_sweep.py       ->  results/sweep.csv          (coupling vs. clustering quality)
```

## Layout

```
config.py            all knobs in one dataclass
catalog.py           formats x subjects (what can be checked out)
traits.py            sample latent patron traits
behavior.py          traits -> checkout events
features.py          events -> per-patron feature table
cluster.py           kmeans / GMM / hierarchical wrappers
evaluate.py          internal metrics, stability, post-hoc trait comparison
run_simulation.py    CLI
run_clustering.py    CLI
run_sweep.py         CLI
```

## Realism checklist (the "make it messy" part)

- [ ] Heavy-tailed activity (most patrons check out little, a few check out constantly)
- [ ] Overlapping interests (researcher who also reads fiction)
- [ ] Mixed-use household cards (parent card: kid DVDs + adult fiction)
- [ ] Seasonality (summer reading, holidays)
- [ ] Sparse patrons (1-2 checkouts all year)
- [ ] Calibrate marginals to open data (Seattle Public Library checkouts)

## Evaluation without planted labels

No ground-truth clusters exist, so:
1. **Internal quality**: silhouette, Davies-Bouldin, BIC for GMM
2. **Stability**: ARI between clusterings on bootstrap resamples
3. **Post-hoc interpretation**: do clusters line up with latent traits
   (dominant interest x age band)? This is a soft reference, not a planted truth.

## Open decisions

- Number of patrons / time window (default 2000 patrons, 1 year)
- Which features to engineer (see features.py)
- Whether to calibrate against Seattle data in v1 or v2

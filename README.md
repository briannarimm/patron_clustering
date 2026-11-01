# Patron clustering

A small experiment for my ML for Libraries topic exploration (ISLT 7301). Inspired by discussion of k-means clustering analysis in Lund (2026, pp. 88–89).

I simulate patron records. In the future, I want to incorporate data from public libraries, especially [the data available from Seattle Public Libraries](https://data.seattle.gov/Community-and-Culture/Checkouts-by-Title/tmmm-ytt6/about_data).

AI Disclosure: While I tried to hand code as much of this program as I could based on some old anthropology notes (see [this repo](https://github.com/briannarimm/ml-applications-in-anthropology), if you're interested in that experiment!), I used Claude to help with debugging the clustering code, helping to create the patron data simulation code, and to refactor all of the code (mostly as a sanity check).

## Contents

1. [How it works](#how-it-works)
2. [Run it](#run-it)
3. [What comes out](#what-comes-out)
4. [Limits](#limits)
5. [Files](#files)
6. [References](#references)

## How it works

1. **Simulate patrons.** Each patron gets an age, an activity level, a household card flag, and a mix of interests. The generator has no patron "types" in it. A kids-and-DVDs patron can only appear because age tilts the interest mix toward kids' books and shows. The `coupling` setting controls how strong that tilt is. At 0, interests have nothing to do with age.
2. **Simulate a year of checkouts** from those traits.
3. **Build one row per patron** Cluster the rows with k-means, a Gaussian mixture, or hierarchical clustering.

Only after clustering does the code look back at the hidden traits, to ask whether the clusters mean anything.

## Run it

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

`demo.py` does all three steps, prints the cluster profiles, and saves a figure to `results/`. The figure shows the patrons in two dimensions, what each cluster looks like in the features, and how each cluster lines up with the hidden interests.

Things to try:

- `python demo.py --coupling 0` and `python demo.py --coupling 2`
- `python demo.py --algo gmm --k 4`
- `python demo.py --show` to open the figure in a window

The steps also run on their own: `run_simulation.py` writes the CSVs to `data/`, `run_clustering.py` clusters them, `run_sweep.py` repeats the whole thing across coupling values and seeds, and `plot_sweep.py` plots that.

## What comes out

With the default settings, k-means with five clusters gives roughly: heavy users who read a bit of everything, light users, a kids-and-DVDs group, an academic nonfiction group, and adult fiction readers with the large print share. The silhouette score is low (around 0.2). The clusters overlap, as real patrons would. Picking k is not clean either. These are the problems with k-means pointed out in Trivedi et al. (2017).

## Limits

The clustering uses simulated data. As mentioned above, I want to rework this program to analyze Seattle Public Library's public circulation data.

Lund (2026, p. 89) cites to Jin and He (2017). I have not yet had the chance to read this article, but I'm sure it will provide insight for future work on this project.

## Files

```
config.py            all settings in one place
catalog.py           formats, subjects, and how interests map to them
traits.py            hidden patron traits
behavior.py          traits -> checkout events
features.py          events -> one row per patron
cluster.py           preprocessing and the three algorithms
evaluate.py          metrics, bootstrap stability, post-hoc comparison
demo.py              runs everything
run_simulation.py    step 1 and 2
run_clustering.py    step 3
run_sweep.py         coupling x algorithm x seed -> results/sweep.csv
plot_sweep.py        plot the sweep
```

## References

Jin, W., & He, S. (2017). Research on management of libraries in universities and  colleges based on k-means clustering algorithm under big data environment. *Revista de la Facultad de Ingeria*, 32(8), 177–81.

Lund, B. D. (2026). Artificial intelligence for libraries. Bloomsbury Libraries Unlimited.

Trivedi, Y., Joshi, S., Kansaria, K., & Vora, D. (2017). LibX — An automated bibliotheca. *2017 International Conference on Intelligent Sustainable Systems (ICISS)*, 239–243. https://doi.org/10.1109/ISS1.2017.8389406

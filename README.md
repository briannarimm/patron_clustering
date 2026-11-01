# Patron clustering

A small experiment for my ML for Libraries topic exploration (ISLT 7301). The question: can you use clustering to find patterns in circulation data?

I simulate patron records. In the future, I want to incorporate data from public libraries, especially [the data available from Seattle Public Libraries](https://data.seattle.gov/Community-and-Culture/Checkouts-by-Title/tmmm-ytt6/about_data).

## How it works

1. **Simulate patrons.** Each patron gets an age, an activity level, a household card flag, and a mix of interests. The generator has no patron "types" in it. A kids-and-DVDs patron can only appear because age tilts the interest mix toward kids' books and shows. The `coupling` setting controls how strong that tilt is. At 0, interests have nothing to do with age.
2. **Simulate a year of checkouts** from those traits. The result looks like what an ILS would export: patron id, date, format, subject. Nothing else.
3. **Build one row per patron** from the checkouts (share of DVDs, share of kids' subjects, total checkouts, and so on) and cluster the rows with k-means, a Gaussian mixture, or hierarchical clustering.

Only after clustering does the code look back at the hidden traits, to ask whether the clusters mean anything. The traits never go into the clustering.

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

With the default settings, k-means with five clusters gives roughly: heavy users who read a bit of everything, light users, a kids-and-DVDs group, an academic nonfiction group, and adult fiction readers with the large print share. The silhouette score is low (around 0.2). The clusters overlap, as real patrons would. Picking k is not clean either. That is part of the point.

## Limits

The clustering uses simulated data. As mentioned above, I want to rework this program to analyze Seattle Public Library's public circulation data.

## Files

```
config.py            all settings in one place
catalog.py           formats, subjects, and how interests map to them
traits.py            hidden patron traits
behavior.py          traits -> checkout events
features.py          events -> one row per patron
cluster.py           preprocessing and the three algorithms
evaluate.py          metrics, bootstrap stability, post-hoc comparison
demo.py              everything in one run, with a figure
run_simulation.py    step 1 and 2
run_clustering.py    step 3
run_sweep.py         coupling x algorithm x seed -> results/sweep.csv
plot_sweep.py        plot the sweep
```

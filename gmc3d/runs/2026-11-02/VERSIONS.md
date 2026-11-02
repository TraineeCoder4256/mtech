# Where plain Monte Carlo lost its speed, version by version

Speed thread, 2 Nov 2026.  Branch `gmc-speed` (v1.6.1 merged in as
9add80b).  Nothing in `gmc3d/core/` or `gmc3d/ball/` was changed: every
measurement runs the released code out of a git worktree of its own
commit, and every ablation is a throwaway copy under the scratch
directory.

The question was: our plain Monte Carlo has been getting slower as the
versions went out, and nobody knew exactly why.  Answered here per step,
with the cause proved by taking the suspect out and measuring again, not
by reading the diff.

## How it was measured

Ten worktrees, one per released commit: v1.0 `177c5db`, v1.1.0 `8b80020`,
v1.1.3 `0764f63`, v1.2.0 `0013942`, v1.3.0 `052abb5`, v1.4.0 `3db14a2`,
v1.4.1 `7b5ed2f`, v1.5.0 `ef479a3`, v1.6.0 `641c6c4`, v1.6.1 `da4221c`.
Each one runs the same three problems (sphere, slab, cask), 100,000
histories x 5 batches, plain MC (no ball backend), on this container's 4
cores.

Three separate measurements, because they answer different questions:

* **the whole run with no tallies at all** -- geometry plus flights plus
  the per-batch bookkeeping, and the only column that is comparable
  across versions without further care;
* **`_descend` on its own**, from a tight compiled loop over 5,000 fixed
  interior points with both questions asked (nearest surface and ray
  distance).  Same points and directions in every version;
* **the source sampler on its own**, one batch of 100,000 sites.

One warning about the fourth column anyone would reach for.  **"With
tallies" is not comparable across versions**, because the problems' own
tally sets grew: at v1.1.3 the cask went from 3 tallies to 6 (a material
tally with four scores, a 2D mesh and a rectilinear mesh) and the slab
and sphere gained scores.  So the runs were repeated with the tally set
cut back to v1.0's in every version -- v1.0's tallies, by name, scoring
flux and absorption only.
## Plain MC, no tallies at all: seconds for 500,000 histories

### no tallies

| version | sphere | slab | cask | sphere vs v1.0 | slab vs v1.0 | cask vs v1.0 |
|---|---|---|---|---|---|---|
| v1.0 | 1.18 | 1.79 | 2.93 | 1.00x | 1.00x | 1.00x |
| v1.1.0 | 1.34 | 1.71 | 2.83 | 1.14x | 0.96x | 0.97x |
| v1.1.3 | 1.09 | 1.75 | 3.05 | 0.92x | 0.98x | 1.04x |
| v1.2.0 | 1.72 | 2.35 | 3.55 | 1.46x | 1.32x | 1.21x |
| v1.3.0 | 1.91 | 2.60 | 3.70 | 1.62x | 1.45x | 1.26x |
| v1.4.0 | 2.00 | 2.76 | 3.99 | 1.69x | 1.55x | 1.36x |
| v1.4.1 | 2.19 | 3.05 | 4.56 | 1.86x | 1.71x | 1.56x |
| v1.5.0 | 2.24 | 3.11 | 4.63 | 1.90x | 1.74x | 1.58x |
| v1.6.0 | 2.49 | 3.19 | 4.67 | 2.11x | 1.79x | 1.59x |
| v1.6.1 | 2.71 | 3.50 | 5.29 | 2.30x | 1.96x | 1.81x |

## Where that time sits

### advance s / rest s, then ns per _descend, then source ms per batch

| version | sphere adv/rest | slab adv/rest | cask adv/rest | sphere desc ns | slab desc ns | cask desc ns | sphere src ms | slab src ms | cask src ms |
|---|---|---|---|---|---|---|---|---|---|
| v1.0 | 1.11 / 0.07 | 1.71 / 0.07 | 2.85 / 0.08 | 648 | 822 | 653 | - | - | - |
| v1.1.0 | 1.26 / 0.08 | 1.64 / 0.07 | 2.75 / 0.08 | 668 | 765 | 721 | - | - | - |
| v1.1.3 | 1.02 / 0.07 | 1.68 / 0.07 | 2.98 / 0.08 | 667 | 707 | 732 | - | - | - |
| v1.2.0 | 1.65 / 0.07 | 2.27 / 0.08 | 3.48 / 0.07 | 672 | 757 | 647 | 126 | 135 | 124 |
| v1.3.0 | 1.80 / 0.11 | 2.50 / 0.09 | 3.60 / 0.09 | 684 | 820 | 717 | 115 | 129 | 122 |
| v1.4.0 | 1.91 / 0.09 | 2.55 / 0.22 | 3.90 / 0.09 | 645 | 786 | 703 | 114 | 133 | 118 |
| v1.4.1 | 2.11 / 0.08 | 2.92 / 0.13 | 4.47 / 0.09 | 914 | 1161 | 898 | 141 | 154 | 151 |
| v1.5.0 | 2.15 / 0.09 | 2.94 / 0.17 | 4.54 / 0.10 | 874 | 1152 | 962 | 123 | 124 | 124 |
| v1.6.0 | 2.40 / 0.09 | 3.09 / 0.10 | 4.53 / 0.14 | 878 | 1180 | 869 | 135 | 137 | 148 |
| v1.6.1 | 2.61 / 0.10 | 3.39 / 0.11 | 5.21 / 0.08 | 883 | 1218 | 940 | 139 | 149 | 149 |

## The tally side

### tally cost: the problem's own set, then v1.0's set, then how many tallies v1.0's set is

| version | sphere own | slab own | cask own | sphere fixed | slab fixed | cask fixed | sphere n | slab n | cask n |
|---|---|---|---|---|---|---|---|---|---|
| v1.0 | 0.55 | 0.60 | 0.14 | 0.36 | 0.30 | 0.41 | 3 | 2 | 3 |
| v1.1.0 | 0.30 | 0.26 | 0.80 | 0.31 | 0.52 | 0.82 | 3 | 2 | 3 |
| v1.1.3 | 1.05 | 1.58 | 4.27 | 0.97 | 0.95 | 1.38 | 3 | 2 | 3 |
| v1.2.0 | 0.94 | 1.88 | 4.11 | 1.05 | 0.91 | 1.27 | 3 | 2 | 3 |
| v1.3.0 | 0.97 | 2.42 | 5.58 | 0.97 | 1.19 | 2.00 | 3 | 2 | 3 |
| v1.4.0 | 0.93 | 2.06 | 5.43 | 1.09 | 1.02 | 1.97 | 3 | 2 | 3 |
| v1.4.1 | 0.79 | 1.86 | 5.23 | 1.06 | 1.08 | 1.73 | 3 | 2 | 3 |
| v1.5.0 | 3.43 | 3.37 | 4.93 | 1.18 | 1.19 | 1.98 | 3 | 2 | 3 |
| v1.6.0 | 3.58 | 3.36 | 5.51 | 0.80 | 1.13 | 1.90 | 3 | 2 | 3 |
| v1.6.1 | 3.88 | 3.17 | 5.18 | 0.98 | 1.19 | 1.85 | 3 | 2 | 3 |

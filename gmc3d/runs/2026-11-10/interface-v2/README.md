# Interface v2: what was measured (10 Nov 2026)

Interface v2 was agreed between Innovation and Development on 10 Nov 2026 (Phani: "discuss with development for network related stuff"). Spec: `ball/interface.py`. Laws: `ball/laws.py`. Rows: `ball/data.make_aniso`.

## How many Legendre moments does a ball's exit depend on?

Script: `checks/aniso_truncation.py`. Logs:
- `truncation.log`: L = 1, 2, 3, 5, 8.
- `truncation_high_L.log`: L = 8, 12, 16 for the peaked laws.
- `truncation_clipped.log`: a first version, superseded (see its note in the script).

Method:
- For a law A, take a second positive law B with exactly A's first L moments: the maximum-entropy law with those moments.
- Run 100,000 exact anisotropic walks with each, at R = 1, 3, 10, 30 mfp.
- Compare the four outputs. z is the difference of means in standard errors; KS D is the two-sample distance.
- The noise floor is A against A with other random numbers: KS D 0.004-0.006.

Largest KS D over the four outputs:

| law A (g1) | L=1 | L=2 | L=3 | L=5 | L=8 | L=12 | L=16 |
|---|---|---|---|---|---|---|---|
| HG 0.3 (0.30), R=1 | 0.015 | 0.006 | noise | noise | noise | | |
| HG 0.6 (0.60), R=1 | 0.081 | 0.028 | 0.012 | 0.006 | noise | | |
| histogram fwd (0.50) | 0.009 | 0.008 | noise | noise | noise | | |
| clipped P3 (0.45), R=1 | 0.065 | 0.060 | 0.014 | 0.009 | noise | | |
| HG 0.9 (0.90), R=1 | 0.318 | 0.240 | 0.170 | 0.099 | 0.048 | 0.018 | 0.014 |
| HG 0.9, R=3 | 0.273 | 0.192 | 0.125 | 0.064 | 0.027 | 0.010 | 0.009 |
| HG 0.9, R=10 | 0.177 | 0.093 | 0.040 | 0.012 | noise | | |
| HG 0.9, R=30 | 0.054 | 0.014 | noise | noise | noise | | |
| 2 lobes 0.7 x HG 0.9 + 0.3 x HG -0.5 (0.50), R=1 | 0.332 | 0.203 | 0.141 | 0.077 | 0.036 | 0.013 | 0.011 |

Findings:
- Moderate laws (g1 up to about 0.6) need 3 to 5 moments.
- Peaked laws need more in small balls. HG 0.9 at R = 1 is still off with 8 moments: the exit azimuth phi is 5 standard errors off, KS 0.048. With 12 moments it is mostly closed (KS 0.018).
- Large balls forget the law: at R = 30, 3 moments are enough even for HG 0.9.
- So the interface carries 12 moments (`N_MOMENTS = 12`), not the 8 first proposed, and a model may use fewer.
- At L = 12 and 16 the remaining gap is partly the 401-point table of B. Its g1 drifts by +0.002 to +0.010, so these columns are an upper bound.

## Cost of rows

- 812,600 anisotropic rows/s (56.5 scatters per walk on average, R log-uniform 1-30), against 1,321,357 for v1. Same 4-core container.
- So 40M rows cost about a minute of walking.
- Building the 20,000-law pool and its moments takes about 4 minutes (Python).

## Reference dataset

`data.dataset_aniso(1_000_000, 1, 30, seed=1)`, with the default pool of 20,000 laws, seed 1:
- sha256 (`data.data_hash`): `e9969835711a7e93d8499e7f1244115edc51989ebeaada3bafc5b5d82dcbbc31`
- 0 rows outside the interface ranges.
- mean 56.2 scatters per walk; g1 from -0.4999 to 0.9500.

The file is `data/ball/aniso_walks_1000000_1_30_1.npz` (gitignored; remade by the call above).

Held out (`laws.held_out`): the 7 laws of `checks/aniso.py`. None of them is in the pool, which is tested.

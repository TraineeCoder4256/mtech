#!/bin/bash
# Three ablations, each a throwaway copy with ONE thing taken out, to see
# what that one thing was costing.  Nothing in the repo is changed.
set -e
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
rm -rf $S/abl-noinst $S/abl-oldsurf $S/abl-noframe
cp -r $S/wt-head/gmc3d $S/abl-noinst         # v1.4.1
cp -r $S/wt-head/gmc3d $S/abl-oldsurf        # v1.4.1
cp -r $S/wt-v150/gmc3d $S/abl-noframe        # v1.5.0
for d in abl-noinst abl-oldsurf abl-noframe; do
  find $S/$d -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  find $S/$d \( -name "*.nbi" -o -name "*.nbc" \) -delete 2>/dev/null || true
done
cp $S/wt-v140/gmc3d/core/surfaces.py $S/abl-oldsurf/core/surfaces.py
python3 - "$S" <<'PY'
import sys
S = sys.argv[1]


def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        if s.count(old) != 1:
            raise SystemExit(f"FAILED {path}: {s.count(old)} matches for "
                             f"{old[:50]!r}")
        s = s.replace(old, new)
    open(path, "w").write(s)


# (a) v1.4.1 without the distributed-material instance tracking
g = f"{S}/abl-noinst/core/geometry.py"
s = open(g).read()
s = s.replace("        inst += g.cell_off[cell]\n", "", 2)
s = s.replace("            inst += g.lat_off[g.lat_univ_start[lat] + k]\n",
              "", 2)
s = s.replace("m = g.leaf_mat[inst] if g.cell_dist[cell] "
              "else g.cell_fill[cell]", "m = g.cell_fill[cell]")
s = s.replace("return g.leaf_mat[inst] if g.cell_dist[cell] "
              "else g.cell_fill[cell]", "return g.cell_fill[cell]")
if "inst +=" in s or "leaf_mat[inst]" in s:
    raise SystemExit("FAILED abl-noinst")
open(g, "w").write(s)

# (c) v1.5.0 without the winning surface's frame kept down the descent
patch(f"{S}/abl-noframe/core/geometry.py", [
    ("    hx, hy, hz, hu, hv, hw = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0   "
     "# winner's frame",
     "    hx, hy, hz, hu, hv, hw = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0   # ABL"),
    ("                    hx, hy, hz, hu, hv, hw = x, y, z, u, v, w\n", ""),
])
print("ABL_READY")
PY

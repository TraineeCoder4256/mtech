#!/bin/bash
# A dose-response curve: v1.0's Geometry padded by 2 and by 4 unused
# arrays, to go with the 0 and 8 already measured.
set -e
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
for n in 2 4; do
  rm -rf $S/padgeo$n
  cp -r $S/wt-v100/gmc3d $S/padgeo$n
  find $S/padgeo$n -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  find $S/padgeo$n \( -name "*.nbi" -o -name "*.nbc" \) -delete 2>/dev/null || true
  python3 - "$S/padgeo$n" "$n" <<'PY'
import sys
d, n = sys.argv[1], int(sys.argv[2])
names = ", ".join(f'"pad{i}"' for i in range(n))
g = d + "/core/geometry.py"
s = open(g).read()
old = '    "root",                                                 # root universe\n])'
assert s.count(old) == 1
s = s.replace(old, old[:-2] + f"    {names},  # PAD\n])")
open(g, "w").write(s)
o = d + "/core/openmc_import.py"
s = open(o).read()
old = "        univ_index[geo.root_universe.id])"
assert s.count(old) == 1
s = s.replace(old, "        univ_index[geo.root_universe.id],\n"
              f"        *[np.zeros(8) for _ in range({n})])      # PAD")
open(o, "w").write(s)
print(f"PAD3_READY_{n}")
PY
done

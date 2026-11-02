#!/bin/bash
# Build padded copies of v1.0 to isolate TWO mechanisms from the new logic:
#   padcoef : surf_coef rows widened 4 -> 11 columns (v1.4.0 made it 8,
#             v1.6.1 made it 11).  The extra columns are never read.
#   padgeo  : 8 unused float64 arrays appended to the Geometry namedtuple
#             (v1.0 had 24 fields, v1.6.1 has 32).
# Nothing in the repo is touched; these are throwaway copies.
set -e
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
rm -rf $S/padcoef $S/padgeo
cp -r $S/wt-v100/gmc3d $S/padcoef
cp -r $S/wt-v100/gmc3d $S/padgeo
find $S/padcoef $S/padgeo -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find $S/padcoef $S/padgeo -name "*.nbi" -delete 2>/dev/null || true
find $S/padcoef $S/padgeo -name "*.nbc" -delete 2>/dev/null || true

# --- padcoef ---------------------------------------------------------
sed -i 's/^    coef = np.zeros(4)$/    coef = np.zeros(11)   # PAD/' $S/padcoef/core/openmc_import.py
sed -i 's/np.asarray(coefs, float).reshape(-1, 4)/np.asarray(coefs, float).reshape(-1, 11)/' $S/padcoef/core/openmc_import.py
grep -n "PAD\|reshape(-1, 11)" $S/padcoef/core/openmc_import.py

# --- padgeo ----------------------------------------------------------
python3 - "$S/padgeo" <<'PY'
import sys, re
d = sys.argv[1]
g = d + "/core/geometry.py"
s = open(g).read()
s = s.replace('''    "root",                                                 # root universe
])''', '''    "root",                                                 # root universe
    "pad0", "pad1", "pad2", "pad3", "pad4", "pad5", "pad6", "pad7",  # PAD
])''')
open(g, "w").write(s)
o = d + "/core/openmc_import.py"
s = open(o).read()
s = s.replace('''        univ_index[geo.root_universe.id])''',
'''        univ_index[geo.root_universe.id],
        *[np.zeros(8) for _ in range(8)])      # PAD''')
open(o, "w").write(s)
PY
grep -n "PAD" $S/padgeo/core/geometry.py $S/padgeo/core/openmc_import.py
echo PADS_READY

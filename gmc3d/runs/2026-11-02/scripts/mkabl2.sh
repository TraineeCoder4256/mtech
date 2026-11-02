#!/bin/bash
# Retry of the oldsurf ablation: v1.4.1 with v1.4.0's surface kernels.
# v1.4.1's importer names the torus and quadric codes, so the old file
# needs those names to exist; the three test problems never use them
# (planes, spheres and cylinders only), so no kernel for them is needed.
set -e
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
rm -rf $S/abl-oldsurf
cp -r $S/wt-head/gmc3d $S/abl-oldsurf
find $S/abl-oldsurf -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find $S/abl-oldsurf \( -name "*.nbi" -o -name "*.nbc" \) -delete 2>/dev/null || true
cp $S/wt-v140/gmc3d/core/surfaces.py $S/abl-oldsurf/core/surfaces.py
python3 - "$S/abl-oldsurf/core/surfaces.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "CYLINDER, CONE = 11, 12          # any axis (OpenMC's Cylinder, Cone)"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, old + "\n# names the v1.4.1 importer asks for; the test "
              "problems never use them\nQUADRIC = 13\n"
              "X_TORUS, Y_TORUS, Z_TORUS = 14, 15, 16")
old2 = '    Z_CONE: "z-cone", CYLINDER: "cylinder", CONE: "cone",'
assert s.count(old2) == 1
s = s.replace(old2, old2 + '\n    QUADRIC: "quadric", X_TORUS: "x-torus",'
              '\n    Y_TORUS: "y-torus", Z_TORUS: "z-torus",')
open(p, "w").write(s)
print("ABL2_READY")
PY

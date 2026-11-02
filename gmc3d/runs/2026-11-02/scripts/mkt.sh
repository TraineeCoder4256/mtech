#!/bin/bash
# Two throwaway copies of v1.6.1 to size the two structural costs in the
# scorer.  Both make the physics wrong on purpose; only the clock matters.
#   add1  : _add writes one score per bin instead of four
#   noexp : _decayed returns w * length, no exp / expm1 per voxel
set -e
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
rm -rf $S/abl-add1 $S/abl-noexp
cp -r /home/claude/mtech/gmc3d $S/abl-add1
cp -r /home/claude/mtech/gmc3d $S/abl-noexp
for d in abl-add1 abl-noexp; do
  find $S/$d -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  find $S/$d \( -name "*.nbi" -o -name "*.nbc" \) -delete 2>/dev/null || true
  rm -rf $S/$d/runs $S/$d/models 2>/dev/null || true
done
python3 - "$S" <<'PY'
import sys
S = sys.argv[1]


def patch(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, s.count(old))
    open(path, "w").write(s.replace(old, new))


patch(f"{S}/abl-add1/core/tallies.py", """    row[at] += flux
    row[at + 1] += sa * flux
    row[at + 2] += ss * flux
    row[at + 3] += (sa + ss) * flux""", """    row[at] += flux       # ABL: one score, not four""")

patch(f"{S}/abl-noexp/core/tallies.py", """    if sa * (lb - la) < 1.0e-12:
        return w * np.exp(-sa * la) * (lb - la)
    return w * np.exp(-sa * la) * (-np.expm1(-sa * (lb - la))) / sa""",
      """    return w * (lb - la)      # ABL: no exponential per voxel""")
print("T_ABL_READY")
PY

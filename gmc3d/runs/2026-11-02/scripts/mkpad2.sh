#!/bin/bash
# Two more padded copies of v1.0: the namedtuples _track is handed per
# particle, rather than the one _descend is handed per flight.
set -e
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
rm -rf $S/padphys $S/padpart
cp -r $S/wt-v100/gmc3d $S/padphys
cp -r $S/wt-v100/gmc3d $S/padpart
for d in padphys padpart; do
  find $S/$d -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  find $S/$d \( -name "*.nbi" -o -name "*.nbc" \) -delete 2>/dev/null || true
done
python3 - "$S" <<'PY'
import sys
S = sys.argv[1]


def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        if s.count(old) != 1:
            raise SystemExit(f"FAILED {path}: {s.count(old)} for {old[:40]!r}")
        s = s.replace(old, new)
    open(path, "w").write(s)


patch(f"{S}/padphys/core/transport.py", [
    ('    "mesh_cap",                       # \'cap\' rule for mesh tallies\n])',
     '    "mesh_cap",                       # \'cap\' rule for mesh tallies\n'
     '    "pad0", "pad1", "pad2", "pad3", "pad4", "pad5", "pad6",  # PAD\n])'),
    ('                 float(r_star), float(r_max or INF), '
     'mesh_rule == "cap")',
     '                 float(r_star), float(r_max or INF), '
     'mesh_rule == "cap",\n                 *[np.zeros(8) for _ in range(7)])'),
])
patch(f"{S}/padpart/core/transport.py", [
    ('    "mu", "s",                        # the backend\'s answer (s in mfp)\n])',
     '    "mu", "s",                        # the backend\'s answer (s in mfp)\n'
     '    "pad0", "pad1",                   # PAD\n])'),
    ("                     np.zeros(n, np.int64), f(), f(), f(), f())",
     "                     np.zeros(n, np.int64), f(), f(), f(), f(),\n"
     "                     f(), f())"),
])
print("PAD2_READY")
PY

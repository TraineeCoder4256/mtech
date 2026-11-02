#!/bin/bash
# interleaved rounds: round by round, every version, so drift hits all equally
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
VERS="v1.0:wt-v100 v1.1.0:wt-v110 v1.1.3:wt-v113 v1.2.0:wt-v12 v1.3.0:wt-v13 v1.4.0:wt-v140 v1.4.1:wt-head v1.5.0:wt-v150 v1.6.0:wt-v160 v1.6.1:wt-v161"
for r in 1 2 3 4 5; do
  for pair in $VERS; do
    tag=${pair%%:*}; d=${pair##*:}
    cd $S/$d/gmc3d || { echo "$tag MISSING"; continue; }
    PYTHONPATH=. /opt/mm/root/envs/gmc/bin/python -I $S/interleave.py "$tag" "$r" sphere,slab,cask 2>&1 | grep -E "^ROW|Error|Traceback"
  done
done
echo ROUNDS_DONE

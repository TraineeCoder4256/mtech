#!/bin/bash
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
for pair in base:/home/claude/mtech/gmc3d add1:$S/abl-add1 noexp:$S/abl-noexp; do
  tag=${pair%%:*}; d=${pair##*:}
  cd $d || continue
  PYTHONPATH=. /opt/mm/root/envs/gmc/bin/python -I $S/interleave.py "$tag" tabl sphere,slab,cask 2>&1 | grep -E "^ROW|Error|Traceback" | head -8
  PYTHONPATH=. /opt/mm/root/envs/gmc/bin/python -I $S/owntal.py "$tag" sphere,slab,cask 2>&1 | grep -E "^OWN|Error|Traceback" | head -8
done
echo TABL_DONE

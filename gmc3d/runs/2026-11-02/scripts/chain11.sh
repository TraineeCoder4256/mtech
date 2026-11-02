set -u
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
PY=/opt/mm/root/envs/gmc/bin/python
L=$S/phase11.log
while ! grep -q CHAIN10_DONE $L 2>/dev/null; do sleep 15; done
cd /home/claude/mtech/gmc3d && PYTHONPATH=. $PY -I $S/balltal.py sphere,slab,cask $S >> $L 2>&1
echo CHAIN11_DONE >> $L

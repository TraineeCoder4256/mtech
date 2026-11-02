set -u
S=/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad
PY=/opt/mm/root/envs/gmc/bin/python
L=$S/phase11.log
P=sphere,slab,cask
: > $L
cd /home/claude/mtech/gmc3d && PYTHONPATH=. $PY -I $S/openref.py $P $S >> $L 2>&1
for r in 1 2; do
  cd /home/claude/mtech/gmc3d && PYTHONPATH=. $PY -I $S/ballrace.py v161 $r $P profile >> $L 2>&1
  cd $S/wt-v100/gmc3d && PYTHONPATH=. $PY -I $S/ballrace.py v100 $r $P centre >> $L 2>&1
done
echo CHAIN10_DONE >> $L

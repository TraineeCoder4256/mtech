#!/bin/bash
# v1.6.1: every check behind the score, then speed against v1.6.0 (run from gmc3d/)
export PATH=/opt/mm/root/envs/gmc/bin:$PATH
D=runs/2026-10-30/v1.6.1
find . -name "*.nbi" -delete; find . -name "*.nbc" -delete
python -m pytest -q -p no:cacheprovider tests --junitxml=$D/tests.xml > $D/tests.log 2>&1
python run.py detcheck --out $D > /dev/null 2>&1
python $D/detnet.py > $D/detnet.log 2>&1
python run.py robust --out $D > /dev/null 2>&1
python run.py validate --out $D > /dev/null 2>&1
python run.py score --out $D > /dev/null 2>&1
echo "same machine, same session" > $D/speed.log
python /tmp/claude-0/-home-claude-mtech/8631efaa-684a-5889-ab2b-c9c5fe1334f5/scratchpad/speed15.py . | sed 's/^/v1.6.1 /' >> $D/speed.log 2>&1
(cd /tmp/claude-0/-home-claude-mtech/8631efaa-684a-5889-ab2b-c9c5fe1334f5/scratchpad/wt160/gmc3d && find . -name "*.nbi" -delete; find . -name "*.nbc" -delete; python /tmp/claude-0/-home-claude-mtech/8631efaa-684a-5889-ab2b-c9c5fe1334f5/scratchpad/speed15.py .) | sed 's/^/v1.6.0 /' >> $D/speed.log 2>&1
echo done > $D/all.done

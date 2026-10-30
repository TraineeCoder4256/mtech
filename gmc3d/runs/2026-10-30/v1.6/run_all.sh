#!/bin/bash
# v1.6: every check behind the score, in order (run from gmc3d/)
export PATH=/opt/mm/root/envs/gmc/bin:$PATH
D=runs/2026-10-30/v1.6
python $D/slab_corr.py data/openmc/v16 > $D/slab_corr.log 2>&1
python run.py detcheck --out $D > /dev/null 2>&1
python run.py robust --out $D > /dev/null 2>&1
echo done-stage1 > $D/stage1.done

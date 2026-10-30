#!/bin/bash
# v1.6, second pass: every check behind the score (run from gmc3d/)
export PATH=/opt/mm/root/envs/gmc/bin:$PATH
D=runs/2026-10-30/v1.6
find . -name "*.nbi" -delete; find . -name "*.nbc" -delete
python run.py detcheck --out $D > /dev/null 2>&1
python run.py robust --out $D > /dev/null 2>&1
python -m pytest -q -p no:cacheprovider tests --junitxml=$D/tests.xml > $D/tests.log 2>&1
python run.py validate --out $D > /dev/null 2>&1
python run.py score --out $D > /dev/null 2>&1
echo done > $D/stage2.done

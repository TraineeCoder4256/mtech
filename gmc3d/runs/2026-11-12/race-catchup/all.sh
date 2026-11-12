#!/bin/bash
# 12 Nov catch-up: 5 anisochecks with v2 models, then the v1 race, seeds 11-13
cd "$(dirname "$0")/../../.."
O=runs/2026-11-12/aniso-catchup; mkdir -p $O
for m in nflow_v2_seed1 cfm_v2_seed0 cfm_v2_seed1 cvae_v2_seed1 cgan_v2_seed0; do
  PYTHONPATH=. /opt/mm/root/envs/gmc/bin/python run.py anisocheck --backend $m --out $O > $O/anisocheck_$m.log 2>&1
  echo "anisocheck $m: $(grep 'cases pass' $O/anisocheck_$m.log)" >> runs/2026-11-12/race-catchup/progress.txt
done
for s in 11 12 13; do
  PYTHONPATH=. /opt/mm/root/envs/gmc/bin/python runs/2026-11-12/race-catchup/race_models.py $s > runs/2026-11-12/race-catchup/stdout_$s.txt 2>&1
  echo "race seed $s done" >> runs/2026-11-12/race-catchup/progress.txt
done
touch runs/2026-11-12/race-catchup/all.done

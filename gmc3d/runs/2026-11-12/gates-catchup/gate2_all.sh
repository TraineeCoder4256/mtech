#!/bin/bash
# gate2 with the final rule (v2.1.6+) on the v2 models that were gated only
# under the older gate code (m2.1 runs of 10 Nov): nflow, cfm seeds 0 and 1,
# cvae seed 1.  cvae_v2_seed0 (11 Nov) and cgan (12 Nov) are already done.
cd "$(dirname "$0")/../../.."
P=/opt/mm/root/envs/gmc/bin/python
H=runs/2026-11-12/gates-catchup
for m in nflow_v2_seed0 nflow_v2_seed1 cfm_v2_seed0 cfm_v2_seed1 cvae_v2_seed1; do
  $P run.py gate2 $m --out $H > $H/gate2_$m.log 2>&1
done
echo FINISHED

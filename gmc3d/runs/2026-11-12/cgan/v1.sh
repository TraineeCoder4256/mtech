#!/bin/bash
# cGAN at interface v1: seeds 0 and 1 (8,000 steps each), then every
# family side by side (run.py compare, the m1.x comparison).
set -e
cd "$(dirname "$0")/../../.."
P=/opt/mm/root/envs/gmc/bin/python
H=runs/2026-11-12/cgan
for s in 0 1; do $P run.py train --family cgan --seed $s --out $H > $H/train_cgan_seed$s.log 2>&1; done
$P run.py compare --backends nflow nflow_seed1 cfm cfm_seed1 cvae cvae_seed1 cgan cgan_seed1 oracle table --out $H > $H/compare_v1.log 2>&1
echo FINISHED

#!/bin/bash
# Phani chose 1 seed (12 Nov 09:58 UTC): stop the runner once seed 11 is done
cd "$(dirname "$0")"
until grep -q "race seed 11 done" progress.txt 2>/dev/null; do sleep 20; done
pkill -f race-catchup/all.sh; sleep 1; pkill -f "race-catchup/race_models.py 12"
echo "stopped after seed 11 (Phani: 1 seed)" >> progress.txt
touch all.done

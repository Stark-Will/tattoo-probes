#!/bin/bash
cd /workspace/tatany-paper/tattoo-probes; . ../venv/bin/activate
while pgrep -f "score_rewards.py data/manifest_ir.csv" >/dev/null; do sleep 30; done
python scripts/score_rewards.py data/manifest_hpd_ts.csv results/rewards_hpd.csv > logs/rewards_hpd.log 2>&1
python scripts/score_rewards.py data/manifest_degr60.csv results/rewards_degr.csv > logs/rewards_degr.log 2>&1
echo QUEUE_DONE >> logs/rewards_degr.log

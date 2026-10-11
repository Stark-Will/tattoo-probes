#!/bin/bash
cd /workspace/tatany-paper/tattoo-probes; . ../venv/bin/activate
python scripts/classify.py data/manifest_pap.csv data/cls_pap.csv > logs/pap_classify.log 2>&1
python scripts/run_probes.py data/manifest_pap.csv results/probes_pap.csv > logs/pap_probes.log 2>&1
python scripts/score_rewards.py data/manifest_pap.csv results/rewards_pap.csv > logs/rewards_pap.log 2>&1
echo PAP_DONE >> logs/rewards_pap.log

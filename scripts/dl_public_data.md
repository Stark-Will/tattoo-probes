# Downloading public data (not redistributed)

This repo ships indexes under `data/` (`manifest_*.csv`, `pairs_*.csv`, `cls_*.csv`, …).
Images must be fetched from the original releases into `data/raw/` (gitignored).

## HPDv2 (Apache-2.0)
- Upstream: Hugging Face HPDv2 / HPSv2 release used by Wu et al.
- Use the IDs / relative paths in `data/manifest_hpd.csv` and `data/hpd_tattoo_paths.txt`.
- Example (needs `huggingface_hub` and acceptance of the dataset card):

```bash
# illustrative — adjust repo id / split to the card you use
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="yuvalkirstain/HPDv2", repo_type="dataset", local_dir="data/raw/hpdv2")
PY
```

Then point `scripts/run_probes.py` / existing analysis scripts at the extracted files matching `manifest_hpd.csv`.

## ImageRewardDB (Apache-2.0)
- Upstream: ImageReward dataset release (Xu et al.).
- Indexes: `data/manifest_ir.csv`, `data/ir_tattoo.csv`, `data/cls_ir.csv`.

## Pick-a-Pic v2 (licence unverified in this work)
- **Do not** commit or re-upload images.
- Metadata only: `data/manifest_pap.csv`, `data/pairs_pap*.csv`, `data/pap_tattoo_pairs.csv`.
- Original HF card was removed; treat redistribution as unsafe until the licence is clear.

## Drozdik/tattoo_v0 (CC BY-NC-SA 4.0, research only)
- **Do not** redistribite images from this repo or mirrors of this work.
- Manifests of IDs used for defect / scale-spread runs: `data/manifest_degr.csv`, `data/manifest_degr60.csv`.
- Download the upstream set yourself for non-commercial research, then regenerate degradations with `scripts/make_degradations.py` if needed.

## What you can use without re-download
- All CSVs in `results/` (probe scores, reward scores, agreement, fusion, scale/spread summaries).
- Paper figures in `figs/` and `paper/figures/` (author-generated plots + author-owned illustrative panel).

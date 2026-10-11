# Data licences and provenance (verified 2026-10-11, Asia/Shanghai)

This repository redistributes **indexes, pair tables, classification labels and derived probe/reward scores only**.
It does **not** redistribute third-party images.

| Dataset | Source | Licence (where verified) | Redistributed here | Use in paper |
|---|---|---|---|---|
| **HPDv2** | HF `yuvalkirstain/HPDv2` / related HPSv2 release | **Apache-2.0** | Prompt/ID indexes, pair tables, classification CSVs, probe & reward score CSVs in `results/` | Main preference vs probe comparison (tattoo-subject subset) |
| **ImageRewardDB** | ImageReward public release | **Apache-2.0** | Same class of indexes/scores | Secondary preference comparison (small tattoo-subject *n*) |
| **Pick-a-Pic v2** | Public mirrors (original HF card removed; mirrors state MIT — **unverified**) | Unverified | Prompt/pair **metadata only**; **no images** | Exploratory; licence caveat stated in paper |
| **Drozdik/tattoo_v0** | Public tattoo-design set | **CC BY-NC-SA 4.0** | **Nothing image-like**; only manifests listing IDs used for the controlled-defect and scale/spread stress tests | Research-only defect + stress-test source; download yourself |

## Obligations
- **HPDv2 / ImageRewardDB (Apache-2.0)**: attribution; cite the papers. Images must be fetched from the original releases (`scripts/dl_public_data.md`).
- **Drozdik/tattoo_v0 (CC BY-NC-SA 4.0)**: attribution + share-alike + **non-commercial**. This repo does not ship its images. Derived defect/stress statistics in `results/` are for non-commercial research under the same spirit; do not use them commercially without clearing the upstream licence.
- **Pick-a-Pic**: do not redistribute images from this work. Treat the upstream licence as unresolved until the original card is restored.
- No tatany.app generations, models or user data were used in any scored experiment (see paper competing-interest statement). The illustrative figure in the paper is author-owned marketing material, not scored.

## Re-download
See [`scripts/dl_public_data.md`](../scripts/dl_public_data.md).

# Phase 1 summary (2026-10-10, Asia/Shanghai). All numbers copied from CSVs in results/. Seed 0. Env: ../venv, requirements.lock.txt (torch 2.14.1+cpu, open_clip 3.3.0, transformers 4.46.3, image-reward 1.5).

## Data
- HPDv2 train, prompts matching "tattoo": 2,010 pairs / 327 prompts / 1,340 images (stream-extracted from 31.7 GB tar, 23 min, ~60 MB on disk).
- ImageRewardDB train+val tattoo prompts: 370 rows, 47 images are 0-byte in the upstream zips -> 323 usable images, 809 within-prompt rank pairs / 51 prompts.
- Pick-a-Pic v2 (liuhuohuo2 parquet mirror, 672 shards, metadata only): 2,608 tattoo rows, 1,687 labeled & different pairs, 185 prompts. Images not fetched. License unverified (original card removed; mirrors say MIT).
- Drozdik/tattoo_v0 (CC-BY-NC-SA-4.0, research only): 120 designs sampled; 6 defect types x 2 levels = 1,560 images.
- Classification (keyword + CLIP ViT-B/32 zero-shot): HPDv2 design 334 / on_skin 87 / incidental 674 / other 146 / ambiguous 99; ImageRewardDB design 48 / on_skin 12 / incidental 238 / other 18 / ambiguous 7. "tattoo_subject" = design+on_skin -> HPDv2 463 pairs / 119 prompts; ImageRewardDB 148 pairs / 9 prompts. Visual spot-check of 24 random "design" images: HPDv2 ~21/24 tattoo-like, ImageRewardDB 24/24 (figs/spot_*.jpg).

## Human agreement (pairwise accuracy, 95% prompt-cluster bootstrap CI, results/human_agreement.csv)
Rewards on HPDv2 were scored only on the tattoo_subject images, so their "all" row equals the tattoo_subject row.
| scorer | HPDv2 tattoo_subject (463) | IRDB all (809) | IRDB tattoo_subject (148) |
|---|---|---|---|
| probe_composite@5cm | 0.305 [0.256,0.353] | 0.472 [0.426,0.518] | 0.486 [0.375,0.584] |
| edge_transition_mm@5 | 0.454 | 0.539 | 0.507 |
| CLIPScore | 0.685 | 0.617 | 0.554 |
| PickScore | 0.724 | 0.624 | 0.547 |
| HPSv2.1 | 0.790 [0.748,0.831] | 0.640 | 0.493 |
| ImageReward | 0.683 | 0.726 [0.670,0.776] | 0.615 [0.438,0.757] |
Caveats: HPSv2.1 was trained on HPDv2 train and ImageReward on ImageRewardDB -> in-distribution leakage. Probe composite is significantly BELOW 0.5 on HPDv2: annotators prefer more detailed/shaded images, i.e. the opposite of the tattooability heuristics. IRDB tattoo_subject has only 9 prompts (wide CIs).

## Controlled-defect detection (P(clean scored better), results/degr_detection_withrewards.csv, n=30 designs per cell; probe-only run on n=120 in degr_detection.csv)
- Probe composite@5: haze 1.00/1.00, speckle 1.00/1.00, but blur 0.27/0.23, thicken 0.00/0.00, detail_loss 0.53/0.20, thin 0.57/0.43.
- Single probes cover complementary defects: edge_transition_mm (added post hoc, v0.2) blur 0.97/0.97, detail_loss 1.00/0.97, thicken 0.90/1.00; midtone_frac haze 1.00, speckle 1.00.
- HPSv2.1: >=0.80 on all 12 cells except none (min 0.80 blur1). PickScore weak on blur1 (0.53). CLIPScore fails on speckle (0.20/0.33). ImageReward weak on speckle2 (0.47).
- Interpretation: the unweighted composite is not a usable score; "thicken"/"blur" raise line width, which the needle-width probe rewards by design.

## Scale x ink-spread stress test (results/scale_spread.csv, 120 designs)
Mean skeleton F1 at spread 0.3 mm: 3 cm 0.562, 5 cm 0.617, 10 cm 0.688, 15 cm 0.718; at 0.5 mm: 0.467 / 0.483 / 0.532 / 0.564. Gap survival at 0.3 mm: 0.891 / 0.914 / 0.945 / 0.963.
Predictors at 5 cm, spread 0.3 mm (Spearman, n=120): frac_tight_gap vs gap_survival rho=-0.813 (partly circular by construction); edge_density vs gap_survival -0.588; edge_transition_mm vs skel_f1 -0.414; line_w_p10_mm vs skel_f1 +0.284.
Caveat: spread model is an isotropic Gaussian assumption, not validated on healed skin; skel F1 at spread 0 drops with size (0.985 -> 0.931), a resampling artifact to fix.

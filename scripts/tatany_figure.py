"""Illustrative figure only (NOT an evaluation): probes at 5 cm + ink-spread preview on tatany.app product case images.
Source images: copies of the user's local tatany repo public assets, cropped; see paper/DRAFT_NOTES.md for provenance."""
import sys, glob, os, pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, '.'); from tattooprobes import compute_probes, simulate_on_skin, retention
SHOW = ['line_w_p10_mm', 'frac_sub_needle', 'frac_tight_gap', 'midtone_frac', 'open_ends_per_cm', 'edge_transition_mm']
fs = sorted(glob.glob('../paper/figs/tatany/*.png')); rows = []
fig, ax = plt.subplots(len(fs), 3, figsize=(8.5, 2.1 * len(fs)), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
for i, f in enumerate(fs):
    im = Image.open(f).convert('RGB'); p = compute_probes(im, 5.0); r = retention(im, 5.0, 0.2)
    rows.append(dict(image=os.path.basename(f), **p, **{f'ret_{k}': v for k, v in r.items()}))
    ax[i, 0].imshow(im); ax[i, 0].set_title(os.path.basename(f)[:-4], fontsize=7)
    ax[i, 1].imshow(simulate_on_skin(im, 5.0, 0.2), cmap='gray'); ax[i, 1].set_title('5 cm, spread 0.20 mm (sim.)', fontsize=7)
    ax[i, 2].axis('off'); ax[i, 2].text(0, 0.95, '\n'.join([f'{k}: {p[k]:.3f}' for k in SHOW] + [f'skeleton F1 @0.20 mm: {r["skel_f1"]:.3f}', f'gap survival @0.20 mm: {r["gap_survival"]:.3f}']), fontsize=7, va='top', family='monospace')
    for a in ax[i, :2]: a.axis('off')
fig.tight_layout(); fig.savefig('../paper/figures/tatany_illustrative.png', dpi=160)
pd.DataFrame(rows).to_csv('results/tatany_illustrative.csv', index=False); print(pd.DataFrame(rows)[['image'] + SHOW].round(3).to_string())

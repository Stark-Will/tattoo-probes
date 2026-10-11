import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
d = pd.read_csv('results/scale_spread.csv')
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for s, g in d.groupby('size_cm'):
    m = g.groupby('spread_mm')[['skel_f1', 'gap_survival']].mean()
    ax[0].plot(m.index, m.skel_f1, marker='o', label=f'{s} cm'); ax[1].plot(m.index, m.gap_survival, marker='o', label=f'{s} cm')
ax[0].set(xlabel='simulated ink spread sigma (mm)', ylabel='skeleton F1 vs. design', title='Line structure retention')
ax[1].set(xlabel='simulated ink spread sigma (mm)', ylabel='background-gap survival', title='Negative-space retention'); ax[0].legend()
fig.tight_layout(); fig.savefig('figs/scale_spread.png', dpi=150)
f = 'results/degr_detection_withrewards.csv' if os.path.exists('results/degr_detection_withrewards.csv') else 'results/degr_detection.csv'
r = pd.read_csv(f); keep = [c for c in r.scorer.unique() if '@10' not in c]
pv = r[r.scorer.isin(keep)].pivot_table(index='scorer', columns=['op', 'level'], values='acc')
fig, ax = plt.subplots(figsize=(11, 0.35 * len(pv) + 1.5)); im = ax.imshow(pv.values, cmap='RdBu', vmin=0, vmax=1, aspect='auto')
ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index, fontsize=7); ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels([f'{a}{b}' for a, b in pv.columns], rotation=45, fontsize=7)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]): ax.text(j, i, f'{pv.values[i, j]:.2f}', ha='center', va='center', fontsize=6)
fig.colorbar(im, label='P(clean scored better)'); ax.set_title(f'Controlled-defect detection ({os.path.basename(f)})'); fig.tight_layout(); fig.savefig('figs/degr_detection.png', dpi=150)
print('ok')
# radar: per-dimension agreement with human preference (0.5 = chance)
import numpy as np
pdim = pd.read_csv('results/per_dimension.csv')
dims = [s for s in pdim.scorer.unique()]
fig = plt.figure(figsize=(6.5, 6.5)); ax = fig.add_subplot(111, polar=True)
ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist(); ang += ang[:1]
for ds, lab in [('hpd', 'HPDv2 tattoo-subject'), ('ir', 'ImageRewardDB tattoo-subject'), ('pap', 'Pick-a-Pic tattoo-subject')]:
    g = pdim[pdim.dataset == ds].set_index('scorer').reindex(dims).acc.values.tolist()
    if not np.isfinite(g).any(): continue
    ax.plot(ang, g + g[:1], marker='o', ms=3, label=lab)
ax.plot(ang, [0.5] * len(ang), 'k--', lw=0.8, label='chance')
ax.set_xticks(ang[:-1]); ax.set_xticklabels([d.replace('@5', '') + ('*' if d.startswith('edge_transition') else '') for d in dims], fontsize=7); ax.set_ylim(0.2, 0.85)
ax.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), fontsize=7); ax.set_title('Pairwise agreement with human preference (* post hoc)', fontsize=9)
fig.tight_layout(); fig.savefig('figs/radar_per_dimension.png', dpi=150)

import pandas as pd, numpy as np, sys, json
sys.path.insert(0, 'scripts'); from analysis_lib import *
m = pd.read_csv('data/manifest_degr.csv'); p = pd.read_csv('results/probes_degr.csv').set_index('id')
rw = pd.read_csv('results/rewards_degr.csv').set_index('id') if len(sys.argv) > 1 else None
scores = {}
for s in (5, 10):
    scores[f'probe_composite@{s}'] = composite(p, s)
    for k, sg in PROBE_SIGNS.items(): scores[f'{k}@{s}'] = sg * p[f'{k}@{s}']
if rw is not None:
    for c in rw.columns: scores[c] = rw[c]
rows = []
for (op, lv), g in m[m.op != 'clean'].groupby(['op', 'level']):
    pairs = pd.DataFrame({'win': g.src.astype(str) + '_clean', 'lose': g.id, 'src': g.src})
    if rw is not None: pairs = pairs[pairs.win.isin(rw.index) & pairs.lose.isin(rw.index)]
    for name, sc in scores.items():
        acc, n = pair_acc(pairs, sc); rows.append(dict(op=op, level=lv, scorer=name, acc=acc, n=n))
r = pd.DataFrame(rows); tag = '_withrewards' if rw is not None else ''
r.to_csv(f'results/degr_detection{tag}.csv', index=False)
keep = [k for k in scores if k.startswith('probe_composite') or '@' not in k] + ['frac_sub_needle@5', 'frac_tight_gap@5', 'midtone_frac@5', 'specks_per_cm2@5', 'line_w_p10_mm@5']
print(r[r.scorer.isin(keep)].pivot_table(index=['op', 'level'], columns='scorer', values='acc').round(2).to_string())
print('n per cell', r.n.min(), r.n.max())

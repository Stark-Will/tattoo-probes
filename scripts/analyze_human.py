"""Agreement of probes / reward models with existing human labels (HPDv2 pairs, ImageRewardDB within-prompt ranks)."""
import pandas as pd, numpy as np, sys, os, itertools
sys.path.insert(0, 'scripts'); from analysis_lib import *
def scorers(probe_csv, reward_csv):
    p = pd.read_csv(probe_csv).set_index('id'); S = {}
    for s in (5, 10): S[f'probe_composite@{s}'] = composite(p, s)
    for k, sg in PROBE_SIGNS.items(): S[f'{k}@5'] = sg * p[f'{k}@5']
    if os.path.exists(reward_csv):
        r = pd.read_csv(reward_csv).set_index('id')
        for c in r.columns: S[c] = r[c]
    return S
out = []
# HPDv2
cls = pd.read_csv('data/cls_hpd.csv').set_index('id').cls; ts = cls.isin(['design', 'on_skin'])
P = pd.read_csv('data/pairs_hpd.csv'); P['both_ts'] = P.win.map(ts) & P.lose.map(ts)
S = scorers('results/probes_hpd.csv', 'results/rewards_hpd.csv')
for subset, PP in [('all_tattoo_prompts', P), ('tattoo_subject', P[P.both_ts])]:
    for n, sc in S.items():
        a, k = pair_acc(PP, sc)
        if k == 0: continue
        lo, hi = boot_ci(PP[PP.win.isin(sc.dropna().index) & PP.lose.isin(sc.dropna().index)], sc, n=2000, group='prompt')
        out.append(dict(dataset='HPDv2', subset=subset, scorer=n, acc=a, ci_lo=lo, ci_hi=hi, n_pairs=k, n_prompts=PP.prompt.nunique()))
# ImageRewardDB: within-prompt pairs from 'rank' (lower rank = better)
ir = pd.read_csv('data/cls_ir.csv'); meta = pd.read_csv('data/ir_tattoo.csv'); meta['id'] = 'ir_' + meta.image_path.str.split('/').str[-1]
ir = ir.merge(meta[['id', 'rank', 'overall_rating', 'prompt_id']], on='id')
pairs = []
for pid, g in ir.groupby('prompt_id'):
    for a, b in itertools.combinations(g.itertuples(), 2):
        if a.rank == b.rank: continue
        w, l = (a, b) if a.rank < b.rank else (b, a)
        pairs.append(dict(win=w.id, lose=l.id, prompt=pid, both_ts=w.cls in ('design', 'on_skin') and l.cls in ('design', 'on_skin')))
P = pd.DataFrame(pairs); S = scorers('results/probes_ir.csv', 'results/rewards_ir.csv')
for subset, PP in [('all_tattoo_prompts', P), ('tattoo_subject', P[P.both_ts])]:
    for n, sc in S.items():
        a, k = pair_acc(PP, sc)
        if k == 0: continue
        lo, hi = boot_ci(PP[PP.win.isin(sc.dropna().index) & PP.lose.isin(sc.dropna().index)], sc, n=2000, group='prompt')
        out.append(dict(dataset='ImageRewardDB', subset=subset, scorer=n, acc=a, ci_lo=lo, ci_hi=hi, n_pairs=k, n_prompts=PP.prompt.nunique()))
o = pd.DataFrame(out); o.to_csv('results/human_agreement.csv', index=False)
print(o.round(3).to_string())

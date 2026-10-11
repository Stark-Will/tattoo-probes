"""Phase 2 analyses: per-dimension agreement table, prompt-held-out fusion, using manual-reviewed classes (cls_v2).
Pre-registered dimensions (Phase 1, a priori signs): line_w_p10_mm, frac_sub_needle, open_ends_per_cm, frac_tight_gap,
midtone_frac, border_std, edge_density_per_mm2, specks_per_cm2. POST HOC: edge_transition_mm (added after first degradation run)."""
import pandas as pd, numpy as np, sys, os, itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, 'scripts'); from analysis_lib import PROBE_SIGNS, pair_acc, boot_ci
SEED = 0; REW = ['clipscore', 'pickscore', 'hpsv21', 'imagereward']
PROBES = [f'{k}@5' for k in PROBE_SIGNS]
def load(ds):
    p = pd.read_csv(f'results/probes_{ds}.csv').set_index('id')
    rf = [f for f in [f'results/rewards_{ds}.csv', f'results/rewards_{ds}_extra.csv'] if os.path.exists(f)]
    r = pd.concat([pd.read_csv(f) for f in rf]).drop_duplicates('id').set_index('id') if rf else pd.DataFrame(index=p.index)
    return p.join(r, how='left')
def pairs_for(ds):
    if ds == 'hpd':
        c = pd.read_csv('data/cls_hpd.csv').set_index('id').cls_v2; ts = c.isin(['design', 'on_skin'])
        P = pd.read_csv('data/pairs_hpd.csv'); return P[P.win.map(ts) & P.lose.map(ts)].reset_index(drop=True)
    if ds == 'ir':
        ir = pd.read_csv('data/cls_ir.csv'); meta = pd.read_csv('data/ir_tattoo.csv'); meta['id'] = 'ir_' + meta.image_path.str.split('/').str[-1]
        ir = ir.merge(meta[['id', 'rank', 'prompt_id']], on='id'); ir = ir[ir.cls_v2.isin(['design', 'on_skin'])]; out = []
        for pid, g in ir.groupby('prompt_id'):
            for a, b in itertools.combinations(g.itertuples(), 2):
                if a.rank != b.rank: w, l = (a, b) if a.rank < b.rank else (b, a); out.append(dict(win=w.id, lose=l.id, prompt=pid))
        return pd.DataFrame(out)
    if ds == 'pap':
        c = pd.read_csv('data/cls_pap.csv').set_index('id').cls; ts = c.isin(['design', 'on_skin'])
        P = pd.read_csv('data/pairs_pap_all.csv'); P = P[P.win.map(ts) & P.lose.map(ts)].reset_index(drop=True); P.to_csv('data/pairs_pap.csv', index=False); return P
def per_dim(ds, F, P):
    rows = []
    for k in PROBES + [r for r in REW if r in F]:
        sc = F[k] * (PROBE_SIGNS.get(k.split('@')[0], 1))
        a, n = pair_acc(P, sc)
        if n == 0: continue
        lo, hi, dist = boot_ci(P, sc, n=2000, group='prompt', return_dist=True)
        p_boot = min(1.0, 2 * min((dist <= 0.5).mean(), (dist >= 0.5).mean()) + 1 / len(dist))
        h = 2 * np.arcsin(np.sqrt(a)) - 2 * np.arcsin(np.sqrt(0.5))
        rows.append(dict(dataset=ds, scorer=k, effect=a - 0.5, cohens_h=h, p_boot=p_boot, kind='probe' if '@' in k else 'reward', post_hoc=k.startswith('edge_transition'),
                         acc=a, ci_lo=lo, ci_hi=hi, n_pairs=n, n_prompts=P.prompt.nunique()))
    return rows
def fusion(ds, F, P, feats, name):
    P = P[P.win.isin(F.index) & P.lose.isin(F.index)].reset_index(drop=True)
    X = F.loc[P.win, feats].values - F.loc[P.lose, feats].values; ok = ~np.isnan(X).any(1); P, X = P[ok].reset_index(drop=True), X[ok]
    rng = np.random.default_rng(SEED); flip = rng.random(len(X)) < 0.5; Xs = np.where(flip[:, None], -X, X); y = (~flip).astype(int)
    pred = np.zeros(len(y)); g = P.prompt.astype(str).values; k = min(5, len(set(g)))
    for tr, te in GroupKFold(k).split(Xs, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(Xs[tr], y[tr]); pred[te] = m.predict_proba(Xs[te])[:, 1]
    correct = ((pred > 0.5) == y).astype(float)
    acc = correct.mean(); keys, inv = np.unique(g, return_inverse=True); s = np.bincount(inv, correct); c = np.bincount(inv)
    bs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(keys), len(keys)) for _ in range(2000))]
    coef = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(Xs, y)[-1].coef_[0]
    return dict(dataset=ds, model=name, cv='GroupKFold(prompt,k=%d)' % k, acc=acc, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n_pairs=len(y), n_prompts=len(keys),
                coefs=';'.join(f'{f}={w:+.2f}' for f, w in zip(feats, coef)))
if __name__ == '__main__':
    dims, fus = [], []
    for ds in ['hpd', 'ir', 'pap']:
        if ds == 'pap' and not os.path.exists('results/rewards_pap.csv'): continue
        F, P = load(ds), pairs_for(ds)
        dims += per_dim(ds, F, P)
        rew = [r for r in REW if r in F and F[r].notna().any()]
        for name, feats in [('probes_preregistered', PROBES[:-1] if PROBES[-1].startswith('edge_transition') else [p for p in PROBES if not p.startswith('edge_transition')]),
                            ('probes_all', PROBES), ('rewards', rew), ('rewards+probes', rew + PROBES)]:
            if feats: fus.append(fusion(ds, F, P, feats, name))
    D = pd.DataFrame(dims)
    for ds, g in D.groupby('dataset'):  # Holm-Bonferroni within dataset over all scorers
        order = g.p_boot.sort_values().index; m = len(order); run = 0
        for i, ix in enumerate(order): run = max(run, min(1.0, (m - i) * D.at[ix, 'p_boot'])); D.at[ix, 'p_holm'] = run
    D['sig_holm_05'] = D.p_holm < 0.05; dims = D.to_dict('records')
    pd.DataFrame(dims).to_csv('results/per_dimension.csv', index=False); pd.DataFrame(fus).to_csv('results/fusion_lopo.csv', index=False)
    print(pd.DataFrame(dims).round(3).to_string()); print(pd.DataFrame(fus).drop(columns='coefs').round(3).to_string())

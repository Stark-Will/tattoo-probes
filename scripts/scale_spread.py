"""Print-scale x ink-spread stress test on clean tattoo_v0 designs (seed 0). Spread values are assumptions, not dermal measurements."""
import sys, pandas as pd, numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy.stats import spearmanr
sys.path.insert(0, '.')
from tattooprobes import retention, compute_probes
SIZES = [3, 5, 10, 15]; SPREADS = [round(0.05 * i, 2) for i in range(13)]; NOISES = [0.0, 4.0]
def go(r):
    im = Image.open(r[1]).convert('RGB'); out = []
    for s in SIZES:
        pr = compute_probes(im, s)
        for nz in NOISES:
            for sp in SPREADS:
                d = dict(id=r[0], size_cm=s, spread_mm=sp, noise=nz, **retention(im, s, sp, seed=0, noise=nz)); d.update({f'probe_{k}': v for k, v in pr.items()}); out.append(d)
    return out
if __name__ == '__main__':
    m = pd.read_csv('data/manifest_degr.csv'); m = m[m.op == 'clean']
    with Pool(6) as p: rows = sum(p.map(go, list(zip(m.id, m.path))), [])
    d = pd.DataFrame(rows); d.to_csv('results/scale_spread.csv', index=False)
    print(d[d.noise == 0].groupby(['size_cm', 'spread_mm'])[['skel_f1', 'gap_survival']].mean().unstack().round(3).to_string())
    # can probes measured at the design stage predict which designs survive spread=0.3mm?
    rows = []
    for s in SIZES:
        g = d[(d.size_cm == s) & (d.spread_mm == 0.3) & (d.noise == 0)]
        for k in [c for c in d.columns if c.startswith('probe_')]:
            for tgt in ('skel_f1', 'gap_survival'):
                rho, pv = spearmanr(g[k], g[tgt]); rows.append(dict(size_cm=s, probe=k, target=tgt, rho=rho, p=pv, n=len(g)))
    c = pd.DataFrame(rows); c.to_csv('results/scale_spread_predict.csv', index=False)
    print(c[c.size_cm == 5].round(3).to_string())

import pandas as pd, numpy as np
d = pd.read_csv('results/scale_spread.csv'); d = d[d.noise == 0]
m = d.groupby(['size_cm', 'spread_mm']).skel_f1.mean().reset_index()
rows = []
for sp, g in m.groupby('spread_mm'):
    for thr in (0.7, 0.8):
        ok = g[g.skel_f1 >= thr].size_cm
        rows.append(dict(spread_mm=sp, threshold=thr, min_size_cm=ok.min() if len(ok) else np.nan))
r = pd.DataFrame(rows); r.to_csv('results/spread_sensitivity.csv', index=False); print(r.pivot(index='spread_mm', columns='threshold', values='min_size_cm'))

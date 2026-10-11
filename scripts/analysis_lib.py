import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
from tattooprobes import PROBE_SIGNS
RNG_SEED = 0
def composite(df, size):
    """Unweighted mean of signed robust z-scores (median/IQR over the pool). Signs fixed a priori."""
    zs = []
    for k, s in PROBE_SIGNS.items():
        x = df[f'{k}@{size:g}']; iqr = x.quantile(.75) - x.quantile(.25)
        zs.append(s * (x - x.median()) / (iqr if iqr > 0 else (x.std() or 1)))
    return pd.concat(zs, axis=1).mean(1)
def pair_acc(pairs, score):
    """pairs: DataFrame with win, lose ids; score: Series indexed by id. ties count 0.5"""
    a = score.reindex(pairs.win).values; b = score.reindex(pairs.lose).values
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    return float(((a > b) + 0.5 * (a == b)).mean()), int(ok.sum())
def boot_ci(pairs, score, n=1000, group=None, seed=RNG_SEED, return_dist=False):
    """Cluster bootstrap over prompts (group) of pairwise accuracy; vectorised."""
    rng = np.random.default_rng(seed)
    a = score.reindex(pairs.win).values; b = score.reindex(pairs.lose).values
    ok = ~(np.isnan(a) | np.isnan(b)); c = ((a > b) + 0.5 * (a == b))[ok]
    g = pairs[group].values[ok] if group else np.arange(ok.sum())
    keys, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=c, minlength=len(keys)); cnts = np.bincount(inv, minlength=len(keys))
    accs = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys)); accs.append(sums[pick].sum() / max(cnts[pick].sum(), 1))
    if return_dist: return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)), np.array(accs)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))

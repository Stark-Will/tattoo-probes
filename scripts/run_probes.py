"""Compute probes for a manifest at given physical sizes. usage: run_probes.py manifest out.csv [sizes]"""
import sys, pandas as pd
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, '.')
from tattooprobes import compute_probes
SIZES = [float(s) for s in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['5', '10'])]
def go(r):
    im = Image.open(r[1]).convert('RGB'); d = {'id': r[0]}
    for s in SIZES:
        for k, v in compute_probes(im, s).items(): d[f'{k}@{s:g}'] = v
    return d
if __name__ == '__main__':
    m = pd.read_csv(sys.argv[1])
    with Pool(6) as p: rows = p.map(go, list(zip(m.id, m.path)), chunksize=4)
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False); print(len(rows))

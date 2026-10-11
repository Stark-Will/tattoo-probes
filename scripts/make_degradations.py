"""Controlled defects on Drozdik/tattoo_v0 designs (CC-BY-NC-SA-4.0; research use only, not redistributed)."""
import pandas as pd, numpy as np, cv2, io, os
from PIL import Image
SEED = 0; N = 120
rng = np.random.default_rng(SEED)
d = pd.read_parquet('data/raw/tattoo_v0/0.parquet')
idx = rng.choice(len(d), N, replace=False)
os.makedirs('data/degr', exist_ok=True); rows = []
def thin(g, k): return cv2.dilate(g, np.ones((k, k), np.uint8))          # ink is dark -> dilating gray erodes ink
def thicken(g, k): return cv2.erode(g, np.ones((k, k), np.uint8))        # closes gaps / blobs lines
def haze(g, a): return np.clip(g * (1 - a) + 160 * a + rng.normal(0, 8, g.shape), 0, 255)
def speckle(g, p):
    g = g.copy(); m = rng.random(g.shape) < p; g[m] = 0; return g
def blur(g, s): return cv2.GaussianBlur(g, (0, 0), s)
def detail_loss(g, f):
    h, w = g.shape; return cv2.resize(cv2.resize(g, (w // f, h // f), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_NEAREST)
OPS = {'thin': (thin, [3, 5]), 'thicken': (thicken, [3, 5]), 'haze': (haze, [0.3, 0.5]),
       'speckle': (speckle, [0.003, 0.01]), 'blur': (blur, [1.5, 3.0]), 'detail_loss': (detail_loss, [4, 8])}
for i in idx:
    im = Image.open(io.BytesIO(d.image[i]['bytes'])).convert('L'); g = np.asarray(im).astype(np.float32)
    prompt = f"{d.text[i]}, tattoo design"
    p = f'data/degr/{i}_clean.png'; Image.fromarray(g.astype(np.uint8)).save(p)
    rows.append(dict(id=f'{i}_clean', src=i, op='clean', level=0, path=p, prompt=prompt))
    for op, (fn, lv) in OPS.items():
        for L, a in enumerate(lv, 1):
            o = np.clip(fn(g, a), 0, 255).astype(np.uint8); p = f'data/degr/{i}_{op}{L}.png'; Image.fromarray(o).save(p)
            rows.append(dict(id=f'{i}_{op}{L}', src=i, op=op, level=L, path=p, prompt=prompt))
pd.DataFrame(rows).to_csv('data/manifest_degr.csv', index=False); print(len(rows))

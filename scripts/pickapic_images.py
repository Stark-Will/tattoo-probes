"""Fetch images only for Pick-a-Pic v2 tattoo pairs whose prompt matches the design keyword rule (row-group range reads)."""
import pyarrow.parquet as pq, fsspec, pandas as pd, sys, os, re
from multiprocessing import Pool
sys.path.insert(0, 'scripts'); from classify import DESIGN_KW
BASE = 'https://huggingface.co/api/datasets/liuhuohuo2/pick-a-pic-v2/parquet/default/'
os.makedirs('data/raw/pap', exist_ok=True)
def go(args):
    shard, uids = args
    try:
        with fsspec.filesystem('https', skip_instance_cache=True).open(BASE + shard, block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh); got = 0
            for rg in range(pf.num_row_groups):
                c = pf.read_row_group(rg, columns=['image_0_uid', 'image_1_uid']).to_pandas()
                if not (c.image_0_uid.isin(uids) | c.image_1_uid.isin(uids)).any(): continue
                t = pf.read_row_group(rg, columns=['image_0_uid', 'image_1_uid', 'jpg_0', 'jpg_1']).to_pandas()
                for _, r in t.iterrows():
                    for k in (0, 1):
                        u = r[f'image_{k}_uid']
                        if u in uids and not os.path.exists(f'data/raw/pap/{u}.jpg'):
                            open(f'data/raw/pap/{u}.jpg', 'wb').write(r[f'jpg_{k}']); got += 1
        return got
    except Exception as e:
        print('FAIL', shard, e, flush=True); return 0
if __name__ == '__main__':
    d = pd.read_csv('data/pap_tattoo_pairs.csv'); d = d[d.has_label.astype(bool) & d.are_different.astype(bool) & d.caption.str.contains(DESIGN_KW)]
    jobs = [(s, set(g.image_0_uid) | set(g.image_1_uid)) for s, g in d.groupby('shard')]
    print('pairs', len(d), 'shards', len(jobs), flush=True); tot = 0
    with Pool(6) as p:
        for i, n in enumerate(p.imap_unordered(go, jobs)):
            tot += n
            if i % 10 == 0: print(i, tot, flush=True)
    print('DONE', tot)

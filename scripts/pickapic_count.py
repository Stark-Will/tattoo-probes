# Count tattoo pairs in Pick-a-Pic v2 (liuhuohuo2 parquet mirror), reading only text/label columns via HTTP range reads.
import pyarrow.parquet as pq, fsspec, json, re, sys, pandas as pd
from multiprocessing import Pool
cols=['caption','label_0','label_1','has_label','are_different','image_0_uid','image_1_uid','model_0','model_1','ranking_id','user_id','image_0_url','image_1_url']
r=re.compile('tattoo',re.I)
def go(f):
    err=None
    for a in range(3):
        try:
            with fsspec.filesystem('https',skip_instance_cache=True).open(f,block_size=1<<20) as fh:
                t=pq.ParquetFile(fh).read(columns=cols).to_pandas()
            t=t[t.caption.str.contains(r,na=False)].copy();t['shard']=f.split('/')[-2]+'/'+f.split('/')[-1];return t
        except Exception as e: err=e
    print('FAIL',f,err,file=sys.stderr,flush=True);return None
if __name__=='__main__':
    files=json.load(open('data/pap_files.json'));out=[]
    with Pool(8) as ex:
        for i,t in enumerate(ex.imap_unordered(go,files)):
            if t is not None: out.append(t)
            if i%25==0: print(i,sum(len(x) for x in out),flush=True)
    d=pd.concat(out);d.to_csv('data/pap_tattoo_pairs.csv',index=False)
    print('shards_ok',len(out),'of',len(files),'rows',len(d),'labeled&different',int((d.has_label&d.are_different).sum()),'prompts',d.caption.nunique())

"""Score images with CLIPScore, PickScore, HPSv2.1, ImageReward on CPU. One model at a time.
usage: python score_rewards.py manifest.csv out.csv   (manifest: id,path,prompt)"""
import sys, os, torch, pandas as pd
from PIL import Image
torch.set_num_threads(8); torch.manual_seed(0)
man = pd.read_csv(sys.argv[1]); out = sys.argv[2]
res = pd.read_csv(out) if os.path.exists(out) else man[['id']].copy()
models = sys.argv[3].split(',') if len(sys.argv) > 3 else ['clipscore', 'pickscore', 'hpsv21', 'imagereward']
def batches(n=8):
    for i in range(0, len(man), n): yield man.iloc[i:i+n]
@torch.no_grad()
def run(name, fn):
    if name in res.columns: print('skip', name); return
    vals = []
    for b in batches():
        vals += fn([Image.open(p).convert('RGB') for p in b.path], list(b.prompt))
        print(name, len(vals), flush=True)
    res[name] = vals; res.to_csv(out, index=False)
for name in models:
    if name in ('clipscore', 'hpsv21'):
        import open_clip
        if name == 'clipscore':
            m, _, pre = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai'); tok = open_clip.get_tokenizer('ViT-L-14')
        else:
            from huggingface_hub import hf_hub_download
            m, _, pre = open_clip.create_model_and_transforms('ViT-H-14', pretrained=None)
            sd = torch.load(hf_hub_download('xswu/HPSv2', 'HPS_v2.1_compressed.pt'), map_location='cpu')
            m.load_state_dict(sd['state_dict']); tok = open_clip.get_tokenizer('ViT-H-14')
        m.eval()
        def fn(ims, ps):
            i = m.encode_image(torch.stack([pre(x) for x in ims])); t = m.encode_text(tok(ps))
            i = i / i.norm(dim=-1, keepdim=True); t = t / t.norm(dim=-1, keepdim=True)
            s = (i * t).sum(-1)
            return (torch.clamp(s, min=0) * 2.5 if name == 'clipscore' else s).tolist()
        run(name, fn); del m
    elif name == 'pickscore':
        from transformers import AutoProcessor, AutoModel
        proc = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K'); m = AutoModel.from_pretrained('yuvalkirstain/PickScore_v1').eval()
        def fn(ims, ps):
            i = m.get_image_features(**proc(images=ims, return_tensors='pt'))
            t = m.get_text_features(**proc(text=ps, padding=True, truncation=True, max_length=77, return_tensors='pt'))
            i = i / i.norm(dim=-1, keepdim=True); t = t / t.norm(dim=-1, keepdim=True)
            return (m.logit_scale.exp() * (i * t).sum(-1)).tolist()
        run(name, fn); del m
    elif name == 'imagereward':
        import ImageReward as RM
        m = RM.load('ImageReward-v1.0', device='cpu')
        def fn(ims, ps): return [float(m.score(p, [x])) for x, p in zip(ims, ps)]
        run(name, fn); del m

"""Classify tattoo-prompt images into design / on_skin / incidental using prompt keywords + CLIP zero-shot (ViT-B-32, openai)."""
import re, sys, torch, pandas as pd, open_clip
from PIL import Image
torch.manual_seed(0); torch.set_num_threads(4)
DESIGN_KW = re.compile(r"tattoo (design|flash|stencil|sketch|art\b|style|drawing|illustration|concept)|flash sheet|line ?art|(on|over) (a )?white (paper|background)|vector|logo|sticker|tattoo of an? |tattoo,? (black|blackwork|minimal|fine ?line|traditional|neo)", re.I)
INCID_KW = re.compile(r"with (intricate |many |full[- ]body |face |colorful )?tattoos|tattooed|covered in tattoos|portrait of|(woman|man|girl|boy|warrior|samurai|punk|biker|cyborg|elf|lady|person|model)\b.*tattoos", re.I)
LABELS = ['a tattoo design drawing on plain paper or a white background', 'a close-up photo of a tattoo on skin',
          'a picture of a person who has tattoos', 'an image of something else']
CLS = ['design', 'on_skin', 'incidental', 'other']
def kw(p):
    if DESIGN_KW.search(p): return 'design'
    if INCID_KW.search(p): return 'incidental'
    return 'unknown'
if __name__ == '__main__':
    man = pd.read_csv(sys.argv[1]); out = sys.argv[2]
    m, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); tok = open_clip.get_tokenizer('ViT-B-32'); m.eval()
    with torch.no_grad():
        t = m.encode_text(tok(LABELS)); t /= t.norm(dim=-1, keepdim=True); preds = []; probs = []
        for i in range(0, len(man), 32):
            ims = torch.stack([pre(Image.open(p).convert('RGB')) for p in man.path[i:i+32]])
            f = m.encode_image(ims); f /= f.norm(dim=-1, keepdim=True)
            pr = (100 * f @ t.T).softmax(-1); preds += [CLS[j] for j in pr.argmax(-1)]; probs += pr.max(-1).values.tolist()
    man['clip_cls'] = preds; man['clip_p'] = probs; man['kw_cls'] = man.prompt.map(kw)
    # final rule: keyword decides when explicit; otherwise CLIP; disagreement design(kw) vs incidental(clip) -> 'ambiguous'
    def final(r):
        if r.kw_cls == 'design': return 'design' if r.clip_cls in ('design', 'on_skin') else 'ambiguous'
        if r.kw_cls == 'incidental': return 'incidental' if r.clip_cls != 'design' else 'ambiguous'
        return r.clip_cls
    man['cls'] = man.apply(final, axis=1); man.to_csv(out, index=False)
    print(man.cls.value_counts().to_string()); print(pd.crosstab(man.kw_cls, man.clip_cls).to_string())

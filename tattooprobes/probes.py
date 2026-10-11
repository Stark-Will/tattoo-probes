import numpy as np, cv2
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

NEEDLE_MM = 0.30   # approx. single-needle/3RL line (practitioner guidance; assumption, see paper)
GAP_MM = 0.50      # gaps narrower than this are assumed to close after healing (assumption)
WORK_PX_PER_MM = 10.0  # analysis resolution after physical rescaling

def to_gray(img):
    a = np.asarray(img.convert('RGB') if hasattr(img, 'convert') else img)
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a

def physical_resample(gray, size_cm):
    """Rescale so that the longest side spans size_cm at WORK_PX_PER_MM."""
    h, w = gray.shape
    target = int(round(size_cm * 10 * WORK_PX_PER_MM))
    s = target / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)

def ink_mask(gray):
    t = threshold_otsu(gray) if gray.std() > 1 else 128
    return gray < t

def _skel_widths(mask):
    if mask.sum() == 0: return np.array([])
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(mask)
    return 2 * dt[sk] / WORK_PX_PER_MM, sk

def compute_probes(img, size_cm=5.0):
    g = physical_resample(to_gray(img), size_cm)
    m = ink_mask(g)
    out = {}
    area_mm2 = m.size / WORK_PX_PER_MM**2
    ink_frac = m.mean(); out['ink_frac'] = float(ink_frac)
    # line width
    r = _skel_widths(m)
    if len(r) and len(r[0]):
        w, sk = r
        out['line_w_p10_mm'] = float(np.percentile(w, 10))
        out['frac_sub_needle'] = float((w < NEEDLE_MM).mean())
        # contour closure: skeleton endpoints per cm of skeleton
        k = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
        ends = ((k == 2) & sk).sum()
        out['open_ends_per_cm'] = float(ends / max(sk.sum() / WORK_PX_PER_MM / 10, 1e-6))
    else:
        out.update(line_w_p10_mm=0.0, frac_sub_needle=1.0, open_ends_per_cm=0.0)
    # gaps: background skeleton inside ink bounding box
    ys, xs = np.where(m)
    if len(ys):
        bg = ~m[ys.min():ys.max()+1, xs.min():xs.max()+1]
        rg = _skel_widths(bg)
        out['frac_tight_gap'] = float((rg[0] < GAP_MM).mean()) if len(rg) and len(rg[0]) else 0.0
    else:
        out['frac_tight_gap'] = 0.0
    # background cleanliness: mid-tone pixels (neither ink nor paper) and border non-uniformity
    mid = ((g > 60) & (g < 200)).mean(); out['midtone_frac'] = float(mid)
    b = np.concatenate([g[:5].ravel(), g[-5:].ravel(), g[:, :5].ravel(), g[:, -5:].ravel()])
    out['border_std'] = float(b.std())
    # detail density / specks
    e = cv2.Canny(g, 50, 150) > 0
    # edge crispness: ink/paper contrast divided by median gradient at edges -> transition width in mm (post-hoc addition, v0.2)
    if m.any() and (~m).any() and e.any():
        gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8
        gm = np.hypot(gx, gy)[e]; contrast = float(np.median(g[~m]) - np.median(g[m]))
        out['edge_transition_mm'] = float(contrast / max(np.median(gm), 1e-3) / WORK_PX_PER_MM)
    else: out['edge_transition_mm'] = 0.0
    out['edge_density_per_mm2'] = float(e.sum() / WORK_PX_PER_MM / area_mm2)
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    tiny = (st[1:, cv2.CC_STAT_AREA] < (NEEDLE_MM * WORK_PX_PER_MM) ** 2).sum()
    out['specks_per_cm2'] = float(tiny / (area_mm2 / 100))
    return out

# sign: +1 means higher is better for tattooability, -1 worse. Fixed a priori, not fitted.
PROBE_SIGNS = {'line_w_p10_mm': +1, 'frac_sub_needle': -1, 'open_ends_per_cm': -1, 'frac_tight_gap': -1,
               'midtone_frac': -1, 'border_std': -1, 'edge_density_per_mm2': -1, 'specks_per_cm2': -1, 'edge_transition_mm': -1}

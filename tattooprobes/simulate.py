import numpy as np, cv2
from skimage.morphology import skeletonize
from .probes import physical_resample, ink_mask, to_gray, WORK_PX_PER_MM

TOL_MM = 0.2

def simulate_on_skin(img, size_cm, spread_mm, seed=0, noise=0.0):
    """Rescale to physical size, apply isotropic Gaussian ink spread (sigma=spread_mm) + mild pigment noise.
    A stress test, NOT a validated dermal model."""
    g = physical_resample(to_gray(img), size_cm).astype(np.float32)
    if spread_mm > 0:
        g = cv2.GaussianBlur(g, (0, 0), spread_mm * WORK_PX_PER_MM)
    rng = np.random.default_rng(seed)
    g = np.clip(g + rng.normal(0, noise, g.shape), 0, 255) if noise > 0 else g
    return g.astype(np.uint8)

def retention(img, size_cm, spread_mm, seed=0, noise=0.0):
    ref = physical_resample(to_gray(img), size_cm)
    m0 = ink_mask(ref)
    deg = simulate_on_skin(img, size_cm, spread_mm, seed, noise)
    # v0.3 fix: threshold the simulated image with the SAME Otsu threshold as the reference, and
    # compare skeletons with a tolerance expressed in mm (0.2 mm) instead of a fixed 5-px window,
    # which previously made the match stricter at larger physical sizes (resize artifact).
    from skimage.filters import threshold_otsu
    t0 = threshold_otsu(ref) if ref.std() > 1 else 128
    m1 = deg < t0
    s0, s1 = skeletonize(m0), skeletonize(m1)
    k = 2 * int(round(TOL_MM * WORK_PX_PER_MM)) + 1
    tol = cv2.dilate(s1.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    tol0 = cv2.dilate(s0.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    p = (s1 & tol0).sum() / max(s1.sum(), 1); r = (s0 & tol).sum() / max(s0.sum(), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    # gap survival: fraction of original background pixels inside the ink bbox that remain background
    ys, xs = np.where(m0)
    if len(ys):
        sl = (slice(ys.min(), ys.max()+1), slice(xs.min(), xs.max()+1))
        bg0 = ~m0[sl]; gap_surv = float((bg0 & ~m1[sl]).sum() / max(bg0.sum(), 1))
    else: gap_surv = 1.0
    return {'skel_f1': float(f1), 'gap_survival': gap_surv}

# tattoo-probes

**Deterministic, CPU-only tattooability probes for AI tattoo generation — and an audit of whether preference reward models measure the same thing.**

Code, pair indexes, derived scores and figures for the paper

> YUZHU CONG. *Preference ≠ Tattooability: Auditing Reward Models for AI Tattoo Generation.* 2026.
> PDF: [`paper/main.pdf`](paper/main.pdf) (15 pages) · Zenodo: *DOI pending* · Hugging Face indexes: *pending* · arXiv: *pending* · v1.0 Release: see *Releases* after push.

## TL;DR

- Physically scaled CPU probes (needle 0.30 mm, gap 0.50 mm, 10 px/mm) evaluated at print sizes 3 / 5 / 10 / 15 cm.
- On HPDv2 tattoo-subject pairs, a-priori probes agree with human preference **below chance**; the contrast survives a Holm correction across scorers.
- CLIPScore, PickScore, HPSv2.1 and ImageReward track preference (and flag many injected defects), but HPSv2.1 / ImageReward are in-distribution on their training sets.
- A print-scale × ink-spread stress test (isotropic Gaussian assumption, **not** a healed-skin model) shows skeleton retention falling as size shrinks and spread widens.
- **Nothing here ranks a commercial AI tattoo generator.**

All headline numbers trace to CSVs in `results/` and macros in `paper/numbers.tex` (checked by `paper/check_numbers.py`).

## Repository layout

| Path | What |
|---|---|
| `tattooprobes/` | probe + simulator implementation (single source of thresholds) |
| `scripts/` | classification, reward scoring, analyses, figures, download notes |
| `data/` | **indexes / manifests / pair tables / labels only** — no third-party images |
| `results/` | probe scores, reward scores, agreement, defect detection, scale/spread tables |
| `figs/` | paper plot PNGs (author-generated) |
| `paper/` | LaTeX source, figures, bibliography, compiled PDF, `check_numbers.py` |

## Install

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # CPU torch + vision stack; large
# minimal probe-only path (no reward models):
.venv/bin/pip install numpy opencv-python-headless scikit-image pillow pandas
```

## Reproduce (after you download images locally)

Third-party images are **not** in this repo. Follow [`scripts/dl_public_data.md`](scripts/dl_public_data.md), then:

```bash
# Probe scores on manifests you have populated under data/raw/
.venv/bin/python scripts/run_probes.py

# Re-derive paper macros from CSVs and verify against committed numbers.tex
.venv/bin/python paper/check_numbers.py
```

Seeds: `0` everywhere (numpy / torch / bootstrap). Exact lockfile: `requirements.lock.txt`.

## Data and licences

See [`data/LICENSES.md`](data/LICENSES.md). Short version:

- **No** HPDv2 / ImageRewardDB / Pick-a-Pic / Drozdik images are redistributed.
- Drozdik/tattoo_v0 is **CC BY-NC-SA 4.0** (research only).
- Pick-a-Pic upstream licence is **unverified** here; only metadata indexes ship.
- Code is MIT (`LICENSE`). Paper text and author figures are CC BY 4.0.

## Cite

```bibtex
@misc{cong2026tattooability,
  author    = {CONG, YUZHU},
  title     = {Preference $\neq$ Tattooability: Auditing Reward Models for AI Tattoo Generation},
  year      = {2026},
  note      = {Code: https://github.com/Stark-Will/tattoo-probes
               Zenodo DOI and arXiv id: fill after deposit},
  url       = {https://tatany.app}
}
```

Also see `CITATION.cff`.

## Competing interest

The author operates [tatany.app](https://tatany.app), a commercial AI tattoo generator.
That product was **not** evaluated or ranked; none of its outputs appear in the scored datasets.
Contact: support@tatany.app

## Resources (fill after publish)

- GitHub: `https://github.com/Stark-Will/tattoo-probes` *(pending push)*
- Zenodo DOI: *pending*
- Hugging Face dataset (indexes only): *pending*
- arXiv: *pending*
- Site: https://tatany.app

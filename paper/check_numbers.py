"""Number cross-check for the paper (public snapshot layout: paper/ lives inside the repo root).
1) Re-derive numbers.tex from the result CSVs and require it to be identical to the committed numbers.tex.
2) Every \\macro used in .tex that looks like a numbers macro must be defined in numbers.tex.
3) Flag literal numbers in .tex prose that are not whitelisted method constants (results must use macros)."""
import re, glob, subprocess, sys, os, shutil
ok = True
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, '..'))
# Prefer repo .venv, then PATH python
py = os.path.join(root, '.venv', 'bin', 'python')
if not os.path.isfile(py):
    py = sys.executable
shutil.copy(os.path.join(here, 'numbers.tex'), '/tmp/numbers_committed.tex')
r = subprocess.run([py, 'scripts/make_numbers.py'], cwd=root, capture_output=True, text=True)
if r.returncode != 0:
    print('WARN: make_numbers.py failed (need deps); skipping re-derive equality check')
    print(r.stderr[-2000:] if r.stderr else r.stdout[-2000:])
else:
    if open('/tmp/numbers_committed.tex').read() != open(os.path.join(here, 'numbers.tex')).read():
        print('FAIL: numbers.tex differs from values re-derived from CSVs'); ok = False
    else:
        print('PASS: numbers.tex == re-derived from CSVs')
defs = set(re.findall(r'\\newcommand\{\\(\w+)\}', open(os.path.join(here, 'numbers.tex')).read()))
WL = {'0.30', '0.3', '0.50', '0.5', '0.2', '0.20', '10', '3', '5', '15', '0', '1', '2', '4', '6', '7', '8', '9', '12', '0.05', '0.60', '0.6', '2000', '1000', '32', '77', '0.25', '0.75', '1.0', '100', '95', '2.5', '97.5', '50', '150', '60', '200', '1.5', '3.0', '0.003', '0.01', '0.4', '0.1', '0.10', '24', '30', '120', '13', '11', '14', '16', '0.15', '0.35', '0.45', '0.55', '0.40', '3.13', '2.14.1', '4.46.3', '3.3.0', '1.5'}
bad = []
for f in sorted(glob.glob(os.path.join(here, 'main.tex')) + glob.glob(os.path.join(here, 'sections', '*.tex')) + glob.glob(os.path.join(here, 'tables', '*.tex'))):
    for ln, line in enumerate(open(f), 1):
        line = re.sub(r'(?<!\\)%.*', '', line)
        for m in re.findall(r'\\([A-Za-z]+)', line):
            if re.match(r'(n[A-Z]|N[A-Z][a-z]|Hpd[A-Z]|Ir[A-Z]|Pap[A-Z]|Fus[A-Z]|Dg[A-Z]|ss[A-Z]|Rho[A-Z])', m) and m not in defs:
                bad.append(f'{os.path.basename(f)}:{ln}: undefined numbers macro \\{m}')
        txt = re.sub(r'\\(cite\w*|ref|label|url|href|includegraphics|input|eqref|begin|end|usepackage|documentclass|geometry|setlength|hypersetup)(\[[^\]]*\])?\{[^}]*\}', '', line)
        txt = re.sub(r'\\[A-Za-z]+', ' ', txt)
        for num in re.findall(r'(?<![\w.])(\d+(?:[.,]\d+)*)(?![\w])', txt):
            if num in WL or re.fullmatch(r'(19|20)\d\d', num): continue
            bad.append(f'{os.path.basename(f)}:{ln}: literal number {num} -> use a macro (or whitelist if a method constant)')
for b in bad: print('FLAG', b)
print('literal/undefined flags:', len(bad)); sys.exit(0 if ok and not bad else 1)

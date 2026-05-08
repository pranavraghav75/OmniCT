# Report

The NeurIPS-style final report for OmniCT.

## Files

- `main.tex`        — paper source. All 7 required sections are stubbed
                       with content seeded from the project proposal.
                       Search for `\TODO{...}` to find every place that
                       still needs writer attention.
- `references.bib`  — pre-populated with $\geq 10$ citations covering
                       the workshop baselines, the proposal's reading
                       list, plus methodology grounding (DINO, LoRA,
                       Grad-CAM).
- `neurips_2025.sty` — official NeurIPS 2025 style file (the 2026 sty
                       was not directly downloadable from media.neurips.cc
                       at the time of writing; layout is virtually
                       identical, swap if your TA strictly requires the
                       2026 file).
- `figures/`         — drop generated PDFs (`data_efficiency.pdf`,
                       `saliency_panel.pdf`) here.
- `tables/`          — drop generated CSVs / `.tex` table fragments here.
- `Makefile`         — `make` to build, `make clean` to remove aux files.

## Build

You need a TeX distribution with `pdflatex` and `bibtex`. On macOS:

```bash
brew install --cask mactex-no-gui     # large but standard
# or:
brew install basictex                 # small; `tlmgr install` extras
sudo tlmgr install booktabs caption microtype natbib subcaption
```

Then:

```bash
cd report/
make
open main.pdf
```

## Drafting workflow

1. Search for `\TODO{}` in `main.tex`. Each one is a small, scoped
   writing task with the rubric criterion it serves in a comment above.
2. Run `make watch` for live recompilation while writing.
3. As experiments finish, drop the figure PDFs into `figures/` and the
   table CSV into `tables/`. The `\includegraphics` lines already
   reference the expected filenames.
4. Before submitting, do a final search for `\TODO`, `\PLACE`, and
   `[TODO:` and confirm the PDF has none.

## Page budget

Hard limit: **8 pages excluding references** (course rubric).
The skeleton is currently around 6 pages of text once `\TODO`s are
replaced—comfortably under the cap.

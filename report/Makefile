# Build the NeurIPS-formatted final report.
# Usage:
#   make            # build main.pdf
#   make clean      # remove latex auxiliary files
#   make watch      # auto-rebuild on save (requires latexmk)

MAIN = main
TEX  = $(MAIN).tex
PDF  = $(MAIN).pdf

all: $(PDF)

$(PDF): $(TEX) references.bib
	pdflatex -interaction=nonstopmode $(TEX)
	bibtex $(MAIN) || true
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PDF)

.PHONY: all watch clean distclean

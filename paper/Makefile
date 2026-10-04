# Build the paper.
#
#   make            compile main.pdf (figures are rebuilt only if missing)
#   make figures    regenerate figures from data/features and runs/default
#   make all        figures + PDF
#   make watch      recompile whenever a source file changes (needs entr)
#   make clean      remove LaTeX by-products; keep main.pdf and figures
#   make distclean  also remove main.pdf and figures
#
# Requires tectonic (https://tectonic-typesetting.github.io), which runs
# BibTeX and repeated passes automatically and fetches packages on first use.
# Figures require the project environment (uv) and a finished
# `magreg train -o runs/default`.

TEX      := tectonic
PY       := uv run --project .. python
FIGS     := $(addprefix figures/,fig_selection.pdf fig_pred_vs_true.pdf fig_bias.pdf fig_importance.pdf fig_fc_scaling.pdf)

.PHONY: pdf all figures watch clean distclean

pdf: main.pdf

all: figures pdf

main.pdf: main.tex refs.bib $(FIGS)
	$(TEX) --keep-logs main.tex

$(FIGS):
	$(PY) make_figures.py

figures:
	$(PY) make_figures.py

watch:
	ls main.tex refs.bib | entr -c $(MAKE) pdf

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc

distclean: clean
	rm -f main.pdf $(FIGS)

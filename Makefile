PYTHON ?= python3
.PHONY: paper figures tables all clean
all: tables figures paper
paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
tables:
	$(PYTHON) scripts/build_results.py
figures:
	$(PYTHON) scripts/make_figures.py
clean:
	latexmk -c main.tex

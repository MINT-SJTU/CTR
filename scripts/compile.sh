#!/usr/bin/env bash
# Portable build, including environments that name the binary bibtex.original.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v pdflatex >/dev/null || { echo "pdflatex is required." >&2; exit 1; }
if command -v bibtex >/dev/null; then BIB=bibtex
elif command -v bibtex.original >/dev/null; then BIB=bibtex.original
else echo "BibTeX is required." >&2; exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
"$BIB" main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

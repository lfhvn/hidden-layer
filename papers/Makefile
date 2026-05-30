# Build automation for Hidden Layer papers.
#
#   make results        Regenerate numbers/tables for bound papers (no LaTeX needed)
#   make pdf            Build every paper to PDF (requires latexmk)
#   make <name>.pdf     Build one paper, e.g. make metacognition-calibration.pdf
#   make clean          Remove LaTeX build artifacts
#
# "results" re-runs the experiments behind bound papers and rewrites generated/*.tex,
# so cited numbers always match the code. It needs only Python; PDF building needs a
# LaTeX toolchain (latexmk + a TeX distribution).

PYTHON ?= python3
LATEXMK := $(shell command -v latexmk 2>/dev/null)

SObjects := $(wildcard *.tex)
PDFS := $(SObjects:.tex=.pdf)

.PHONY: all results pdf clean help check

help:
	@echo "Targets: results | pdf | <name>.pdf | clean | check"
	@echo "  results : regenerate generated/*.tex from experiments (Python only)"
	@echo "  pdf     : build all papers (needs latexmk)"
	@echo "  check   : verify bound papers regenerate and report toolchain status"

all: results pdf

# Regenerate the figures/macros/tables that bound papers \input.
results:
	@echo "Regenerating bound-paper results..."
	@$(PYTHON) generate_results.py

# Papers with generated results depend on `results` being fresh.
metacognition-calibration.pdf: results

%.pdf: %.tex
ifndef LATEXMK
	@echo "ERROR: latexmk not found. Install a LaTeX toolchain (e.g. 'brew install --cask mactex' or 'apt-get install texlive-full latexmk') to build PDFs."
	@echo "       The reproducible numbers were still (re)generated via 'make results'."
	@exit 1
else
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $<
endif

pdf: results $(PDFS)
ifndef LATEXMK
	@echo "NOTE: latexmk absent; ran 'results' only. No PDFs produced."
endif

# CI-friendly: regenerate results and confirm bound papers reference real artifacts,
# without requiring a TeX toolchain.
check: results
	@test -f references.bib && echo "  ok references.bib"
	@test -f generated/metacognition_macros.tex && echo "  ok generated/metacognition_macros.tex"
	@test -f generated/metacognition_table.tex && echo "  ok generated/metacognition_table.tex"
	@test -f generated/multi_agent_macros.tex && echo "  ok generated/multi_agent_macros.tex"
	@test -f generated/multi_agent_table.tex && echo "  ok generated/multi_agent_table.tex"
	@if [ -n "$(LATEXMK)" ]; then echo "  latexmk: $(LATEXMK)"; else echo "  latexmk: ABSENT (PDF build skipped)"; fi
	@echo "OK"

clean:
ifdef LATEXMK
	-$(LATEXMK) -C
endif
	@rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.toc
	@echo "Cleaned LaTeX build artifacts."

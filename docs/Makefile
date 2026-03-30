# Minimal Sphinx makefile

.PHONY: help clean html

PYTHON ?= python
SPHINXBUILD ?= $(PYTHON) -m sphinx
MPLBACKEND ?= Agg
MPLCONFIGDIR ?= /tmp/mne-denoise-mpl
DOCS_HOME ?= /tmp/mne-denoise-home
MNE_HOME ?= /tmp/mne-denoise-mne
NUMBA_CACHE_DIR ?= /tmp/mne-denoise-numba

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean    remove built documentation"
	@echo "  html     build HTML documentation"

clean:
	@rm -rf _build

html:
	@mkdir -p $(MPLCONFIGDIR) $(DOCS_HOME) $(MNE_HOME) $(NUMBA_CACHE_DIR)
	@HOME=$(DOCS_HOME) MNE_HOME=$(MNE_HOME) NUMBA_CACHE_DIR=$(NUMBA_CACHE_DIR) MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) MNE_DONTWRITE_HOME=true $(SPHINXBUILD) -M html . _build -W --keep-going

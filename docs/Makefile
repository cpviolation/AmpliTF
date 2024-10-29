###############################################################################
# (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for SPHINXOPTS.
SPHINXPREFIX  = _env.$(BINARY_TAG)
SPHINXOPTS    ?=
SPHINXBUILD   = $(SPHINXPREFIX)/bin/sphinx-build
RUNENV        = PYTHONPATH=$(SPHINXPREFIX):$(PYTHONPATH)
SOURCEDIR     = .
BUILDDIR      = _build
GIT_COMMIT    = $(shell git rev-parse --short HEAD)

#GRAPH_SCRIPTS := $(wildcard scripts/*.py)
#CONTROL_GRAPH_SOURCES := $(patsubst scripts/%.py, graphviz/%_control_flow.gv, $(GRAPH_SCRIPTS))
#DATA_GRAPH_SOURCES := $(patsubst scripts/%.py, graphviz/%_data_flow.gv, $(GRAPH_SCRIPTS))

# Put it first so that "make" without argument is like "make help".
help: $(SPHINXBUILD)
	$(RUNENV) $(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# we install certifi (matching the LCG version) so that we can patch
# cacert.pem (== `python -m certifi`)
$(SPHINXBUILD): requirements.txt Makefile
	python -m pip install --disable-pip-version-check --no-warn-conflicts --ignore-installed \
		--target=$(SPHINXPREFIX) -r $< \
		certifi==`python -c 'import certifi; print(certifi.__version__)'`
	@touch $@

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
#html pdf: Makefile graphs functor_docs $(SPHINXBUILD)
html pdf: Makefile $(SPHINXBUILD)#Makefile graphs $(SPHINXBUILD)
	$(RUNENV) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

#linkcheck: Makefile graphs functor_docs $(SPHINXBUILD)
linkcheck: Makefile $(SPHINXBUILD)# Makefile graphs $(SPHINXBUILD)
	cat _certificates/*.crt >> `$(RUNENV) python -m certifi`  # install CERN CA certificates
	$(RUNENV) $(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean: Makefile $(SPHINXBUILD)#rm -rf graphviz
	$(RUNENV) $(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

purge:
	rm -rf $(SPHINXPREFIX) $(BUILDDIR) 

# summary_plots:
# 	python scripts/make_summary_pages.py

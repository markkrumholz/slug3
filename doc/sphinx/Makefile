# Minimal Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment, for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
# The final built HTML lands in doc/html (not doc/sphinx/_build/html),
# so it sits at a stable, predictable path outside doc/sphinx/'s own
# source tree; doctrees still cache under _build/, since those are a
# pure build-speed cache, not something anyone browses.
HTMLDIR       = ../html

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html clean

html:
	@$(SPHINXBUILD) -b html -d "$(BUILDDIR)/doctrees" "$(SOURCEDIR)" "$(HTMLDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf "$(HTMLDIR)"

# Catch-all target: route all other targets (latex, text, etc.) to
# Sphinx using the "make mode" option, under the usual _build/ layout
# -- only the common "html" case above gets its own output location.
# $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

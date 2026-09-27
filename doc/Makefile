# Builds the MFTL2 manual (MFTL2.pdf). Needs pdflatex with TikZ and listings; the library build
# never needs LaTeX. The listings are read from ../examples/.
#   make            -> MFTL2.pdf in this directory
#   make OUT=<dir>  -> MFTL2.pdf in <dir> (used by the CMake target `manual`)
#   make clean
PDFLATEX ?= pdflatex
OUT      ?= .
SOURCES  := MFTL2.tex $(wildcard chapters/*.tex) $(wildcard ../examples/manual_*.c)

all: $(OUT)/MFTL2.pdf

$(OUT)/MFTL2.pdf: $(SOURCES)
	@mkdir -p $(OUT)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(OUT) MFTL2.tex > $(OUT)/MFTL2.build.log
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(OUT) MFTL2.tex > $(OUT)/MFTL2.build.log
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=$(OUT) MFTL2.tex > $(OUT)/MFTL2.build.log
	@! grep -q "LaTeX Warning: Reference" $(OUT)/MFTL2.log || { echo "undefined references, see $(OUT)/MFTL2.log"; exit 1; }
	@echo "built $(OUT)/MFTL2.pdf"

clean:
	rm -f $(OUT)/MFTL2.pdf $(OUT)/MFTL2.aux $(OUT)/MFTL2.log $(OUT)/MFTL2.toc $(OUT)/MFTL2.out $(OUT)/MFTL2.build.log

.PHONY: all clean

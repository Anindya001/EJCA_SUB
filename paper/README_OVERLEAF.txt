Elsevier CAS package -- European Journal of Control submission
  "Two-sided sampling-period design windows for digital PI/PID control
   under dead-time and identification uncertainty", A. Bhattacharyya

MAIN DOCUMENT: main_sc.tex  (cas-sc, single column). Set this as the
main document in Overleaf. Compiles with pdfLaTeX + BibTeX.
Reference build: 25 pages, no errors, one overfull box.

GENERATED INPUTS -- DO NOT EDIT BY HAND
  validation_macros.tex        every number quoted in the text
  benchmark_summary_table.tex  Table 5
  soptd_validation_table.tex   Table 6
  application_table.tex        Table 7
  fig01..fig12 (PDF)           every figure
All of these are written by ../github_repro_EJCA/validation_ejc.py. To
refresh them after changing the code:

  cd ../github_repro_EJCA && python validation_ejc.py
  cp fig*.pdf *_table.tex validation_macros.tex ../paper/

Because the text pulls its numbers from validation_macros.tex, the
manuscript cannot silently disagree with the code. test_validation_ejc.py
additionally asserts that every generated macro the manuscript references
is actually emitted.

FIGURE NUMBERING
  Files are named in document order (fig01..fig12), so the file name and
  the printed figure number agree. Four figures are new relative to the
  previous revision (fig02 delay architecture, fig08 window map, fig09
  design chart, fig12 applications); two pairs were merged to keep the
  figure count down (fig06 now carries the sampling zero together with
  the command-activity panels, fig07 carries both fast-sampling lower
  bounds).

FLOAT PLACEMENT
  The cas-common.sty class redefines figure/table with a key-value
  optional argument, so the float package's [H] specifier is NOT
  available -- it raises "Unknown float option `H'". Placement is
  instead controlled by:
    - [pos=!htb] on every float (the ! suspends the fraction tests),
    - relaxed \topfraction / \bottomfraction / \textfraction,
    - placeins loaded with [section] so no float crosses a section head.
  All figure*/table* were converted to figure/table: in the single-column
  cas-sc layout the starred forms go through \@dblfloat, which cannot
  honour "here" placement and was the cause of the drifting figures.

FIGURE SCALING
  Every graphic uses width=\linewidth. The figures are authored at their
  final aspect ratio in matplotlib (wide and short for the three-panel
  figures), so no height cap is needed and no font is rescaled by
  LaTeX -- label sizes on the page match what the plotting code sets.

PAGE BUDGET
  The submission is capped at 25 pages. If material is added, the usual
  levers are: reduce a 2x2 figure to 1x2 in the plotting code (this is
  what fig03 and fig10 already do), or tighten the discussion in
  Section 5.9. Changing \includegraphics widths is not the right lever,
  since it shrinks the figure fonts.

Class files, elsarticle-num.bst and thumbnails/ come from the official
Elsevier CAS bundle and must stay with the sources.

BEFORE SUBMISSION
  Fill \ead{...} with the corresponding-author email and add the ORCID in
  the \author options.

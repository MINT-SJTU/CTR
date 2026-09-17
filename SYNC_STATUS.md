# Sync status

This repository now contains the current rewritten manuscript source used for the CTR paper discussion, including:

- the redefined five experimental questions;
- Q2/Q3 separation into brittleness vs. robustness recovery;
- the new Q4 phase-conditioned cross-arm sensitivity experiment on simulated Scan Object (independent pickup -> dependent scanning -> independent placement);
- Q5 combining paired arm-pause intervention and held-out joint-progress generalization;
- updated appendix metric definitions for cross-arm sensitivity and the selectivity gap;
- bibliography, ICLR 2027 style, and main-text result tables.

The figure assets, complete appendix result tables, data, and build scripts previously present only in MINT-SJTU/CTR were imported from commit f8b1571fea1e03fcb35249a78efae33364f7c187. Existing files in this repository were retained, including the updated manuscript, relation terminology, and task overview figure. Generated LaTeX build files were not imported.

The task overview is available as a PDF and an editable, self-contained HTML source under `task_overview/`. The new Scan Object sensitivity section still contains a TODO until those measurements are run and verified; no missing experimental results have been fabricated.

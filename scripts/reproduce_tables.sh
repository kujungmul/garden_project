#!/bin/bash
# Regenerates every number in Section IV of the paper from results/ (no API calls, no clustering).
set -e
cd "$(dirname "$0")/../garden"
python3 paper_tables.py > ../results/paper_tables.txt     # Tables II-VIII and the Cost paragraph
python3 report_wmean.py                                    # results/report_wmean.md (all rows, all metrics)
python3 error_analysis.py cora citeseer                    # results/error_analysis.md, error_nodes_*.csv
python3 judge_summary.py                                   # judge robustness (repeat / class-order / Sonnet passes)
echo "wrote results/paper_tables.txt, results/report_wmean.md, results/error_analysis.md"

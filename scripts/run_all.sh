#!/bin/bash
# Full clustering protocol of the paper for one dataset: bash scripts/run_all.sh cora 0 1 2 3 4
# Runs sequentially (one clustering process at a time; ogbn-arxiv needs ~2 GB and ~16 min per method and seed).
# Rows are appended to results/<dataset>.jsonl; partitions go to results/preds/. Re-running a (dataset, method,
# tag, seed) appends a duplicate row, so delete the old row and .npy first if you repeat a run.
set -e
cd "$(dirname "$0")/../garden"
D=$1; shift
for s in "$@"; do
  echo "### $(date '+%H:%M:%S') $D seed=$s baselines (Louvain, SN-modularity, SemMod, SemMod-desc)"
  python3 run.py --dataset $D --methods louvain,sn,rdm-centroid,garden --lam 0.5 --sem-norm --seed $s --tag audit
  echo "### $(date '+%H:%M:%S') $D seed=$s GARDEN (agreement) and the two single-view controls"
  python3 run.py --dataset $D --methods garden-cagree --lam 0.5 --sem-norm --seed $s --tag cagree
  python3 run.py --dataset $D --methods garden-cexp   --lam 0.5 --sem-norm --seed $s --tag cexp
  python3 run.py --dataset $D --methods garden-dexp   --lam 0.5 --sem-norm --seed $s --tag dexp
done
echo "### $(date '+%H:%M:%S') $D shuffled-text control (seed 0)"
python3 run.py --dataset $D --methods louvain,rdm-centroid,garden-cagree --lam 0.5 --sem-norm --seed 0 --tag shuffle2 --shuffle-text
echo "### $(date '+%H:%M:%S') $D post-hoc description evaluation (needs ANTHROPIC_API_KEY unless fully cached)"
python3 desc_eval.py --dataset $D --glob "${D}_*_audit.npy"               --out ../results/desc_audit_$D.jsonl
python3 desc_eval.py --dataset $D --glob "${D}_garden-*_0.5_*_*.npy"      --out ../results/desc_audit_$D.jsonl
echo "### DONE $D $(date '+%H:%M:%S')"

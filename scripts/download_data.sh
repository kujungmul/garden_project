#!/bin/bash
# Downloads the four datasets into data/. Run from anywhere: bash scripts/download_data.sh
set -e
cd "$(dirname "$0")/.."
HF=https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs/resolve/main
for ds in cora citeseer wikics; do
  for f in processed_data.pt raw_texts.pt categories.csv; do
    [ -f data/$ds/$f ] || curl -L -o data/$ds/$f $HF/$ds/$f
  done
done
# ogbn-arxiv: graph + labels from OGB, titles/abstracts from the OGB misc bundle
cd data/arxiv
[ -f arxiv.zip ] || curl -L -o arxiv.zip http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
[ -d arxiv ] || unzip -q arxiv.zip            # -> data/arxiv/arxiv/{raw,mapping,...}
[ -f titleabs.tsv.gz ] || curl -L -o titleabs.tsv.gz https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
[ -f titleabs.tsv ] || gunzip -k titleabs.tsv.gz
echo "done"

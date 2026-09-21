Run `bash scripts/download_data.sh` from the repository root to populate this folder.

Expected layout (data.py):
```
data/cora/{processed_data.pt, raw_texts.pt, categories.csv, emb_minilm.npy}
data/citeseer/{...same...}
data/wikics/{...same...}
data/arxiv/arxiv/{raw/edge.csv.gz, raw/node-label.csv.gz, mapping/nodeidx2paperid.csv.gz, mapping/labelidx2arxivcategeory.csv.gz}
data/arxiv/titleabs.tsv
```
`emb_minilm.npy` is the all-MiniLM-L6-v2 embedding cache. It is shipped for Cora, CiteSeer and WikiCS
(so that the LLM prompt cache is hit exactly) and is computed on first use for ogbn-arxiv (~260 MB, ~10 min).

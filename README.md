# GARDEN: Graph-Aware Representative Descriptions for Enhanced Network Clustering

Code, cached LLM responses and result logs for the paper. Everything in Section IV of the paper can be
regenerated from `results/` without any API call; every clustering run of the paper can be re-executed
from the shipped caches, and the shipped caches make the Cora/CiteSeer/WikiCS runs bit-for-bit
reproducible without an API key (verified: the seed-0 GARDEN partition on Cora is identical).

## Layout

```
garden/        source code (see table below)
scripts/       download_data.sh, run_all.sh (clustering protocol), reproduce_tables.sh (paper tables)
data/          datasets (download with scripts/download_data.sh); MiniLM embedding caches for Cora/CiteSeer/WikiCS
cache/         LLM responses keyed by the exact prompt: 9,837 description generations (Claude Haiku 4.5),
               7,313 judge answers (Claude Opus 5 / Sonnet 5)
results/       {dataset}.jsonl        one line per clustering run (metrics, tag, seed, LLM calls, gate log)
               desc_audit_{ds}.jsonl post-hoc description evaluation of every partition (retrieval, judge)
               preds/                final partitions  {ds}_{method}_{lam}_{seed}_{tag}.npy
               paper_tables.txt      LaTeX rows of Tables II-VIII and the Cost paragraph (paper_tables.py)
               report_wmean.md       all rows and metrics in one report (report_wmean.py)
               error_analysis*.md    residual-error analysis (error_analysis.py)
               judge_audit.jsonl     judge robustness study, 4 passes (judge_audit.py / judge_summary.py)
```

| file | role |
|---|---|
| `data.py` | dataset loading, adjacency lists, all-MiniLM-L6-v2 embeddings (cached in `data/{ds}/emb_minilm.npy`) |
| `core.py` | `GraphState`; objectives `Modularity`, `SNModularity`, `RDM` (semantic modularity, centroid or description prototypes), `RDMAgree` (candidate expansion: agreement / centroid-only / description-only); `local_move`, `merge_to_k` |
| `methods.py` | `louvain_init`, `run_louvain_k`, `run_sn_centroid`, `run_rdm(proto=...)` |
| `llm.py` | evidence-set selection, description generation, candidate scoring, lazy refresh, prompt cache |
| `evaluate.py` | NMI / ARI / ACC (Hungarian) / purity / Q |
| `run.py` | clustering CLI, appends to `results/{ds}.jsonl`, saves partition |
| `desc_eval.py` | post-hoc descriptions of any partition; held-out retrieval; LLM judge (JSON-schema answer, failures never cached) |
| `paper_tables.py`, `report_wmean.py`, `error_analysis.py`, `judge_summary.py` | tables and analyses of Section IV |
| `judge_audit.py` | judge robustness passes (repeat, shuffled class order, Sonnet 5) |
| `instrument.py`, `analyze_moves.py`, `failure_taxonomy.py` | move-level logging and description failure taxonomy (supplementary analyses) |
| `test_cache.py` | 6 cache-consistency tests with a fake LLM; `python3 test_cache.py` must print `ALL PASS` |

## Method names (`--methods`) and paper names

| `--methods` | paper | `--tag` used in `results/` |
|---|---|---|
| `louvain` | Louvain | `audit` |
| `sn` | SN-modularity | `audit` |
| `rdm-centroid` | SemMod (centroid prototypes, adjacent moves) | `audit` |
| `garden` | SemMod-desc (description prototypes, adjacent moves) | `audit` |
| `garden-cagree` | **GARDEN** (centroid objective + agreement-guided non-local candidate) | `cagree` |
| `garden-cexp` | non-local candidate from the centroid view alone (Table VI) | `cexp` |
| `garden-dexp` | non-local candidate from the description view alone (Table VI) | `dexp` |
| any of the above with `--shuffle-text` | reliability-gate control (Table VIII) | `shuffle2` |

The tag `audit` marks runs of the final code (an earlier code version was withdrawn after an internal audit;
its rows are not included here).

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh          # ~1 GB, mostly ogbn-arxiv
export ANTHROPIC_API_KEY=...            # only needed on a cache miss (see "Determinism")
cd garden && python3 test_cache.py      # ALL PASS
```

Tested with Python 3.11.7 on macOS (Apple M4). Run one clustering process at a time; ogbn-arxiv needs
about 2 GB of memory and 16 minutes per method and seed (plus ~10 minutes for the one-time embedding).

## Reproducing the paper

**Tables only (no clustering, no API):**

```
bash scripts/reproduce_tables.sh
```

writes `results/paper_tables.txt`, whose rows are Tables II (clustering quality and Q), III (centroid vs.
description prototypes), IV (label-trapped errors), V (moved nodes, seeds/followers, connectivity),
VI (agreement vs. single-view candidates), VII (retrieval, judge), VIII (reliability gate under shuffled text) and the Cost paragraph.
`python3 paper_tables.py` loads all four datasets, so the data must be downloaded first.

**Full clustering protocol for one dataset** (five seeds, all methods, shuffled-text control, description
evaluation):

```
bash scripts/run_all.sh cora 0 1 2 3 4
```

`run.py` appends to `results/{ds}.jsonl`; delete the existing rows and `.npy` files of a
(dataset, method, tag, seed) before repeating it, otherwise the report scripts see duplicates.
Individual runs:

```
cd garden
python3 run.py --dataset cora --methods garden-cagree --lam 0.5 --sem-norm --seed 0 --tag cagree
python3 run.py --dataset cora --methods louvain,rdm-centroid,garden-cagree --lam 0.5 --sem-norm --seed 0 --tag shuffle2 --shuffle-text
python3 desc_eval.py --dataset cora --glob "cora_garden-cagree_0.5_*_cagree.npy" --out ../results/desc_audit_cora.jsonl
```

`--sem-norm` enables the calibration constant c of Eq. (9); all paper results use it with `--lam 0.5`.
Other options: `--max-iter` (T, default 6), `--lazy` (τ, default 0.9), `--K`, `--no-reliability`,
`--llm-model` (default `claude-haiku-4-5`); `desc_eval.py --judge-model` (default `claude-opus-5`).

Hyper-parameters of Section III are constants in the code: `n_min=5` (`min_llm_size` in `methods.py`),
`r0=40, r=10, θ=0.8`, 500 characters per document (`representative_docs` in `llm.py`), `q=3, L=20`
(`generate_candidates`), `n_U=512` (`select_candidate`), `n_rel=20` (`reliability_test` in `core.py`),
`ε=1e-12` and 20 sweeps (`local_move`).

## Determinism and the LLM cache

* Louvain initialisation, node visiting order and the permutation null are seeded by `--seed`; the
  remaining randomness is the language model.
* Every generation request is cached under `sha1("v3" + model + system prompt + user message)`. The user
  message contains the evidence documents, the community size, the domain line, q and L, so an identical
  request never reaches the API. With the shipped `cache/` and embedding caches, re-running any
  Cora/CiteSeer/WikiCS configuration of the paper makes zero API calls and returns the archived partition.
* ogbn-arxiv embeddings are recomputed on first use. Floating-point differences in a recomputed embedding can
  change the order of evidence documents of a community and therefore miss the cache for that community;
  the regenerated description is then sampled anew (the Anthropic API exposes no temperature/seed for these
  models), and results can differ slightly from the archived ones.
* Judge answers are cached only when valid; `desc_eval.py` retries failures up to three times and reports
  unresolved queries separately (`judge_unresolved`, 0 in all archived rows).

## Result file formats

`results/{ds}.jsonl`: `dataset, method, lam, seed, K_target, tag, time, K, NMI, ARI, ACC, purity, Q, K0,
iters, llm_calls, Qsem, retrieval_acc, sem_scale, reliability[{acc, chance, lam_eff} per block], descriptions,
lazy, shuffled`.

`results/desc_audit_{ds}.jsonl`: `method, lam, seed, tag, pred_file, K, retrieval_acc, retrieval_heldout,
Qsem, judge_acc, judge_unresolved, judge_version, align_cos, descriptions{cid: {text, majority, judged, size}}`.
`retrieval_heldout` excludes the nodes whose documents were shown to the language model; it is the
"Retrieval" column of Table VII.

## Datasets

Cora, CiteSeer and WikiCS come from the Hugging Face dataset `Graph-COM/Text-Attributed-Graphs`
(raw texts, labels, edges); ogbn-arxiv from OGB with the official `titleabs.tsv`. Graphs are undirected with
self-loops and duplicate edges removed; disconnected components are kept.

"""Description-quality evaluation of saved partitions (results/preds/*.npy).

For every partition, descriptions are generated post hoc with the same procedure GARDEN
uses in the loop (representative docs -> candidates -> encoder-scored selection), so the
comparison asks: *which partitions admit faithful, discriminative descriptions?*

Metrics
  retrieval_acc       fraction of nodes whose nearest description is their own community's
  retrieval_heldout   same, only on nodes whose documents the LLM did not see
  Qsem                mean contrastive margin (paper's semantic modularity)
  judge_acc           an LLM judge maps each description to one of the dataset's class names;
                      correct if it equals the community's majority class
  align_cos           cos(description, majority-class name text)
Usage: python desc_eval.py --dataset cora [--glob "cora_*_0.npy"]
"""
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np

from data import load, node_embeddings, adjacency, encode
from llm import Describer, client, CACHE
from run import DOMAIN

ARXIV_NAMES = {"cs.AI": "Artificial Intelligence", "cs.AR": "Hardware Architecture", "cs.CC": "Computational Complexity",
 "cs.CE": "Computational Engineering, Finance, and Science", "cs.CG": "Computational Geometry", "cs.CL": "Computation and Language",
 "cs.CR": "Cryptography and Security", "cs.CV": "Computer Vision and Pattern Recognition", "cs.CY": "Computers and Society",
 "cs.DB": "Databases", "cs.DC": "Distributed, Parallel, and Cluster Computing", "cs.DL": "Digital Libraries", "cs.DM": "Discrete Mathematics",
 "cs.DS": "Data Structures and Algorithms", "cs.ET": "Emerging Technologies", "cs.FL": "Formal Languages and Automata Theory",
 "cs.GL": "General Literature", "cs.GR": "Graphics", "cs.GT": "Computer Science and Game Theory", "cs.HC": "Human-Computer Interaction",
 "cs.IR": "Information Retrieval", "cs.IT": "Information Theory", "cs.LG": "Machine Learning", "cs.LO": "Logic in Computer Science",
 "cs.MA": "Multiagent Systems", "cs.MM": "Multimedia", "cs.MS": "Mathematical Software", "cs.NA": "Numerical Analysis",
 "cs.NE": "Neural and Evolutionary Computing", "cs.NI": "Networking and Internet Architecture", "cs.OH": "Other Computer Science",
 "cs.OS": "Operating Systems", "cs.PF": "Performance", "cs.PL": "Programming Languages", "cs.RO": "Robotics", "cs.SC": "Symbolic Computation",
 "cs.SD": "Sound", "cs.SE": "Software Engineering", "cs.SI": "Social and Information Networks", "cs.SY": "Systems and Control"}

JUDGE = """A clustering algorithm produced a community of documents from a {domain} and described it as:

"{desc}"

Which ONE of the following categories does this description best correspond to? Answer with the category's number only.

{options}"""


JUDGE_SCHEMA = {"type": "object", "properties": {"category": {"type": "integer"}},
                "required": ["category"], "additionalProperties": False}


def judge_v2(desc, names, domain, model="claude-opus-5", tag="p1", max_tokens=4096, tries=3):
    """Corrected judge (audit, 2026-09-21). v1 used max_tokens=16 and a regex over free text; on
    Opus 5 adaptive thinking consumed the budget, ~10-17% of answers came back empty, were parsed
    as -1 (= wrong) and cached permanently. v2: JSON-schema output with an integer category,
    max_tokens 4096, stop_reason recorded, failed answers retried and never cached.
    Returns a record dict; record["answer"] is a 0-based class index or None (unresolved)."""
    import hashlib
    opts = "\n".join(f"{i+1}. {n}" for i, n in enumerate(names))
    msg = JUDGE.format(domain=domain, desc=desc, options=opts)
    key = hashlib.sha1(("judge-v2" + model + tag + msg).encode()).hexdigest()
    p = CACHE / f"{key}.json"
    if p.exists():
        return json.load(open(p))
    rec = None
    for t in range(tries):
        r = client().messages.create(model=model, max_tokens=max_tokens,
                                     output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                                     messages=[{"role": "user", "content": msg}])
        text = "".join(b.text for b in r.content if b.type == "text")
        try:
            cat = int(json.loads(text)["category"]) - 1
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            cat = None
        valid = cat is not None and 0 <= cat < len(names) and r.stop_reason == "end_turn"
        rec = dict(answer=cat if valid else None, raw=text, stop_reason=r.stop_reason, model=model,
                   output_tokens=r.usage.output_tokens, attempts=t + 1, valid=valid, empty=(text.strip() == ""))
        if valid:
            json.dump(rec, open(p, "w"))          # only successful answers are cached
            break
    return rec


def judge(desc, names, domain, model="claude-opus-5"):
    """Backward-compatible wrapper: 0-based class index, or None when unresolved."""
    return judge_v2(desc, names, domain, model)["answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--llm-model", default="claude-haiku-4-5", help="description generator")
    ap.add_argument("--judge-model", default="claude-opus-5")
    ap.add_argument("--min-size", type=int, default=5, help="only communities with >= this many nodes are described/judged")
    ap.add_argument("--out", default=None, help="output jsonl (default results/desc_{dataset}.jsonl)")
    args = ap.parse_args()
    ds = load(args.dataset); E = node_embeddings(ds); nbrs, deg = adjacency(ds); y = ds["y"]
    names = [ARXIV_NAMES.get(n, n) for n in ds["label_names"]] if args.dataset == "arxiv" else ds["label_names"]
    name_emb = encode(names)
    root = Path(__file__).resolve().parent.parent / "results"
    pat = args.glob or f"{args.dataset}_*.npy"
    files = sorted(glob.glob(str(root / "preds" / pat)))
    outp = Path(args.out) if args.out else root / f"desc_{args.dataset}.jsonl"
    done = set()
    if outp.exists():
        done = {json.loads(l)["pred_file"] for l in open(outp)}
    fout = open(outp, "a")
    for f in files:
        base = os.path.basename(f)
        if base in done or base.endswith("_shuf.npy"):
            continue
        m = re.match(rf"{args.dataset}_(.+)_(None|[\d.]+)_(\d+)(?:_([A-Za-z][\w.]*))?\.npy", base)
        if m is None:
            continue
        method, lam, seed, tag = m.group(1), m.group(2), int(m.group(3)), (m.group(4) or "final")
        pred = np.load(f); K = pred.max() + 1
        sizes = np.bincount(pred, minlength=K)
        big = np.flatnonzero(sizes >= args.min_size)            # communities that get a description
        node_mask = np.isin(pred, big)
        d = Describer(ds, nbrs, E, args.llm_model, DOMAIN[args.dataset], verbose=False)
        d.update(pred, big)
        P = d.prototypes(K); P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        S = (1 + E @ P.T) / 2
        S[:, sizes < args.min_size] = -np.inf                    # undescribed communities cannot be retrieved
        own = S[np.arange(len(y)), pred]; S2 = S.copy(); S2[np.arange(len(y)), pred] = -np.inf
        marg = own - S2.max(1); top = S.argmax(1)
        held = ~d.fit_mask(len(y)) & node_mask
        maj = {int(c): int(np.bincount(y[pred == c], minlength=len(names)).argmax()) for c in big}
        texts = d.texts()
        judged = {int(c): judge(texts[c], names, DOMAIN[args.dataset], args.judge_model) for c in big}
        resolved = [c for c in big if judged[c] is not None]
        align = np.array([float(P[c] @ name_emb[maj[c]]) for c in big])
        row = dict(dataset=args.dataset, method=method, lam=None if lam == "None" else float(lam), seed=seed, tag=tag, pred_file=base, K=int(K),
                   K_eval=int(len(big)), node_coverage=float(node_mask.mean()),
                   retrieval_acc=float(np.mean(top[node_mask] == pred[node_mask])), retrieval_heldout=float(np.mean(top[held] == pred[held])),
                   Qsem=float(marg[node_mask].mean()),
                   judge_acc=float(np.mean([judged[c] == maj[c] for c in resolved])) if resolved else None,
                   judge_unresolved=int(len(big) - len(resolved)), judge_version="v2", align_cos=float(align.mean()),
                   descriptions={int(c): dict(text=texts[c], majority=names[maj[c]], judged=names[judged[c]] if judged[c] is not None else None,
                                              size=int(sizes[c])) for c in big})
        print(f"  {method:13s} lam={lam:5s} seed={seed}  K={K} evaluated={len(big)} (nodes {row['node_coverage']:.2f})  retr={row['retrieval_acc']:.3f} "
              f"held={row['retrieval_heldout']:.3f} Qsem={row['Qsem']:.4f} judge={row['judge_acc'] if row['judge_acc'] is None else round(row['judge_acc'], 3)} (unresolved {row['judge_unresolved']}) align={row['align_cos']:.3f}")
        fout.write(json.dumps(row) + "\n"); fout.flush()


if __name__ == "__main__":
    main()

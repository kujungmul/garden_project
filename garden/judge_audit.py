"""Corrected LLM judge + robustness audit (D).

Differences from desc_eval.judge (v1):
  * max_tokens 4096 (v1: 16 -> Opus 5's adaptive thinking consumed the budget, empty text)
  * structured output (JSON schema with an integer category) instead of regex over free text
  * stop_reason / raw text stored; failed queries are NEVER cached and are retried; still-failed
    queries are recorded as unresolved (not counted as wrong)
  * passes: p1 = original class order, p2 = seeded random class order, p3 = repeat of p1
    (independent sample), p4 = alternative judge model (Sonnet 5), original order
Writes one record per query to results/judge_audit.jsonl.
Usage: python judge_audit.py [--datasets cora,citeseer,wikics,arxiv] [--passes p1,p2,p3,p4]
"""
import argparse, hashlib, json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from data import load
from desc_eval import ARXIV_NAMES, JUDGE, judge_v2
from llm import client, CACHE
from run import DOMAIN

ROOT = Path(__file__).resolve().parent.parent / "results"
LOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cora,citeseer,wikics,arxiv")
    ap.add_argument("--passes", default="p1,p2,p3,p4")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    passes = args.passes.split(",")
    out = open(ROOT / "judge_audit.jsonl", "a")
    done = set()
    if (ROOT / "judge_audit.jsonl").exists():
        done = {(r["pred_file"], r["cid"], r["pass"]) for r in map(json.loads, open(ROOT / "judge_audit.jsonl"))}
    jobs = []
    for ds in args.datasets.split(","):
        d = load(ds)
        names = [ARXIV_NAMES.get(n, n) for n in d["label_names"]] if ds == "arxiv" else list(d["label_names"])
        for r in map(json.loads, open(ROOT / f"desc_{ds}.jsonl")):
            for cid, info in r["descriptions"].items():
                for ps in passes:
                    if (r["pred_file"], int(cid), ps) in done:
                        continue
                    jobs.append((ds, r, int(cid), info, ps, names))
    print(f"{len(jobs)} judge queries to run", flush=True)

    def run(job):
        ds, r, cid, info, ps, names = job
        model = "claude-sonnet-5" if ps == "p4" else "claude-opus-5"
        if ps == "p2":
            perm = np.random.default_rng(hash((r["pred_file"], cid)) % (2**32)).permutation(len(names))
            shown = [names[i] for i in perm]
        else:
            perm = np.arange(len(names)); shown = names
        rec = judge_v2(info["text"], shown, DOMAIN[ds], model, ps)
        ans = int(perm[rec["answer"]]) if rec["answer"] is not None else None
        row = dict(dataset=ds, method=r["method"], seed=r["seed"], pred_file=r["pred_file"], cid=cid, size=info["size"],
                   text=info["text"], majority=info["majority"], v1_judged=info["judged"], **{"pass": ps},
                   model=model, judged=names[ans] if ans is not None else None, valid=rec["valid"],
                   stop_reason=rec["stop_reason"], attempts=rec["attempts"], output_tokens=rec["output_tokens"],
                   empty=rec["empty"], raw=rec["raw"][:200], correct=(names[ans] == info["majority"]) if ans is not None else None)
        with LOCK:
            out.write(json.dumps(row) + "\n"); out.flush()
        return row

    n = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for row in ex.map(run, jobs):
            n += 1
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

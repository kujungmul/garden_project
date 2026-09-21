"""LLM description generation for communities (Anthropic SDK), with disk cache.

For each community we
  1. pick graph-aware representative documents (internal-degree ranked, diversity-filtered),
  2. ask the model for several candidate one-sentence descriptions,
  3. score candidates with the fixed encoder (contrastive: members vs. a sample of non-members)
     and keep the best.
"""
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data import encode

SCHEMA = {"type": "object",
          "properties": {"candidates": {"type": "array", "items": {"type": "string"}}},
          "required": ["candidates"], "additionalProperties": False}

CACHE = Path(__file__).resolve().parent.parent / "cache"
CACHE.mkdir(exist_ok=True)
PROMPT_VERSION = "v3"      # v3 (audit, 2026-09-21): key = full prompt text + system + model
LEGACY_KEY = False         # True: also look up entries under the v2 key (read-only; used to replay pre-audit runs)
# NOTE: the installed anthropic SDK (1.x) no longer exposes sampling parameters (temperature/top_p), so
# generation is not deterministic; identical inputs can yield different candidates on a cache miss.

SYSTEM = (
    "You write concise, specific, discriminative descriptions of what a group of documents "
    "has in common. A good description names the shared topic precisely enough that a reader "
    "could tell these documents apart from documents on neighbouring topics."
)

USER_TMPL = """Below are {r} representative documents (out of {size} total) from one community of a {domain}.

{docs}

{contrast}Write {ncand} alternative one-sentence descriptions (at most {L} words each) of the topic shared by this community. Each should be a noun phrase like "algorithms for detecting cohesive communities in large attributed networks". Be specific: avoid generic phrases such as "research on computer science". Do not mention the number of documents or the word "community".

Return only JSON: {{"candidates": ["...", "...", "..."]}}"""

_client = None
_lock = threading.Lock()


def client():
    global _client
    with _lock:
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic(max_retries=5)
    return _client


def _cache_path(key):
    return CACHE / f"{key}.json"


def representative_docs(members, nbrs, comm, E, texts, r=10, pool=40, max_chars=500, div=0.8):
    """Graph-aware selection: candidate pool = members with most neighbours inside the
    community (structural core); from the pool greedily take r documents ordered by
    similarity to the community centroid, skipping near-duplicates (cos >= div)."""
    cid = comm[members[0]]
    internal = np.array([np.count_nonzero(comm[nbrs[v]] == cid) for v in members])
    pool_idx = members[np.argsort(-internal, kind="stable")][:pool]
    mu = E[members].mean(0)
    order = pool_idx[np.argsort(-(E[pool_idx] @ mu), kind="stable")]
    chosen = []
    for v in order:
        if not texts[v].strip():
            continue
        if chosen and (E[chosen] @ E[v]).max() >= div:
            continue
        chosen.append(v)
        if len(chosen) >= r:
            break
    if not chosen:
        chosen = list(order[:r])
    return [(int(v), texts[v][:max_chars].replace("\n", " ")) for v in chosen]


def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            c = json.loads(m.group(0)).get("candidates", [])
            c = [str(x).strip() for x in c if str(x).strip()]
            if c:
                return c
        except json.JSONDecodeError:
            pass
    lines = [re.sub(r'^[\-\d\.\)\s"]+', "", l).strip(' "') for l in text.splitlines()]
    return [l for l in lines if len(l.split()) >= 3][:5] or [text.strip()[:200]]


def generate_candidates(docs, size, domain, model, ncand=3, L=20):
    body = "\n\n".join(f"[Document {i+1}] {t}" for i, (_, t) in enumerate(docs))
    msg = USER_TMPL.format(r=len(docs), size=size, domain=domain, docs=body, ncand=ncand, L=L, contrast="")
    # The key covers everything that determines the request: prompt version, model, system prompt,
    # the full user message (documents, community size, ncand, L, domain) and decoding parameters.
    # (v2 keyed only on model+domain+docs, so requests differing in community size shared entries.)
    key = hashlib.sha1((PROMPT_VERSION + model + SYSTEM + msg).encode()).hexdigest()
    p = _cache_path(key)
    if p.exists():
        return json.load(open(p))["candidates"]
    if LEGACY_KEY:                               # pre-audit entries (v2 key: model + domain + docs only)
        p2 = _cache_path(hashlib.sha1(("v2" + model + domain + json.dumps(docs)).encode()).hexdigest())
        if p2.exists():
            return json.load(open(p2))["candidates"]
    out_cfg = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if "haiku" not in model:                     # effort is not accepted by Haiku 4.5
        out_cfg["effort"] = "low"
    resp = client().messages.create(
        model=model, max_tokens=1024, system=SYSTEM,
        output_config=out_cfg,
        messages=[{"role": "user", "content": msg}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        cands = [str(c).strip() for c in json.loads(text)["candidates"] if str(c).strip()]
    except (json.JSONDecodeError, KeyError, TypeError):
        cands = []
    if not cands:
        cands = _parse(text)
    json.dump({"candidates": cands, "raw": text, "stop_reason": resp.stop_reason, "usage": resp.usage.model_dump(),
               "model": model, "prompt_version": PROMPT_VERSION, "size": size, "prompt": msg}, open(p, "w"), indent=1)
    return cands


def select_candidate(cands, members, E, rng, n_neg=512):
    """Contrastive scoring with the fixed encoder: mean s(member, z) - mean s(random non-member, z)."""
    Z = encode(cands)
    n = E.shape[0]
    neg = rng.choice(np.setdiff1d(np.arange(n), members, assume_unique=True), size=min(n_neg, n - len(members)), replace=False)
    pos = (1 + E[members] @ Z.T) / 2
    ngt = (1 + E[neg] @ Z.T) / 2
    score = pos.mean(0) - ngt.mean(0)
    i = int(np.argmax(score))
    return cands[i], Z[i], float(score[i])


class Describer:
    """Generates and caches a description per community id, with lazy updates:
    a community keeps its description while its member set stays Jaccard-close to the
    member set the description was generated from."""

    def __init__(self, ds, nbrs, E, model, domain, lazy_jaccard=0.9, workers=8, seed=0, verbose=True):
        self.ds, self.nbrs, self.E, self.model, self.domain = ds, nbrs, E, model, domain
        self.lazy = lazy_jaccard
        self.workers = workers
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.desc = {}          # cid -> (text, emb, member_set_at_generation)
        self.fit = {}           # cid -> nodes whose documents were shown to the LLM
        self.calls = 0

    def _one(self, cid, members, comm, alive):
        docs = representative_docs(members, self.nbrs, comm, self.E, self.ds["texts"])
        cands = generate_candidates(docs, len(members), self.domain, self.model)
        text, z, score = select_candidate(cands, members, self.E, np.random.default_rng(cid))
        return cid, text, z, set(members.tolist()), [v for v, _ in docs]

    def update(self, comm, alive):
        todo = []
        for cid in alive:
            members = np.flatnonzero(comm == cid)
            if cid in self.desc:
                old = self.desc[cid][2]
                cur = set(members.tolist())
                jac = len(old & cur) / len(old | cur)
                if jac >= self.lazy:
                    continue
            todo.append((cid, members))
        if self.verbose:
            print(f"    [LLM] generating descriptions for {len(todo)}/{len(alive)} communities")
        with ThreadPoolExecutor(self.workers) as ex:
            for cid, text, z, mem, fit in ex.map(lambda t: self._one(t[0], t[1], comm, alive), todo):
                self.desc[cid] = (text, z, mem)
                self.fit[cid] = fit
        self.calls += len(todo)

    def prototypes(self, K):
        P = np.zeros((K, self.E.shape[1]), dtype=np.float32)
        for cid, (t, z, _) in self.desc.items():
            P[cid] = z
        return P

    def fit_mask(self, n):
        m = np.zeros(n, dtype=bool)
        for v in self.fit.values():
            m[v] = True
        return m

    def texts(self):
        return {cid: t for cid, (t, _, _) in self.desc.items()}

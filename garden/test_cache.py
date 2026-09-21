"""Cache-correctness tests (audit B). Run: python test_cache.py
The LLM is replaced by a deterministic fake so that cache behaviour is tested independently
of sampling noise. Each test prints PASS/FAIL; the script exits non-zero if any fails.

 T1  cache-disabled vs cache-enabled runs give identical descriptions/partitions
 T2  deleting and rebuilding the cache reproduces the same results
 T3  a membership change that changes the representative documents cannot hit a stale entry
 T4  two requests whose PROMPTS differ (same docs, different community size) must not share a key
 T5  judge: a failed/empty answer must not be cached
"""
import json, os, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np

import llm

FAILS = []


def check(name, cond, note=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {note}")
    if not cond:
        FAILS.append(name)


class FakeResp:
    def __init__(self, text):
        self.content = [types.SimpleNamespace(type="text", text=text)]
        self.usage = types.SimpleNamespace(model_dump=lambda: {"output_tokens": 1}, output_tokens=1)
        self.stop_reason = "end_turn"


class FakeMessages:
    """Deterministic 'LLM': the answer is a hash of the full prompt it receives."""
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        prompt = kw["messages"][0]["content"]
        self.calls.append(prompt)
        h = abs(hash(prompt)) % 10**6
        return FakeResp(json.dumps({"candidates": [f"topic {h} a", f"topic {h} b", f"topic {h} c"]}))


def install_fake():
    fake = types.SimpleNamespace(messages=FakeMessages())
    llm._client = fake
    return fake.messages


def tiny_dataset(seed=0):
    rng = np.random.default_rng(seed)
    n = 60
    texts = [f"document {i} about {'graphs' if i < 30 else 'databases'} and things {rng.integers(1000)}" for i in range(n)]
    E = rng.normal(size=(n, 8)).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    nbrs = [np.array([(i + 1) % n, (i - 1) % n]) for i in range(n)]
    ds = {"texts": texts, "n": n}
    return ds, nbrs, E


def run_describer(ds, nbrs, E, comm, alive, cache_dir, model="fake-model"):
    llm.CACHE = Path(cache_dir)
    llm.CACHE.mkdir(exist_ok=True, parents=True)
    d = llm.Describer(ds, nbrs, E, model, "test domain", verbose=False, workers=1)
    d.update(comm, alive)
    return d.texts()


def main():
    ds, nbrs, E = tiny_dataset()
    comm = np.array([0] * 30 + [1] * 30); alive = np.array([0, 1])
    tmp = tempfile.mkdtemp()
    real_encode = llm.encode
    llm.encode = lambda texts: np.stack([np.ones(8, dtype=np.float32) * (abs(hash(t)) % 7 + 1) for t in texts])  # deterministic fake encoder

    # T1 cache-disabled (fresh dir each call = no hits) vs cache-enabled (same dir twice)
    fake = install_fake()
    t_nocache1 = run_describer(ds, nbrs, E, comm, alive, os.path.join(tmp, "a"))
    t_nocache2 = run_describer(ds, nbrs, E, comm, alive, os.path.join(tmp, "b"))
    t_cache1 = run_describer(ds, nbrs, E, comm, alive, os.path.join(tmp, "c"))
    ncalls_before = len(fake.calls)
    t_cache2 = run_describer(ds, nbrs, E, comm, alive, os.path.join(tmp, "c"))
    check("T1 cache-disabled == cache-enabled", t_nocache1 == t_nocache2 == t_cache1 == t_cache2,
          f"(second cached run made {len(fake.calls) - ncalls_before} LLM calls; expected 0)")
    check("T1b cached run makes no LLM calls", len(fake.calls) == ncalls_before)

    # T2 delete and rebuild
    shutil.rmtree(os.path.join(tmp, "c"))
    t_rebuilt = run_describer(ds, nbrs, E, comm, alive, os.path.join(tmp, "c"))
    check("T2 delete+rebuild reproduces", t_rebuilt == t_cache1)

    # T3 membership change that changes the representative docs -> different description
    comm2 = comm.copy(); comm2[:15] = 1                       # move 15 nodes: rep docs of both communities change
    t_changed = run_describer(ds, nbrs, E, comm2, alive, os.path.join(tmp, "c"))
    check("T3 changed membership -> changed descriptions", t_changed != t_cache1)

    # T4 same representative docs, different community size -> prompt differs -> must NOT share a key
    fake.calls.clear()
    docs = llm.representative_docs(np.flatnonzero(comm == 0), nbrs, comm, E, ds["texts"])
    llm.CACHE = Path(os.path.join(tmp, "d")); llm.CACHE.mkdir()
    c_small = llm.generate_candidates(docs, size=30, domain="test domain", model="fake-model")
    c_big = llm.generate_candidates(docs, size=3000, domain="test domain", model="fake-model")
    check("T4 different prompt (size) -> different cache entry", len(fake.calls) == 2 and c_small != c_big,
          f"(LLM calls made: {len(fake.calls)}; expected 2)")

    # T5 judge must not cache a failed answer
    import desc_eval
    class FailThenOK:
        def __init__(self): self.n = 0
        def create(self, **kw):
            self.n += 1
            return FakeResp("" if self.n == 1 else '{"category": 2}')
    llm._client = types.SimpleNamespace(messages=FailThenOK())
    llm.CACHE = Path(os.path.join(tmp, "e")); llm.CACHE.mkdir()
    desc_eval.CACHE = llm.CACHE
    a1 = desc_eval.judge("some description", ["x", "y", "z"], "test domain", model="fake-model")
    a2 = desc_eval.judge("some description", ["x", "y", "z"], "test domain", model="fake-model")
    check("T5 failed judge answer is not cached (second call retries and succeeds)", a1 in (None, 1) and a2 == 1,
          f"(answers: {a1}, {a2}; expected first None/-1 then 1)")

    llm.encode = real_encode
    shutil.rmtree(tmp)
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

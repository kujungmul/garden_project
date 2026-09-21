"""Report for the corrected-code experiments + description-weighted centroid (wmean) study.
Reads results/{ds}.jsonl (tags: audit, wmean*, selfw*, ...) and results/desc_audit_{ds}.jsonl.
Writes results/report_wmean.md and results/report_wmean.html.
Usage: python report_wmean.py"""
import html, json
from collections import defaultdict
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "results"
DS = ["cora", "citeseer", "wikics", "arxiv"]
DSN = {"cora": "Cora", "citeseer": "CiteSeer", "wikics": "WikiCS", "arxiv": "ogbn-arxiv", "photo": "Photo", "history": "History"}
K = {"cora": 7, "citeseer": 6, "wikics": 10, "arxiv": 40, "photo": 12, "history": 12}
# (tag, method, label) rows of the main tables
ROWS = [("audit", "louvain", "Graph only (Louvain)"), ("audit", "sn", "SN-modularity"),
        ("audit", "rdm-centroid", "Graph + centroid"), ("audit", "garden", "Graph + description (GARDEN)"),
        ("wmean3", "garden-wmean", "Graph + desc-weighted centroid, τ=3"), ("wmean5", "garden-wmean", "Graph + desc-weighted centroid, τ=5"),
        ("selfw3", "garden-selfw", "Control: self-weighted centroid, τ=3"), ("selfw5", "garden-selfw", "Control: self-weighted centroid, τ=5"),
        ("cagree", "garden-cagree", "GARDEN-cagree: centroid + agreement-licensed non-local moves"),
        ("cexp", "garden-cexp", "Control: non-local moves to nearest centroid (no description)"),
        ("dexp", "garden-dexp", "Control: non-local moves to nearest description (no centroid)")]
CL_METRICS = [("NMI", "NMI"), ("ARI", "ARI"), ("ACC", "ACC (Hungarian)"), ("purity", "purity"), ("Q", "Q (graph modularity)")]
DE_METRICS = [("retrieval_heldout", "retrieval (held-out)"), ("judge_acc", "judge acc (corrected judge v2)"), ("Qsem", "Q_sem (post-hoc descriptions)")]

cl = defaultdict(list); de = defaultdict(list)
for ds in DS:
    p = ROOT / f"{ds}.jsonl"
    if p.exists():
        for r in map(json.loads, open(p)):
            if r.get("lam") in (None, 0.5) and not r.get("shuffled"):
                cl[(ds, r["tag"], r["method"])].append(r)
    p = ROOT / f"desc_audit_{ds}.jsonl"
    if p.exists():
        for r in map(json.loads, open(p)):
            de[(ds, r.get("tag", "final"), r["method"])].append(r)


def vals(store, ds, tag, m, key):
    return [r[key] for r in store.get((ds, tag, m), []) if r.get(key) is not None]


def fmt(v, d=3):
    return f"{np.mean(v):.{d}f}±{np.std(v):.{d}f}" if v else "—"


def table(store, key, higher=True):
    """rows x datasets; bold = best, underline = second best (over rows with n>0)."""
    cells = {}; ns = {}
    for ds in DS:
        col = [(np.mean(vals(store, ds, t, m, key)) if vals(store, ds, t, m, key) else None) for t, m, _ in ROWS]
        order = sorted([i for i, x in enumerate(col) if x is not None], key=lambda i: -col[i] if higher else col[i])
        for i, (t, m, lab) in enumerate(ROWS):
            v = vals(store, ds, t, m, key); ns[(ds, i)] = len(v)
            s = fmt(v)
            if v and order and i == order[0]:
                s = f"**{s}**"
            elif v and len(order) > 1 and i == order[1]:
                s = f"<u>{s}</u>"
            cells[(ds, i)] = s
    return cells, ns


def n_seeds(store, ds):
    return {lab: len(store.get((ds, t, m), [])) for t, m, lab in ROWS}


md = [f"# GARDEN — corrected-code results and description-weighted centroid study ({date.today()})", "",
      "All numbers come from the **audited implementation** (merge-cache fix, v3 description cache key, corrected LLM judge). "
      "Rows labelled `audit` are the four original methods re-run from scratch; `wmean τ` is the new prototype "
      "P_i = Σ_v w_v x_v with w_v = softmax_v(τ·cos(x_v, z_i)) (members weighted by similarity to the community's LLM description z_i); "
      "`selfw τ` is the control that weights by similarity to the plain centroid instead (no description). "
      "Everything else (Louvain init, λ=0.5 with the normalised semantic term, K = #classes, 6-phase cap, reliability gate, encoder all-MiniLM-L6-v2, "
      "descriptions by claude-haiku-4-5, judge claude-opus-5) is identical across rows. Mean ± std over Louvain seeds; bold = best, underline = second best.", ""]

md.append("## 1. Seeds available per setting\n")
md.append("| setting | " + " | ".join(DSN[d] for d in DS) + " |")
md.append("|---|" + "---|" * len(DS))
for t, m, lab in ROWS:
    md.append(f"| {lab} | " + " | ".join(str(len(cl.get((d, t, m), []))) for d in DS) + " |")
md.append("")

md.append("## 2. Clustering quality vs. ground-truth classes\n")
for key, name in CL_METRICS:
    cells, ns = table(cl, key)
    md.append(f"**{name}**\n")
    md.append("| setting | " + " | ".join(f"{DSN[d]} (K={K[d]})" for d in DS) + " |")
    md.append("|---|" + "---|" * len(DS))
    for i, (t, m, lab) in enumerate(ROWS):
        md.append(f"| {lab} | " + " | ".join(cells[(d, i)] for d in DS) + " |")
    md.append("")

md.append("## 3. Description quality of the final partitions (post-hoc descriptions, same generator for every row)\n")
for key, name in DE_METRICS:
    cells, ns = table(de, key)
    md.append(f"**{name}**\n")
    md.append("| setting | " + " | ".join(DSN[d] for d in DS) + " |")
    md.append("|---|" + "---|" * len(DS))
    for i, (t, m, lab) in enumerate(ROWS):
        md.append(f"| {lab} | " + " | ".join(cells[(d, i)] for d in DS) + " |")
    md.append("")

md.append("## 4. Paired comparison against Graph + centroid (same seed)\n")
md.append("Δ = setting − centroid, mean over seeds; (w/n) = seeds on which the setting is strictly better.\n")
for key in ["NMI", "ACC", "purity"]:
    md.append(f"**Δ{key}**\n")
    md.append("| setting | " + " | ".join(DSN[d] for d in DS) + " |")
    md.append("|---|" + "---|" * len(DS))
    for t, m, lab in ROWS[3:]:
        cells = []
        for d in DS:
            cen = {r["seed"]: r[key] for r in cl.get((d, "audit", "rdm-centroid"), [])}
            diffs = [r[key] - cen[r["seed"]] for r in cl.get((d, t, m), []) if r["seed"] in cen]
            cells.append(f"{np.mean(diffs):+.3f} ({sum(x > 0 for x in diffs)}/{len(diffs)})" if diffs else "—")
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    md.append("")

md.append("## 5. Appendix — τ sensitivity on Cora (exploratory; main tables use τ ∈ {3, 5} only)\n")
md.append("| prototype | tag | n | NMI | ARI | ACC | purity | retrieval (held-out) | judge v2 |")
md.append("|---|---|---|---|---|---|---|---|---|")
extra = [("audit", "rdm-centroid", "centroid (τ=0)"), ("audit", "garden", "description")]
extra += sorted([(t, m, f"{m.split('-')[1]} {t}") for (d, t, m) in cl if d == "cora" and m.startswith("garden-")], key=lambda x: (x[1], x[0]))
extra += [("w3_0.4_0.1", "rdm3", "3-factor (0.5,0.4,0.1)"), ("w3_0.3_0.2", "rdm3", "3-factor (0.5,0.3,0.2)"), ("w3_0.25_0.25", "rdm3", "3-factor (0.5,0.25,0.25)")]
for t, m, lab in extra:
    rs = cl.get(("cora", t, m), [])
    if not rs:
        continue
    ds_ = de.get(("cora", t, m), [])
    md.append(f"| {lab} | {t} | {len(rs)} | {fmt([r['NMI'] for r in rs])} | {fmt([r['ARI'] for r in rs])} | {fmt([r['ACC'] for r in rs])} | {fmt([r['purity'] for r in rs])} | "
              f"{fmt([r['retrieval_heldout'] for r in ds_])} | {fmt([r['judge_acc'] for r in ds_ if r.get('judge_acc') is not None])} |")
md.append("")

md += ["## 6. Definitions and caveats", "",
       "- **Q** = (1−λ)·Q_graph + λ·c·Q_sem, λ = 0.5, c = 2 / mean_v(max_j s_vj − mean_j s_vj) fixed per run; Q_sem = mean contrastive margin s(x_v, P_c(v)) − max_j≠c(v) s(x_v, P_j), s = (1+cos)/2.",
       "- Prototypes (of every kind) are frozen within a phase and recomputed at each phase start; at most 6 phases (WikiCS/arXiv hit the cap — results there are 6-phase results, not fixed points).",
       "- **wmean** uses the community's current LLM description z_i only to re-weight its own members; the prototype stays a convex combination of member embeddings. τ = 0 recovers the centroid; large τ approaches the description-nearest member.",
       "- **selfw** replaces z_i by the centroid itself in the weights: any gain it reproduces is a robust-mean effect, not a description effect.",
       "- **judge acc** (v2): Opus 5 maps each post-hoc description to a class name via JSON-schema output; 0 unresolved queries. Pre-audit v1 numbers (max_tokens=16, failures cached as wrong) are not comparable.",
       "- **retrieval (held-out)**: fraction of nodes not shown to the LLM whose nearest post-hoc description is their own community's.",
       "- Bugs fixed before these runs: stale pair-gain cache in the RDM merge step (changed rdm-centroid/GARDEN partitions, ARI 0.80–0.93 vs exact on Cora), generation-cache key without community size, judge truncation/failure caching. See `results/audit_report.md`.", ""]

(ROOT / "report_wmean.md").write_text("\n".join(md))

# --- minimal HTML rendering of the markdown (headings, paragraphs, tables, bold/underline) ---
def inline(s):
    s = html.escape(s, quote=False).replace("&lt;u&gt;", "<u>").replace("&lt;/u&gt;", "</u>")
    out = ""; b = False
    while "**" in s:
        i = s.index("**"); out += s[:i] + ("</b>" if b else "<b>"); b = not b; s = s[i + 2:]
    return out + s

body = []; in_table = False; in_list = False
for line in md:
    if line.startswith("|"):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set(line.replace("|", "").strip()) <= set("-: "):
            continue
        if not in_table:
            body.append("<table>"); in_table = True; body.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>"); continue
        body.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>"); continue
    if in_table:
        body.append("</table>"); in_table = False
    if line.startswith("- "):
        if not in_list:
            body.append("<ul>"); in_list = True
        body.append(f"<li>{inline(line[2:])}</li>"); continue
    if in_list:
        body.append("</ul>"); in_list = False
    if line.startswith("# "):
        body.append(f"<h1>{inline(line[2:])}</h1>")
    elif line.startswith("## "):
        body.append(f"<h2>{inline(line[3:])}</h2>")
    elif line.strip():
        body.append(f"<p>{inline(line)}</p>")
if in_table:
    body.append("</table>")
if in_list:
    body.append("</ul>")
css = ("body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#222;line-height:1.45}"
       "table{border-collapse:collapse;margin:8px 0 22px;font-size:13.5px}th,td{border:1px solid #ddd;padding:5px 9px;text-align:right;white-space:nowrap}"
       "th:first-child,td:first-child{text-align:left}th{background:#f4f4f4}tr:nth-child(even) td{background:#fafafa}h2{margin-top:34px}u{text-decoration:underline}")
(ROOT / "report_wmean.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>GARDEN corrected results</title><style>{css}</style></head><body>{''.join(body)}</body></html>")
print("wrote", ROOT / "report_wmean.md", "and report_wmean.html")

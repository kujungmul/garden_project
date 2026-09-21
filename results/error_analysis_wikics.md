# Error analysis: misassigned nodes in the corrected graph+centroid partitions

Partition: `rdm-centroid`, tag `audit`, seeds 0–4. 'wrong' = majority GT class of the node's community ≠ its GT class. Candidates = own community ∪ neighbours' communities. Signals restricted to candidates. Per-node data: `error_nodes_{ds}_{seed}.csv`.

## wikics (n=11701 nodes, 5 seeds pooled)

- wrong nodes: 3927 per seed (33.6%)
- of the wrong nodes, a community of the right class is among the candidates for 45.7% (the rest cannot be fixed by any local move; their class has no adjacent community)
- boundary check — centroid margin (top-1 − top-2 among candidates): wrong nodes median 0.083, correct nodes median 0.154; wrong nodes with margin in the lowest quartile of all nodes: 36.0%
- fraction of neighbours in own community: wrong nodes 0.58, correct nodes 0.78

**Which signal points to the right class?** (fraction of nodes whose pick has the node's GT class)

| nodes | centroid | description | graph vote | current |
|---|---|---|---|---|
| wrong (fixable: right class available) (n=1796/seed) | 0.141 | 0.277 | 0.163 | 0.000 |
| correct (n=7774/seed) | 0.941 | 0.875 | 0.957 | 1.000 |

**Correction rules applied to all nodes** (per seed averages; net = fixes − breaks; a rule helps only if net > 0)

| rule | moved | fixes | breaks | net | net as % of wrong |
|---|---|---|---|---|---|
| R1 move to nearest description | 2505 | 498 | 974 | -476 | -12.1% |
| R2 move to graph vote | 991 | 293 | 332 | -39 | -1.0% |
| R3 move to nearest centroid | 1076 | 253 | 462 | -209 | -5.3% |
| R4 desc == graph, both ≠ current | 168 | 72 | 44 | +28 | +0.7% |
| R5 desc == centroid, both ≠ current | 530 | 145 | 204 | -59 | -1.5% |
| R6 desc == graph == centroid ≠ current | 1 | 0 | 0 | +0 | +0.0% |
| R4' as R4 but only low-margin nodes (bottom quartile) | 107 | 50 | 23 | +27 | +0.7% |

**Most frequent confusions (GT class → class of assigned community), per seed**

| GT class | assigned to | count/seed |
|---|---|---|
| web technology | distributed computing architecture | 326 |
| computer architecture | operating systems | 310 |
| computational linguistics | programming language topics | 271 |
| computer security | internet protocols | 252 |
| operating systems | computer architecture | 237 |
| computer file systems | operating systems | 185 |
| computer security | operating systems | 145 |
| databases | distributed computing architecture | 132 |

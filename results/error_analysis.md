# Error analysis: misassigned nodes in the corrected graph+centroid partitions

Partition: `rdm-centroid`, tag `audit`, seeds 0–4. 'wrong' = majority GT class of the node's community ≠ its GT class. Candidates = own community ∪ neighbours' communities. Signals restricted to candidates. Per-node data: `error_nodes_{ds}_{seed}.csv`.

## cora (n=2708 nodes, 5 seeds pooled)

- wrong nodes: 767 per seed (28.3%)
- of the wrong nodes, a community of the right class is among the candidates for 13.2% (the rest cannot be fixed by any local move; their class has no adjacent community)
- boundary check — centroid margin (top-1 − top-2 among candidates): wrong nodes median 1.000, correct nodes median 1.000; wrong nodes with margin in the lowest quartile of all nodes: 100.0%
- fraction of neighbours in own community: wrong nodes 0.92, correct nodes 0.95

**Which signal points to the right class?** (fraction of nodes whose pick has the node's GT class)

| nodes | centroid | description | graph vote | current |
|---|---|---|---|---|
| wrong (fixable: right class available) (n=101/seed) | 0.295 | 0.349 | 0.093 | 0.000 |
| correct (n=1941/seed) | 0.982 | 0.976 | 0.991 | 1.000 |

**Correction rules applied to all nodes** (per seed averages; net = fixes − breaks; a rule helps only if net > 0)

| rule | moved | fixes | breaks | net | net as % of wrong |
|---|---|---|---|---|---|
| R1 move to nearest description | 107 | 35 | 47 | -12 | -1.6% |
| R2 move to graph vote | 29 | 9 | 17 | -7 | -0.9% |
| R3 move to nearest centroid | 81 | 30 | 34 | -5 | -0.6% |
| R4 desc == graph, both ≠ current | 2 | 1 | 1 | -1 | -0.1% |
| R5 desc == centroid, both ≠ current | 55 | 22 | 23 | -1 | -0.2% |
| R6 desc == graph == centroid ≠ current | 0 | 0 | 0 | +0 | +0.0% |
| R4' as R4 but only low-margin nodes (bottom quartile) | 2 | 1 | 1 | -1 | -0.1% |

**Most frequent confusions (GT class → class of assigned community), per seed**

| GT class | assigned to | count/seed |
|---|---|---|
| Rule Learning | Theory | 127 |
| Probabilistic Methods | Neural Networks | 84 |
| Case Based | Theory | 83 |
| Neural Networks | Theory | 77 |
| Neural Networks | Probabilistic Methods | 60 |
| Rule Learning | Neural Networks | 40 |
| Theory | Neural Networks | 37 |
| Probabilistic Methods | Theory | 33 |

## citeseer (n=3186 nodes, 5 seeds pooled)

- wrong nodes: 1558 per seed (48.9%)
- of the wrong nodes, a community of the right class is among the candidates for 2.3% (the rest cannot be fixed by any local move; their class has no adjacent community)
- boundary check — centroid margin (top-1 − top-2 among candidates): wrong nodes median 1.000, correct nodes median 1.000; wrong nodes with margin in the lowest quartile of all nodes: 100.0%
- fraction of neighbours in own community: wrong nodes 0.94, correct nodes 0.98

**Which signal points to the right class?** (fraction of nodes whose pick has the node's GT class)

| nodes | centroid | description | graph vote | current |
|---|---|---|---|---|
| wrong (fixable: right class available) (n=36/seed) | 0.506 | 0.416 | 0.022 | 0.000 |
| correct (n=1628/seed) | 0.996 | 0.993 | 0.999 | 1.000 |

**Correction rules applied to all nodes** (per seed averages; net = fixes − breaks; a rule helps only if net > 0)

| rule | moved | fixes | breaks | net | net as % of wrong |
|---|---|---|---|---|---|
| R1 move to nearest description | 35 | 15 | 11 | +4 | +0.2% |
| R2 move to graph vote | 3 | 1 | 2 | -1 | -0.1% |
| R3 move to nearest centroid | 33 | 18 | 7 | +11 | +0.7% |
| R4 desc == graph, both ≠ current | 0 | 0 | 0 | +0 | +0.0% |
| R5 desc == centroid, both ≠ current | 22 | 12 | 4 | +7 | +0.5% |
| R6 desc == graph == centroid ≠ current | 0 | 0 | 0 | +0 | +0.0% |
| R4' as R4 but only low-margin nodes (bottom quartile) | 0 | 0 | 0 | +0 | +0.0% |

**Most frequent confusions (GT class → class of assigned community), per seed**

| GT class | assigned to | count/seed |
|---|---|---|
| database (DB) | machine learning(ML) | 172 |
| machine learning(ML) | database (DB) | 135 |
| Agents | machine learning(ML) | 117 |
| human-computer interaction (HCI) | machine learning(ML) | 110 |
| information retrieval (IR) | machine learning(ML) | 106 |
| machine learning(ML) | information retrieval (IR) | 100 |
| information retrieval (IR) | database (DB) | 98 |
| human-computer interaction (HCI) | database (DB) | 84 |

## Global reassignment (nearest prototype among ALL communities, ignoring graph adjacency)

| dataset | rule | moved/seed | fixes | breaks | net | net as % of wrong |
|---|---|---|---|---|---|---|
| cora | global desc | 885 | 242 | 400 | -158 | -20.6% |
| cora | global cent | 666 | 244 | 241 | +2 | +0.3% |
| cora | global desc==cent | 422 | 182 | 137 | +46 | +6.0% |
| cora | global desc==cent & margin>0.05 | 286 | 135 | 86 | +48 | +6.3% |
| citeseer | global desc | 1636 | 683 | 299 | +384 | +24.7% |
| citeseer | global cent | 1480 | 749 | 186 | +564 | +36.2% |
| citeseer | global desc==cent | 987 | 559 | 104 | +455 | +29.2% |
| citeseer | global desc==cent & margin>0.05 | 726 | 443 | 67 | +377 | +24.2% |

## Reading of the results

1. **The errors are not boundary errors.** On Cora only 13 % of the wrong nodes (101 of 767 per seed) have a community of their own class among the communities of their neighbours; on CiteSeer 2 % (36 of 1,558). The other 87–98 % sit inside a neighbourhood that is entirely in a community of another class (92–94 % of their neighbours are in their own community). No node-move rule — with any signal, including an LLM — can reach them; the errors are community-level (a "Theory" community that absorbs Rule-Learning and Case-Based papers; on CiteSeer, DB/ML/IR/HCI/Agents papers pooled into a few large communities).
2. **On the few fixable nodes, no local correction rule has a positive net effect on Cora** (fixes ≈ breaks for description, centroid and graph vote alike); on CiteSeer the net is +0.2–0.7 % of the wrong nodes. Descriptions do not help here.
3. **Global (graph-ignoring) reassignment is a different story.** Moving every node to its nearest prototype among all K communities fixes many more nodes than it breaks on CiteSeer (centroid alone: +564 per seed, 36 % of the wrong nodes) — the CiteSeer graph is simply a weak guide compared with the text. On Cora the centroid alone is neutral (+2), but reassigning only when **nearest description and nearest centroid agree** gives +46 per seed (+6 % of wrong nodes, ≈ +1.7 pt ACC), because the agreement acts as a confidence filter (fix/break 182/137 vs 244/241 for the centroid alone). This is the only place in the whole study where the description adds information beyond the centroid — as an *agreement gate* for non-local moves, not as a prototype.
4. Consequence for the method: the gain requires letting a node leave its graph neighbourhood, which Louvain-style local moves never do. A candidate-expansion rule (allow the move to the agreed community as an extra candidate; the objective is unchanged and still pays the modularity penalty) is being tested as `garden-cagree`.

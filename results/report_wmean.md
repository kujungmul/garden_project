# GARDEN — corrected-code results and description-weighted centroid study (2026-09-21)

All numbers come from the **audited implementation** (merge-cache fix, v3 description cache key, corrected LLM judge). Rows labelled `audit` are the four original methods re-run from scratch; `wmean τ` is the new prototype P_i = Σ_v w_v x_v with w_v = softmax_v(τ·cos(x_v, z_i)) (members weighted by similarity to the community's LLM description z_i); `selfw τ` is the control that weights by similarity to the plain centroid instead (no description). Everything else (Louvain init, λ=0.5 with the normalised semantic term, K = #classes, 6-phase cap, reliability gate, encoder all-MiniLM-L6-v2, descriptions by claude-haiku-4-5, judge claude-opus-5) is identical across rows. Mean ± std over Louvain seeds; bold = best, underline = second best.

## 1. Seeds available per setting

| setting | Cora | CiteSeer | WikiCS | ogbn-arxiv |
|---|---|---|---|---|
| Graph only (Louvain) | 5 | 5 | 5 | 5 |
| SN-modularity | 5 | 5 | 5 | 5 |
| Graph + centroid | 5 | 5 | 5 | 5 |
| Graph + description (GARDEN) | 5 | 5 | 5 | 5 |
| Graph + desc-weighted centroid, τ=3 | 5 | 5 | 5 | 5 |
| Graph + desc-weighted centroid, τ=5 | 5 | 5 | 5 | 5 |
| Control: self-weighted centroid, τ=3 | 5 | 5 | 5 | 5 |
| Control: self-weighted centroid, τ=5 | 0 | 0 | 0 | 0 |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | 5 | 5 | 5 | 5 |
| Control: non-local moves to nearest centroid (no description) | 5 | 5 | 5 | 5 |
| Control: non-local moves to nearest description (no centroid) | 5 | 5 | 5 | 5 |

## 2. Clustering quality vs. ground-truth classes

**NMI**

| setting | Cora (K=7) | CiteSeer (K=6) | WikiCS (K=10) | ogbn-arxiv (K=40) |
|---|---|---|---|---|
| Graph only (Louvain) | 0.381±0.015 | 0.135±0.016 | 0.397±0.008 | 0.397±0.003 |
| SN-modularity | 0.418±0.015 | 0.202±0.017 | 0.424±0.011 | 0.406±0.004 |
| Graph + centroid | 0.505±0.010 | 0.239±0.010 | 0.461±0.013 | 0.457±0.004 |
| Graph + description (GARDEN) | 0.494±0.021 | 0.241±0.017 | 0.435±0.012 | 0.430±0.006 |
| Graph + desc-weighted centroid, τ=3 | 0.518±0.007 | 0.243±0.018 | 0.459±0.005 | 0.462±0.002 |
| Graph + desc-weighted centroid, τ=5 | 0.505±0.014 | 0.243±0.009 | **0.467±0.013** | 0.458±0.001 |
| Control: self-weighted centroid, τ=3 | 0.515±0.006 | 0.239±0.028 | <u>0.466±0.013</u> | 0.457±0.004 |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | **0.562±0.015** | **0.418±0.011** | 0.462±0.006 | 0.478±0.001 |
| Control: non-local moves to nearest centroid (no description) | <u>0.556±0.011</u> | <u>0.404±0.027</u> | 0.455±0.011 | <u>0.482±0.003</u> |
| Control: non-local moves to nearest description (no centroid) | 0.554±0.015 | 0.402±0.028 | 0.463±0.006 | **0.482±0.006** |

**ARI**

| setting | Cora (K=7) | CiteSeer (K=6) | WikiCS (K=10) | ogbn-arxiv (K=40) |
|---|---|---|---|---|
| Graph only (Louvain) | 0.286±0.018 | 0.092±0.011 | 0.360±0.006 | 0.255±0.008 |
| SN-modularity | 0.329±0.016 | 0.127±0.011 | 0.361±0.010 | 0.260±0.009 |
| Graph + centroid | 0.459±0.014 | 0.193±0.024 | 0.410±0.019 | 0.303±0.020 |
| Graph + description (GARDEN) | 0.438±0.030 | 0.189±0.020 | 0.385±0.019 | 0.278±0.015 |
| Graph + desc-weighted centroid, τ=3 | 0.481±0.010 | 0.194±0.019 | 0.415±0.006 | **0.334±0.025** |
| Graph + desc-weighted centroid, τ=5 | 0.452±0.008 | 0.187±0.025 | **0.424±0.017** | <u>0.330±0.025</u> |
| Control: self-weighted centroid, τ=3 | 0.476±0.012 | 0.196±0.029 | <u>0.420±0.015</u> | 0.305±0.020 |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | **0.519±0.028** | **0.425±0.012** | 0.414±0.008 | 0.291±0.023 |
| Control: non-local moves to nearest centroid (no description) | <u>0.508±0.020</u> | 0.401±0.034 | 0.405±0.013 | 0.282±0.015 |
| Control: non-local moves to nearest description (no centroid) | 0.501±0.033 | <u>0.406±0.032</u> | 0.414±0.007 | 0.304±0.037 |

**ACC (Hungarian)**

| setting | Cora (K=7) | CiteSeer (K=6) | WikiCS (K=10) | ogbn-arxiv (K=40) |
|---|---|---|---|---|
| Graph only (Louvain) | 0.517±0.040 | 0.360±0.028 | 0.508±0.013 | 0.370±0.010 |
| SN-modularity | 0.569±0.038 | 0.402±0.014 | 0.525±0.011 | 0.376±0.010 |
| Graph + centroid | 0.680±0.019 | 0.471±0.013 | 0.571±0.024 | 0.401±0.015 |
| Graph + description (GARDEN) | 0.675±0.019 | 0.477±0.016 | **0.587±0.021** | 0.390±0.007 |
| Graph + desc-weighted centroid, τ=3 | 0.703±0.018 | 0.477±0.021 | 0.568±0.009 | **0.431±0.017** |
| Graph + desc-weighted centroid, τ=5 | 0.669±0.009 | 0.472±0.022 | 0.578±0.026 | <u>0.428±0.021</u> |
| Control: self-weighted centroid, τ=3 | 0.700±0.017 | 0.470±0.045 | <u>0.579±0.025</u> | 0.407±0.014 |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | <u>0.706±0.020</u> | **0.651±0.006** | 0.568±0.005 | 0.396±0.017 |
| Control: non-local moves to nearest centroid (no description) | **0.716±0.015** | 0.624±0.020 | 0.563±0.010 | 0.394±0.012 |
| Control: non-local moves to nearest description (no centroid) | 0.701±0.024 | <u>0.636±0.018</u> | 0.569±0.003 | 0.410±0.026 |

**purity**

| setting | Cora (K=7) | CiteSeer (K=6) | WikiCS (K=10) | ogbn-arxiv (K=40) |
|---|---|---|---|---|
| Graph only (Louvain) | 0.584±0.025 | 0.407±0.012 | 0.598±0.007 | 0.502±0.006 |
| SN-modularity | 0.605±0.007 | 0.452±0.020 | 0.603±0.010 | 0.507±0.006 |
| Graph + centroid | 0.717±0.007 | 0.511±0.020 | 0.664±0.014 | 0.523±0.008 |
| Graph + description (GARDEN) | 0.695±0.025 | 0.520±0.010 | 0.645±0.015 | 0.514±0.014 |
| Graph + desc-weighted centroid, τ=3 | 0.726±0.003 | 0.520±0.014 | 0.665±0.008 | 0.533±0.004 |
| Graph + desc-weighted centroid, τ=5 | 0.710±0.010 | 0.519±0.017 | **0.672±0.016** | 0.528±0.003 |
| Control: self-weighted centroid, τ=3 | 0.725±0.002 | 0.508±0.039 | <u>0.671±0.016</u> | 0.524±0.008 |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | **0.754±0.014** | **0.689±0.013** | 0.667±0.006 | 0.561±0.004 |
| Control: non-local moves to nearest centroid (no description) | <u>0.751±0.010</u> | 0.672±0.031 | 0.662±0.012 | **0.580±0.005** |
| Control: non-local moves to nearest description (no centroid) | 0.748±0.014 | <u>0.679±0.027</u> | 0.668±0.005 | <u>0.570±0.007</u> |

**Q (graph modularity)**

| setting | Cora (K=7) | CiteSeer (K=6) | WikiCS (K=10) | ogbn-arxiv (K=40) |
|---|---|---|---|---|
| Graph only (Louvain) | <u>0.766±0.002</u> | <u>0.788±0.001</u> | **0.645±0.003** | **0.712±0.002** |
| SN-modularity | **0.769±0.002** | **0.789±0.003** | <u>0.643±0.002</u> | <u>0.711±0.001</u> |
| Graph + centroid | 0.740±0.005 | 0.775±0.013 | 0.615±0.004 | 0.686±0.002 |
| Graph + description (GARDEN) | 0.733±0.009 | 0.781±0.009 | 0.584±0.011 | 0.666±0.002 |
| Graph + desc-weighted centroid, τ=3 | 0.740±0.003 | 0.781±0.021 | 0.616±0.004 | 0.681±0.005 |
| Graph + desc-weighted centroid, τ=5 | 0.734±0.008 | 0.780±0.014 | 0.614±0.004 | 0.679±0.006 |
| Control: self-weighted centroid, τ=3 | 0.740±0.003 | 0.763±0.027 | 0.615±0.005 | 0.685±0.002 |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | 0.715±0.002 | 0.750±0.009 | 0.613±0.004 | 0.655±0.006 |
| Control: non-local moves to nearest centroid (no description) | 0.708±0.002 | 0.757±0.008 | 0.611±0.003 | 0.648±0.002 |
| Control: non-local moves to nearest description (no centroid) | 0.713±0.006 | 0.747±0.016 | 0.612±0.004 | 0.650±0.007 |

## 3. Description quality of the final partitions (post-hoc descriptions, same generator for every row)

**retrieval (held-out)**

| setting | Cora | CiteSeer | WikiCS | ogbn-arxiv |
|---|---|---|---|---|
| Graph only (Louvain) | 0.513±0.025 | 0.379±0.017 | 0.505±0.016 | 0.447±0.009 |
| SN-modularity | 0.528±0.037 | 0.420±0.025 | 0.526±0.020 | 0.462±0.016 |
| Graph + centroid | 0.666±0.012 | 0.479±0.013 | 0.613±0.019 | 0.576±0.011 |
| Graph + description (GARDEN) | 0.651±0.010 | 0.489±0.012 | **0.685±0.012** | 0.633±0.008 |
| Graph + desc-weighted centroid, τ=3 | 0.652±0.014 | 0.470±0.016 | 0.635±0.014 | 0.610±0.010 |
| Graph + desc-weighted centroid, τ=5 | 0.642±0.023 | 0.496±0.009 | 0.631±0.026 | 0.611±0.022 |
| Control: self-weighted centroid, τ=3 | 0.650±0.021 | 0.474±0.021 | 0.639±0.011 | 0.588±0.007 |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | 0.720±0.009 | <u>0.685±0.016</u> | 0.666±0.012 | 0.627±0.024 |
| Control: non-local moves to nearest centroid (no description) | **0.741±0.029** | **0.695±0.029** | 0.652±0.018 | <u>0.638±0.007</u> |
| Control: non-local moves to nearest description (no centroid) | <u>0.725±0.022</u> | 0.677±0.028 | <u>0.670±0.013</u> | **0.648±0.013** |

**judge acc (corrected judge v2)**

| setting | Cora | CiteSeer | WikiCS | ogbn-arxiv |
|---|---|---|---|---|
| Graph only (Louvain) | 0.743±0.167 | 0.767±0.133 | 0.780±0.075 | 0.765±0.034 |
| SN-modularity | 0.686±0.190 | 0.733±0.133 | 0.840±0.080 | 0.725±0.035 |
| Graph + centroid | 0.857±0.090 | 0.833±0.105 | <u>0.880±0.075</u> | 0.742±0.030 |
| Graph + description (GARDEN) | 0.800±0.114 | 0.867±0.067 | 0.840±0.049 | 0.781±0.027 |
| Graph + desc-weighted centroid, τ=3 | 0.800±0.114 | <u>0.900±0.082</u> | 0.820±0.040 | 0.756±0.039 |
| Graph + desc-weighted centroid, τ=5 | **0.914±0.070** | 0.867±0.125 | 0.840±0.080 | 0.766±0.057 |
| Control: self-weighted centroid, τ=3 | 0.800±0.114 | 0.867±0.125 | **0.900±0.063** | 0.799±0.052 |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | <u>0.886±0.057</u> | 0.867±0.067 | 0.840±0.049 | 0.804±0.025 |
| Control: non-local moves to nearest centroid (no description) | 0.771±0.070 | 0.900±0.082 | 0.860±0.080 | **0.860±0.037** |
| Control: non-local moves to nearest description (no centroid) | 0.857±0.090 | **0.933±0.082** | 0.860±0.049 | <u>0.859±0.056</u> |

**Q_sem (post-hoc descriptions)**

| setting | Cora | CiteSeer | WikiCS | ogbn-arxiv |
|---|---|---|---|---|
| Graph only (Louvain) | 0.006±0.004 | -0.027±0.004 | 0.001±0.003 | -0.013±0.002 |
| SN-modularity | 0.010±0.005 | -0.017±0.005 | 0.005±0.003 | -0.010±0.003 |
| Graph + centroid | 0.034±0.001 | -0.004±0.004 | 0.018±0.003 | 0.013±0.001 |
| Graph + description (GARDEN) | 0.032±0.001 | -0.003±0.004 | **0.029±0.002** | <u>0.022±0.001</u> |
| Graph + desc-weighted centroid, τ=3 | 0.032±0.004 | -0.006±0.004 | 0.021±0.002 | 0.017±0.002 |
| Graph + desc-weighted centroid, τ=5 | 0.030±0.003 | -0.000±0.004 | 0.020±0.004 | 0.019±0.004 |
| Control: self-weighted centroid, τ=3 | 0.033±0.003 | -0.003±0.005 | 0.022±0.002 | 0.015±0.002 |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | 0.045±0.002 | **0.044±0.004** | 0.025±0.001 | 0.021±0.004 |
| Control: non-local moves to nearest centroid (no description) | **0.051±0.003** | <u>0.044±0.006</u> | 0.024±0.002 | 0.022±0.002 |
| Control: non-local moves to nearest description (no centroid) | <u>0.047±0.003</u> | 0.042±0.005 | <u>0.027±0.002</u> | **0.023±0.002** |

## 4. Paired comparison against Graph + centroid (same seed)

Δ = setting − centroid, mean over seeds; (w/n) = seeds on which the setting is strictly better.

**ΔNMI**

| setting | Cora | CiteSeer | WikiCS | ogbn-arxiv |
|---|---|---|---|---|
| Graph + description (GARDEN) | -0.011 (1/5) | +0.002 (3/5) | -0.026 (0/5) | -0.026 (0/5) |
| Graph + desc-weighted centroid, τ=3 | +0.012 (5/5) | +0.004 (2/5) | -0.003 (3/5) | +0.006 (5/5) |
| Graph + desc-weighted centroid, τ=5 | -0.000 (3/5) | +0.004 (3/5) | +0.006 (4/5) | +0.002 (3/5) |
| Control: self-weighted centroid, τ=3 | +0.009 (5/5) | -0.000 (3/5) | +0.004 (4/5) | +0.001 (3/5) |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | +0.056 (5/5) | +0.179 (5/5) | +0.000 (3/5) | +0.021 (5/5) |
| Control: non-local moves to nearest centroid (no description) | +0.051 (5/5) | +0.165 (5/5) | -0.007 (2/5) | +0.025 (5/5) |
| Control: non-local moves to nearest description (no centroid) | +0.048 (5/5) | +0.163 (5/5) | +0.001 (3/5) | +0.026 (5/5) |

**ΔACC**

| setting | Cora | CiteSeer | WikiCS | ogbn-arxiv |
|---|---|---|---|---|
| Graph + description (GARDEN) | -0.005 (2/5) | +0.006 (4/5) | +0.015 (4/5) | -0.011 (2/5) |
| Graph + desc-weighted centroid, τ=3 | +0.023 (4/5) | +0.006 (3/5) | -0.003 (4/5) | +0.031 (5/5) |
| Graph + desc-weighted centroid, τ=5 | -0.011 (2/5) | +0.001 (4/5) | +0.006 (4/5) | +0.027 (4/5) |
| Control: self-weighted centroid, τ=3 | +0.019 (5/5) | -0.001 (3/5) | +0.007 (5/5) | +0.007 (4/5) |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | +0.025 (5/5) | +0.180 (5/5) | -0.004 (3/5) | -0.005 (1/5) |
| Control: non-local moves to nearest centroid (no description) | +0.036 (5/5) | +0.153 (5/5) | -0.009 (2/5) | -0.006 (2/5) |
| Control: non-local moves to nearest description (no centroid) | +0.021 (5/5) | +0.164 (5/5) | -0.002 (4/5) | +0.009 (4/5) |

**Δpurity**

| setting | Cora | CiteSeer | WikiCS | ogbn-arxiv |
|---|---|---|---|---|
| Graph + description (GARDEN) | -0.022 (1/5) | +0.009 (4/5) | -0.019 (1/5) | -0.008 (2/5) |
| Graph + desc-weighted centroid, τ=3 | +0.009 (4/5) | +0.009 (4/5) | +0.001 (4/5) | +0.010 (5/5) |
| Graph + desc-weighted centroid, τ=5 | -0.007 (2/5) | +0.007 (4/5) | +0.008 (5/5) | +0.006 (3/5) |
| Control: self-weighted centroid, τ=3 | +0.008 (5/5) | -0.003 (3/5) | +0.007 (5/5) | +0.002 (3/5) |
| Control: self-weighted centroid, τ=5 | — | — | — | — |
| GARDEN-cagree: centroid + agreement-licensed non-local moves | +0.037 (5/5) | +0.178 (5/5) | +0.003 (4/5) | +0.038 (5/5) |
| Control: non-local moves to nearest centroid (no description) | +0.034 (5/5) | +0.161 (5/5) | -0.002 (3/5) | +0.057 (5/5) |
| Control: non-local moves to nearest description (no centroid) | +0.031 (5/5) | +0.168 (5/5) | +0.004 (4/5) | +0.048 (5/5) |

## 5. Appendix — τ sensitivity on Cora (exploratory; main tables use τ ∈ {3, 5} only)

| prototype | tag | n | NMI | ARI | ACC | purity | retrieval (held-out) | judge v2 |
|---|---|---|---|---|---|---|---|---|
| centroid (τ=0) | audit | 5 | 0.505±0.010 | 0.459±0.014 | 0.680±0.019 | 0.717±0.007 | 0.666±0.012 | 0.857±0.090 |
| description | audit | 5 | 0.494±0.021 | 0.438±0.030 | 0.675±0.019 | 0.695±0.025 | 0.651±0.010 | 0.800±0.114 |
| cagree cagree | cagree | 5 | 0.562±0.015 | 0.519±0.028 | 0.706±0.020 | 0.754±0.014 | 0.720±0.009 | 0.886±0.057 |
| cexp cexp | cexp | 5 | 0.556±0.011 | 0.508±0.020 | 0.716±0.015 | 0.751±0.010 | 0.741±0.029 | 0.771±0.070 |
| dexp dexp | dexp | 5 | 0.554±0.015 | 0.501±0.033 | 0.701±0.024 | 0.748±0.014 | 0.725±0.022 | 0.857±0.090 |
| selfw selfw10 | selfw10 | 5 | 0.523±0.010 | 0.490±0.009 | 0.714±0.018 | 0.729±0.003 | 0.659±0.022 | 0.800±0.114 |
| selfw selfw3 | selfw3 | 5 | 0.515±0.006 | 0.476±0.012 | 0.700±0.017 | 0.725±0.002 | 0.650±0.021 | 0.800±0.114 |
| shift shift0.5 | shift0.5 | 5 | 0.500±0.019 | 0.450±0.030 | 0.681±0.036 | 0.699±0.025 | 0.651±0.013 | 0.714±0.090 |
| trim trim0.5 | trim0.5 | 5 | 0.500±0.013 | 0.451±0.020 | 0.684±0.024 | 0.703±0.016 | 0.662±0.011 | 0.829±0.107 |
| wmargin wmargin10 | wmargin10 | 5 | 0.487±0.011 | 0.417±0.013 | 0.655±0.035 | 0.677±0.022 | 0.665±0.010 | 0.743±0.057 |
| wmean wmean1 | wmean1 | 5 | 0.513±0.008 | 0.472±0.015 | 0.694±0.020 | 0.724±0.004 | 0.668±0.009 | 0.800±0.070 |
| wmean wmean10 | wmean10 | 5 | 0.494±0.013 | 0.431±0.017 | 0.670±0.009 | 0.692±0.007 | 0.650±0.027 | 0.743±0.107 |
| wmean wmean2 | wmean2 | 5 | 0.506±0.017 | 0.460±0.020 | 0.696±0.019 | 0.714±0.012 | 0.648±0.013 | 0.743±0.107 |
| wmean wmean3 | wmean3 | 5 | 0.518±0.007 | 0.481±0.010 | 0.703±0.018 | 0.726±0.003 | 0.652±0.014 | 0.800±0.114 |
| wmean wmean5 | wmean5 | 5 | 0.505±0.014 | 0.452±0.008 | 0.669±0.009 | 0.710±0.010 | 0.642±0.023 | 0.914±0.070 |

## 6. Definitions and caveats

- **Q** = (1−λ)·Q_graph + λ·c·Q_sem, λ = 0.5, c = 2 / mean_v(max_j s_vj − mean_j s_vj) fixed per run; Q_sem = mean contrastive margin s(x_v, P_c(v)) − max_j≠c(v) s(x_v, P_j), s = (1+cos)/2.
- Prototypes (of every kind) are frozen within a phase and recomputed at each phase start; at most 6 phases (WikiCS/arXiv hit the cap — results there are 6-phase results, not fixed points).
- **wmean** uses the community's current LLM description z_i only to re-weight its own members; the prototype stays a convex combination of member embeddings. τ = 0 recovers the centroid; large τ approaches the description-nearest member.
- **selfw** replaces z_i by the centroid itself in the weights: any gain it reproduces is a robust-mean effect, not a description effect.
- **judge acc** (v2): Opus 5 maps each post-hoc description to a class name via JSON-schema output; 0 unresolved queries. Pre-audit v1 numbers (max_tokens=16, failures cached as wrong) are not comparable.
- **retrieval (held-out)**: fraction of nodes not shown to the LLM whose nearest post-hoc description is their own community's.
- Bugs fixed before these runs: stale pair-gain cache in the RDM merge step (changed rdm-centroid/GARDEN partitions, ARI 0.80–0.93 vs exact on Cora), generation-cache key without community size, judge truncation/failure caching. See `results/audit_report.md`.

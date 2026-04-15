<div align="center">

# 📐 Chalkline

### *Unsupervised Career Mapping for Maine's Construction Trades*

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-f7931e.svg)](https://scikit-learn.org/)
[![Hamilton](https://img.shields.io/badge/Hamilton-DAG-8B5CF6.svg)](https://hamilton.dagworks.io/)
[![Marimo](https://img.shields.io/badge/UI-Marimo-009485.svg)](https://marimo.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🔩 Quick Start

Requires Python **3.14+** and [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

```bash
git clone https://github.com/Jybbs/chalkline.git
cd chalkline
uv sync
```

Chalkline operates in two stages. First, `fit` encodes the posting corpus with a sentence transformer, clusters the embeddings into career families, assigns [O\*NET](https://www.onetonline.org/) occupations via posting-level MaxSim late-interaction scoring, derives per-cluster wages from the joined labor table, and builds a stepwise career graph that attaches credentials to each pathway on demand. Results are cached to disk via Hamilton's content-addressed store so subsequent runs with unchanged code and config serve instantly.

```bash
uv run chalkline fit              # fit the pipeline, print a summary
uv run chalkline fit -v           # same, with diagnostic logs
```

Then `launch` starts the Marimo reactive notebook where you upload a resume and receive a personalized career report.

```bash
uv run chalkline launch           # open the career report in your browser
```

> The posting corpus from AGC Maine is proprietary and not included in the repository. Place posting data in `data/postings/` before fitting.

---

## 🏗️ Background

The [Green Buildings Career Map](https://greenbuildingscareermap.org/) organized **55** jobs across **4** sectors with **300+** advancement routes, demonstrating that structured career maps change how workers navigate trades[^23]. Chalkline asks whether the same kind of structure can be constructed algorithmically from job postings, complementing expert-curated maps with a data-driven approach that can be re-fitted as the labor market shifts.

The premise is that postings encode implicit structure about how occupations relate to one another, which skills bridge adjacent roles, and what credentials separate one career level from the next. Occupational modeling at scale has confirmed this, showing that millions of unstructured postings yield taxonomies comparable to expert-curated frameworks[^2], and network models built from skill overlap reveal the same latent mobility structure[^1][^36]. Data-driven taxonomies extracted directly from online adverts have reached similar conclusions at smaller scale[^31], reinforcing that the signal is in the postings themselves.

Chalkline works with **2,154** postings scraped from [AGC Maine](https://www.agcmaine.org/)'s listings and covers **60** [O\*NET](https://www.onetonline.org/) [SOC](https://www.bls.gov/soc/) codes across three sectors (*Building Construction, Construction Managers, Heavy Highway Construction*). A sentence transformer[^57] encodes each posting into a 768-dimensional embedding, Ward-linkage HAC[^27] clusters those embeddings into **20** career families, and a stepwise k-NN graph routes advancement and lateral moves enriched by **836** credentials (*19 apprenticeships, 787 certifications, 30 educational programs*) on a per-route basis. A joined labor table of [BLS OEWS](https://www.bls.gov/oes/) wages, growth projections, and [O\*NET Bright Outlook](https://www.onetonline.org/find/bright) designations covers **53** of the SOC codes, driving the cluster-level wage expectations that appear on every career card. Upload a resume, and the system chunks it into sentences, encodes each chunk, and projects into the same space for personalized skill-gap analysis[^44].

A chalk line snaps a straight reference path between two points. Chalkline does the same for careers.

---

## 🪚 How It Works

Chalkline is a single-track embedding pipeline orchestrated by [Hamilton](https://hamilton.dagworks.io/)[^58], wherein each processing step is a DAG node whose parameter names declare its dependencies. Hamilton resolves execution order automatically, caches every node result to disk under a content-addressed key of `hash(code_version + input_data_versions)`, and serves from cache on subsequent calls with unchanged code and config, so that editing a curation script, a lexicon JSON, or an individual node function reliably invalidates only that node and its downstream dependents, rather than requiring a blunt wipe of the cache directory. The pipeline draws on recent work in job ad segmentation via NLP and clustering[^8], and on end-to-end transformer pipelines for resume matching[^45].

| Step | Node | Technique | Module |
|------|------|-----------|--------|
| 1 | **Corpus Loading** | Deduplicate and filter JobSpy-collected postings, normalize companies and locations | `collection.collector` |
| 2 | **Sentence Encoding** | [`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) via ONNX with CLS pooling | `pipeline.encoder` |
| 3 | **Dimensionality Reduction** | L2-normalize embeddings, then [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) to **10** components | `pipeline.steps` |
| 4 | **Clustering** | [Ward-linkage HAC](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) at $`k = 20`$ career families | `pipeline.steps` |
| 5 | **SOC Assignment** | Posting-level ColBERTv2 MaxSim against Task embeddings of all **60** O\*NET occupations | `pathways.scoring` |
| 6 | **Per-Cluster Wage** | Top-K softmax expectation over labor wages weighted by SOC similarity | `pathways.clusters` |
| 7 | **Career Graph** | Stepwise k-NN backbone (*lateral at same Job Zone, upward at next*) with per-pair dual-threshold credential filter on demand | `pathways.graph` |
| 8 | **Resume Matching** | Sentence chunking, per-task MaxSim, BM25-weighted gap ranking, SVD projection for centroid distance | `matching.matcher` |

The `SentenceEncoder` in `pipeline/encoder.py` downloads the ONNX model from HuggingFace on first use and runs inference via `onnxruntime` in fixed-size batches with CLS pooling followed by L2 normalization, with the ~430 MB model file deliberately instantiated outside the DAG, so that Hamilton's disk cache only serializes NumPy array outputs rather than the encoder weights themselves. Cold-start time for subsequent sessions drops from **~10.4s** to **~0.35s** because the encoder loads tokenizer files through `Tokenizer.from_file` and reuses `try_to_load_from_cache` rather than re-resolving the HuggingFace revision each time.

The fitted pipeline assembles into a `Chalkline` dataclass that exposes four attributes (*`clusters`, `config`, `graph`, `matcher`*) and a single `match(pdf_bytes)` method which extracts resume text via `pdfplumber`, splits it into sentences, encodes each chunk with the same sentence transformer used for posting encoding, projects the mean chunk vector through the fitted SVD, assigns the nearest career family, computes per-task MaxSim gap analysis, and returns a `MatchResult` carrying reach exploration and credential metadata. Because the matcher reuses every fitted transformation rather than re-encoding the reference corpus, per-match latency stays under a second once the encoder is warm.

Fit-time timing is logged per Hamilton node via `run_after_node_execution` in `pipeline/progress.py`, resulting in a diagnostic run that surfaces which step dominates wall-clock time without external profiling, and the `chalkline cache` CLI subcommand inspects Hamilton's SQLite metadata store to show which cached node output maps to which on-disk file when code changes do not invalidate the downstream subtree in the way you expect.

### Corpus Collection

Posting collection sits upstream of the Hamilton DAG, because raw scraping is a stateful process that should not re-run on every pipeline fit. The `collection/` subpackage wraps [`python-jobspy`](https://github.com/Bunsly/JobSpy) to issue searches against multiple aggregators for a curated list of construction search terms, concatenates the returned records into a single DataFrame, and passes them through `clean_text` normalization before the collector deduplicates on a composite key derived from the company and title slugs. Each posting receives a deterministic id via `python-slugify`, so that the same listing encountered twice across different boards collapses to a single record, resulting in a stable corpus that can be diffed between collection runs. The collector writes to `data/postings/` as a JSON array consumed by the Hamilton `corpus` node, and the pipeline treats everything under that directory as read-only input.

---

## ⚙️ Mathematical Framework

### Encoding and Reduction

Each posting description is fed through a sentence transformer ([`gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)) that converts text into a **768**-dimensional vector capturing its semantic meaning. Every vector is scaled to unit length (*L2-normalized*), so that $`\hat{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|_2`$, meaning that dot products between any two vectors directly measure their directional similarity.

**768** dimensions is more than the downstream steps need, and high-dimensional spaces introduce a well-documented problem wherein all pairwise distances converge toward the same value[^34], making it harder to tell similar postings apart from dissimilar ones. TruncatedSVD[^17] compresses the space by decomposing the posting embedding matrix into its most informative components:

$`
\hspace{0.5cm} \displaystyle
\mathbf{M} \approx \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top
`$  
<br>

The pipeline retains $`k = 10`$ components, reducing each posting from **768** coordinates to **10** that capture the dominant structure of the original space. This generalizes latent semantic analysis[^18] to dense transformer embeddings, and the randomized SVD algorithm[^17] keeps the factorization efficient even for large matrices. Evidence suggests that cutting sentence embedding dimensions by roughly half can actually improve downstream clustering[^53].

### Ward-Linkage Hierarchical Clustering

The pipeline groups postings into career families using Ward-linkage hierarchical agglomerative clustering[^27]. Starting with each posting as its own cluster, the algorithm repeatedly merges the two clusters whose combination increases total within-cluster variance the least. The cost of merging clusters $`A`$ and $`B`$ with centroids $`\bar{\mathbf{a}}`$ and $`\bar{\mathbf{b}}`$ is

$`
\hspace{0.5cm} \displaystyle
d_{\text{Ward}}(A, B) = \sqrt{\frac{2 \cdot |A| \cdot |B|}{|A| + |B|}} \; \|\bar{\mathbf{a}} - \bar{\mathbf{b}}\|_2
`$  
<br>

This builds a full merge hierarchy that is then cut at $`k = 20`$ to produce twenty career families. Ward linkage is the chosen criterion, because its variance-minimization objective produces the most cohesive families under the construction corpus's tight within-sector similarity.

### Cluster Quality and Connectivity

The methods tab surfaces two analytical primitives that together describe how usable the fitted partition actually is. Silhouette analysis[^29] validates the quality of the partition by measuring how well each posting fits its assigned family versus its nearest alternative family, with the per-posting silhouette coefficient defined as

$`
\hspace{0.5cm} \displaystyle
s(i) = \frac{b(i) - a(i)}{\max\bigl(a(i), b(i)\bigr)}
`$  
<br>

where $`a(i)`$ is the mean cosine distance from posting $`i`$ to every other posting in its assigned cluster and $`b(i)`$ is the minimum mean cosine distance from $`i`$ to any other cluster. Coefficients near **+1** indicate a well-separated assignment, coefficients near **0** indicate a posting on the boundary between two families, and negative coefficients indicate likely misassignment. The methods tab renders per-cluster mean silhouette as a horizontal bar chart ordered by score, letting the user see at a glance which families are tight (*Electricians, HVAC Mechanics*) versus which are diffuse (*Construction Managers, Project Managers*).

Brokerage centrality on the career graph[^49] complements the silhouette view by measuring how often each cluster appears on the shortest path between other pairs of clusters. For a cluster $`v`$ on the career graph $`G = (V, E)`$, brokerage is

$`
\hspace{0.5cm} \displaystyle
B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}
`$  
<br>

where $`\sigma_{st}`$ is the number of shortest paths from $`s`$ to $`t`$ and $`\sigma_{st}(v)`$ is the number of those paths that pass through $`v`$. Clusters with high brokerage function as career stepping stones, meaning they bridge several pairs of families that would otherwise be separated by more than one Job Zone, whereas clusters with low brokerage sit at the periphery of the graph. The methods tab pairs silhouette on one axis with brokerage on the other, and the resulting scatter surfaces clusters that are both well-defined and well-connected in the upper right quadrant.

### SOC Assignment via Posting-Level MaxSim

Each cluster needs an occupational identity drawn from the **60**-code O\*NET reference set, and the assignment scorer uses ColBERTv2-style late-interaction MaxSim[^60][^61], which preserves the multi-vector structure on both sides of the comparison rather than collapsing either into a pooled mean. For each cluster $`C`$ (*a set of posting embeddings*), and each SOC $`S`$ (*a set of Task embeddings*), the score is

$`
\hspace{0.5cm} \displaystyle
\text{score}(C, S) = \frac{1}{|C|} \sum_{p \in C} \max_{t \in S} \cos(\mathbf{p}, \mathbf{t})
`$  
<br>

Each posting casts its best-matching single task against each SOC, and the cluster-level score is the mean of those maxes. The `SOCScorer` dataclass in `pathways/scoring.py` stacks every SOC's task matrix into one contiguous array at construction, and resolves every cluster-occupation pair with a single BLAS matmul plus `np.maximum.reduceat` for per-occupation max-pooling, so that the `soc_similarity` Hamilton node collapses to a three-line delegation.

The full `(n_clusters, n_occupations)` similarity matrix feeds two downstream consumers. The argmax assigns each cluster's SOC title and sector, and a softmax over each row at temperature $`\tau = 0.02`$ produces per-cluster occupation weights that drive wage expectation and Job Zone voting. Using task-only descriptions for assignment is deliberate, because task specificity discriminates adjacent trades whose aggregated descriptions otherwise blur together under pooled scoring[^65]. The downstream resume matcher uses a different regime, namely Task+DWA with BM25 weighting, because the gap view's purpose is to surface the matched occupation's full activity profile, rather than to re-pick the occupation.

### Job Zones via Neighbor Voting

[Job Zone](https://www.onetonline.org/help/online/zones) assignment (*ranging from 1 for minimal preparation to 5 for extensive*) uses a smoothed vote from the top $`k = 3`$ most similar occupations rather than relying on a single nearest neighbor:

$`
\hspace{0.5cm} \displaystyle
\text{JobZone}(c) = \text{median}\bigl(\{z_i : i \in \text{top-}k \; \text{neighbors of } c\}\bigr)
`$  
<br>

where $`z_i`$ is the Job Zone of the $`i`$-th most similar O\*NET occupation. The median smooths over cases where the single nearest occupation would produce a misleading preparation level.

### Resume Chunking and Per-Task MaxSim

The matcher splits an uploaded resume into sentences via NLTK's Punkt tokenizer, encodes each chunk $`\mathbf{c}_i`$ independently, and defines per-task similarity as the MaxSim across chunks[^16][^60], because a single document-level vector drowns task-level signal in document-level noise[^57] and leaves no natural zero point for the demonstrated-versus-gap split:

$`
\hspace{0.5cm} \displaystyle
\text{sim}(r, t_j) = \max_i \cos(\mathbf{c}_i, \mathbf{t}_j)
`$  
<br>

A resume line like *"Installed commercial electrical systems for 8 years"* scores **0.6–0.8** against its matching task, while unrelated tasks stay close to the neutral **~0.30** cosine floor that any coherent English text produces against construction content. Cluster assignment continues to use the mean chunk vector projected through the fitted SVD, so that centroid distance stays comparable across sessions even as per-task scoring benefits from the chunk-level resolution.

### BM25-Weighted Gap Ranking

Generic verbs like *"prepare"*, *"use"*, and *"assist"* appear in the task descriptions of almost every occupation, so that raw per-task MaxSim would reward resumes that mention them regardless of whether the underlying work matches. The matcher re-weights each task's similarity by a BM25 term-weighting function[^44] over its stemmed content words, which suppresses high-document-frequency terms, and amplifies domain-specific ones such as *"conduit"*, *"circuit"*, and *"journeyman"*. The BM25 term-frequency component with length normalization is

$`
\hspace{0.5cm} \displaystyle
w_{\text{tf}}(f, \ell) = \frac{(s + 1) \cdot f}{s \cdot \bigl((1 - b) + b \cdot \ell / \bar{\ell}\bigr) + f}
`$  
<br>

where $`f`$ is the term's frequency in the task, $`\ell`$ is the task length, $`\bar{\ell}`$ is the average task length, $`s = 1.5`$ is the saturation parameter, and $`b = 0.75`$ is the length-weight parameter. The Zipf frequency threshold for stop-filtering (*cutoff at zipf < 6.0, sourced from the `wordfreq` corpus*) removes terms that carry little occupation-specific signal before the weighting runs. Demonstrated tasks rank by descending weighted similarity, gaps by ascending (*largest deficits first*).

### Career Graph and Credential Filter

The career graph connects the **20** career families with directed, weighted edges representing plausible career moves[^1]. Graph-based representations of occupational transitions capture mobility patterns that flat taxonomies miss[^47][^55], and the stepwise constraint ensures edges only link clusters at the same Job Zone (*lateral pivots*) or one level apart (*upward advancement*), preventing unrealistic tier-skipping jumps[^48]. Each cluster gets $`k_\text{lateral} = 2`$ bidirectional edges to clusters at the same Job Zone and $`k_\text{upward} = 2`$ unidirectional edges to clusters at the next Job Zone level.

Credentials attach per route rather than per edge, meaning that `CareerPathwayGraph.credentials_for(source_id, destination_id)` applies a dual-threshold cosine filter to the full credential set on demand, so that every route the user explores receives a freshly computed, pair-specific credential set. For a route from cluster $`\mathbf{s}`$ to cluster $`\mathbf{d}`$, a credential $`\mathbf{c}`$ passes when

$`
\hspace{0.5cm} \displaystyle
\cos(\mathbf{c}, \mathbf{d}) \geq \tau_\text{dest}(\mathbf{d}) \quad \wedge \quad \cos(\mathbf{c}, \mathbf{s}) \geq \tau_\text{src}
`$  
<br>

The destination threshold $`\tau_\text{dest}(\mathbf{d})`$ is the **95th** percentile of credential similarities to that specific target cluster, selecting only the most closely aligned credentials. The source threshold $`\tau_\text{src}`$ is the **75th** percentile of the global credential distribution, ensuring each credential is relevant to the worker's current position rather than a non-sequitur. Because the filter runs per route, the credential set adapts to every source-destination pair the user explores rather than being frozen at fit time.

### Gap Coverage via Greedy Targeted-Ratio Selection

Once a route's gap set and credential set are known, the route recipe picks up to five credentials that jointly cover as many gaps as possible. The greedy picker scores each credential by the ratio of its remaining-gap affinity to its total affinity on this route:

$`
\hspace{0.5cm} \displaystyle
\text{score}(c \mid R) = \frac{\sum_{g \in R} \mathbb{1}[\mathbf{A}_{c,g} > 0] \cdot \mathbf{A}_{c,g}}{\sum_{g} \mathbf{A}_{c,g}}
`$  
<br>

where $`\mathbf{A}_{c,g}`$ is the stem-gated cosine affinity between credential $`c`$ and gap task $`g`$, and $`R`$ is the still-uncovered residual. Narrow credentials whose entire reach lies in the residual win over broad credentials whose affinity is mostly spent on already-covered gaps. Absolute remaining-affinity acts as a tiebreaker so a wholly-targeted weak credential does not beat a wholly-targeted stronger one. Each picked credential records its incremental `positions: frozenset[int]` (*intersection of its reach with the residual at pick time*), which the UI uses to check off only the gaps that credential newly contributes rather than every gap the credential can cover in isolation.

### Per-Cluster Wage Expectation

Every career card shows a median annual wage derived from the BLS OEWS table joined to O\*NET SOC codes. Because SOC assignment is probabilistic (*the softmax row over `soc_similarity` gives each cluster a distribution over occupations*), the wage is computed as a top-K expectation rather than a single-SOC lookup. For cluster $`c`$ with softmax-weighted SOC distribution $`\mathbf{w}_c`$ and per-occupation wages $`\mathbf{W}`$:

$`
\hspace{0.5cm} \displaystyle
\text{wage}(c) = \text{round}\!\left(\frac{\sum_{k \in \mathcal{K}_c} \mathbf{w}_{c,k} \cdot \mathbf{W}_k}{\sum_{k \in \mathcal{K}_c} \mathbf{w}_{c,k}}, \; r\right)
`$  
<br>

where $`\mathcal{K}_c`$ is the top $`K = 3`$ occupations by softmax weight restricted to those with non-null wage data, and $`r = 10`$ is the rounding granularity matching the source labor records. The non-null mask ensures occupations missing wage data cannot drag the expectation toward zero, and the top-K cap stabilizes the estimate against the long tail of low-weight SOCs. Wage lives on each `Cluster` as a post-init attribute alongside `display_title` and `soc_weights`, so downstream consumers (*map nodes, route verdicts, the wage-filter slider*) read per-cluster values without re-running the computation.

### Distinctive Vocabulary and Sub-Role Discovery

The data tab characterizes the matched career family through two complementary views that both rest on TF-IDF over clusters-as-documents. The distinctive vocabulary view ranks every word appearing in the matched cluster's postings by

$`
\hspace{0.5cm} \displaystyle
\text{tfidf}(w, c) = \frac{\text{count}(w, c)}{\sum_{w'} \text{count}(w', c)} \cdot \log\!\frac{|C|}{\text{df}(w)}
`$  
<br>

where $`\text{count}(w, c)`$ is the word's frequency in cluster $`c`$, $`\sum_{w'} \text{count}(w', c)`$ is the cluster's total word count, $`|C|`$ is the number of career families in the corpus, and $`\text{df}(w)`$ is the number of clusters containing the word at least once. Words are partitioned into three tiers by their document frequency, resulting in a treemap that separates *unique to this family* vocabulary (*`df = 1`*), *rare across the corpus* vocabulary (*`2 ≤ df ≤ 4`*), and *notable vocabulary* that still ranks high in the matched family despite appearing in five or more families. Each tier is sized independently, so that a sparser tier never gets visually crowded by a denser one, and words below a minimum raw count threshold are filtered out first to suppress single-occurrence noise.

Sub-role discovery operates at a finer grain by running k-means on the matched cluster's posting embeddings and labeling each sub-cluster with its top-two TF-IDF words where the "documents" are the sub-clusters themselves. For sub-cluster $`j`$ within the matched cluster, each word scores

$`
\hspace{0.5cm} \displaystyle
\text{score}(w, j) = \frac{\text{count}(w, j)}{\text{total}(j)} \cdot \log\!\frac{k}{\text{df}_{\text{sub}}(w)}
`$  
<br>

where $`k`$ is the number of sub-clusters, $`\text{df}_{\text{sub}}(w)`$ is the number of sub-clusters containing the word, and the denominator filters out words appearing in all $`k`$ sub-clusters, because such words cannot discriminate between them. The top-2 scoring words per sub-cluster concatenate with a middle-dot separator to produce labels like *"Conduit · Circuit"* or *"Supervisor · Foreman"*, and a numbered fallback surfaces when no word survives the filtering pass. The resulting labels attach to each color band in the t-SNE projection on the data tab, resulting in a visual breakdown of sub-roles within the matched family that is both semantically grounded and geometrically readable.

### Display Title Cascade

Cluster labels need to be unique within the corpus so two cards never collide in the picker or the map. Because multiple clusters can legitimately share a SOC title (*two *Operating Engineers* clusters that differ in specialty*), the pipeline resolves collisions through a three-level cascade applied asymmetrically per collision group:

$`
\hspace{0.5cm} \displaystyle
\ell(c) = \begin{cases} \text{soc\_title}(c) & \text{if unique at level 0} \\ \text{modal\_title}(c) & \text{if unique at level 1} \\ \text{soc\_title}(c) \; + \; \text{``(\#''} + \text{id}(c) + \text{``)''} & \text{otherwise} \end{cases}
`$  
<br>

At each pass, the resolver groups clusters by their current label, and, for any group with more than one cluster, promotes the smaller members to the next level, breaking ties by descending cluster size with cluster id as the secondary key. The largest *Civil Engineers* cluster keeps the bare title, while smaller colliding clusters advance to their modal posting title, or to the numbered fallback. The loop runs at most three iterations, because the level-2 fallback is guaranteed unique via the cluster id, and the `(#id)` form carries SOC context in the rare case where modal titles also collide.

---

## 🪜 The Career Report

The Marimo notebook opens to a splash page showing the fitted landscape at a glance (*corpus size, occupation count, sector distribution, credential totals*) with a drag-and-drop upload zone. Drop a PDF resume and the system extracts text, chunks it into sentences, encodes each chunk, projects through the fitted SVD, and matches to the nearest career family. The splash then dismisses and the three-tab dashboard takes over.

### Map Tab

The primary view is an interactive D3 force-directed [career map](https://d3js.org/d3-force) rendered via [AnyWidget](https://anywidget.dev/) so Python state (*click selection, wage filter*) flows reactively back through traitlets. Horizontal position encodes wage, node rendering tier distinguishes the immediate career neighborhood from distant options, and the matched career renders as an enriched hero card integrated into the SVG. Clicking any cluster swaps the route panel below the map to describe the transition from the matched career to the selected destination.

The route panel owns the substantive career-planning content:

- **Verdict**: fit percentage (*calibrated from SVD centroid distance*), wage comparison bars for the source and destination, bold narrative verdict, and open-positions count
- **Evidence Drawer**: the eight strongest demonstrated skills and eight largest gaps, each rendered as a skill card with cosine-weighted progress bars
- **Recipe**: stacked credential path cards with per-credential gap shelves, where each shelf lists the route's full gap set and checks off only the tasks this credential contributes. Multiple strategies surface side by side (*bang-for-your-buck, work-based path, certification stack*) so the user can compare approaches
- **Postings**: up to ten destination-cluster postings ranked by cosine against the resume, rendered as compact cards
- **Resources Drawer**: the full credential catalog for this route, fuzzy-matched AGC member companies with career-page URLs, and sector-filtered job boards

A wage-floor slider at the top of the map prunes tier-2 cards whose median wage falls below the chosen threshold, defaulting to the corpus floor so every cluster is visible on first render. Debounce mode defers the map's re-render to slider release rather than every tick.

### Data Tab

The data tab surfaces corpus statistics that contextualize the match by describing both the ambient job market and the internal structure of the user's assigned career family. The top row aggregates posting counts, sector shares, wage percentiles, and location distribution, so that a reader opening the tab sees the scale of the evidence before diving into specifics. A posting timeline plots every matched posting along its collection date, resulting in a temporal strip where hover text surfaces the company name, so that a reader can reason about seasonality or recent hiring bursts without leaving the notebook.

The matched cluster's internal composition comes from two analytical pieces that reuse the cluster's stored embeddings rather than re-encoding anything at render time. A [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) projection of the posting embeddings maps them to two dimensions with PCA initialization for stability, and k-means sub-clustering on the same high-dimensional vectors colors each point by its sub-role assignment, resulting in color bands labeled through the in-cluster TF-IDF formula described above. The distinctive vocabulary treemap sits alongside the projection and partitions the cluster's words into the three tiers (*unique to this family, rare across the corpus, notable vocabulary*), giving the reader a textual counterpart to the geometric sub-role view. An employer roll-up identifies which companies are hiring most in the matched family, a credential catalog filtered to the destination cluster surfaces ranked certifications and programs, and the tab closes with a relevant job boards listing filtered by sector relevance.

### Methods Tab

The methods tab documents the pipeline's design choices for technical audiences by combining a visual walkthrough of the Hamilton DAG with the analytical primitives that justified each step's configuration. The tab opens with a process flow diagram rendering every node's parameter-level dependency graph, accompanied by per-node timing pulled from the fit log, so that the reader can see which step dominates wall-clock cost. Bar charts of SVD explained variance reveal how much of the original 768-dimensional signal survives in the 10-component reduction, whereas sector cluster sizes and per-cluster silhouette coefficients describe the partition's balance and separation quality. A scatter plot pairs silhouette against brokerage centrality on the career graph, resulting in a two-dimensional view that distinguishes families that are well-defined but peripheral from families that are both well-defined and well-connected, and a matching brokerage bar chart ranks every cluster by its stepping-stone role in the graph. SOC-similarity heatmaps show how each cluster ranks against every O\*NET occupation, providing direct evidence for the MaxSim assignment decisions, and a node-to-file table mirrors `chalkline cache` output, so that a reader verifying an invalidation subtree can confirm which cached artifacts Hamilton will rebuild on the next fit.

Interactive glossary tooltips sit throughout both analytical tabs via pipeline-specific substitutions, meaning technical terms like *silhouette*, *betweenness*, *MaxSim*, and *TruncatedSVD* render as underlined popover triggers that reveal rich definitions sourced from `display/tabs/shared/glossary.toml` without requiring the reader to leave the notebook for external documentation.

---

## 🔧 CLI Reference

Chalkline's CLI is built on [Typer](https://typer.tiangolo.com/) with Rich markup. Running `chalkline` with no arguments prints help.

```bash
uv run chalkline --help
```

### `fit`

Encode postings, cluster into career families, run SOC assignment, build the career graph, and cache the fitted pipeline. All directory flags default to sensible project-relative paths that work when running from the repository root.

```bash
uv run chalkline fit                  # fit with default paths
uv run chalkline fit --verbose        # same, with debug-level logs
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--lexicon-dir` | | `data/lexicons` | Path to lexicon JSONs (*O\*NET, credentials, labor*) |
| `--postings-dir` | | `data/postings` | Path to the corpus directory |
| `--verbose` | `-v` | `False` | Show diagnostic logs |

### `launch`

Pre-fit the pipeline (*hitting cache on unchanged code and config*), then start `marimo run` on the career report notebook. Must be run from the project root where `app/main.py` exists.

```bash
uv run chalkline launch
uv run chalkline launch --verbose
```

### `cache`

Inspect Hamilton's content-addressed disk cache, listing every cached node, the SHA it keys against, and the on-disk file size. Useful when a code change does not seem to have invalidated what you expected.

```bash
uv run chalkline cache                            # inspect default .cache/hamilton
uv run chalkline cache --cache-dir path/to/cache  # custom cache root
```

---

## 🧱 Core Dependencies

| Component | Technology | Role |
|-----------|------------|------|
| **Sentence Encoding** | [`onnxruntime`](https://onnxruntime.ai/) + [`tokenizers`](https://github.com/huggingface/tokenizers) | ONNX inference for [`gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) with HuggingFace fast tokenization |
| **Machine Learning** | [`scikit-learn`](https://scikit-learn.org/) | TruncatedSVD, Ward HAC, t-SNE, k-means, L2 normalization, cosine similarity, silhouette |
| **Pipeline Orchestration** | [`sf-hamilton[diskcache]`](https://github.com/DAGWorks-Inc/hamilton) | DAG resolution from function signatures with node-level content-addressed disk caching[^58] |
| **Career Graph** | [`NetworkX`](https://networkx.org/) | Directed weighted graph for stepwise k-NN backbone, reach queries, and betweenness centrality[^49] |
| **Corpus Collection** | [`python-jobspy`](https://github.com/Bunsly/JobSpy) | Multi-board job aggregation from Indeed and other sources |
| **PDF Extraction** | [`pdfplumber`](https://github.com/jsvine/pdfplumber) | Resume text extraction with layout-aware parsing |
| **UI** | [`Marimo`](https://marimo.io/) + [`AnyWidget`](https://anywidget.dev/) | Reactive notebook with custom D3 career-map widget |
| **HTML Composition** | [`htpy`](https://htpy.dev/) + [`MarkupSafe`](https://palletsprojects.com/projects/markupsafe/) | Typed HTML element trees for display-layer composition |
| **Visualization** | [`Plotly`](https://plotly.com/python/) | Interactive charts for landscape, variance, heatmaps, treemaps |
| **Vocabulary Filtering** | [`wordfreq`](https://github.com/rspeer/wordfreq) + [`nltk`](https://www.nltk.org/) | Zipf-frequency stop filtering and Snowball stemming for BM25 weighting |
| **CLI** | [`Typer`](https://typer.tiangolo.com/) | `fit`, `launch`, and `cache` subcommands with Rich markup |
| **Configuration** | [`Pydantic`](https://pydantic.dev/) | `PipelineConfig` with `extra="forbid"` and tuned defaults |
| **Logging** | [`Loguru`](https://github.com/Delgan/loguru) | Structured pipeline progress and per-node timing |
| **Utilities** | [`python-slugify`](https://github.com/un33k/python-slugify) | Deterministic posting id construction |

---

## 🏠 Project Structure

```
chalkline/
├── app/
│   ├── chalkline.css                      Dashboard theme (dark, Lora serif, sector palette)
│   └── main.py                            Marimo reactive notebook (career report)
│
├── data/
│   ├── labor/                             BLS OEWS raw curations (committed)
│   │   ├── outlook.json                   O*NET Bright Outlook flags for 53 SOCs
│   │   ├── projections.json               10-year employment projections for 51 SOCs
│   │   └── wages.json                     Annual wage percentiles for 50 SOCs
│   ├── lexicons/                          Pipeline inputs (committed)
│   │   ├── credentials.json               836 credentials (19 apprenticeships, 787 certs, 30 programs)
│   │   ├── labor.json                     Joined wage + projection + outlook table for 53 SOCs
│   │   └── onet.json                      60 SOC codes with Tasks, DWAs, Technology Skills, KSAs
│   ├── postings/                          Scraped AGC corpus (gitignored, 2154 records)
│   └── stakeholder/                       AGC Maine reference data (gitignored)
│       ├── additions/                     Scope extensions (apprenticeship SOCs, program SOCs)
│       └── reference/                     Members, apprenticeships, programs, job boards, etc.
│
├── scripts/                               Repeatable data curation (not part of the package)
│   ├── curate_credentials.py              Build credentials.json from stakeholder refs + enrichment
│   ├── curate_labor.py                    Join wages + projections + outlook into labor.json
│   ├── curate_onet.py                     Fetch O*NET Tasks, DWAs, Technology Skills, KSAs
│   ├── explore_embeddings.py              Diagnostic tool for SOC assignment investigations
│   ├── parse_agc_workbook.py              Extract stakeholder workbook sheets into reference JSONs
│   ├── parse_certifications.py            Transform CareerOneStop certification scrapes
│   └── parse_labor.py                     Parse raw BLS OEWS sheets into the labor subdirectory
│
├── src/chalkline/
│   ├── cli/                               Typer CLI with fit, launch, and cache subcommands
│   │   ├── cache.py                       Hamilton cache inspector
│   │   ├── fit.py                         Pipeline fitting with cache-or-compute
│   │   └── launch.py                      Marimo notebook launcher with pre-fit
│   │
│   ├── collection/                        Corpus loading and posting schemas
│   │   ├── collector.py                   Filter and key postings from storage
│   │   ├── schemas.py                     Posting Pydantic models
│   │   └── storage.py                     File-backed posting persistence
│   │
│   ├── display/                           Presentation layer
│   │   ├── charts.py                      Plotly chart builders (variance, sector, silhouette, heatmap, scatter)
│   │   ├── forms.py                       Marimo UI composers (wage-filter slider)
│   │   ├── loaders.py                     ContentLoader + Layout composer for htpy assembly
│   │   ├── routes.py                      Route card builders (verdict, evidence, recipe, postings, resources)
│   │   ├── schemas.py                     RouteDetail, MapGeometry, CredentialPath, PathItem, MlMetrics, ...
│   │   ├── theme.py                       Plotly templates, sector palette, CSS custom property forwarding
│   │   └── tabs/
│   │       ├── data/render.py             Data tab renderer
│   │       ├── map/render.py              Map tab renderer
│   │       ├── map/widget.py              PathwayMap AnyWidget (D3 force-directed)
│   │       ├── methods/render.py          Methods tab renderer
│   │       ├── shared/content.toml        Shared UI labels
│   │       ├── shared/glossary.toml       Glossary tooltip definitions
│   │       └── splash/render.py           Splash page renderer
│   │
│   ├── matching/                          Resume-to-career matching
│   │   ├── matcher.py                     Sentence chunking, per-task MaxSim, BM25 weighting, SVD projection
│   │   ├── reader.py                      PDF text extraction via pdfplumber
│   │   └── schemas.py                     MatchResult, BM25Config, ScoredTask models
│   │
│   ├── pathways/                          Career graph construction and cluster domain
│   │   ├── clusters.py                    Cluster and Clusters dataclasses (wage, display_title cascade)
│   │   ├── graph.py                       NetworkX stepwise k-NN backbone with per-pair credentials_for
│   │   ├── loaders.py                     LaborLoader and StakeholderReference
│   │   ├── schemas.py                     Credential, EncodedOccupation, Occupation, SkillType
│   │   └── scoring.py                     SOCScorer with ColBERTv2 MaxSim late interaction
│   │
│   └── pipeline/                          Orchestration and shared types
│       ├── encoder.py                     ONNX sentence transformer wrapper with CLS pooling
│       ├── orchestrator.py                Hamilton DAG driver → fitted Chalkline dataclass
│       ├── progress.py                    Loguru + Rich progress with per-node timing
│       ├── schemas.py                     PipelineConfig (Pydantic, extra="forbid")
│       └── steps.py                       Hamilton node functions (the full DAG)
│
├── tests/                                 Pytest suite mirroring src/ structure
├── pyproject.toml                         Build config, dependencies, CLI entry point
└── uv.lock                                Locked dependency versions
```

Each domain subpackage (*`collection/`, `matching/`, `pathways/`, `pipeline/`, `display/`*) owns its schemas and logic. The `pipeline/` subpackage orchestrates the others through Hamilton, where each function in `steps.py` is a DAG node whose parameter names declare its dependencies. The `display/` subpackage is organized tab-per-directory, so that each tab's `render.py` owns the Marimo cell composition for that tab, with shared primitives (*`Layout`, `Routes`, `Charts`, `Forms`, `Theme`*) sitting at the package root.

---

## 🤝 AGC Maine

[AGC Maine](https://www.agcmaine.org/) (*Associated General Contractors of Maine*) represents **222** member companies and has been the state's primary construction trade association since 1951. The association operates the [Maine Construction Academy](https://buildingmaine.com/) with tuition-free pre-apprenticeship programs expanding to five community colleges in 2026 and manages **19** registered [apprenticeship](https://www.apprenticeship.gov/) pathways spanning trades from carpentry and welding to crane operation and solar installation.

AGC provided the posting corpus, the stakeholder reference data defining the project's SOC scope and three sectors, and the credential records (*apprenticeships, certifications, educational programs*) that enrich the career graph. The collaboration connects algorithmic career mapping to a real training pipeline[^52][^56], where outputs directly inform which programs AGC recommends to workers entering or advancing through the trades.

---

## 📚 References

[^1]: del Rio-Chanona, et al. 2021. "Occupational Mobility and Automation: A Data-Driven Network Model." *Journal of the Royal Society Interface* 18 (174): 20200898. https://doi.org/10.1098/rsif.2020.0898

[^2]: Dixon, et al. 2023. "Occupational Models from 42 Million Unstructured Job Postings." *Patterns* 4 (7): 100757. https://doi.org/10.1016/j.patter.2023.100757

[^8]: Lukauskas, et al. 2023. "Enhancing Skills Demand Understanding through Job Ad Segmentation Using NLP and Clustering Techniques." *Applied Sciences* 13 (10): 6119. https://doi.org/10.3390/app13106119

[^16]: Rosenberger, et al. 2025. "CareerBERT: Matching Resumes to ESCO Jobs in a Shared Embedding Space for Generic Job Recommendations." *Expert Systems with Applications* 275: 127043. https://doi.org/10.1016/j.eswa.2025.127043

[^17]: Halko, Martinsson & Tropp. 2011. "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." *SIAM Review* 53 (2): 217-288. https://doi.org/10.1137/090771806

[^18]: Deerwester, Dumais, Furnas, Landauer & Harshman. 1990. "Indexing by Latent Semantic Analysis." *Journal of the American Society for Information Science* 41 (6): 391-407. https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9

[^23]: Hamilton. 2012. "Career Pathway and Cluster Skill Development: Promising Models from the United States." *OECD Local Economic and Employment Development (LEED) Papers* 2012/14. https://doi.org/10.1787/5k94g1s6f7td-en

[^27]: Ward. 1963. "Hierarchical Grouping to Optimize an Objective Function." *Journal of the American Statistical Association* 58 (301): 236-244. https://doi.org/10.1080/01621459.1963.10500845

[^29]: Rousseeuw. 1987. "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis." *Journal of Computational and Applied Mathematics* 20: 53-65. https://doi.org/10.1016/0377-0427(87)90125-7

[^31]: Djumalieva & Sleeman. 2018. "An Open and Data-driven Taxonomy of Skills Extracted from Online Job Adverts." *ESCoE Discussion Paper 2018-13*. https://www.escoe.ac.uk/publications/an-open-and-data-driven-taxonomy-of-skills-extracted-from-online-job-adverts/

[^34]: Aggarwal, Hinneburg & Keim. 2001. "On the Surprising Behavior of Distance Metrics in High Dimensional Space." *Database Theory (ICDT 2001), Lecture Notes in Computer Science* 1973: 420-434. https://doi.org/10.1007/3-540-44503-X_27

[^36]: Alabdulkareem, et al. 2018. "Unpacking the Polarization of Workplace Skills." *Science Advances* 4 (7): eaao6030. https://doi.org/10.1126/sciadv.aao6030

[^44]: de Groot, et al. 2021. "Job Posting-Enriched Knowledge Graph for Skills-based Matching." *RecSys in HR '21 Workshop, CEUR Workshop Proceedings, Vol. 2967*. https://arxiv.org/abs/2109.02554

[^45]: Khelkhal & Lanasri. 2025. "Smart-Hiring: An Explainable End-to-End Pipeline for CV Information Extraction and Job Matching." *arXiv preprint arXiv:2511.02537*. https://doi.org/10.48550/arXiv.2511.02537

[^47]: Avlonitis, et al. 2023. "Career Path Recommendations for Long-term Income Maximization: A Reinforcement Learning Approach." *RecSys in HR '23 Workshop, CEUR Workshop Proceedings, Vol. 3490*. https://ceur-ws.org/Vol-3490/RecSysHR2023-paper_2.pdf

[^48]: Senger, et al. 2025. "Toward More Realistic Career Path Prediction: Evaluation and Methods." *Frontiers in Big Data* 8: 1564521. https://doi.org/10.3389/fdata.2025.1564521

[^49]: Freeman. 1977. "A Set of Measures of Centrality Based on Betweenness." *Sociometry* 40 (1): 35-41. https://doi.org/10.2307/3033543

[^52]: Frej, et al. 2024. "Course Recommender Systems Need to Consider the Job Market." *Proceedings of the 47th ACM SIGIR Conference*. https://doi.org/10.1145/3626772.3657847

[^53]: Zhang, Zhou & Bollegala. 2024. "Evaluating Unsupervised Dimensionality Reduction Methods for Pretrained Sentence Embeddings." *Proceedings of LREC-COLING 2024*: 6530-6543. https://aclanthology.org/2024.lrec-main.579/

[^55]: Boškoski, et al. 2024. "Career Path Discovery through Bipartite Graphs." *Journal of Decision Systems* 33 (sup1): 140-153. https://doi.org/10.1080/12460125.2024.2354585

[^56]: Alonso, et al. 2025. "A Novel Approach for Job Matching and Skill Recommendation Using Transformers and the O\*NET Database." *Big Data Research* 39: 100509. https://doi.org/10.1016/j.bdr.2025.100509

[^57]: Ortakci. 2024. "Revolutionary Text Clustering: Investigating Transfer Learning Capacity of SBERT Models through Pooling Techniques." *Engineering Science and Technology, an International Journal* 55: 101730. https://doi.org/10.1016/j.jestch.2024.101730

[^58]: Krawczyk, et al. 2022. "Hamilton: Enabling Software Engineering Best Practices for Data Transformations via Generalized Dataflow Graphs." *1st International Workshop on Data Ecosystems (DEco@VLDB 2022), CEUR Workshop Proceedings, Vol. 3306*: 41-50. https://ceur-ws.org/Vol-3306/paper5.pdf

[^60]: Khattab and Zaharia. 2020. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval*: 39-48. https://doi.org/10.1145/3397271.3401075

[^61]: Santhanam, et al. 2022. "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*: 3715-3734. https://doi.org/10.18653/v1/2022.naacl-main.272

[^65]: Achananuparp, et al. 2025. "A Multi-Stage Framework with Taxonomy-Guided Reasoning for Occupation Classification Using Large Language Models." *arXiv preprint, accepted at ICWSM 2026*. https://doi.org/10.48550/arXiv.2503.12989

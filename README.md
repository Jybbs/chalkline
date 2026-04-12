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

Chalkline operates in two stages. First, `fit` encodes the posting corpus with a sentence transformer, clusters the embeddings into career families, assigns [O\*NET](https://www.onetonline.org/) occupations, and builds a stepwise career graph with credential enrichment. Results are cached to disk so that subsequent runs with unchanged code and config serve instantly.

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

Chalkline works with **922** postings from Maine's construction industry, scraped from [AGC Maine](https://www.agcmaine.org/)'s listings and covering **21** [O\*NET](https://www.onetonline.org/) [SOC](https://www.bls.gov/soc/) codes across three sectors. A sentence transformer[^57] encodes each posting into a 768-dimensional embedding, Ward-linkage HAC[^27] clusters those embeddings into **20** career families, and a stepwise k-NN graph maps **96** advancement and lateral edges enriched by **325** credentials. Upload a resume, and the system projects it into the same space for personalized skill gap analysis[^44].

A chalk line snaps a straight reference path between two points. Chalkline does the same for careers.

---

## 🪚 How It Works

Chalkline is a single-track embedding pipeline orchestrated by [Hamilton](https://hamilton.dagworks.io/)[^58], where each processing step is a DAG node whose parameter names declare dependencies. Hamilton resolves execution order automatically, caches every node result to disk, and serves from cache on subsequent calls with unchanged code and config. The pipeline draws on recent work in job ad segmentation via NLP and clustering[^8] and end-to-end transformer pipelines for resume matching[^45].

| Step | Node | Technique | Module |
|------|------|-----------|--------|
| 1 | **Corpus Loading** | Filter and key postings from [JobSpy](https://github.com/Bunsly/JobSpy) collection | `collection.collector` |
| 2 | **Sentence Encoding** | [`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) via ONNX with CLS pooling | `pipeline.encoder` |
| 3 | **Dimensionality Reduction** | L2-normalize embeddings, then [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) to 10 components | `pipeline.steps` |
| 4 | **Clustering** | [Ward-linkage HAC](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) at $`k = 20`$ career families | `pipeline.steps` |
| 5 | **SOC Assignment** | Top-3 median cosine similarity against [O\*NET](https://www.onetonline.org/) Task+DWA embeddings | `pipeline.steps` |
| 6 | **Career Graph** | Stepwise k-NN backbone with per-edge dual-threshold credential enrichment | `pathways.graph` |
| 7 | **Resume Matching** | SVD projection, nearest centroid, per-task cosine gap analysis | `matching.matcher` |

The `SentenceEncoder` in `pipeline/encoder.py` downloads the [ONNX](https://onnxruntime.ai/) model from HuggingFace on first use (*cached locally thereafter*), runs inference via `onnxruntime` in fixed-size batches with CLS pooling, and L2-normalizes the output. Because the ~430 MB model file should not be serialized into Hamilton's disk cache, the orchestrator in `pipeline/orchestrator.py` instantiates the encoder outside the DAG and passes it as an input alongside the `PipelineConfig`, so that all encoding node outputs (*NumPy arrays*) cache normally while the encoder itself is excluded.

The fitted pipeline is assembled into a `Chalkline` dataclass that exposes a single `match(pdf_bytes)` method. This method extracts text from an uploaded PDF via `pdfplumber`, encodes it with the same sentence transformer, projects through the fitted SVD, assigns the nearest career family, computes per-task gap analysis, and returns a `MatchResult` with reach exploration and credential metadata.

---

## ⚙️ Mathematical Framework

### Encoding and Reduction

Each posting description is fed through a sentence transformer ([`gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)) that converts text into a 768-dimensional vector capturing its semantic meaning. Every vector is then scaled to unit length (*L2-normalized*) so that $`\hat{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|_2`$, which means dot products between any two vectors directly measure their similarity.

768 dimensions is far more than the pipeline needs, and high-dimensional spaces introduce a well-documented problem where all pairwise distances converge toward the same value[^34], making it harder to tell similar postings apart from dissimilar ones. TruncatedSVD[^17] compresses the space by decomposing the embedding matrix into its most informative components:

$`
\hspace{0.5cm} \displaystyle
\mathbf{M} \approx \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top
`$  
<br>

The pipeline retains $`k = 10`$ components, reducing each posting from 768 coordinates to 10 that capture the dominant structure of the original space. This generalizes latent semantic analysis[^18] to dense transformer embeddings, and the randomized SVD algorithm[^17] keeps the factorization efficient even for large matrices. Evidence suggests that cutting sentence embedding dimensions by roughly half can actually improve downstream clustering[^53].

### Ward-Linkage HAC

The pipeline groups postings into career families using Ward-linkage hierarchical agglomerative clustering[^27]. Starting with each posting as its own cluster, the algorithm repeatedly merges the two clusters whose combination increases total within-cluster variance the least. The cost of merging clusters $`A`$ and $`B`$ with centroids $`\bar{\mathbf{a}}`$ and $`\bar{\mathbf{b}}`$ is:

$`
\hspace{0.5cm} \displaystyle
d_{\text{Ward}}(A, B) = \sqrt{\frac{2 \cdot |A| \cdot |B|}{|A| + |B|}} \; \|\bar{\mathbf{a}} - \bar{\mathbf{b}}\|_2
`$  
<br>

This builds a full merge hierarchy that is then cut at $`k = 20`$ to produce 20 career families. Silhouette analysis[^29] validates the quality of the resulting partition by measuring how well each posting fits its assigned family versus its nearest alternative.

### SOC Assignment and Job Zones

Each cluster needs an occupational identity. The pipeline computes the mean posting embedding for each cluster (*in the full 768-dimensional space, L2-normalized*), then compares it against sentence embeddings of all **21** [O\*NET](https://www.onetonline.org/) occupations' Task and DWA descriptions. The most similar occupation by cosine similarity becomes the cluster's label.

[Job Zone](https://www.onetonline.org/help/online/zones) assignment (*ranging from 1 for minimal preparation to 5 for extensive*) uses a smoothed vote from the three most similar occupations rather than relying on a single nearest neighbor:

$`
\hspace{0.5cm} \displaystyle
\text{JZ}(c) = \text{median}\bigl(\{z_i : i \in \text{top-}k\; \text{neighbors of } c\}\bigr), \quad k = 3
`$  
<br>

where $`z_i`$ is the Job Zone of the $`i`$-th most similar O\*NET occupation. The median smooths over cases where the single nearest occupation would produce a misleading preparation level.

### Cosine Similarity

Cosine similarity[^43] is the central metric throughout the pipeline. It measures how closely two embeddings point in the same direction, regardless of their magnitude. For any two vectors $`\mathbf{a}`$ and $`\mathbf{b}`$:

$`
\hspace{0.5cm} \displaystyle
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \; \|\mathbf{b}\|}
`$  
<br>

A score of 1.0 means identical direction (*maximally similar*), 0 means unrelated, and negative values mean opposing. Since all vectors in the pipeline are L2-normalized, this simplifies to a dot product. The same metric drives SOC assignment, per-task gap analysis, and credential filtering. Recent work on O\*NET-enriched transformer matching[^56] and shared embedding space approaches[^16] has validated cosine similarity as an effective signal for occupational proximity when both job descriptions and resumes are encoded by the same model.

### Resume Matching

When a user uploads a resume, the system encodes it with the same sentence transformer and projects it through the fitted SVD into the 10-dimensional space shared with all postings[^45]. The resume is then assigned to whichever career family's centroid is closest:

$`
\hspace{0.5cm} \displaystyle
k^* = \underset{k}{\mathrm{argmin}} \; \|\mathbf{r} - \mathbf{c}_k\|_2
`$  
<br>

The full distance ranking across all 20 clusters is preserved, letting the career report show proximity to every family rather than only the assigned one. After assignment, each of the matched cluster's [O\*NET](https://www.onetonline.org/) tasks is individually compared against the resume embedding in the full 768-dimensional space. A median-split threshold separates tasks the resume demonstrates from tasks it does not[^44]:

$`
\hspace{0.5cm} \displaystyle
\cos(\mathbf{r}, \mathbf{t}_i) \geq \tilde{S} \implies \text{demonstrated}, \quad \cos(\mathbf{r}, \mathbf{t}_i) < \tilde{S} \implies \text{gap}
`$  
<br>

where $`\tilde{S}`$ is the median similarity across all tasks for the matched cluster. Demonstrated tasks rank by descending similarity (*strongest first*), gaps by ascending (*largest deficits first*).

### Career Graph

The career graph connects the 20 career families with directed, weighted edges representing plausible career moves[^1]. Graph-based representations of occupational transitions capture mobility patterns that flat taxonomies miss[^47][^55], and the stepwise constraint here ensures edges only link clusters at the same [Job Zone](https://www.onetonline.org/help/online/zones) (*lateral pivots*) or one level apart (*upward advancement*), preventing unrealistic tier-skipping jumps[^48]. Each cluster gets $`k_{\text{lateral}} = 2`$ bidirectional same-JZ edges and $`k_{\text{upward}} = 2`$ unidirectional next-JZ edges, yielding **96** edges total.

Each edge is then annotated with relevant credentials (*19 [apprenticeships](https://www.apprenticeship.gov/), 276 certifications, 30 educational programs*) using a dual-threshold filter. For an edge from a worker's current cluster $`\mathbf{s}`$ to a target cluster $`\mathbf{d}`$, a credential $`\mathbf{c}`$ is attached when it meets both conditions:

$`
\hspace{0.5cm} \displaystyle
\cos(\mathbf{c}, \mathbf{d}) \geq \tau_{\text{dest}} \quad \wedge \quad \cos(\mathbf{c}, \mathbf{s}) \geq \tau_{\text{src}}
`$  
<br>

The destination threshold $`\tau_{\text{dest}}`$ (*95th percentile of credential similarities to the target cluster*) selects credentials closely aligned with where the worker is headed. The source threshold $`\tau_{\text{src}}`$ (*75th percentile of the global distribution*) ensures the credential is also relevant to the worker's current position rather than a non-sequitur. Credentials passing both thresholds are ranked by destination affinity and attached to the edge.

---

## 🪜 The Career Report

The Marimo notebook opens to a splash page showing the fitted landscape at a glance with a drag-and-drop upload zone. Drop a PDF resume, and the system encodes it with the same sentence transformer, projects through the fitted SVD, and matches to the nearest career family.

The resulting report is a seven-panel accordion that expands lazily as you open each section:

- **Career Landscape:** Scatter plot of every career family in the SVD coordinate space, with node sizes scaled by betweenness centrality[^49] and the resume overlaid as a gold star showing where you sit relative to the full landscape
- **Skill Analysis:** Demonstrated competencies and skill gaps ranked by cosine similarity against the cluster's [O\*NET](https://www.onetonline.org/) tasks, with gaps ordered by deficit magnitude
- **Career Pathways:** Spring-layout network of advancement and lateral edges from a target cluster, with [apprenticeship](https://www.apprenticeship.gov/) programs and hour requirements annotated on each edge
- **Education & Training:** Registered apprenticeships with [RAPIDS](https://www.apprenticeship.gov/) codes and educational programs reachable through career graph edges from the target cluster
- **Employer Connections:** Posting companies fuzzy-matched against the AGC Maine member directory with career page URLs
- **Job Boards:** Maine and national boards filtered by sector relevance
- **Pipeline Details:** Underlying DAG visualization, cluster profiles, and model metadata for technical audiences

A downloadable plain-text report is available from the sidebar at any point.

---

## 🔧 CLI Reference

Chalkline's CLI is built on [Typer](https://typer.tiangolo.com/) with Rich markup. Running `chalkline` with no arguments prints help.

```bash
uv run chalkline --help
```

### `fit`

Encode postings, cluster into career families, build the career graph, and cache the fitted pipeline. All directory flags are optional, defaulting to sensible project-relative paths that work when running from the repository root.

```bash
uv run chalkline fit                  # fit with default paths
uv run chalkline fit --verbose        # same, with debug-level logs
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--postings-dir` | | `data/postings` | Custom path to posting corpus (*optional*) |
| `--lexicon-dir` | | `data/lexicons` | Custom path to lexicon JSONs (*optional*) |
| `--output-dir` | | `.cache/pipeline` | Custom cache directory (*optional*) |
| `--verbose` | `-v` | `False` | Show diagnostic logs |

### `launch`

Start `marimo run` on the career report notebook. Must be run from the project root where `app/main.py` exists.

```bash
uv run chalkline launch
```

---

## 🧱 Core Dependencies

| Component | Technology | Role |
|-----------|------------|------|
| **Sentence Encoding** | [`onnxruntime`](https://onnxruntime.ai/) + [`tokenizers`](https://github.com/huggingface/tokenizers) | ONNX inference for [`gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) with HuggingFace fast tokenization |
| **Machine Learning** | [`scikit-learn`](https://scikit-learn.org/) | TruncatedSVD, Ward HAC, L2 normalization, cosine similarity |
| **Pipeline Orchestration** | [`sf-hamilton`](https://github.com/DAGWorks-Inc/hamilton) | DAG resolution from function signatures with node-level disk caching |
| **Career Graph** | [`NetworkX`](https://networkx.org/) | Directed weighted graph for stepwise k-NN backbone and reach queries |
| **Corpus Collection** | [`python-jobspy`](https://github.com/Bunsly/JobSpy) | Multi-board job aggregation from Indeed and other sources |
| **PDF Extraction** | [`pdfplumber`](https://github.com/jsvine/pdfplumber) | Resume text extraction with layout-aware parsing |
| **UI** | [`Marimo`](https://marimo.io/) | Reactive notebook with drag-and-drop resume upload |
| **Visualization** | [`Plotly`](https://plotly.com/python/) | Interactive career landscape scatters and graph visualizations |
| **CLI** | [`Typer`](https://typer.tiangolo.com/) | `fit` and `launch` subcommands with Rich markup |
| **Configuration** | [`Pydantic`](https://pydantic.dev/) | `PipelineConfig` with `extra="forbid"` and tuned defaults |
| **Logging** | [`Loguru`](https://github.com/Delgan/loguru) | Structured pipeline progress and debug output |

---

## 🏠 Project Structure

```
chalkline/
├── app/
│   └── main.py                        Marimo reactive notebook (career report)
│
├── data/
│   ├── lexicons/                      Curated domain knowledge (committed)
│   │   ├── credentials.json           Apprenticeships, certifications, programs
│   │   └── onet.json                  21 SOC codes with Tasks, DWAs, Technology Skills, KSAs
│   ├── postings/                      Scraped AGC corpus (gitignored)
│   └── stakeholder/                   AGC Maine reference data (gitignored)
│       └── reference/                 7 JSON files: members, apprenticeships, programs, etc.
│
├── scripts/                           Repeatable data curation (not part of the package)
│   ├── curate_onet.py                 Fetch O*NET tasks, DWAs, technology skills
│   ├── curate_osha.py                 Fetch OSHA standards from eCFR API
│   └── parse_stakeholder.py           Extract AGC workbook into reference JSONs
│
├── src/chalkline/
│   ├── cli/                           Typer CLI with fit and launch subcommands
│   │   ├── __init__.py                App registration and subcommand wiring
│   │   ├── fit.py                     Pipeline fitting with cache-or-compute
│   │   └── launch.py                  Marimo notebook launcher
│   │
│   ├── collection/                    Corpus loading and posting schemas
│   │   ├── collector.py               Filter and key postings from storage
│   │   ├── schemas.py                 Posting Pydantic models
│   │   └── storage.py                 File-backed posting persistence
│   │
│   ├── display/                       Notebook presentation layer
│   │   ├── figures.py                 Plotly figure builders (landscape, pathways)
│   │   ├── layout.py                  Marimo layout helpers (stat rows, filtered accordions)
│   │   └── tables.py                  Row builders for panel tables and text reports
│   │
│   ├── matching/                      Resume-to-career matching
│   │   ├── matcher.py                 SVD projection, nearest centroid, cosine gap analysis
│   │   ├── reader.py                  PDF text extraction via pdfplumber
│   │   └── schemas.py                 MatchResult and gap/demonstrated models
│   │
│   ├── pathways/                      Career graph construction
│   │   ├── graph.py                   NetworkX stepwise k-NN backbone with credential edges
│   │   ├── loaders.py                 Credential and cluster data loading
│   │   └── schemas.py                 Clusters, ClusterProfile, Reach, Edge models
│   │
│   └── pipeline/                      Orchestration and shared types
│       ├── encoder.py                 ONNX sentence transformer wrapper
│       ├── orchestrator.py            Hamilton DAG driver → fitted Chalkline dataclass
│       ├── progress.py                Loguru progress reporting
│       ├── schemas.py                 PipelineConfig (Pydantic, extra="forbid")
│       └── steps.py                   Hamilton node functions (the full DAG)
│
├── tests/                             Pytest suite mirroring src/ structure
├── pyproject.toml                     Build config, dependencies, CLI entry point
└── uv.lock                            Locked dependency versions
```

Each domain subpackage (*`collection/`, `matching/`, `pathways/`, `pipeline/`, `display/`*) owns its schemas and logic. The `pipeline/` subpackage orchestrates the others through Hamilton, where each function in `steps.py` is a DAG node whose parameter names declare its dependencies. The `display/` subpackage separates figure construction, table building, and layout composition so that `app/main.py` stays thin, wiring Marimo cells to display methods without inline chart logic.

---

## 🤝 AGC Maine

[AGC Maine](https://www.agcmaine.org/) (*Associated General Contractors of Maine*) represents **222** member companies and has been the state's primary construction trade association since 1951. The association operates the [Maine Construction Academy](https://buildingmaine.com/) with tuition-free pre-apprenticeship programs expanding to five community colleges in 2026 and manages **19** registered [apprenticeship](https://www.apprenticeship.gov/) pathways spanning trades from carpentry and welding to crane operation and solar installation.

AGC provided the posting corpus, the stakeholder reference data defining the project's **21** [SOC](https://www.bls.gov/soc/) codes and three sectors, and the credential records (*apprenticeships, certifications, educational programs*) that enrich the career graph. The collaboration connects algorithmic career mapping to a real training pipeline[^52][^56], where outputs directly inform which programs AGC recommends to workers entering or advancing through the trades.

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

[^43]: Levy, Shalom & Chalamish. 2025. "A Guide to Similarity Measures and Their Data Science Applications." *Journal of Big Data* 12: 188. https://doi.org/10.1186/s40537-025-01227-1

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

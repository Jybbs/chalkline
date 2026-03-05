<div align="center">

# 📐 Chalkline

### *Unsupervised Career Mapping for Maine's Construction Trades*

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-f7931e.svg)](https://scikit-learn.org/)
[![Marimo](https://img.shields.io/badge/UI-Marimo-009485.svg)](https://marimo.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🏗️ The Problem

Maine's construction industry faces a visibility gap[^1]. Job seekers don't know what career paths exist, employers can't articulate what skills bridge one role to the next, and training programs lack a data-driven map of how trades connect[^9]. The [Green Buildings Career Map](https://greenbuildingscareermap.org/) demonstrated that career mapping works, organizing **55** jobs across **4** sectors with **300+** advancement routes, but it was built by expert panels rather than from the data itself.

Chalkline takes a different approach. Built in partnership with [AGC Maine](https://www.agcmaine.org/) (*Associated General Contractors of Maine*), it applies unsupervised machine learning to construct career pathways algorithmically from job postings. Skills are extracted and normalized against trade-specific lexicons, postings are clustered into career families, and a co-occurrence network reveals which skills travel together. Upload a resume, and the system shows you where you sit in Maine's construction landscape, what skills separate you from your next role, and how to get there.

A chalk line snaps a straight reference path between two points. Chalkline does the same for careers.

---

## 📐 How It Works

The pipeline forks into two parallel tracks after skill extraction, then merges at pathway generation.

**Geometry track** positions jobs and resumes in a shared coordinate space. Skill extraction uses TF-IDF over normalized skill tokens, an approach that recent surveys find competitive with heavier deep-learning methods[^3][^7], especially under weak supervision[^2]. PCA reduces the resulting space to interpretable axes, and postings are clustered into career families, following occupational-modeling work that has scaled this pattern to tens of millions of job postings[^4][^10]. Incoming resumes project into the same space, where distance-based matching identifies the closest career family and highlights skill gaps[^6][^11].

> Skill Extraction → PCA → Clustering → Resume Matching

**Co-occurrence track** reveals which skills cluster in job postings. Pointwise mutual information surfaces skill pairs that appear together more often than chance, revealing the latent mobility network between occupations[^5]. Apriori association mining identifies broader skill bundles, the kind of structured relationships that knowledge-graph approaches have shown improve matching quality[^12].

> Skill Extraction → PMI Network → Pathway Graph

Step 6 merges both tracks, drawing on career trajectory modeling[^13][^15] and reinforcement-learning approaches to pathway optimization[^14] to recommend skill-bridged routes through the career landscape.

| Step | Name | Technique | Purpose |
|------|------|-----------|---------|
| 1 | Text Mining & Skill Extraction | TF-IDF on normalized skills | Convert postings to numerical skill signatures |
| 2 | Dimensionality Reduction | PCA via TruncatedSVD | Reduce sparse skill space to interpretable axes |
| 3 | Clustering | HAC Ward + K-Means/DBSCAN/Mean Shift comparison | Group postings into nested career families |
| 4 | Association Mining | PMI co-occurrence network + Apriori (*DS5230 coverage*) | Discover which skills predict each other |
| 5 | Resume Matching | Euclidean distance in PCA space + Jaccard | Locate a resume in the career landscape |
| 6 | Pathway Generation | Co-occurrence edges + NetworkX + centrality analysis | Recommend skill-bridged career routes |

---

## 🧱 Core Dependencies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **NLP** | [NLTK](https://www.nltk.org/) + [ahocorasick-rs](https://github.com/jcrist/ahocorasick-rs) | Tokenization, lemmatization, multi-pattern matching |
| **ML** | [scikit-learn](https://scikit-learn.org/) | TF-IDF, PCA, K-Means, HAC, StandardScaler |
| **Hierarchical Clustering** | [SciPy](https://scipy.org/) | Ward linkage, dendrogram, cophenetic correlation |
| **Graph Analysis** | [NetworkX](https://networkx.org/) | Pathway construction, centrality, widest path |
| **Association Rules** | [mlxtend](https://rasbt.github.io/mlxtend/) | Apriori frequent itemsets |
| **PDF Extraction** | [pdfplumber](https://github.com/jsvine/pdfplumber) | Resume and posting text extraction |
| **UI** | [Marimo](https://marimo.io/) | Reactive notebook with drag-and-drop upload |
| **Data** | [NumPy](https://numpy.org/) + [pandas](https://pandas.pydata.org/) | Array operations and tabular data processing |
| **Visualization** | [Plotly](https://plotly.com/python/) + [matplotlib](https://matplotlib.org/) | Interactive career graphs, PCA scatters, dendrograms |

---

## 💻 Quick Start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

```bash
git clone https://github.com/jamesParkington/chalkline.git
cd chalkline

uv sync                # install dependencies
uv run chalkline       # launch the Marimo app
```

---

## 📁 Project Structure

Each pipeline step becomes a subpackage under `src/chalkline/` as its issue is implemented. The target architecture:

```
chalkline/
├── app/
│   └── main.py                    Marimo reactive notebook
│
├── data/
│   └── lexicons/                  Curated skill normalization lexicons
│
├── src/chalkline/
│   ├── association/               Step 4: Skill co-occurrence
│   │   ├── apriori.py             Frequent itemsets and association rules
│   │   └── cooccurrence.py        PMI network and Louvain community detection
│   │
│   ├── clustering/                Step 3: Career family identification
│   │   ├── comparison.py          K-Means, DBSCAN, Mean Shift benchmarking
│   │   └── hierarchical.py        HAC Ward linkage, dendrogram, cophenetic correlation
│   │
│   ├── extraction/                Step 1: TF-IDF and skill extraction
│   │   ├── extractor.py           Skill extraction, lemmatization, normalization
│   │   ├── lexicons.py            OSHA and O*NET lexicon normalization
│   │   ├── parsing.py             PDF and text extraction via pdfplumber
│   │   └── vectorizer.py          TF-IDF and binary matrix construction
│   │
│   ├── matching/                  Step 5: Resume-to-career matching
│   │   └── matcher.py             PCA projection, distance metrics, skill gaps
│   │
│   ├── pathways/                  Step 6: Career route generation
│   │   ├── graph.py               NetworkX graph construction and career pathway edges
│   │   └── routing.py             Centrality analysis and widest-path computation
│   │
│   ├── reduction/                 Step 2: Dimensionality reduction
│   │   └── pca.py                 TruncatedSVD, scree analysis, component loadings
│   │
│   ├── __main__.py                Console script entry point (uv run chalkline)
│   ├── config.py                  Pipeline configuration (Pydantic models)
│   ├── enums.py                   Shared StrEnum definitions
│   └── pipeline.py                End-to-end orchestrator
│
└── tests/                         Test suite
```

---

## 🤝 Partnership

Chalkline is developed in partnership with [AGC Maine](https://www.agcmaine.org/), the state's primary construction trade association since 1951. AGC operates the [Maine Construction Academy](https://buildingmaine.com/) with tuition-free pre-apprenticeship programs expanding to five community colleges in 2026, along with **18+** registered apprenticeship pathways spanning trades from carpentry and welding to crane operation and solar installation. Aligning those training programs with actual labor-market demand is an active area of research[^8] and a core motivation for this project.

This project is part of DS5230: Unsupervised Machine Learning at the [Roux Institute](https://roux.northeastern.edu/), Northeastern University.

---

## 📚 References

[^1]: Truitt, Sarah, Juliana Williams, and Madeline Salzman. 2020. "Building the Efficiency Workforce." *National Renewable Energy Laboratory* NREL/CP-5500-75497. https://www.nrel.gov/docs/fy20osti/75497.pdf

[^2]: Zhang, Mike, Kristian Nørgaard Jensen, Rob van der Goot, and Barbara Plank. 2022. "Skill Extraction from Job Postings using Weak Supervision." *RecSys in HR '22: The 2nd Workshop on Recommender Systems for Human Resources, 16th ACM Conference on Recommender Systems. CEUR Workshop Proceedings, Vol. 3218*. https://arxiv.org/abs/2209.08071

[^3]: Senger, Elena, Mike Zhang, Rob van der Goot, and Barbara Plank. 2024. "Deep Learning-based Computational Job Market Analysis: A Survey on Skill Extraction and Classification from Job Postings." *Proceedings of the First Workshop on Natural Language Processing for Human Resources (NLP4HR 2024)*: 1–15. https://doi.org/10.18653/v1/2024.nlp4hr-1.1

[^4]: Dixon, Nile, Marcelle Goggins, Ethan Ho, Mark Howison, Joe Long, Emma Northcott, Karen Shen, and Carrie Yeats. 2023. "Occupational Models from 42 Million Unstructured Job Postings." *Patterns* 4 (7): 100757. https://doi.org/10.1016/j.patter.2023.100757

[^5]: del Rio-Chanona, R. Maria, Penny Mealy, Mariano Beguerisse-Díaz, François Lafond, and J. Doyne Farmer. 2021. "Occupational Mobility and Automation: A Data-Driven Network Model." *Journal of the Royal Society Interface* 18 (174): 20200898. https://doi.org/10.1098/rsif.2020.0898

[^6]: Khelkhal, Kenza, and Dihia Lanasri. 2025. "Smart-Hiring: An Explainable End-to-End Pipeline for CV Information Extraction and Job Matching." *arXiv preprint arXiv:2511.02537*. https://arxiv.org/abs/2511.02537

[^7]: Otani, Naoki, Nikita Bhutani, and Estevam Hruschka. 2025. "Natural Language Processing for Human Resources: A Survey." *Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Industry Track)*: 583–597. https://doi.org/10.18653/v1/2025.naacl-industry.47

[^8]: Frej, Jibril, Anna Dai, Syrielle Montariol, Antoine Bosselut, and Tanja Käser. 2024. "Course Recommender Systems Need to Consider the Job Market." *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24)*. https://doi.org/10.1145/3626772.3657847

[^9]: Hamilton, Virginia. 2012. "Career Pathway and Cluster Skill Development: Promising Models from the United States." *OECD Local Economic and Employment Development (LEED) Papers* No. 2012/14. https://doi.org/10.1787/5k94g1s6f7td-en

[^10]: Lukauskas, Mantas, Viktorija Šarkauskaitė, Vaida Pilinkienė, Alina Stundžienė, Andrius Grybauskas, and Jurgita Bruneckienė. 2023. "Enhancing Skills Demand Understanding through Job Ad Segmentation Using NLP and Clustering Techniques." *Applied Sciences* 13 (10): 6119. https://doi.org/10.3390/app13106119

[^11]: Rosenberger, Julian, Lukas Wolfrum, Sven Weinzierl, Mathias Kraus, and Patrick Zschech. 2025. "CareerBERT: Matching Resumes to ESCO Jobs in a Shared Embedding Space for Generic Job Recommendations." *Expert Systems with Applications* 275: 127043. https://doi.org/10.1016/j.eswa.2025.127043

[^12]: de Groot, Maurits, Jelle Schutte, and David Graus. 2021. "Job Posting-Enriched Knowledge Graph for Skills-based Matching." *RecSys in HR '21: Workshop on Recommender Systems for Human Resources, 15th ACM Conference on Recommender Systems. CEUR Workshop Proceedings, Vol. 2967*. https://arxiv.org/abs/2109.02554

[^13]: Lee, Yeon-Chang, JaeHyun Lee, Michiharu Yamashita, Dongwon Lee, and Sang-Wook Kim. 2025. "CAPER: Enhancing Career Trajectory Prediction using Temporal Knowledge Graph and Ternary Relationship." *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '25)*: 647–658. https://doi.org/10.1145/3690624.3709329

[^14]: Avlonitis, Spyros, Dor Lavi, Masoud Mansoury, and David Graus. 2023. "Career Path Recommendations for Long-term Income Maximization: A Reinforcement Learning Approach." *RecSys in HR '23 Workshop, 17th ACM Conference on Recommender Systems. CEUR Workshop Proceedings, Vol. 3490*. https://arxiv.org/abs/2309.05391

[^15]: Senger, Elena, Yuri Campbell, Rob van der Goot, and Barbara Plank. 2025. "Toward More Realistic Career Path Prediction: Evaluation and Methods." *Frontiers in Big Data* 8: 1564521. https://doi.org/10.3389/fdata.2025.1564521

[^16]: Ramshaw, Lance A., and Mitchell P. Marcus. 1995. "Text Chunking Using Transformation-Based Learning." *Proceedings of the Third Workshop on Very Large Corpora*: 82–94. https://aclanthology.org/W95-0107/

[^17]: Porter, Martin F. 1980. "An Algorithm for Suffix Stripping." *Program* 14 (3): 130–137. https://doi.org/10.1108/eb046814

[^18]: Spärck Jones, Karen. 1972. "A Statistical Interpretation of Term Specificity and Its Application in Retrieval." *Journal of Documentation* 28 (1): 11–21. https://doi.org/10.1108/eb026526

[^19]: Robertson, Stephen. 2004. "Understanding Inverse Document Frequency: On Theoretical Arguments for IDF." *Journal of Documentation* 60 (5): 503–520. https://doi.org/10.1108/00220410410560582

[^20]: Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. 2011. "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." *SIAM Review* 53 (2): 217–288. https://doi.org/10.1137/090771806

[^21]: Deerwester, Scott, Susan T. Dumais, George W. Furnas, Thomas K. Landauer, and Richard Harshman. 1990. "Indexing by Latent Semantic Analysis." *Journal of the American Society for Information Science* 41 (6): 391–407. https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9

[^22]: Cattell, Raymond B. 1966. "The Scree Test for the Number of Factors." *Multivariate Behavioral Research* 1 (2): 245–276. https://doi.org/10.1207/s15327906mbr0102_10

[^23]: Ward, Joe H., Jr. 1963. "Hierarchical Grouping to Optimize an Objective Function." *Journal of the American Statistical Association* 58 (301): 236–244. https://doi.org/10.1080/01621459.1963.10500845

[^24]: Sokal, Robert R., and F. James Rohlf. 1962. "The Comparison of Dendrograms by Objective Methods." *Taxon* 11 (2): 33–40. https://doi.org/10.2307/1217208

[^25]: Hubert, Lawrence, and Phipps Arabie. 1985. "Comparing Partitions." *Journal of Classification* 2 (1): 193–218. https://doi.org/10.1007/BF01908075

[^26]: Caliński, Tadeusz, and Jerzy Harabasz. 1974. "A Dendrite Method for Cluster Analysis." *Communications in Statistics* 3 (1): 1–27. https://doi.org/10.1080/03610927408827101

[^27]: Davies, David L., and Donald W. Bouldin. 1979. "A Cluster Separation Measure." *IEEE Transactions on Pattern Analysis and Machine Intelligence* PAMI-1 (2): 224–227. https://doi.org/10.1109/TPAMI.1979.4766909

[^28]: Rousseeuw, Peter J. 1987. "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis." *Journal of Computational and Applied Mathematics* 20: 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

[^29]: Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 1996. "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD-96)*: 226–231. https://dl.acm.org/doi/10.5555/3001460.3001507

[^30]: Campello, Ricardo J. G. B., Davoud Moulavi, and Jörg Sander. 2013. "Density-Based Clustering Based on Hierarchical Density Estimates." *Advances in Knowledge Discovery and Data Mining (PAKDD 2013), Lecture Notes in Computer Science* 7819: 160–172. https://doi.org/10.1007/978-3-642-37456-2_14

[^31]: Comaniciu, Dorin, and Peter Meer. 2002. "Mean Shift: A Robust Approach Toward Feature Space Analysis." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 24 (5): 603–619. https://doi.org/10.1109/34.1000236

[^32]: Satopaa, Ville, Jeannie Albrecht, David Irwin, and Barath Raghavan. 2011. "Finding a 'Kneedle' in a Haystack: Detecting Knee Points in System Behavior." *2011 31st International Conference on Distributed Computing Systems Workshops (ICDCSW)*: 166–171. https://doi.org/10.1109/ICDCSW.2011.20

[^33]: Church, Kenneth Ward, and Patrick Hanks. 1990. "Word Association Norms, Mutual Information, and Lexicography." *Computational Linguistics* 16 (1): 22–29. https://aclanthology.org/J90-1003/

[^34]: Bouma, Gerlof. 2009. "Normalized (Pointwise) Mutual Information in Collocation Extraction." *Proceedings of the Biennial GSCL Conference 2009*: 31–40. https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf

[^35]: Dunning, Ted. 1993. "Accurate Methods for the Statistics of Surprise and Coincidence." *Computational Linguistics* 19 (1): 61–74. https://aclanthology.org/J93-1003/

[^36]: Blondel, Vincent D., Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. 2008. "Fast Unfolding of Communities in Large Networks." *Journal of Statistical Mechanics: Theory and Experiment* 2008 (10): P10008. https://doi.org/10.1088/1742-5468/2008/10/P10008

[^37]: Newman, M. E. J., and Michelle Girvan. 2004. "Finding and Evaluating Community Structure in Networks." *Physical Review E* 69 (2): 026113. https://doi.org/10.1103/PhysRevE.69.026113

[^38]: Agrawal, Rakesh, and Ramakrishnan Srikant. 1994. "Fast Algorithms for Mining Association Rules in Large Databases." *Proceedings of the 20th International Conference on Very Large Data Bases (VLDB '94)*: 487–499. https://www.vldb.org/conf/1994/P487.PDF

[^39]: Cha, Sung-Hyuk. 2007. "Comprehensive Survey on Distance/Similarity Measures between Probability Density Functions." *International Journal of Mathematical Models and Methods in Applied Sciences* 1 (4): 300–307. https://www.naun.org/main/NAUN/ijmmas/mmmas-49.pdf

[^40]: Pollack, Maurice. 1960. "Letter to the Editor—The Maximum Capacity Through a Network." *Operations Research* 8 (5): 733–736. https://doi.org/10.1287/opre.8.5.733

[^41]: Freeman, Linton C. 1977. "A Set of Measures of Centrality Based on Betweenness." *Sociometry* 40 (1): 35–41. https://doi.org/10.2307/3033543

[^42]: Page, Lawrence, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999. "The PageRank Citation Ranking: Bringing Order to the Web." *Stanford InfoLab Technical Report 1999-66*. http://ilpubs.stanford.edu:8090/422/

---

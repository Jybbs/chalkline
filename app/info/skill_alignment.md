Each task from O\*NET's occupation profile is encoded as a **768-dimensional sentence embedding** using `all-mpnet-base-v2`, the same model applied to every job posting in the corpus. Your resume receives the same treatment, producing a single embedding vector. **Cosine similarity** measures the directional alignment between your resume vector and each task vector:

$$\text{sim}(r, t) = \frac{r \cdot t}{\|r\| \|t\|}$$

Scores closer to 1.0 mean your resume's language closely mirrors the task description. Tasks above the cluster median are strengths; tasks below are growth areas. The **alignment percentile** compares your overall similarity against the distribution of posting similarities within the matched career family, answering the question: how does your resume stack up against the postings employers actually wrote?

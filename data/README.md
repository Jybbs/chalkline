# Data

Chalkline's data pipeline flows through four directories, each with different visibility and purpose. This document describes what lives where, how to regenerate derived artifacts, and what data the pipeline still needs.

---

## Stakeholder Data (*gitignored*)

AGC Maine provided a single Excel workbook in March 2026 containing reference data that scopes the project. It defines which occupations, employers, apprenticeships, educational programs, and job boards are relevant to Maine's commercial and heavy highway construction industry.

The entire `data/stakeholder/` directory is gitignored because the AGC member list carries a privacy restriction (*"to be used to search job postings, LinkedIn profiles, and other public data for project — not to be used for any other purposes"*). The directory is organized into two subdirectories:

- **`raw/`** — the original workbook (`agc-maine-2026.xlsx`)
- **`reference/`** — extracted JSON files, one per logical dataset

### Extracted Files

Run `uv run python data/stakeholder/extract.py` to regenerate all JSON from the workbook into `reference/`. The script resolves embedded Excel hyperlinks to actual URLs and unwraps Google Search redirects.

**`onet_codes.json`** (21 records) — O\*NET occupation codes curated by the stakeholder across three sectors. These 21 codes define the project's occupation scope and feed directly into CL-02's lexicon curation.

| Field | Description |
|-------|-------------|
| `soc_code` | Standard Occupational Classification code (*e.g., 47-2031.00*) |
| `title` | O\*NET occupational title |
| `role_description` | Stakeholder-written description of the role in commercial projects |
| `sector` | Building Construction, Heavy Highway Construction, or Construction Managers |

SOC prefix coverage: `11-` (3), `13-` (1), `17-` (2), `47-` (15). Five codes fall outside the `47-` construction prefix, namely Civil Engineers (17-2051.00), Transportation Engineers (17-2051.01), Project Management Specialists (13-1082.00), Architectural & Engineering Managers (11-9041.00), and Facilities Managers (11-3013.00).

**`apprenticeships.json`** (19 records) — AGC-sponsored registered apprenticeship programs with RAPIDS codes (*DOL program identifiers, not O\*NET codes*) and term lengths. Term lengths range from 2,000 to 8,000 hours, with some expressed as ranges. This data enriches the career pathway graph in CL-11.

| Field | Description |
|-------|-------------|
| `rapids_code` | DOL Registered Apprenticeship program number |
| `title` | Apprenticeship title (*e.g., Construction Carpenter, Crane Operator*) |
| `term_hours` | Required training hours, as string (*some are ranges like "4500 - 5000"*) |

**`agc_members.json`** (222 records) — AGC Maine member companies classified by type. **Privacy-restricted**, use only for job posting searches and public data lookups per the stakeholder's terms.

| Field | Description |
|-------|-------------|
| `name` | Company name |
| `type` | General Contractors (60), Specialty Contractors (67), Suppliers (36), Service Providers (58), Developer (2) |

**`dot_contractors.json`** (56 records) — MaineDOT prequalified bidders with notes on where they post jobs. About 23 companies overlap with the AGC member list.

| Field | Description |
|-------|-------------|
| `company` | Contractor name |
| `website_note` | Job board or website reference (*many are just the company name repeated*) |

**`cc_programs.json`** — Maine Community College System construction programs, split into two sections.

*Degrees* (14 records):

| Field | Description |
|-------|-------------|
| `college` | College abbreviation (*CMCC, EMCC, KVCC, NMCC, SMCC, WCCC, YCCC*) |
| `program` | Program name |
| `credential` | AAS Degree, Certificate, or Short-term Training |
| `url` | Direct link to the program page |

*Initiatives* (6 records): Workforce development programs connected to the community college system, including the Maine Construction Academy immersion programs and the TREC heat pump installer pipeline.

| Field | Description |
|-------|-------------|
| `initiative` | Initiative name |
| `description` | What the program offers |
| `best_for` | Target audience |
| `url` | Direct link to the initiative page |

**`umaine_programs.json`** (10 records) — University of Maine System construction and engineering degree programs across UMaine, UMA, USM, and the UMA/UMPI pathway.

| Field | Description |
|-------|-------------|
| `campus` | UMaine, UMA, USM, or UMA/UMPI |
| `category` | Construction, Architecture, Engineering, or Pathways |
| `degree` | B.S., B.Arch, or 1+3 Program |
| `program` | Program name |
| `url` | Direct link to the program page |

**`job_boards.json`** — Job boards for finding construction postings, split into `maine` (12 records) and `national` (10 records) sections.

| Field | Description |
|-------|-------------|
| `category` | Government, Industry Association, Private Sector, etc. |
| `name` | Board or organization name |
| `focus` | Primary industry focus |
| `best_for` | Target roles or job types |

---

## Lexicons (*tracked*)

`data/lexicons/` holds curated skill normalization CSVs built by CL-01 and CL-02. These are the pipeline's vocabulary, used by the lexicon registry (CL-05) and skill extractor (CL-06) to match raw posting text against canonical skill names.

| File | Issue | Description |
|------|-------|-------------|
| `osha.csv` | CL-01 | OSHA 10/30-hour safety course topics |
| `onet.csv` | CL-02 | O\*NET occupation-skill mappings for the 21 stakeholder-scoped codes |

None of these exist yet. CL-02 will use `data/stakeholder/reference/onet_codes.json` as its scope definition, pulling skills, knowledge, abilities, and Job Zone levels from the O\*NET 30.0 database for those 21 SOC codes.

---

## Postings (*gitignored, not yet received*)

`data/postings/` will hold the job posting corpus that feeds the pipeline's skill extraction step. This is the primary input the project still needs. Kelly from AGC Maine is visiting class on March 12th to clarify how postings will be sourced, whether that is a direct data handoff, scraping guidance using the employer lists above, or another approach.

---

## Processed (*gitignored*)

`data/processed/` holds intermediate and final pipeline artifacts: TF-IDF matrices, PCA projections, cluster assignments, PMI co-occurrence networks, and career pathway graphs. Generated by the pipeline, not checked into version control.

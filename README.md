# Future Interns - Machine Learning Task 3 (2026)
# Resume / Candidate Screening & Ranking Decision-Support System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green)](https://www.nltk.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-REST--API-000000?logo=flask&logoColor=white)](https://palletsprojects.com/p/flask/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Executive Overview
Talent acquisition teams receive hundreds of resumes for a single job opening. Manually screening every resume is slow, inconsistent, fatigue-inducing, and vulnerable to unconscious human bias. High inbound volume often causes high-potential candidates to sit in queues for weeks or get overlooked entirely.

This repository provides an enterprise-grade **Natural Language Processing (NLP) Candidate Screening & Decision-Support System** designed to:
1. **Automate Resume Ingestion & Parsing**: Extracts unstructured text from **PDF, DOCX, and TXT** documents, as well as bulk tabular datasets.
2. **Standardize Skills via a 500+ Taxonomy**: Employs boundary-aware regex matching and canonical normalization to extract technical and domain competencies across 7 major disciplines.
3. **Deconstruct Job Requisitions**: Parses job descriptions into **Must-Have Core Skills**, **Nice-to-Have Preferred Skills**, and minimum experience/degree targets.
4. **Rank Candidates using a Multi-Criteria Composite Fit Index (CFI)**: Blends sublinear TF-IDF semantic cosine similarity with hard skill coverage metrics and experience alignment.
5. **Diagnose Skill Gaps with Full Explainability**: Generates plain-English executive rationale for why each candidate ranked where they did, itemizing matched competencies and missing critical prerequisites.
6. **Enforce Ethical Blind Screening**: Optional PII masking replaces names, emails, phone numbers, and URLs with generic tokens to mitigate demographic bias.

---

## 🏗️ System Architecture

```
[ Inbound Candidate Resumes (PDF / DOCX / TXT / CSV) ]
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│           Text Preprocessing & Privacy Layer           │
│  - Multi-format text extraction (pypdf, python-docx)   │
│  - Optional PII Anonymization (<EMAIL>, <PHONE>, <URL>)│
│  - Lemmatization, Stopword Removal, Unicode Cleaners   │
│  - Structural Section Partitioning                     │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│ Boundary-Aware Taxonomy │ │  Sublinear TF-IDF Space │
│ - 500+ Skills Catalog   │ │ - Unigram + Bigram NLP  │
│ - 7 Domain Categories   │ │ - Calibrated Cosine Fit │
│ - Density & Gaps Log    │ │ - Semantic Vocabulary   │
└────────────┬────────────┘ └────────────┬────────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│          Job Description Requirements Profiler         │
│  - Partition: Must-Have Skills vs. Nice-to-Have Skills │
│  - Extract Experience Target (Yrs) & Seniority Level   │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│         Multi-Criteria Scoring Engine (CFI)            │
│  - Required Skill Match Score (w_req = 0.45)           │
│  - Semantic TF-IDF Similarity (w_sem = 0.35)           │
│  - Preferred / Bonus Skills   (w_pref = 0.10)          │
│  - Experience Alignment Fit   (w_exp = 0.10)           │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│        Recruiter Decision-Support Output Suite         │
│  - Tier Classification: Shortlist, Review, Upskill, Pass│
│  - Transparent Skill Gap Diagnosis (Matched vs Missing)│
│  - Plain-English Executive Recruiter Rationale         │
│  - Interactive Streamlit Dashboard & Flask REST API    │
└────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation: Composite Fit Index (CFI)

In real hiring scenarios, relying exclusively on keyword counts allows keyword-stuffed resumes to dominate, while relying solely on dense embeddings or TF-IDF misses strict non-negotiable requirements.

Our system calculates a normalized composite index:

$$\text{CFI} = w_{\text{req}} \cdot S_{\text{required}} + w_{\text{sem}} \cdot S_{\text{semantic}} + w_{\text{pref}} \cdot S_{\text{preferred}} + w_{\text{exp}} \cdot S_{\text{exp}}$$

### Component Formulations:
1. **Required Skill Coverage ($S_{\text{required}}$)**:
   $$S_{\text{required}} = \frac{|\text{Skills}_{\text{resume}} \cap \text{Skills}_{\text{required}}|}{|\text{Skills}_{\text{required}}|}$$
2. **Semantic Similarity ($S_{\text{semantic}}$)**:
   $$S_{\text{semantic}} = \min\left(1.0, \frac{\mathbf{v}_{\text{resume}} \cdot \mathbf{v}_{\text{job}}}{\|\mathbf{v}_{\text{resume}}\| \|\mathbf{v}_{\text{job}}\| \cdot \tau}\right)$$
   Where vectors are computed using sublinear TF-IDF $(1, 2)$-grams and scaled by empirical alignment factor $\tau = 0.30$.
3. **Preferred / Bonus Skill Boost ($S_{\text{preferred}}$)**:
   $$S_{\text{preferred}} = \frac{|\text{Skills}_{\text{resume}} \cap \text{Skills}_{\text{preferred}}|}{|\text{Skills}_{\text{preferred}}|}$$
4. **Experience Alignment Score ($S_{\text{exp}}$)**:
   $$S_{\text{exp}} = \min\left(1.0, \frac{\text{Exp}_{\text{candidate}}}{\text{Exp}_{\text{required}}}\right)$$

### Operational Tier Thresholds:
- 🟢 **Strong Fit (Shortlist)**: $\text{CFI} \ge 75\%$
- 🔵 **Moderate Fit (Review)**: $55\% \le \text{CFI} < 75\%$
- 🟡 **Needs Upskilling**: $40\% \le \text{CFI} < 55\%$
- 🔴 **Low Fit (Reject)**: $\text{CFI} < 40\%$

---

## 📁 Repository Structure

```
FUTURE_ML_03/
├── data/
│   ├── job_descriptions.json       # 5 production-grade enterprise job specs
│   ├── resumes.csv                 # 20 diverse benchmark candidate profiles
│   └── sample_resumes/             # Physical sample PDF & TXT resumes for live upload testing
│       ├── CAND_001_Dr_Sarah_Chen.pdf
│       ├── CAND_001_Dr_Sarah_Chen.txt
│       ├── CAND_002_Marcus_Vance.pdf
│       └── ...
├── notebooks/
│   └── resume_screening_system.ipynb # End-to-end interactive Jupyter notebook with executed outputs
├── outputs/
│   ├── candidate_rankings.csv      # Full ranked candidate scoring records
│   ├── skill_gap_analysis.csv      # Matched skills, missing gaps, and recruiter explanations
│   └── figures/                    # High-resolution analytical visualization charts
│       ├── 01_candidate_score_distribution.png
│       ├── 02_skill_coverage_comparison.png
│       ├── 03_radar_chart_candidate_comparison.png
│       └── 04_tfidf_vs_skill_score_correlation.png
├── src/
│   ├── __init__.py                 # Clean package exports
│   ├── preprocessing.py            # PDF/DOCX readers, text normalization, PII masking, section splitter
│   ├── skill_extractor.py          # 500+ multi-domain skill taxonomy, boundary regex, categorizer
│   ├── job_parser.py               # Requirement extraction (must-have vs preferred, seniority, experience)
│   ├── matcher.py                  # Sublinear TF-IDF, cosine similarity, Composite Fit Index (CFI) math
│   ├── ranker.py                   # Candidate ranking, tier labeling, gap analysis, plain-English explanations
│   └── generator.py                # Synthetic benchmark generator and PDF resume renderer (ReportLab)
├── templates/
│   └── index.html                  # Responsive modern web UI for Flask / Vercel
├── app.py                          # Flask REST API with endpoints (/api/screen, /api/jobs, /api/parse_resume)
├── build_notebook.py               # Automation script to compile the comprehensive Jupyter notebook
├── BUSINESS_STAKEHOLDER_REPORT.md  # Executive briefing on ROI, ethics, and ATS integration
├── requirements.txt                # Production dependency specification
├── run_pipeline.py                 # End-to-end execution script
├── streamlit_app.py                # Interactive multi-tab recruiter decision-support dashboard
└── vercel.json                     # Serverless deployment configuration
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/srujan1-creator/FUTURE_ML_03.git
cd FUTURE_ML_03
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the End-to-End Screening Pipeline (CLI)
Executes the full NLP matching engine across the benchmark candidate pool, prints leaderboard tables, and saves analytical figures into `outputs/figures/`:
```bash
python run_pipeline.py
```

### 5. Launch the Interactive Streamlit Web App
Launches the full recruiter decision-support dashboard with dynamic weight tuning, radar charts, candidate deep-dive inspection, and live PDF/DOCX file upload:
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 6. Run the Flask REST API & Web UI
```bash
python app.py
```
Access the web interface at `http://localhost:5000` or test API endpoints:
```bash
# Health check
curl http://localhost:5000/api/health

# Screen candidates against custom JD
curl -X POST http://localhost:5000/api/screen \
     -H "Content-Type: application/json" \
     -d '{"job_description": "We need a Senior ML Engineer with Python, PyTorch, Docker, and AWS."}'
```

### 7. Explore the Jupyter Notebook
```bash
jupyter notebook notebooks/resume_screening_system.ipynb
```

---

## 📊 Benchmark Results & Performance

Screening a cohort of 20 benchmark candidates against the **Senior Machine Learning Engineer** role produces the following top shortlist:

| Rank | Candidate Name | Fit Status | Composite Fit (CFI) | Core Skills Match | Semantic Fit | Experience | Missing Critical Skills |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | **Dr. Sarah Chen** | 🟢 **Strong Fit** | **82.8%** | 94.7% | 76.8% | 6.5 yrs | *GCP (Has AWS)* |
| **#2** | **Marcus Vance** | 🔵 **Moderate Fit** | **67.7%** | 89.5% | 49.9% | 5.0 yrs | *GCP, SciPy* |
| **#3** | **Elena Rostova** | 🟡 **Needs Upskilling**| **54.3%** | 68.4% | 47.2% | 3.5 yrs | *CI/CD, GCP, System Design* |
| **#4** | **Kenji Takahashi** | 🟡 **Needs Upskilling**| **47.8%** | 36.8% | 44.6% | 4.5 yrs | *Kubernetes, Tabular ML, SQL* |
| **#5** | **Jordan Lee** | 🔴 **Low Fit** | **35.8%** | 52.6% | 29.0% | 1.0 yrs | *AWS, CI/CD, Kubernetes, 5+ yrs exp* |

### Generated Analytical Figures (in `outputs/figures/`):
- `01_candidate_score_distribution.png`: Horizontal bar leaderboard color-coded by fit tier.
- `02_skill_coverage_comparison.png`: Multi-factor subscore comparison across top candidates.
- `03_radar_chart_candidate_comparison.png`: 5-axis competency radar benchmark (Required Skills, Semantic Fit, Bonus Skills, Experience, Skill Breadth).
- `04_tfidf_vs_skill_score_correlation.png`: Quadrant analysis proving why hybrid scoring separates top talent from keyword outliers.

---

## 💡 Explainable AI & Recruiter Decision-Support

The platform resolves the "black-box" dilemma in HR by providing human-readable explanations:

> **Example Rationale for Rank #1 (Dr. Sarah Chen):**  
> *"Dr. Sarah Chen is a top-tier candidate with 94.7% required skill coverage (18/19 skills: Python, PyTorch, Scikit-Learn, Docker). Meets required experience (6.5 yrs vs 5.0 yrs required). Only misses secondary skill: GCP. Fast-track to Recruiter Phone Screen / Technical Round 1."*

> **Example Rationale for Rank #4 (Kenji Takahashi - GenAI Specialist):**  
> *"Kenji Takahashi demonstrates foundational familiarity but misses critical must-have competencies (Kubernetes, Classical ML, SQL, Tabular ML). Matches only 7 of 19 core requirements. Hold for Generative AI / NLP specialist opening; requires upskilling for core ML role."*

---

## 🛡️ Bias Mitigation & Ethical AI Governance
- **Blind Screening Mode**: Anonymizes names, emails, phone numbers, and web links using regex entity masking.
- **Auditable Records**: All evaluations output explicit skill matches and weights to enable retrospective hiring compliance audits.
- **Human-in-the-Loop**: Acts strictly as a prioritization aid; final interview invitations remain strictly under human recruiter authority.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Acknowledgments
Developed for **Future Interns Machine Learning Internship Task 3 (2026)**.  
Built with Python 3.11, Scikit-Learn, NLTK, Streamlit, and ReportLab.

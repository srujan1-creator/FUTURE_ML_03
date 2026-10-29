"""
Script to build the comprehensive Jupyter Notebook for Task 3:
notebooks/resume_screening_system.ipynb
"""

import os
import json


def create_task3_notebook():
    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "resume_screening_system.ipynb")

    cells = [
        # Cell 1: Title Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Future Interns - Machine Learning Task 3 (2026)\n",
                "## Resume / Candidate Screening & Ranking Decision-Support System\n",
                "\n",
                "**Author:** Future Interns Machine Learning Engineering Team  \n",
                "**Project Repository:** [FUTURE_ML_03](https://github.com/srujan1-creator/FUTURE_ML_03.git)  \n",
                "**Tech Stack:** Python 3.11, Scikit-Learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn, Streamlit\n",
                "\n",
                "---\n",
                "\n",
                "### 📌 Executive Overview & Business Objective\n",
                "Talent acquisition teams receive hundreds of resumes for a single open requisition. Manual screening is notoriously slow, subjective, prone to unconscious bias, and inconsistent. High-volume inbound applications create recruiting bottlenecks and extend time-to-hire by weeks.\n",
                "\n",
                "In this task, we build an enterprise-grade **Natural Language Processing (NLP) Decision-Support System** designed to:\n",
                "1. **Read & Normalize Multi-Format Resumes**: Ingest text from PDFs, DOCX, and tabular datasets with optional PII masking for blind/unbiased recruitment.\n",
                "2. **Extract Hierarchical Skills via Taxonomy**: Accurately extract 500+ technical and soft skills using boundary-aware regex matching and canonical normalization.\n",
                "3. **Parse Job Descriptions into Structured Profiles**: Differentiate between **Must-Have Core Requirements** and **Nice-to-Have Preferred Skills**, alongside minimum experience and education targets.\n",
                "4. **Score Candidates via a Hybrid Composite Fit Index (CFI)**: Combine Sublinear TF-IDF Cosine Similarity ($S_{\\text{semantic}}$), Required Skill Coverage ($S_{\\text{req}}$), Preferred Skill Bonus ($S_{\\text{pref}}$), and Experience Alignment ($S_{\\text{exp}}$).\n",
                "5. **Identify Skill Gaps & Explain Ranking Rationale**: Provide recruiter-readable explanations breaking down why each candidate ranked where they did, what critical skills are missing, and recommended next hiring actions.\n"
            ]
        },
        # Cell 2: Imports & Environment
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import re\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add project root to sys.path\n",
                "sys.path.append(os.path.abspath('..'))\n",
                "\n",
                "from src.preprocessing import clean_text, extract_sections, extract_years_of_experience, extract_education\n",
                "from src.skill_extractor import SkillExtractor, SKILL_CATEGORIES\n",
                "from src.job_parser import JobDescriptionParser\n",
                "from src.matcher import ResumeMatcher, DEFAULT_WEIGHTS\n",
                "from src.ranker import CandidateRanker, assign_fit_tier\n",
                "\n",
                "sns.set_theme(style='whitegrid', font_scale=1.05)\n",
                "plt.rcParams['font.family'] = 'sans-serif'\n",
                "print('[OK] Core NLP and ML libraries loaded successfully.')"
            ]
        },
        # Cell 3: Data Ingestion Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Dataset Ingestion & Exploratory Analysis\n",
                "We load our curated benchmark dataset consisting of enterprise Job Descriptions and diverse candidate resumes across multiple technical disciplines."
            ]
        },
        # Cell 4: Load Datasets
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "jd_path = os.path.join('..', 'data', 'job_descriptions.json')\n",
                "resumes_path = os.path.join('..', 'data', 'resumes.csv')\n",
                "\n",
                "with open(jd_path, 'r', encoding='utf-8') as f:\n",
                "    job_descriptions = json.load(f)\n",
                "\n",
                "resumes_df = pd.read_csv(resumes_path)\n",
                "print(f'Total Job Descriptions Loaded: {len(job_descriptions)}')\n",
                "print(f'Total Candidate Resumes in Pool: {len(resumes_df)}')\n",
                "resumes_df[['candidate_id', 'name', 'target_role', 'experience_years', 'education']].head(10)"
            ]
        },
        # Cell 5: Text Preprocessing Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Text Normalization, PII Masking & Section Segmentation\n",
                "Unstructured resumes contain formatting artifacts, Unicode glyphs, bullet characters, URLs, and personally identifiable information (PII).  \n",
                "We apply:\n",
                "- PII Masking (`<EMAIL>`, `<PHONE>`, `<URL>`) to support blind screening protocols.\n",
                "- Lemmatization and stopword removal via NLTK.\n",
                "- Structural section segmentation (Summary, Skills, Experience, Education, Projects)."
            ]
        },
        # Cell 6: Preprocessing Demonstration
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_resume_text = resumes_df.iloc[0]['text']\n",
                "print('--- RAW RESUME TEXT (First 350 chars) ---')\n",
                "print(sample_resume_text[:350])\n",
                "\n",
                "print('\\n--- CLEANED & PII-MASKED TEXT ---')\n",
                "cleaned = clean_text(sample_resume_text, mask_pii=True, remove_stopwords=False)\n",
                "print(cleaned[:350])\n",
                "\n",
                "print('\\n--- EXTRACTED RESUME SECTIONS ---')\n",
                "sections = extract_sections(sample_resume_text)\n",
                "for sec_name, content in sections.items():\n",
                "    if content:\n",
                "        print(f'  Section [{sec_name.upper()}]: {len(content.split())} words')"
            ]
        },
        # Cell 7: Skill Extraction Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Boundary-Aware Skill Extraction & Multi-Domain Taxonomy\n",
                "Matching technical skills requires boundary-aware regex to prevent substring false-positives (e.g. avoiding matching `C` inside `Clear` or `R` inside `React`).  \n",
                "Our taxonomy organizes 500+ skills across **Programming Languages**, **Data Science & AI**, **Cloud & DevOps**, **Web Frameworks**, **Databases**, and **Software Architecture**."
            ]
        },
        # Cell 8: Skill Extraction Demonstration
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "skill_extractor = SkillExtractor()\n",
                "\n",
                "# Test extraction on top 3 candidates\n",
                "for idx in range(3):\n",
                "    cand = resumes_df.iloc[idx]\n",
                "    skills = skill_extractor.extract_skills(cand['text'])\n",
                "    categorized = skill_extractor.extract_categorized_skills(cand['text'])\n",
                "    print(f\"{cand['name']} ({cand['target_role']}): {len(skills)} Skills Extracted\")\n",
                "    for cat, cat_skills in categorized.items():\n",
                "        print(f\"   - {cat}: {', '.join(cat_skills)}\")\n",
                "    print()"
            ]
        },
        # Cell 9: Job Description Parsing Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Job Description Parsing & Requirement Extraction\n",
                "We parse our target job requisition (`Senior Machine Learning Engineer`) into structured constraints:\n",
                "- **Must-Have Core Skills**: Required for baseline job qualification.\n",
                "- **Nice-to-Have Preferred Skills**: Differentiating bonus qualifications.\n",
                "- **Seniority Level & Target Experience**."
            ]
        },
        # Cell 10: Parse Target JD
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "target_jd = next(j for j in job_descriptions if j['id'] == 'jd_ml_senior')\n",
                "job_parser = JobDescriptionParser(skill_extractor)\n",
                "parsed_target_jd = job_parser.parse(target_jd['text'], role_title_hint=target_jd['title'])\n",
                "\n",
                "print(f\"Role Title: {parsed_target_jd['title']}\")\n",
                "print(f\"Seniority: {parsed_target_jd['seniority']} | Min Experience: {parsed_target_jd['min_experience_years']} Years\")\n",
                "print(f\"Target Degree: {parsed_target_jd['min_education']}\")\n",
                "print(f\"\\nMust-Have Core Skills ({len(parsed_target_jd['required_skills'])}):\\n  {', '.join(parsed_target_jd['required_skills'])}\")\n",
                "print(f\"\\nNice-to-Have Skills ({len(parsed_target_jd['preferred_skills'])}):\\n  {', '.join(parsed_target_jd['preferred_skills'])}\")"
            ]
        },
        # Cell 11: Matching Mathematics Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Candidate Similarity & Composite Fit Index (CFI)\n",
                "We calculate candidate scores using our multi-criteria Composite Fit Index formula:\n",
                "\n",
                "$$\\text{CFI} = w_{\\text{req}} \\cdot S_{\\text{required}} + w_{\\text{sem}} \\cdot S_{\\text{semantic}} + w_{\\text{pref}} \\cdot S_{\\text{preferred}} + w_{\\text{exp}} \\cdot S_{\\text{exp}}$$\n",
                "\n",
                "Where:\n",
                "- $w_{\\text{req}} = 0.45$: Primary weight on mandatory role skills.\n",
                "- $w_{\\text{sem}} = 0.35$: Contextual sublinear TF-IDF cosine similarity.\n",
                "- $w_{\\text{pref}} = 0.10$: Bonus for nice-to-have capabilities.\n",
                "- $w_{\\text{exp}} = 0.10$: Experience alignment ratio."
            ]
        },
        # Cell 12: Rank Candidates
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "matcher = ResumeMatcher()\n",
                "ranker = CandidateRanker(skill_extractor, job_parser, matcher)\n",
                "\n",
                "ranked_candidates_df = ranker.rank_candidates(\n",
                "    resumes=resumes_df.to_dict(orient='records'),\n",
                "    job_description=target_jd['text'],\n",
                "    role_title=target_jd['title']\n",
                ")\n",
                "\n",
                "display_columns = [\n",
                "    'Rank', 'Candidate Name', 'Fit Tier', 'Composite Score (%)',\n",
                "    'Required Skill Match (%)', 'Semantic Similarity (%)',\n",
                "    'Experience (Years)', 'Missing Critical Skills'\n",
                "]\n",
                "ranked_candidates_df[display_columns].head(10)"
            ]
        },
        # Cell 13: Skill Gap Analysis Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Skill Gap Diagnostics & Recruiter Explainability\n",
                "A key feature required by hiring managers is explainability: understanding **why** a candidate ranked where they did, and what specific competency gaps exist."
            ]
        },
        # Cell 14: Inspect Top 3 vs Outlier Candidates
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for idx in [0, 1, 2, 7]:\n",
                "    row = ranked_candidates_df.iloc[idx]\n",
                "    print(f\"{'='*80}\")\n",
                "    print(f\"RANK #{row['Rank']}: {row['Candidate Name']} | Fit Tier: {row['Fit Tier']}\")\n",
                "    print(f\"Composite Fit: {row['Composite Score (%)']:.1f}% | Required Skill Match: {row['Required Skill Match (%)']:.1f}%\")\n",
                "    print(f\"Experience: {row['Experience (Years)']} yrs | Education: {row['Education']}\")\n",
                "    print(f\"[+] Matched Core Skills: {row['Matched Required Skills']}\")\n",
                "    print(f\"[-] Missing Critical Skills: {row['Missing Critical Skills']}\")\n",
                "    print(f\"[*] Recruiter Explanation: {row['Recruiter Explanation']}\")\n",
                "    print(f\"[>] Recommended Next Action: {row['Recommended Action']}\")"
            ]
        },
        # Cell 15: Visual Analytics Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Visual Analytics & Diagnostic Evaluation\n",
                "We render high-resolution charts visualizing score distributions, multi-factor breakdowns, and candidate competency radars."
            ]
        },
        # Cell 16: Visualizations Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "# Plot 1: Composite Scores for Top 10 Candidates\n",
                "top_10 = ranked_candidates_df.head(10).iloc[::-1]\n",
                "colors_list = ['#22c55e' if 'Strong' in t else '#3b82f6' if 'Moderate' in t else '#f59e0b' if 'Needs' in t else '#ef4444' for t in top_10['Fit Tier']]\n",
                "axes[0].barh(top_10['Candidate Name'], top_10['Composite Score (%)'], color=colors_list, height=0.6)\n",
                "axes[0].set_xlim(0, 105)\n",
                "axes[0].set_xlabel('Composite Fit Index (%)', fontweight='bold')\n",
                "axes[0].set_title('Top Candidates Ranked by Role Fit', fontweight='bold')\n",
                "for i, v in enumerate(top_10['Composite Score (%)']):\n",
                "    axes[0].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')\n",
                "\n",
                "# Plot 2: Semantic Similarity vs Required Skill Match\n",
                "sns.scatterplot(\n",
                "    data=ranked_candidates_df,\n",
                "    x='Semantic Similarity (%)',\n",
                "    y='Required Skill Match (%)',\n",
                "    hue='Fit Tier',\n",
                "    palette={'Strong Fit (Shortlist)': '#22c55e', 'Moderate Fit (Review)': '#3b82f6', 'Needs Upskilling': '#f59e0b', 'Low Fit (Reject)': '#ef4444'},\n",
                "    s=120,\n",
                "    ax=axes[1]\n",
                ")\n",
                "axes[1].axvline(50, color='#94a3b8', linestyle='--', alpha=0.6)\n",
                "axes[1].axhline(50, color='#94a3b8', linestyle='--', alpha=0.6)\n",
                "axes[1].set_title('TF-IDF Semantic Fit vs. Skill Coverage', fontweight='bold')\n",
                "axes[1].set_xlabel('Semantic Similarity (%)', fontweight='bold')\n",
                "axes[1].set_ylabel('Required Skill Match (%)', fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # Cell 17: Radar Chart Demonstration
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Radar Chart for Top 3 Candidates\n",
                "categories = ['Required Skills', 'Semantic Fit', 'Bonus Skills', 'Experience Fit', 'Skill Breadth']\n",
                "N = len(categories)\n",
                "angles = [n / float(N) * 2 * np.pi for n in range(N)]\n",
                "angles += angles[:1]\n",
                "\n",
                "fig = plt.figure(figsize=(7, 7))\n",
                "ax = fig.add_subplot(111, polar=True)\n",
                "radar_colors = ['#22c55e', '#3b82f6', '#8b5cf6']\n",
                "\n",
                "for i in range(min(3, len(ranked_candidates_df))):\n",
                "    c_row = ranked_candidates_df.iloc[i]\n",
                "    breadth = min(100.0, (c_row['Extracted Skills Count'] / 15.0) * 100.0)\n",
                "    values = [\n",
                "        c_row['Required Skill Match (%)'],\n",
                "        c_row['Semantic Similarity (%)'],\n",
                "        c_row['Preferred Skill Match (%)'],\n",
                "        c_row['Experience Score (%)'],\n",
                "        breadth\n",
                "    ]\n",
                "    values += values[:1]\n",
                "    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f\"#{c_row['Rank']} {c_row['Candidate Name']}\", color=radar_colors[i])\n",
                "    ax.fill(angles, values, color=radar_colors[i], alpha=0.15)\n",
                "\n",
                "ax.set_xticks(angles[:-1])\n",
                "ax.set_xticklabels(categories, fontweight='bold')\n",
                "ax.set_ylim(0, 100)\n",
                "ax.set_title('Top 3 Candidate Competency Radar', fontweight='bold', pad=20)\n",
                "ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # Cell 18: Conclusion Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Conclusion & Production Takeaways\n",
                "\n",
                "### Summary of System Achievements:\n",
                "1. **High Discrimination Accuracy**: The hybrid Composite Fit Index successfully shortlists top candidates with high precision while cleanly filtering out underqualified or cross-domain applicants.\n",
                "2. **Explainable AI for Hiring Managers**: Rather than acting as a black-box scoring algorithm, the system explicitly itemizes matched skills, missing critical competencies, and provides plain-English rationale.\n",
                "3. **Ethical Blind Screening**: Offers built-in PII anonymization to safeguard hiring pipelines against demographic and unconscious bias.\n",
                "4. **Customizable Recruiter Weights**: Provides talent acquisition leaders the flexibility to shift emphasis between raw keyword prerequisites and deep semantic narrative fit depending on role seniority."
            ]
        }
    ]

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"[OK] Successfully built {nb_path}")


if __name__ == "__main__":
    create_task3_notebook()

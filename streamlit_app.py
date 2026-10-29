"""
Streamlit Decision-Support Application for Resume & Candidate Screening.
Future Interns - Machine Learning Task 3 (2026)
Features interactive candidate ranking, multi-criteria weight tuning,
skill gap diagnosis, side-by-side radar charts, and live PDF/DOCX resume upload.
"""

import os
import json
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import clean_text, extract_text_from_file, extract_years_of_experience, extract_education
from src.skill_extractor import SkillExtractor, SKILL_CATEGORIES
from src.job_parser import JobDescriptionParser
from src.matcher import ResumeMatcher, DEFAULT_WEIGHTS
from src.ranker import CandidateRanker, assign_fit_tier


# Page configuration
st.set_page_config(
    page_title="AI Resume Screening & Ranking System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .skill-pill {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pill-green { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .pill-red { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .pill-blue { background-color: #dbeafe; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .pill-purple { background-color: #f3e8ff; color: #7e22ce; border: 1px solid #e9d5ff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_screening_system():
    extractor = SkillExtractor()
    parser = JobDescriptionParser(extractor)
    matcher = ResumeMatcher()
    ranker = CandidateRanker(extractor, parser, matcher)
    return extractor, parser, matcher, ranker


@st.cache_data
def load_datasets():
    jd_path = os.path.join("data", "job_descriptions.json")
    resumes_path = os.path.join("data", "resumes.csv")

    if not os.path.exists(jd_path) or not os.path.exists(resumes_path):
        from src.generator import generate_benchmark_dataset
        generate_benchmark_dataset()

    with open(jd_path, "r", encoding="utf-8") as f:
        jds = json.load(f)

    resumes_df = pd.read_csv(resumes_path)
    return jds, resumes_df


extractor, parser, matcher, ranker = load_screening_system()
job_descriptions, benchmark_resumes_df = load_datasets()


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.image("https://img.shields.io/badge/Future_Interns-ML_Task_3-blue?style=for-the-badge", use_container_width=True)
st.sidebar.title("Screening Controls")

# Job Description Selection
jd_titles = [jd["title"] for jd in job_descriptions] + ["+ Custom Job Description"]
selected_jd_title = st.sidebar.selectbox("🎯 Target Job Role", jd_titles, index=0)

if selected_jd_title == "+ Custom Job Description":
    custom_role_title = st.sidebar.text_input("Role Title", value="Senior Software Engineer")
    custom_jd_text = st.sidebar.text_area(
        "Paste Job Description Text",
        value="""Title: Senior Software Engineer
Requirements:
- 4+ years experience with Python, Docker, and AWS.
- Strong knowledge of PostgreSQL, RESTful APIs, and CI/CD.
- Excellent Problem Solving and System Design skills.
Nice to have:
- Experience with Kubernetes and React.""",
        height=200
    )
    current_jd = {
        "title": custom_role_title,
        "text": custom_jd_text,
        "min_experience_years": 3.0,
        "seniority": "Senior"
    }
else:
    current_jd = next(j for j in job_descriptions if j["title"] == selected_jd_title)

# Parse current JD specifications
parsed_jd = parser.parse(current_jd["text"], role_title_hint=current_jd["title"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Scoring Criteria Weights")
st.sidebar.caption("Adjust recruiter decision-making weights (automatically normalized).")

w_req = st.sidebar.slider("Required Skills Weight", 0.1, 0.8, 0.45, 0.05)
w_sem = st.sidebar.slider("Semantic Similarity Weight", 0.1, 0.8, 0.35, 0.05)
w_pref = st.sidebar.slider("Preferred / Bonus Skills Weight", 0.0, 0.4, 0.10, 0.05)
w_exp = st.sidebar.slider("Experience Fit Weight", 0.0, 0.4, 0.10, 0.05)

custom_weights = {
    'required_skills': w_req,
    'semantic_similarity': w_sem,
    'preferred_skills': w_pref,
    'experience_fit': w_exp
}

st.sidebar.markdown("---")
anonymize_mode = st.sidebar.checkbox("🕶️ Blind Screening (Mask PII)", value=False, help="Anonymizes candidate contact info and names to minimize bias.")
min_fit_filter = st.sidebar.slider("Filter: Min Composite Score (%)", 0, 100, 0, 5)


# -------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------
st.markdown('<div class="main-title">💼 Intelligent Resume & Candidate Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated NLP candidate ranking, skill gap diagnosis, and recruiter decision-support powered by Scikit-Learn & NLTK.</div>', unsafe_allow_html=True)

# Job Role Overview Banner
with st.expander(f"📌 Active Job Role Specification: **{current_jd['title']}** (Click to inspect JD requirements)", expanded=False):
    col_jd1, col_jd2 = st.columns([1, 1])
    with col_jd1:
        st.markdown(f"**Seniority Level:** `{parsed_jd['seniority']}`")
        st.markdown(f"**Target Experience:** `{parsed_jd['min_experience_years']} Years`")
        st.markdown(f"**Target Education:** `{parsed_jd['min_education']}`")
    with col_jd2:
        st.markdown("**Required Core Skills:**")
        req_tags = "".join([f'<span class="skill-pill pill-green">{s}</span>' for s in parsed_jd['required_skills']])
        st.markdown(req_tags or "None detected", unsafe_allow_html=True)
        st.markdown("**Preferred / Bonus Skills:**")
        pref_tags = "".join([f'<span class="skill-pill pill-blue">{s}</span>' for s in parsed_jd['preferred_skills']])
        st.markdown(pref_tags or "None detected", unsafe_allow_html=True)


# Tabs Setup
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Live Screening & Leaderboard",
    "🔍 Candidate Deep-Dive & Gaps",
    "⚖️ Side-by-Side Comparison",
    "📄 Instant Upload Screener",
    "📚 Methodology & Skill Taxonomy"
])


# -------------------------------------------------------------
# TAB 1: SCREENING & LEADERBOARD
# -------------------------------------------------------------
with tab1:
    resumes_input = benchmark_resumes_df.to_dict(orient="records")

    # Perform ranking
    ranked_df = ranker.rank_candidates(
        resumes=resumes_input,
        job_description=current_jd["text"],
        role_title=current_jd["title"],
        custom_weights=custom_weights
    )

    # Filter by min score if specified
    if min_fit_filter > 0:
        ranked_df = ranked_df[ranked_df["Composite Score (%)"] >= min_fit_filter].reset_index(drop=True)
        ranked_df["Rank"] = range(1, len(ranked_df) + 1)

    # Metric Cards
    total_screened = len(ranked_df)
    shortlisted = len(ranked_df[ranked_df["Fit Tier"].str.contains("Strong Fit")])
    in_review = len(ranked_df[ranked_df["Fit Tier"].str.contains("Moderate Fit")])
    upskilling = len(ranked_df[ranked_df["Fit Tier"].str.contains("Needs Upskilling")])
    rejected = len(ranked_df[ranked_df["Fit Tier"].str.contains("Low Fit")])

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.metric("Total Candidates", total_screened)
    with mcol2:
        st.metric("🟢 Shortlisted", shortlisted)
    with mcol3:
        st.metric("🔵 In Review", in_review)
    with mcol4:
        st.metric("🟡 Needs Upskilling", upskilling)
    with mcol5:
        st.metric("🔴 Low Fit", rejected)

    st.markdown("### Candidate Shortlist Leaderboard")

    # Display Table with styled highlights
    table_cols = [
        "Rank", "Candidate Name", "Fit Tier", "Composite Score (%)",
        "Required Skill Match (%)", "Semantic Similarity (%)",
        "Experience (Years)", "Extracted Skills Count", "Missing Critical Skills"
    ]

    display_table = ranked_df[table_cols].copy()
    if anonymize_mode:
        display_table["Candidate Name"] = [f"Candidate #{r}" for r in display_table["Rank"]]

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Composite Score (%)": st.column_config.ProgressColumn(
                "Composite Fit (CFI)",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
            "Required Skill Match (%)": st.column_config.ProgressColumn(
                "Core Skills Match",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
            "Semantic Similarity (%)": st.column_config.ProgressColumn(
                "Semantic Fit",
                format="%.1f%%",
                min_value=0,
                max_value=100
            )
        }
    )

    # Download CSV Button
    csv_bytes = ranked_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Ranked Screening Report (CSV)",
        data=csv_bytes,
        file_name=f"screening_report_{current_jd['title'].replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )

    st.markdown("---")
    # Score Distribution Chart
    st.markdown("#### Candidate Score Ranking Distribution")
    chart_df = ranked_df.head(12).copy()
    if anonymize_mode:
        chart_df["Candidate Name"] = [f"Candidate #{r}" for r in chart_df["Rank"]]

    fig_bar = px.bar(
        chart_df,
        x="Composite Score (%)",
        y="Candidate Name",
        orientation="h",
        color="Fit Tier",
        color_discrete_map={
            "Strong Fit (Shortlist)": "#22c55e",
            "Moderate Fit (Review)": "#3b82f6",
            "Needs Upskilling": "#f59e0b",
            "Low Fit (Reject)": "#ef4444"
        },
        text="Composite Score (%)",
        title=f"Top Candidates Ranked by Composite Fit Index for: {current_jd['title']}"
    )
    fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=420)
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)


# -------------------------------------------------------------
# TAB 2: CANDIDATE DEEP-DIVE & SKILL GAPS
# -------------------------------------------------------------
with tab2:
    st.markdown("### 🔍 Individual Candidate Diagnostics & Skill Gap Inspector")

    cand_names = [f"#{row['Rank']} - {row['Candidate Name']} ({row['Fit Tier']})" for _, row in ranked_df.iterrows()]
    selected_cand_idx = st.selectbox("Select Candidate to Inspect", range(len(cand_names)), format_func=lambda x: cand_names[x])

    cand_row = ranked_df.iloc[selected_cand_idx]

    # Candidate Profile Header
    cand_col1, cand_col2, cand_col3 = st.columns([1.5, 1, 1])
    with cand_col1:
        c_display_name = f"Candidate #{cand_row['Rank']}" if anonymize_mode else cand_row['Candidate Name']
        st.markdown(f"### {c_display_name}")
        st.markdown(f"**Fit Status:** `{cand_row['Fit Tier']}` | **Overall Rank:** `#{cand_row['Rank']} of {len(ranked_df)}`")
    with cand_col2:
        st.metric("Composite Fit Index (CFI)", f"{cand_row['Composite Score (%)']:.1f}%")
    with cand_col3:
        exp_target = parsed_jd.get('min_experience_years', 3.0)
        exp_delta = f"{cand_row['Experience (Years)'] - exp_target:+.1f} yrs vs target"
        st.metric("Experience", f"{cand_row['Experience (Years)']:.1f} Yrs", delta=exp_delta)

    # Recruiter Plain-English Explanation
    st.info(f"💡 **Recruiter Executive Summary:**\n\n{cand_row['Recruiter Explanation']}\n\n**Action Verdict:** *{cand_row['Recommended Action']}*")

    st.markdown("---")
    # Sub-scores breakdown
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(f"**Required Skills:** `{cand_row['Required Skill Match (%)']:.1f}%`")
        st.progress(cand_row['Required Skill Match (%)'] / 100.0)
    with g2:
        st.markdown(f"**Semantic Similarity:** `{cand_row['Semantic Similarity (%)']:.1f}%`")
        st.progress(cand_row['Semantic Similarity (%)'] / 100.0)
    with g3:
        st.markdown(f"**Preferred Skills:** `{cand_row['Preferred Skill Match (%)']:.1f}%`")
        st.progress(cand_row['Preferred Skill Match (%)'] / 100.0)
    with g4:
        st.markdown(f"**Experience Fit:** `{cand_row['Experience Score (%)']:.1f}%`")
        st.progress(cand_row['Experience Score (%)'] / 100.0)

    st.markdown("---")
    # Detailed Skill Gap Analysis
    sg1, sg2 = st.columns(2)
    with sg1:
        st.markdown("#### 🟢 Matched Required Skills (Present in Resume)")
        matched_req = cand_row['_matched_req_list']
        if matched_req:
            tags = "".join([f'<span class="skill-pill pill-green">{s}</span>' for s in matched_req])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.warning("No core required skills matched.")

        st.markdown("#### 🔵 Matched Preferred / Bonus Skills")
        matched_pref = cand_row['_matched_pref_list']
        if matched_pref:
            tags = "".join([f'<span class="skill-pill pill-blue">{s}</span>' for s in matched_pref])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.write("None matched.")

    with sg2:
        st.markdown("#### 🔴 Missing Critical Skills (Gaps to Address)")
        missing_req = cand_row['_missing_req_list']
        if missing_req:
            tags = "".join([f'<span class="skill-pill pill-red">⚠️ {s}</span>' for s in missing_req])
            st.markdown(tags, unsafe_allow_html=True)
            st.caption("These critical skills were requested in the job description but absent from the candidate resume.")
        else:
            st.success("🎉 Zero skill gaps! Candidate covers 100% of required competencies.")

        st.markdown("#### 🟣 Surplus / Additional Capabilities")
        surplus = cand_row['_surplus_list']
        if surplus:
            tags = "".join([f'<span class="skill-pill pill-purple">{s}</span>' for s in surplus[:10]])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.write("No extra skills detected.")

    with st.expander("📄 View Candidate Raw Resume Text", expanded=False):
        st.text(cand_row['_raw_text'])


# -------------------------------------------------------------
# TAB 3: SIDE-BY-SIDE CANDIDATE COMPARISON
# -------------------------------------------------------------
with tab3:
    st.markdown("### ⚖️ Multi-Candidate Side-by-Side Comparison")
    st.write("Select 2 to 4 candidates to benchmark competencies, radar footprints, and skill overlaps.")

    all_cand_names = list(ranked_df["Candidate Name"])
    default_selection = all_cand_names[:3] if len(all_cand_names) >= 3 else all_cand_names
    selected_for_comp = st.multiselect("Select Candidates to Compare", all_cand_names, default=default_selection)

    if len(selected_for_comp) >= 2:
        comp_df = ranked_df[ranked_df["Candidate Name"].isin(selected_for_comp)]

        # Radar Comparison
        categories = ["Required Skills", "Semantic Fit", "Bonus Skills", "Experience Fit", "Skill Breadth"]
        fig_radar = go.Figure()

        colors_radar = ["#22c55e", "#3b82f6", "#8b5cf6", "#ec4899"]
        for idx, (_, row) in enumerate(comp_df.iterrows()):
            breadth = min(100.0, (row["Extracted Skills Count"] / 15.0) * 100.0)
            values = [
                row["Required Skill Match (%)"],
                row["Semantic Similarity (%)"],
                row["Preferred Skill Match (%)"],
                row["Experience Score (%)"],
                breadth
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"#{row['Rank']} {row['Candidate Name']}",
                line_color=colors_radar[idx % len(colors_radar)]
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Candidate Multi-Criteria Competency Comparison",
            height=450
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Skill Overlap Heatmap Table
        st.markdown("#### Required Skill Coverage Matrix")
        req_skills_list = parsed_jd["required_skills"]

        matrix_rows = []
        for s in req_skills_list:
            row_dict = {"Required Skill": s}
            for _, cand in comp_df.iterrows():
                row_dict[cand["Candidate Name"]] = "✅ Yes" if s in cand["_matched_req_list"] else "❌ Missing"
            matrix_rows.append(row_dict)

        matrix_df = pd.DataFrame(matrix_rows)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    else:
        st.info("Please select at least 2 candidates above to render comparison.")


# -------------------------------------------------------------
# TAB 4: INSTANT RESUME UPLOAD SCREENER
# -------------------------------------------------------------
with tab4:
    st.markdown("### 📄 Real-Time File Upload Screener")
    st.write("Upload candidate resumes in **PDF**, **DOCX**, or **TXT** format to automatically parse, score, and rank them against the active Job Description.")

    uploaded_files = st.file_uploader(
        "Upload Resume Files (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s). Processing with NLP engine...")

        custom_resumes = []
        for idx, f in enumerate(uploaded_files):
            file_bytes = f.read()
            text = extract_text_from_file(file_bytes, filename=f.name)
            if not text.strip():
                st.warning(f"Could not extract text from {f.name}. Ensure it contains selectable text.")
                continue

            custom_resumes.append({
                "candidate_id": f"UPLOAD_{idx+1:03d}",
                "name": f.name.rsplit('.', 1)[0].replace('_', ' '),
                "text": text,
                "experience_years": extract_years_of_experience(text)
            })

        if custom_resumes:
            custom_ranked_df = ranker.rank_candidates(
                resumes=custom_resumes,
                job_description=current_jd["text"],
                role_title=current_jd["title"],
                custom_weights=custom_weights
            )

            st.markdown("#### Uploaded Candidates Screening Results")
            up_cols = [
                "Rank", "Candidate Name", "Fit Tier", "Composite Score (%)",
                "Required Skill Match (%)", "Semantic Similarity (%)",
                "Experience (Years)", "Missing Critical Skills", "Recommended Action"
            ]
            st.dataframe(
                custom_ranked_df[up_cols],
                use_container_width=True,
                hide_index=True
            )
    else:
        st.markdown("""
        > **💡 Quick Test Tip:** Sample test resumes are already stored in `data/sample_resumes/` in both `.pdf` and `.txt` format! You can drag and drop them here to test the live parser.
        """)


# -------------------------------------------------------------
# TAB 5: METHODOLOGY & SKILL TAXONOMY
# -------------------------------------------------------------
with tab5:
    st.markdown("### 📚 Algorithmic Methodology & Skill Taxonomy")

    st.markdown("""
    #### 1. Mathematical Formulation: Composite Fit Index (CFI)
    Real-world resume screening cannot rely on a single isolated metric. Pure keyword matching fails on context, while pure cosine similarity misses hard non-negotiable prerequisites.
    
    Our hybrid engine balances both via a normalized weighted composite index:
    """)

    st.latex(r"""
    \text{CFI} = w_{\text{req}} \cdot S_{\text{required}} + w_{\text{sem}} \cdot S_{\text{semantic}} + w_{\text{pref}} \cdot S_{\text{preferred}} + w_{\text{exp}} \cdot S_{\text{exp}}
    """)

    st.markdown("""
    Where:
    - **$S_{\text{required}}$**: $\\frac{|\\text{Skills}_{\\text{resume}} \\cap \\text{Skills}_{\\text{required}}|}{|\\text{Skills}_{\\text{required}}|}$ (Core mandatory skill overlap).
    - **$S_{\text{semantic}}$**: Cosine similarity of sublinear TF-IDF vectors (unigrams + bigrams) between preprocessed resume and job description.
    - **$S_{\text{preferred}}$**: $\\frac{|\\text{Skills}_{\\text{resume}} \\cap \\text{Skills}_{\\text{preferred}}|}{|\\text{Skills}_{\\text{preferred}}|}$ (Bonus qualifications boost).
    - **$S_{\text{exp}}$**: $\\min\\left(1.0, \\frac{\\text{Exp}_{\\text{candidate}}}{\\text{Exp}_{\\text{required}}}\\right)$ (Experience alignment).
    """)

    st.markdown("---")
    st.markdown("#### 2. Comprehensive 500+ Skill Taxonomy Explorer")
    st.write("Our NLP boundary-aware taxonomy captures skills across 7 technological and managerial domains:")

    for cat_name, skills in SKILL_CATEGORIES.items():
        with st.expander(f"📁 {cat_name} ({len(skills)} skills)", expanded=False):
            pills = "".join([f'<span class="skill-pill pill-blue">{s}</span>' for s in skills])
            st.markdown(pills, unsafe_allow_html=True)

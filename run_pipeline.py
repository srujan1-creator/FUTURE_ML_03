"""
End-to-End Execution Pipeline for Candidate Screening & Ranking System.
Loads datasets, parses job roles, executes multi-criteria ranking,
exports tabular reports to outputs/, and generates analytical figures.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import clean_text
from src.skill_extractor import SkillExtractor, SKILL_CATEGORIES
from src.job_parser import JobDescriptionParser
from src.matcher import ResumeMatcher
from src.ranker import CandidateRanker


def run_screening_pipeline():
    print("=" * 75)
    print("   CANDIDATE SCREENING & RESUME RANKING DECISION-SUPPORT PIPELINE")
    print("=" * 75)

    os.makedirs("outputs/figures", exist_ok=True)

    # 1. Load Data
    jd_path = os.path.join("data", "job_descriptions.json")
    resumes_path = os.path.join("data", "resumes.csv")

    if not os.path.exists(jd_path) or not os.path.exists(resumes_path):
        print("Data files not found. Generating benchmark dataset...")
        from src.generator import generate_benchmark_dataset
        generate_benchmark_dataset()

    with open(jd_path, "r", encoding="utf-8") as f:
        job_descriptions = json.load(f)

    resumes_df = pd.read_csv(resumes_path)
    resumes = resumes_df.to_dict(orient="records")

    print(f"[OK] Loaded {len(job_descriptions)} Job Descriptions")
    print(f"[OK] Loaded {len(resumes)} Candidate Resumes")

    # 2. Select Primary Benchmark Role: Senior ML Engineer
    primary_jd = next(jd for jd in job_descriptions if jd["id"] == "jd_ml_senior")
    print(f"\n[>>] Target Screening Role: {primary_jd['title']}")
    print(f"    Required Experience: {primary_jd['min_experience_years']} Years | Seniority: {primary_jd['seniority']}")

    # 3. Initialize NLP & ML Engine
    skill_extractor = SkillExtractor()
    job_parser = JobDescriptionParser(skill_extractor)
    matcher = ResumeMatcher()
    ranker = CandidateRanker(skill_extractor, job_parser, matcher)

    # 4. Execute Ranking
    print("\n[>>] Executing Hybrid NLP & Semantic Matching Engine...")
    ranked_df = ranker.rank_candidates(
        resumes=resumes,
        job_description=primary_jd["text"],
        role_title=primary_jd["title"]
    )

    # 5. Display Leaderboard Summary
    display_cols = [
        "Rank", "Candidate Name", "Fit Tier", "Composite Score (%)",
        "Required Skill Match (%)", "Semantic Similarity (%)",
        "Experience (Years)", "Missing Critical Skills"
    ]
    print("\n" + "=" * 95)
    print("                              TOP SHORTLIST LEADERBOARD")
    print("=" * 95)
    print(ranked_df[display_cols].head(8).to_string(index=False))

    # 6. Export Tabular Results
    rankings_csv = os.path.join("outputs", "candidate_rankings.csv")
    ranked_df.to_csv(rankings_csv, index=False)
    print(f"\n[OK] Exported full candidate rankings to {rankings_csv}")

    # Skill Gap Summary CSV
    gap_cols = [
        "Rank", "Candidate Name", "Fit Tier", "Composite Score (%)",
        "Matched Required Skills", "Missing Critical Skills",
        "Matched Preferred Skills", "Surplus Skills", "Recruiter Explanation"
    ]
    gap_csv = os.path.join("outputs", "skill_gap_analysis.csv")
    ranked_df[gap_cols].to_csv(gap_csv, index=False)
    print(f"[OK] Exported skill gap diagnosis to {gap_csv}")

    # 7. Generate Analytical Visualizations
    generate_visualizations(ranked_df, primary_jd)
    print("[OK] Generated high-resolution visualization figures in outputs/figures/")

    print("\n" + "=" * 75)
    print("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


def generate_visualizations(df: pd.DataFrame, jd: dict):
    """
    Renders 4 high-quality analytical figures highlighting scoring and skill gaps.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["font.family"] = "sans-serif"

    # --- FIGURE 1: Stacked Score Breakdown for Top Candidates ---
    fig, ax = plt.subplots(figsize=(11, 6))
    top_10 = df.head(10).copy().iloc[::-1]  # reverse so rank 1 is on top

    y_pos = np.arange(len(top_10))
    bar_height = 0.55

    # Scores
    c_names = [f"#{row['Rank']} {row['Candidate Name']}" for _, row in top_10.iterrows()]
    scores = top_10["Composite Score (%)"].values
    tiers = top_10["Fit Tier"].values

    palette = {"Strong Fit (Shortlist)": "#22c55e", "Moderate Fit (Review)": "#3b82f6",
               "Needs Upskilling": "#f59e0b", "Low Fit (Reject)": "#ef4444"}
    bar_colors = [palette.get(t, "#64748b") for t in tiers]

    bars = ax.barh(y_pos, scores, height=bar_height, color=bar_colors, edgecolor="none", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(c_names, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Composite Fit Index (CFI %)", fontsize=12, fontweight="bold")
    ax.set_title(f"Candidate Screening Leaderboard: {jd['title']}", fontsize=14, fontweight="bold", pad=15)

    # Annotate score labels
    for bar, score in zip(bars, scores):
        ax.text(score + 1.2, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%",
                va="center", ha="left", fontsize=10, fontweight="bold", color="#1e293b")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#22c55e", label="Strong Fit (≥75%)"),
        Patch(facecolor="#3b82f6", label="Moderate Fit (55-74%)"),
        Patch(facecolor="#f59e0b", label="Needs Upskilling (40-54%)"),
        Patch(facecolor="#ef4444", label="Low Fit (<40%)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    fig.savefig(os.path.join("outputs", "figures", "01_candidate_score_distribution.png"), dpi=300)
    plt.close()

    # --- FIGURE 2: Multi-Factor Sub-Score Comparison for Top 6 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    top_6 = df.head(6)

    sub_cols = ["Required Skill Match (%)", "Semantic Similarity (%)", "Preferred Skill Match (%)", "Experience Score (%)"]
    tidy_df = pd.melt(top_6, id_vars=["Candidate Name"], value_vars=sub_cols, var_name="Evaluation Metric", value_name="Score")

    sns.barplot(data=tidy_df, x="Candidate Name", y="Score", hue="Evaluation Metric",
                palette=["#2563eb", "#06b6d4", "#8b5cf6", "#10b981"], ax=ax)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Candidate", fontsize=12, fontweight="bold")
    ax.set_title("Multi-Factor NLP & Criteria Breakdown (Top 6 Candidates)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(title="Scoring Component", framealpha=0.95, loc="upper right")
    plt.xticks(fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join("outputs", "figures", "02_skill_coverage_comparison.png"), dpi=300)
    plt.close()

    # --- FIGURE 3: Radar Chart for Top 3 Candidates ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    categories = ["Required Skills", "Semantic Fit", "Bonus Skills", "Experience Fit", "Skill Breadth"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close circle

    radar_colors = ["#22c55e", "#3b82f6", "#8b5cf6"]
    for i in range(min(3, len(df))):
        c_row = df.iloc[i]
        breadth_score = min(100.0, (c_row["Extracted Skills Count"] / 15.0) * 100.0)
        values = [
            c_row["Required Skill Match (%)"],
            c_row["Semantic Similarity (%)"],
            c_row["Preferred Skill Match (%)"],
            c_row["Experience Score (%)"],
            breadth_score
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, linestyle="solid", label=f"#{c_row['Rank']} {c_row['Candidate Name']}", color=radar_colors[i])
        ax.fill(angles, values, color=radar_colors[i], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title(f"Candidate Competency Radar: Top 3 Shortlist", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.95)
    plt.tight_layout()
    fig.savefig(os.path.join("outputs", "figures", "03_radar_chart_candidate_comparison.png"), dpi=300)
    plt.close()

    # --- FIGURE 4: Semantic Similarity vs Required Skill Match ---
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = sns.scatterplot(
        data=df,
        x="Semantic Similarity (%)",
        y="Required Skill Match (%)",
        size="Composite Score (%)",
        hue="Fit Tier",
        palette={"Strong Fit (Shortlist)": "#22c55e", "Moderate Fit (Review)": "#3b82f6",
                 "Needs Upskilling": "#f59e0b", "Low Fit (Reject)": "#ef4444"},
        sizes=(60, 240),
        ax=ax
    )
    # Add quadrant reference lines
    ax.axvline(x=50, color="#94a3b8", linestyle="--", alpha=0.7)
    ax.axhline(y=50, color="#94a3b8", linestyle="--", alpha=0.7)
    ax.text(75, 92, "High Semantic + High Skills\n(Primary Shortlist)", fontsize=10, fontweight="bold", color="#15803d")
    ax.text(15, 20, "Low Semantic + Low Skills\n(Clear Outliers)", fontsize=10, fontweight="bold", color="#b91c1c")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_title("TF-IDF Semantic Similarity vs. NLP Required Skill Coverage", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("TF-IDF Semantic Similarity (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Required Skill Match (%)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join("outputs", "figures", "04_tfidf_vs_skill_score_correlation.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    run_screening_pipeline()

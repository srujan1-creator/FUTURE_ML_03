"""
Flask REST API and Web Application.
Future Interns - Machine Learning Task 3 (2026): Resume Screening & Ranking System.
Compatible with standalone execution and Vercel Serverless deployments.
"""

import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template

from src.preprocessing import clean_text, extract_sections, extract_years_of_experience, extract_education
from src.skill_extractor import SkillExtractor, SKILL_CATEGORIES
from src.job_parser import JobDescriptionParser
from src.matcher import ResumeMatcher
from src.ranker import CandidateRanker


app = Flask(__name__, template_folder="templates")

# Initialize models
extractor = SkillExtractor()
job_parser = JobDescriptionParser(extractor)
matcher = ResumeMatcher()
ranker = CandidateRanker(extractor, job_parser, matcher)

# Load default datasets
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
jd_path = os.path.join(DATA_DIR, "job_descriptions.json")
resumes_path = os.path.join(DATA_DIR, "resumes.csv")

if not os.path.exists(jd_path) or not os.path.exists(resumes_path):
    from src.generator import generate_benchmark_dataset
    generate_benchmark_dataset(DATA_DIR)

with open(jd_path, "r", encoding="utf-8") as f:
    JOB_DESCRIPTIONS = json.load(f)

BENCHMARK_RESUMES = pd.read_csv(resumes_path).to_dict(orient="records")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Candidate Screening & Ranking Decision-Support System",
        "version": "3.0.0",
        "internship": "Future Interns ML Task 3 (2026)"
    })


@app.route("/api/jobs", methods=["GET"])
def get_jobs():
    return jsonify(JOB_DESCRIPTIONS)


@app.route("/api/skills", methods=["GET"])
def get_skills():
    return jsonify(SKILL_CATEGORIES)


@app.route("/api/parse_resume", methods=["POST"])
def parse_resume():
    """
    Parses a single resume text into structured features.
    """
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No resume text provided"}), 400

    skills = extractor.extract_skills(text)
    categorized_skills = extractor.extract_categorized_skills(text)
    exp = extract_years_of_experience(text)
    edu = extract_education(text)
    sections = extract_sections(text)

    return jsonify({
        "extracted_skills": skills,
        "categorized_skills": categorized_skills,
        "years_experience": exp,
        "education": edu,
        "sections": {k: len(v) > 0 for k, v in sections.items()}
    })


@app.route("/api/screen", methods=["POST"])
def screen_candidates():
    """
    Screens candidate resumes against a job description.
    Accepts:
    {
        "job_description": "...",
        "role_title": "...",
        "resumes": [ ...optional custom candidate list... ],
        "weights": { ...optional scoring weights... }
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    jd_text = data.get("job_description", "")
    role_title = data.get("role_title", "")
    custom_resumes = data.get("resumes", None)
    weights = data.get("weights", None)

    if not jd_text:
        # Fallback to primary job description if empty
        jd_text = JOB_DESCRIPTIONS[0]["text"]
        role_title = JOB_DESCRIPTIONS[0]["title"]

    resumes_to_screen = custom_resumes if (custom_resumes and len(custom_resumes) > 0) else BENCHMARK_RESUMES

    ranked_df = ranker.rank_candidates(
        resumes=resumes_to_screen,
        job_description=jd_text,
        role_title=role_title,
        custom_weights=weights
    )

    records = ranked_df.to_dict(orient="records")
    # Clean internal fields for JSON response
    for r in records:
        r.pop("_matched_req_list", None)
        r.pop("_missing_req_list", None)
        r.pop("_matched_pref_list", None)
        r.pop("_surplus_list", None)
        r.pop("_raw_text", None)

    return jsonify({
        "role_title": role_title,
        "candidate_count": len(records),
        "rankings": records
    })


# Export application for Vercel WSGI
application = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

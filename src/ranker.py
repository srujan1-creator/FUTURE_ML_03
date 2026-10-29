"""
Candidate Ranking and Decision-Support Engine.
Orchestrates candidate scoring, leaderboard ranking, tier classification,
skill gap diagnosis, and executive explanation generation for recruiters.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from src.preprocessing import clean_text, extract_years_of_experience, extract_education
from src.skill_extractor import SkillExtractor
from src.job_parser import JobDescriptionParser
from src.matcher import ResumeMatcher, DEFAULT_WEIGHTS


def assign_fit_tier(composite_score: float) -> Tuple[str, str]:
    """
    Classifies composite score into recruiter-facing operational tiers with hex color hints.
    """
    if composite_score >= 75.0:
        return 'Strong Fit (Shortlist)', '#22c55e'       # Emerald Green
    elif composite_score >= 55.0:
        return 'Moderate Fit (Review)', '#3b82f6'        # Blue
    elif composite_score >= 40.0:
        return 'Needs Upskilling', '#f59e0b'             # Amber
    else:
        return 'Low Fit (Reject)', '#ef4444'             # Red


class CandidateRanker:
    """
    Orchestrates end-to-end resume screening, ranking, and explainability.
    """

    def __init__(
        self,
        skill_extractor: Optional[SkillExtractor] = None,
        job_parser: Optional[JobDescriptionParser] = None,
        matcher: Optional[ResumeMatcher] = None
    ):
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.job_parser = job_parser or JobDescriptionParser(self.skill_extractor)
        self.matcher = matcher or ResumeMatcher()

    def rank_candidates(
        self,
        resumes: List[Dict[str, Any]],
        job_description: str,
        role_title: str = "",
        custom_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Ranks a pool of candidates against a job description.

        Args:
            resumes: List of dicts, each with keys like:
                     'candidate_id', 'name', 'text' (or 'resume_text'), optional 'experience_years', 'category'.
            job_description: Plain text of the target job role.
            role_title: Optional title hint (e.g. 'Senior Machine Learning Engineer').
            custom_weights: Optional dictionary overriding default scoring weights.

        Returns:
            pd.DataFrame sorted by rank with scores, gaps, and recruiter explanations.
        """
        if not resumes or not job_description:
            return pd.DataFrame()

        # 1. Parse Job Description
        job_profile = self.job_parser.parse(job_description, role_title_hint=role_title)

        # 2. Extract texts and compute batch semantic similarity
        resume_texts = []
        parsed_resumes = []

        for idx, r in enumerate(resumes):
            raw_text = r.get('text') or r.get('resume_text') or r.get('Resume_str') or ""
            c_name = r.get('name') or r.get('candidate_id') or f"Candidate_{idx + 1}"
            c_id = r.get('candidate_id') or f"CAND_{idx + 1:03d}"

            # If years of experience not pre-populated, extract via NLP
            exp_yrs = float(r.get('experience_years', extract_years_of_experience(raw_text)))
            edu_info = extract_education(raw_text)

            # Extract candidate skills
            skills = self.skill_extractor.extract_skills(raw_text)

            resume_texts.append(raw_text)
            parsed_resumes.append({
                'candidate_id': c_id,
                'name': c_name,
                'text': raw_text,
                'experience_years': exp_yrs,
                'education': edu_info['highest_degree'],
                'skills': skills,
                'raw_metadata': r
            })

        semantic_sims = self.matcher.compute_semantic_similarity(resume_texts, job_description)

        # 3. Score each candidate
        results = []
        for i, cand in enumerate(parsed_resumes):
            fit_metrics = self.matcher.compute_single_fit(
                candidate_skills=cand['skills'],
                candidate_years_exp=cand['experience_years'],
                semantic_sim=semantic_sims[i],
                job_profile=job_profile,
                custom_weights=custom_weights
            )

            tier, tier_color = assign_fit_tier(fit_metrics['composite_score'])
            explanation, action = self._generate_recruiter_explanation(
                name=cand['name'],
                composite_score=fit_metrics['composite_score'],
                matched_req=fit_metrics['matched_required_skills'],
                missing_req=fit_metrics['missing_required_skills'],
                matched_pref=fit_metrics['matched_preferred_skills'],
                cand_exp=cand['experience_years'],
                req_exp=job_profile.get('min_experience_years', 3.0),
                tier=tier
            )

            row = {
                'Candidate ID': cand['candidate_id'],
                'Candidate Name': cand['name'],
                'Fit Tier': tier,
                'Composite Score (%)': fit_metrics['composite_score'],
                'Required Skill Match (%)': fit_metrics['required_skill_score'],
                'Semantic Similarity (%)': fit_metrics['semantic_score'],
                'Preferred Skill Match (%)': fit_metrics['preferred_skill_score'],
                'Experience Score (%)': fit_metrics['experience_score'],
                'Experience (Years)': cand['experience_years'],
                'Education': cand['education'],
                'Extracted Skills Count': len(cand['skills']),
                'Matched Required Skills': ', '.join(fit_metrics['matched_required_skills']) or 'None',
                'Missing Critical Skills': ', '.join(fit_metrics['missing_required_skills']) or 'None',
                'Matched Preferred Skills': ', '.join(fit_metrics['matched_preferred_skills']) or 'None',
                'Surplus Skills': ', '.join(fit_metrics['surplus_skills'][:8]) or 'None',
                'Recruiter Explanation': explanation,
                'Recommended Action': action,
                '_tier_color': tier_color,
                '_matched_req_list': fit_metrics['matched_required_skills'],
                '_missing_req_list': fit_metrics['missing_required_skills'],
                '_matched_pref_list': fit_metrics['matched_preferred_skills'],
                '_surplus_list': fit_metrics['surplus_skills'],
                '_raw_text': cand['text']
            }
            results.append(row)

        df = pd.DataFrame(results)
        # Sort by Composite Score descending
        df = df.sort_values(by=['Composite Score (%)', 'Required Skill Match (%)'], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))

        return df

    def _generate_recruiter_explanation(
        self,
        name: str,
        composite_score: float,
        matched_req: List[str],
        missing_req: List[str],
        matched_pref: List[str],
        cand_exp: float,
        req_exp: float,
        tier: str
    ) -> Tuple[str, str]:
        """
        Creates an executive, non-technical explanation of candidate ranking rationale.
        """
        req_total = len(matched_req) + len(missing_req)
        req_pct = round((len(matched_req) / req_total * 100), 1) if req_total > 0 else 100.0

        if composite_score >= 75.0:
            exp_text = f"Meets required experience ({cand_exp:.1f} yrs vs {req_exp:.1f} yrs required)." if cand_exp >= req_exp else f"Slightly below target experience ({cand_exp:.1f} yrs vs {req_exp:.1f} yrs target)."
            gap_text = f"Only misses secondary skills: {', '.join(missing_req)}." if missing_req else "Possesses 100% of required core skills."
            explanation = (
                f"{name} is a top-tier candidate with {req_pct}% required skill coverage ({len(matched_req)}/{req_total} skills: {', '.join(matched_req[:4])}). "
                f"{exp_text} {gap_text}"
            )
            action = "Fast-track to Recruiter Phone Screen / Technical Round 1."
        elif composite_score >= 55.0:
            gap_str = ', '.join(missing_req[:3]) if missing_req else "None"
            explanation = (
                f"{name} demonstrates moderate alignment ({req_pct}% core skill match). "
                f"Strong in {', '.join(matched_req[:3]) or 'foundational topics'}, but has key skill gaps in {gap_str}."
            )
            action = "Review portfolio / assess trainable gaps in technical assessment."
        elif composite_score >= 40.0:
            gap_str = ', '.join(missing_req[:4]) if missing_req else "fundamental requirements"
            explanation = (
                f"{name} has foundational familiarity but misses critical must-have competencies ({gap_str}). "
                f"Matches only {len(matched_req)} of {req_total} core requirements."
            )
            action = "Hold for junior or alternative role opening; requires upskilling."
        else:
            explanation = (
                f"{name} shows substantial divergence from the job profile ({composite_score:.1f}% fit). "
                f"Lacks primary required stack ({', '.join(missing_req[:4]) or 'core skills'})."
            )
            action = "Polite rejection or redirect to matching entry-level opportunities."

        return explanation, action

"""
Resume-to-Job Matching and Similarity Engine.
Implements TF-IDF Vectorization, Cosine Similarity, Skill Coverage Mathematics,
Experience Alignment, and Customizable Composite Fit Index (CFI) Scoring.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import clean_text


DEFAULT_WEIGHTS = {
    'required_skills': 0.45,
    'semantic_similarity': 0.35,
    'preferred_skills': 0.10,
    'experience_fit': 0.10
}


class ResumeMatcher:
    """
    Computes semantic similarity and multi-criteria fit scores between resumes and job requirements.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._normalize_weights()
        self.ngram_range = ngram_range
        self.min_df = min_df

    def _normalize_weights(self):
        """Ensures weights sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def compute_semantic_similarity(
        self,
        resume_texts: List[str],
        job_text: str,
        calibrate: bool = True
    ) -> np.ndarray:
        """
        Computes cosine similarity between TF-IDF representations of resumes and JD text.
        Calibrates raw cosine scores (which naturally hover between 0.05 - 0.35 due to text length disparity)
        into an intuitive 0.0 - 1.0 scale.
        """
        if not resume_texts or not job_text:
            return np.zeros(len(resume_texts))

        # Preprocess texts with lemmatization and stopword removal for cleaner vector space
        clean_resumes = [clean_text(r, remove_stopwords=True, lemmatize=True) for r in resume_texts]
        clean_job = clean_text(job_text, remove_stopwords=True, lemmatize=True)

        corpus = [clean_job] + clean_resumes

        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            min_df=self.min_df,
            max_features=5000
        )

        tfidf_matrix = vectorizer.fit_transform(corpus)
        job_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(resume_vectors, job_vector).flatten()
        similarities = np.clip(similarities, 0.0, 1.0)

        if calibrate:
            # Benchmark factor: a cosine similarity of 0.30+ represents very strong vocabulary alignment
            calibrated = np.clip(similarities / 0.30, 0.0, 1.0)
            return calibrated

        return similarities

    def compute_single_fit(
        self,
        candidate_skills: List[str],
        candidate_years_exp: float,
        semantic_sim: float,
        job_profile: Dict[str, Any],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculates all score components and the Composite Fit Index (CFI) for a candidate.
        """
        w = custom_weights or self.weights
        total_w = sum(w.values())
        if total_w > 0:
            w = {k: v / total_w for k, v in w.items()}

        candidate_skill_set = set(candidate_skills)
        req_skills = set(job_profile.get('required_skills', []))
        pref_skills = set(job_profile.get('preferred_skills', []))

        # 1. Required Skill Coverage
        if req_skills:
            matched_req = candidate_skill_set.intersection(req_skills)
            missing_req = req_skills - candidate_skill_set
            req_score = len(matched_req) / len(req_skills)
        else:
            matched_req = set()
            missing_req = set()
            req_score = 1.0

        # 2. Preferred Skill Coverage
        if pref_skills:
            matched_pref = candidate_skill_set.intersection(pref_skills)
            missing_pref = pref_skills - candidate_skill_set
            pref_score = len(matched_pref) / len(pref_skills)
        else:
            matched_pref = set()
            missing_pref = set()
            pref_score = 0.5  # Neutral default if no bonus skills specified

        # 3. Experience Fit Score
        req_exp = float(job_profile.get('min_experience_years', 3.0))
        if req_exp > 0:
            exp_score = min(1.0, max(0.0, candidate_years_exp / req_exp))
        else:
            exp_score = 1.0

        # 4. Composite Fit Index (CFI)
        cfi_raw = (
            w.get('required_skills', 0.45) * req_score +
            w.get('semantic_similarity', 0.35) * float(semantic_sim) +
            w.get('preferred_skills', 0.10) * pref_score +
            w.get('experience_fit', 0.10) * exp_score
        )

        composite_score = round(float(cfi_raw) * 100.0, 2)

        # Surplus / Bonus skills that candidate possesses beyond the JD requirements
        surplus_skills = candidate_skill_set - req_skills.union(pref_skills)

        return {
            'composite_score': composite_score,
            'semantic_score': round(float(semantic_sim) * 100.0, 2),
            'required_skill_score': round(req_score * 100.0, 2),
            'preferred_skill_score': round(pref_score * 100.0, 2),
            'experience_score': round(exp_score * 100.0, 2),
            'matched_required_skills': sorted(list(matched_req)),
            'missing_required_skills': sorted(list(missing_req)),
            'matched_preferred_skills': sorted(list(matched_pref)),
            'missing_preferred_skills': sorted(list(missing_pref)),
            'surplus_skills': sorted(list(surplus_skills))
        }

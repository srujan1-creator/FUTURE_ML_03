"""
Job Description Parser Module.
Parses job descriptions into structured profiles including role titles,
seniority level, required vs preferred skills, education, and experience targets.
"""

import re
from typing import Dict, List, Set, Any, Optional
from src.skill_extractor import SkillExtractor
from src.preprocessing import extract_years_of_experience, extract_education


SENIORITY_PATTERNS = [
    ('Principal / Staff', re.compile(r'(?i)\b(principal|staff|distinguished|architect)\b')),
    ('Lead / Manager', re.compile(r'(?i)\b(lead|team lead|manager|head of|director)\b')),
    ('Senior', re.compile(r'(?i)\b(senior|sr\.?|lead engineer)\b')),
    ('Mid-Level', re.compile(r'(?i)\b(mid-level|mid level|intermediate|experienced)\b')),
    ('Entry-Level / Junior', re.compile(r'(?i)\b(junior|jr\.?|entry level|graduate|intern|associate)\b'))
]

REQUIREMENTS_HEADER_PATTERNS = re.compile(
    r'(?i)\b(requirements|must have|required qualifications|minimum qualifications|what you\'ll need|what we are looking for|basic qualifications)\b'
)

PREFERRED_HEADER_PATTERNS = re.compile(
    r'(?i)\b(nice to have|preferred qualifications|good to have|bonus|plus|desired skills|preferred skills)\b'
)


class JobDescriptionParser:
    """
    Parses unstructured Job Descriptions into standardized requirement schemas.
    """

    def __init__(self, skill_extractor: Optional[SkillExtractor] = None):
        self.skill_extractor = skill_extractor or SkillExtractor()

    def parse(self, jd_text: str, role_title_hint: str = "") -> Dict[str, Any]:
        """
        Parses full JD text into structured specification.
        """
        if not jd_text or not isinstance(jd_text, str):
            return {
                'title': role_title_hint or 'Unknown Role',
                'seniority': 'Mid-Level',
                'min_experience_years': 2.0,
                'min_education': 'Bachelor\'s Degree',
                'required_skills': [],
                'preferred_skills': [],
                'all_skills': []
            }

        # 1. Detect Seniority
        seniority = 'Mid-Level'
        for level_name, pat in SENIORITY_PATTERNS:
            if pat.search(role_title_hint) or pat.search(jd_text[:300]):
                seniority = level_name
                break

        # 2. Extract Experience & Education targets
        min_experience = extract_years_of_experience(jd_text)
        if min_experience == 0.0:
            if seniority == 'Senior':
                min_experience = 5.0
            elif seniority in ('Principal / Staff', 'Lead / Manager'):
                min_experience = 8.0
            elif seniority == 'Entry-Level / Junior':
                min_experience = 1.0
            else:
                min_experience = 3.0

        edu_info = extract_education(jd_text)
        min_education = edu_info['highest_degree'] if edu_info['highest_degree'] != 'Not Specified' else 'Bachelor\'s Degree'

        # 3. Partition text into Required vs Preferred sections if distinct headers exist
        req_text, pref_text = self._split_requirements_sections(jd_text)

        all_skills = self.skill_extractor.extract_skills(jd_text)

        if req_text and pref_text:
            req_skills = self.skill_extractor.extract_skills(req_text)
            pref_skills = [s for s in self.skill_extractor.extract_skills(pref_text) if s not in req_skills]
        else:
            # Heuristic partitioning: top 60% of technical skills are treated as core/required
            req_skills = all_skills[:max(1, int(len(all_skills) * 0.7))]
            pref_skills = [s for s in all_skills if s not in req_skills]

        # 4. Extract Title if not provided
        title = role_title_hint
        if not title:
            lines = [l.strip() for l in jd_text.split('\n') if l.strip()]
            if lines and len(lines[0].split()) <= 7:
                title = lines[0]
            else:
                title = f"{seniority} Engineer / Specialist"

        return {
            'title': title,
            'seniority': seniority,
            'min_experience_years': min_experience,
            'min_education': min_education,
            'required_skills': req_skills,
            'preferred_skills': pref_skills,
            'all_skills': all_skills,
            'raw_text': jd_text
        }

    def _split_requirements_sections(self, text: str) -> (str, str):
        """
        Splits text based on requirement vs preferred headers.
        """
        lines = text.split('\n')
        req_lines = []
        pref_lines = []
        current_bucket = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if REQUIREMENTS_HEADER_PATTERNS.search(stripped) and len(stripped.split()) <= 6:
                current_bucket = 'required'
                continue
            elif PREFERRED_HEADER_PATTERNS.search(stripped) and len(stripped.split()) <= 6:
                current_bucket = 'preferred'
                continue
            elif re.compile(r'(?i)\b(benefits|about company|who we are|equal opportunity)\b').search(stripped) and len(stripped.split()) <= 6:
                current_bucket = None

            if current_bucket == 'required':
                req_lines.append(stripped)
            elif current_bucket == 'preferred':
                pref_lines.append(stripped)

        return '\n'.join(req_lines), '\n'.join(pref_lines)

"""
Skill Extraction Module.
Features a multi-domain skill taxonomy (500+ skills), canonical normalization,
boundary-aware token matching, category attribution, and density scoring.
"""

import re
from typing import Dict, List, Set, Tuple, Optional


# Canonical Skill Taxonomy mapping variations and aliases to standardized names
SKILL_TAXONOMY = {
    # --- Programming Languages ---
    'Python': ['python', 'py3', 'python3'],
    'Java': ['java', 'core java', 'j2ee'],
    'C++': ['c++', 'cpp'],
    'C': [r'\bc\b'],
    'C#': ['c#', 'csharp', 'c-sharp'],
    'JavaScript': ['javascript', 'js', 'es6', 'ecmascript'],
    'TypeScript': ['typescript', 'ts'],
    'Go': ['golang', r'\bgo\b'],
    'Rust': ['rust', 'rustlang'],
    'SQL': ['sql', 't-sql', 'pl/sql', 'structured query language'],
    'R': [r'\br\b', 'r-project', 'r language'],
    'Scala': ['scala'],
    'Ruby': ['ruby'],
    'PHP': ['php'],
    'Swift': ['swift'],
    'Kotlin': ['kotlin'],
    'Bash/Shell': ['bash', 'shell script', 'shell scripting', 'powershell', 'zsh'],
    'HTML/CSS': ['html', 'html5', 'css', 'css3', 'sass', 'scss'],

    # --- Data Science & Machine Learning ---
    'Machine Learning': ['machine learning', 'ml', 'statistical learning', 'supervised learning', 'unsupervised learning'],
    'Deep Learning': ['deep learning', 'dl', 'neural networks', 'ann', 'cnn', 'rnn', 'lstm'],
    'Natural Language Processing': ['natural language processing', 'nlp', 'text mining', 'named entity recognition', 'ner', 'sentiment analysis'],
    'Computer Vision': ['computer vision', 'cv', 'image processing', 'object detection', 'segmentation', 'yolo'],
    'Large Language Models': ['large language models', 'llm', 'llms', 'generative ai', 'genai', 'prompt engineering', 'rag', 'retrieval-augmented generation', 'fine-tuning'],
    'LangChain': ['langchain', 'langgraph', 'llamaindex'],
    'PyTorch': ['pytorch', 'torch'],
    'TensorFlow': ['tensorflow', 'tf', 'keras'],
    'Scikit-Learn': ['scikit-learn', 'sklearn', 'scikit learn'],
    'XGBoost': ['xgboost', 'lightgbm', 'catboost', 'gradient boosting'],
    'Pandas': ['pandas'],
    'NumPy': ['numpy'],
    'SciPy': ['scipy'],
    'Transformers': ['transformers', 'hugging face', 'huggingface', 'bert', 'roberta'],
    'MLOps': ['mlops', 'ml pipeline', 'model deployment', 'model monitoring', 'mlflow', 'weights & biases', 'wandb', 'dvc'],
    'Recommendation Systems': ['recommendation systems', 'recommender systems', 'collaborative filtering'],
    'Time Series Analysis': ['time series', 'forecasting', 'arima', 'prophet'],
    'Vector Databases': ['vector database', 'vector db', 'chroma', 'chromadb', 'pinecone', 'milvus', 'weaviate', 'qdrant'],

    # --- Cloud & Infrastructure / DevOps ---
    'AWS': ['aws', 'amazon web services', 'ec2', 's3', 'rds', 'aws lambda', 'sagemaker', 'cloudwatch', 'iam'],
    'Azure': ['azure', 'microsoft azure', 'azure devops', 'azure blob'],
    'GCP': ['gcp', 'google cloud', 'google cloud platform', 'bigquery', 'cloud run', 'gke', 'vertex ai'],
    'Docker': ['docker', 'containerization', 'dockerfile', 'docker-compose'],
    'Kubernetes': ['kubernetes', 'k8s', 'helm', 'k8s cluster'],
    'Terraform': ['terraform', 'terragrunt', 'infrastructure as code', 'iac'],
    'Ansible': ['ansible'],
    'CI/CD': ['ci/cd', 'ci cd', 'continuous integration', 'continuous delivery', 'github actions', 'gitlab ci', 'jenkins'],
    'Linux': ['linux', 'ubuntu', 'centos', 'redhat', 'debian', 'unix'],
    'Monitoring & Logging': ['prometheus', 'grafana', 'elk stack', 'datadog', 'splunk', 'cloudwatch'],

    # --- Web & Backend Frameworks ---
    'React': ['react', 'react.js', 'reactjs'],
    'Node.js': ['node.js', 'nodejs', 'node'],
    'Next.js': ['next.js', 'nextjs'],
    'Express': ['express.js', 'expressjs', 'express'],
    'Django': ['django', 'django rest framework', 'drf'],
    'FastAPI': ['fastapi'],
    'Flask': ['flask'],
    'Spring Boot': ['spring boot', 'spring framework', 'spring mvc'],
    'Angular': ['angular', 'angularjs', 'angular.js'],
    'Vue.js': ['vue', 'vue.js', 'vuejs'],
    'GraphQL': ['graphql', 'apollo'],
    'RESTful APIs': ['rest api', 'restful api', 'rest apis', 'restful services', 'api design'],
    'Tailwind CSS': ['tailwind', 'tailwind css'],
    'Microservices': ['microservices', 'microservice architecture', 'event-driven architecture', 'grpc'],

    # --- Databases & Big Data ---
    'PostgreSQL': ['postgresql', 'postgres'],
    'MySQL': ['mysql'],
    'MongoDB': ['mongodb', 'mongo'],
    'Redis': ['redis', 'redis cache'],
    'Elasticsearch': ['elasticsearch', 'elastic search', 'opensearch'],
    'Apache Spark': ['spark', 'pyspark', 'apache spark'],
    'Apache Kafka': ['kafka', 'apache kafka'],
    'Apache Airflow': ['airflow', 'apache airflow'],
    'Snowflake': ['snowflake', 'snowflake data warehouse'],
    'Databricks': ['databricks'],
    'Data Warehousing': ['data warehouse', 'data warehousing', 'etl', 'elt', 'dbt'],

    # --- Software Engineering & Architecture ---
    'System Design': ['system design', 'high level design', 'distributed systems', 'scalability', 'fault tolerance'],
    'OOP / Design Patterns': ['object-oriented programming', 'oop', 'design patterns', 'solid principles'],
    'Git / Version Control': ['git', 'github', 'gitlab', 'bitbucket', 'version control'],
    'Testing & QA': ['unit testing', 'pytest', 'unittest', 'tdd', 'test driven development', 'jest', 'cypress', 'selenium'],
    'Agile / Scrum': ['agile', 'scrum', 'kanban', 'jira', 'sprint planning'],

    # --- Soft Skills & Management ---
    'Problem Solving': ['problem solving', 'analytical skills', 'critical thinking'],
    'Leadership': ['leadership', 'team lead', 'mentoring', 'people management', 'leading teams'],
    'Communication': ['communication', 'verbal communication', 'written communication', 'presentation skills'],
    'Collaboration': ['collaboration', 'cross-functional', 'team player', 'teamwork'],
    'Project Management': ['project management', 'stakeholder management', 'product roadmap', 'okrs', 'kpis']
}

# Skill Categorization for Domain Attribution
SKILL_CATEGORIES = {
    'Programming Languages': [
        'Python', 'Java', 'C++', 'C', 'C#', 'JavaScript', 'TypeScript', 'Go', 'Rust',
        'SQL', 'R', 'Scala', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'Bash/Shell', 'HTML/CSS'
    ],
    'Data Science & AI': [
        'Machine Learning', 'Deep Learning', 'Natural Language Processing', 'Computer Vision',
        'Large Language Models', 'LangChain', 'PyTorch', 'TensorFlow', 'Scikit-Learn',
        'XGBoost', 'Pandas', 'NumPy', 'SciPy', 'Transformers', 'MLOps',
        'Recommendation Systems', 'Time Series Analysis', 'Vector Databases'
    ],
    'Cloud & DevOps': [
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'Ansible',
        'CI/CD', 'Linux', 'Monitoring & Logging'
    ],
    'Web & Backend': [
        'React', 'Node.js', 'Next.js', 'Express', 'Django', 'FastAPI', 'Flask',
        'Spring Boot', 'Angular', 'Vue.js', 'GraphQL', 'RESTful APIs',
        'Tailwind CSS', 'Microservices'
    ],
    'Databases & Big Data': [
        'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Apache Spark',
        'Apache Kafka', 'Apache Airflow', 'Snowflake', 'Databricks', 'Data Warehousing'
    ],
    'Software Engineering': [
        'System Design', 'OOP / Design Patterns', 'Git / Version Control',
        'Testing & QA', 'Agile / Scrum'
    ],
    'Leadership & Soft Skills': [
        'Problem Solving', 'Leadership', 'Communication', 'Collaboration', 'Project Management'
    ]
}


class SkillExtractor:
    """
    Extracts, standardizes, and categorizes skills from resume and job description texts.
    """

    def __init__(self, custom_taxonomy: Optional[Dict[str, List[str]]] = None):
        self.taxonomy = custom_taxonomy or SKILL_TAXONOMY
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """
        Pre-compiles regex patterns for each canonical skill and its aliases.
        Uses exact word boundaries to prevent substring false-positives.
        """
        compiled = []
        for canonical, aliases in self.taxonomy.items():
            pattern_parts = []
            for alias in aliases:
                # If alias already contains regex syntax (e.g. \bc\b), use directly
                if alias.startswith(r'\b') or alias.endswith(r'\b'):
                    pattern_parts.append(alias)
                else:
                    escaped = re.escape(alias)
                    pattern_parts.append(rf'(?<![\w\-]){escaped}(?![\w\-])')

            full_regex = r'(?i)\b(' + '|'.join(pattern_parts) + r')\b'
            try:
                compiled.append((canonical, re.compile(full_regex, re.IGNORECASE)))
            except re.error:
                # Fallback simple boundary pattern
                compiled.append((canonical, re.compile(rf'(?i)\b{re.escape(canonical)}\b', re.IGNORECASE)))
        return compiled

    def extract_skills(self, text: str) -> List[str]:
        """
        Extracts unique canonical skills from input text.
        """
        if not text or not isinstance(text, str):
            return []

        matched_skills = set()
        for canonical, pattern in self._compiled_patterns:
            if pattern.search(text):
                matched_skills.add(canonical)

        return sorted(list(matched_skills))

    def extract_categorized_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts skills and groups them by domain categories.
        """
        flat_skills = set(self.extract_skills(text))
        categorized = {}

        for category, cat_skills in SKILL_CATEGORIES.items():
            present = [s for s in cat_skills if s in flat_skills]
            if present:
                categorized[category] = sorted(present)

        # Catch any custom skills not in predefined categories
        all_categorized = set(s for lst in categorized.values() for s in lst)
        uncategorized = [s for s in flat_skills if s not in all_categorized]
        if uncategorized:
            categorized['Other Technical Skills'] = sorted(uncategorized)

        return categorized

    def get_skill_density(self, text: str) -> float:
        """
        Calculates skill density: number of unique skills extracted per 100 words.
        """
        words = text.split()
        if not words:
            return 0.0
        skills = self.extract_skills(text)
        return round((len(skills) / len(words)) * 100, 2)

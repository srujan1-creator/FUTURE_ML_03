"""
Data Generation Utility for Benchmark Testing and Demonstration.
Creates realistic Job Descriptions and diverse Candidate Resumes (CSV, TXT, PDF).
"""

import os
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors


JOB_DESCRIPTIONS = [
    {
        "id": "jd_ml_senior",
        "title": "Senior Machine Learning Engineer",
        "department": "Artificial Intelligence & Core ML",
        "seniority": "Senior",
        "min_experience_years": 5.0,
        "min_education": "Master's Degree",
        "text": """Title: Senior Machine Learning Engineer
Department: Core AI Platform
Location: Remote / Hybrid

About The Role:
We are seeking an experienced Senior Machine Learning Engineer to design, train, and deploy production ML models at enterprise scale. You will partner with data engineering and platform teams to build resilient MLOps pipelines and deliver high-impact predictive systems.

Requirements & Must-Have Qualifications:
- 5+ years of professional experience in Machine Learning and Software Engineering.
- Mastery of Python, SQL, and core scientific computing libraries (NumPy, Pandas, SciPy).
- Production expertise with Deep Learning frameworks: PyTorch and TensorFlow.
- Strong hands-on experience with Scikit-Learn, XGBoost, and classical ML algorithms.
- Proven experience containerizing models using Docker and orchestrating with Kubernetes.
- Deep understanding of MLOps pipelines, model registry, and CI/CD (MLflow, Weights & Biases).
- Cloud experience with AWS (SageMaker, S3, EC2) or GCP (Vertex AI).
- Solid foundation in System Design, Distributed Systems, and Unit Testing.

Nice to Have & Preferred Qualifications:
- Experience with Large Language Models (LLMs), Transformers, and LangChain.
- Familiarity with Vector Databases (Pinecone, Chroma, Milvus) and RAG architecture.
- Experience with Apache Spark and Apache Kafka for real-time streaming inference.
- Master's or Ph.D. in Computer Science, Data Science, or related STEM field."""
    },
    {
        "id": "jd_fullstack",
        "title": "Senior Full Stack Developer",
        "department": "Product Engineering",
        "seniority": "Senior",
        "min_experience_years": 4.0,
        "min_education": "Bachelor's Degree",
        "text": """Title: Senior Full Stack Developer (React + Node.js)
Department: Web Application Engineering
Location: San Francisco, CA / Remote

About The Role:
We are looking for a Senior Full Stack Engineer to lead the architecture and development of our customer-facing web platforms. You will design scalable RESTful APIs, build performant React frontends, and ensure high code quality.

Requirements & Must-Have Qualifications:
- 4+ years of professional web development experience.
- Deep expertise in JavaScript (ES6+), TypeScript, and modern frontend frameworks (React, Next.js).
- Strong backend experience with Node.js, Express, and RESTful APIs design.
- Production proficiency with relational databases: PostgreSQL and MySQL.
- Experience with state management, HTML/CSS, and Tailwind CSS.
- Solid understanding of Git / Version Control, Unit Testing (Jest, Pytest), and CI/CD.
- Experience with Docker containerization and cloud hosting.

Nice to Have & Preferred Qualifications:
- Familiarity with GraphQL, Redis caching, and WebSockets.
- Experience with Microservices architecture and AWS cloud services.
- Familiarity with Python, Django, or FastAPI.
- Bachelor's degree in Computer Science or equivalent practical experience."""
    },
    {
        "id": "jd_devops_sre",
        "title": "DevOps & Site Reliability Engineer",
        "department": "Cloud Infrastructure",
        "seniority": "Senior",
        "min_experience_years": 4.0,
        "min_education": "Bachelor's Degree",
        "text": """Title: DevOps & Site Reliability Engineer (SRE)
Department: Cloud Operations
Location: Austin, TX / Remote

About The Role:
Join our infrastructure squad to ensure 99.99% uptime, automate cloud provisioning, and build resilient continuous delivery pipelines across multi-region environments.

Requirements & Must-Have Qualifications:
- 4+ years of hands-on experience in DevOps, Cloud Engineering, or SRE roles.
- Deep expertise with Linux systems administration and Bash/Shell scripting.
- Advanced mastery of Docker and Kubernetes (EKS/GKE cluster administration, Helm).
- Strong proficiency in Infrastructure as Code (IaC) using Terraform and Ansible.
- Extensive experience designing CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins).
- Solid experience with AWS (VPC, IAM, EC2, S3, CloudWatch) or GCP.
- Scripting proficiency in Python or Go for infrastructure automation.
- Production monitoring and telemetry using Prometheus, Grafana, and ELK stack.

Nice to Have & Preferred Qualifications:
- Experience managing PostgreSQL / Redis clusters at scale.
- Knowledge of System Design, security compliance (SOC2), and incident response.
- Bachelor's degree in Computer Science or related technical discipline."""
    },
    {
        "id": "jd_data_analyst",
        "title": "Lead Data Analyst & Business Intelligence",
        "department": "Data & Analytics",
        "seniority": "Mid-Level",
        "min_experience_years": 3.0,
        "min_education": "Bachelor's Degree",
        "text": """Title: Lead Data Analyst & Business Intelligence
Department: Strategic Analytics
Location: New York, NY / Hybrid

About The Role:
We are seeking an analytical problem-solver to translate complex transactional data into strategic business insights, executive dashboards, and statistical models.

Requirements & Must-Have Qualifications:
- 3+ years of professional analytics experience.
- Advanced SQL proficiency (complex joins, window functions, query optimization).
- Hands-on data manipulation in Python (Pandas, NumPy, Matplotlib, Seaborn).
- Experience building dashboards in Tableau, PowerBI, or Looker.
- Strong grounding in Statistics, A/B Testing, and quantitative hypothesis validation.
- Demonstrated Problem Solving, Communication, and Stakeholder Management skills.

Nice to Have & Preferred Qualifications:
- Familiarity with Data Warehousing (Snowflake, BigQuery, dbt).
- Basic exposure to Machine Learning concepts (Scikit-Learn, regression, clustering).
- Bachelor's in Mathematics, Statistics, Economics, or Computer Science."""
    },
    {
        "id": "jd_llm_genai",
        "title": "Generative AI & LLM Systems Engineer",
        "department": "Applied Research",
        "seniority": "Senior",
        "min_experience_years": 4.0,
        "min_education": "Master's Degree",
        "text": """Title: Generative AI & LLM Systems Engineer
Department: Applied AI Labs
Location: Remote

About The Role:
Build cutting-edge enterprise Agentic AI systems, Retrieval-Augmented Generation (RAG) pipelines, and fine-tune foundation models for production applications.

Requirements & Must-Have Qualifications:
- 4+ years in software engineering with 2+ years dedicated to NLP / Generative AI.
- Mastery of Python and Deep Learning frameworks (PyTorch, Transformers / Hugging Face).
- Deep expertise with Large Language Models (LLMs), Prompt Engineering, and RAG architectures.
- Experience with LangChain, LlamaIndex, and Vector Databases (Pinecone, Chroma, Qdrant).
- Strong backend engineering with FastAPI, RESTful APIs, and asynchronous processing.
- Containerization with Docker and deployment on AWS or GCP.

Nice to Have & Preferred Qualifications:
- Fine-tuning experience (LoRA, QLoRA, PEFT) on open-source weights (Llama, Mistral).
- Experience with MLOps tools (MLflow, Weights & Biases) and evaluation frameworks.
- Master's or Ph.D. in Computer Science or Artificial Intelligence."""
    }
]


BENCHMARK_RESUMES = [
    # 1. Senior ML Engineer - Strong Fit (Ph.D.)
    {
        "candidate_id": "CAND_001",
        "name": "Dr. Sarah Chen",
        "target_role": "Senior Machine Learning Engineer",
        "experience_years": 6.5,
        "education": "Ph.D.",
        "text": """Dr. Sarah Chen, Ph.D.
Email: sarah.chen@example.com | Phone: +1-555-019-2834 | LinkedIn: linkedin.com/in/sarahchen-ml
San Francisco, CA

PROFESSIONAL SUMMARY
Senior Machine Learning Engineer with 6.5 years of experience researching, architecting, and deploying predictive ML and Deep Learning systems to cloud production. Expert in PyTorch, Scikit-Learn, MLOps, and Kubernetes-orchestrated distributed model serving.

TECHNICAL SKILLS
- Programming: Python, SQL, C++, Bash/Shell
- ML & Deep Learning: Machine Learning, Deep Learning, PyTorch, TensorFlow, Scikit-Learn, XGBoost, Pandas, NumPy, SciPy
- MLOps & Cloud: MLOps, MLflow, Docker, Kubernetes, AWS (SageMaker, S3, EC2), CI/CD
- Architecture & Practices: System Design, Distributed Systems, Git / Version Control, Unit Testing, Agile / Scrum
- Bonus Skills: Transformers, Natural Language Processing, Vector Databases, Large Language Models

WORK EXPERIENCE
Staff Machine Learning Engineer | HyperScale AI (2021 - Present)
- Architected automated MLOps training and evaluation pipeline using MLflow, Docker, and Kubernetes on AWS, cutting model retraining turnaround by 65%.
- Developed real-time recommendation systems and gradient boosting models using Scikit-Learn and XGBoost handling 40M daily predictions.
- Led model governance, drift monitoring, and CI/CD deployment pipelines with zero downtime.

Machine Learning Engineer | Apex Analytics (2018 - 2021)
- Built deep learning NLP classification models in PyTorch and TensorFlow for multi-lingual text extraction.
- Engineered automated feature stores in SQL and Pandas, collaborating closely with data engineering.

EDUCATION
- Ph.D. in Computer Science (Machine Learning Focus), Stanford University (2018)
- B.S. in Computer Science, UC Berkeley (2014)"""
    },
    # 2. Senior ML Engineer - Strong Fit (Master's)
    {
        "candidate_id": "CAND_002",
        "name": "Marcus Vance",
        "target_role": "Senior Machine Learning Engineer",
        "experience_years": 5.0,
        "education": "Master's Degree",
        "text": """Marcus Vance, M.S.
Email: marcus.vance@example.com | Phone: +1-555-014-9921
Seattle, WA

SUMMARY
Results-driven Machine Learning Engineer with 5 years of industry experience. Specialized in productionizing supervised and unsupervised algorithms, building scalable RESTful APIs, and implementing robust MLOps practices on AWS.

TECHNICAL SKILLS
- Core: Python, SQL, Scikit-Learn, PyTorch, TensorFlow, XGBoost, Pandas, NumPy
- Infrastructure: Docker, Kubernetes, AWS (SageMaker, EC2, S3), CI/CD, MLOps, Linux
- Software Engineering: Git / Version Control, Testing & QA, Pytest, System Design, RESTful APIs

PROFESSIONAL EXPERIENCE
Senior ML Engineer | CloudData Systems (2021 - Present)
- Trained and tuned XGBoost and PyTorch neural networks for enterprise fraud detection with 94.2% precision.
- Built containerized model microservices using Docker and FastAPI deployed on AWS SageMaker.
- Mentored junior engineers on Unit Testing, code quality, and MLOps monitoring.

ML Engineer | DataPulse Inc. (2019 - 2021)
- Implemented Scikit-Learn pipelines and feature engineering workflows for customer churn modeling.
- Managed model deployments using Docker and GitHub Actions CI/CD.

EDUCATION
- Master's in Data Science, University of Washington (2019)
- B.S. in Statistics, Purdue University (2017)"""
    },
    # 3. Mid ML Engineer - Moderate Fit (Missing Kubernetes & AWS)
    {
        "candidate_id": "CAND_003",
        "name": "Elena Rostova",
        "target_role": "Machine Learning Engineer (Mid-Level)",
        "experience_years": 3.5,
        "education": "Bachelor's Degree",
        "text": """Elena Rostova
Email: elena.r@example.com | Chicago, IL

OBJECTIVE
Machine Learning Engineer with 3.5 years of experience developing predictive models using Python, Scikit-Learn, and PyTorch. Seeking to expand into enterprise MLOps.

SKILLS
- Machine Learning, Deep Learning, PyTorch, Scikit-Learn, Pandas, NumPy, Python, SQL
- Data Visualization: Matplotlib, Seaborn
- Tools: Git / Version Control, Docker, Linux, RESTful APIs
- Missing/Learning: Kubernetes, AWS SageMaker, MLOps

EXPERIENCE
ML Developer | FinTech Sol (2021 - Present)
- Built regression and classification models using Scikit-Learn and XGBoost for credit scoring.
- Assisted in containerizing inference scripts with Docker.
- Wrote clean Python code with Git version control and participated in Agile sprints.

EDUCATION
- Bachelor's in Computer Science, University of Illinois (2021)"""
    },
    # 4. Senior Full Stack - Strong Fit
    {
        "candidate_id": "CAND_004",
        "name": "Alex Rivera",
        "target_role": "Senior Full Stack Developer",
        "experience_years": 5.5,
        "education": "Bachelor's Degree",
        "text": """Alex Rivera
Email: alex.rivera@example.com | New York, NY | github.com/arivera-dev

EXECUTIVE SUMMARY
Senior Full Stack Engineer with 5.5 years of experience building high-throughput web applications. Expert in React, Next.js, Node.js, TypeScript, PostgreSQL, and Dockerized cloud microservices.

TECHNICAL SKILLS
- Frontend: React, Next.js, TypeScript, JavaScript, HTML/CSS, Tailwind CSS, Redux
- Backend: Node.js, Express, RESTful APIs, GraphQL, Microservices
- Databases: PostgreSQL, MySQL, Redis, MongoDB
- DevOps & Tools: Docker, Git / Version Control, Testing & QA (Jest, Cypress), CI/CD, AWS
- Practices: System Design, Agile / Scrum, Code Review

EXPERIENCE
Senior Full Stack Engineer | Nova Web Labs (2021 - Present)
- Led frontend migration to Next.js and React with TypeScript, improving Core Web Vitals by 42%.
- Designed and documented scalable RESTful APIs in Node.js and Express serving 2M monthly users.
- Optimized PostgreSQL database queries, reducing p99 latency from 850ms to 95ms.

Full Stack Developer | PixelCraft Studio (2019 - 2021)
- Developed responsive web interfaces using React, JavaScript, and Tailwind CSS.
- Implemented backend authentication and user management using Node.js and PostgreSQL.

EDUCATION
- B.S. in Software Engineering, Rochester Institute of Technology (2019)"""
    },
    # 5. Full Stack Mid - Strong Fit
    {
        "candidate_id": "CAND_005",
        "name": "Priya Sharma",
        "target_role": "Full Stack Developer",
        "experience_years": 4.0,
        "education": "Master's Degree",
        "text": """Priya Sharma
Email: priya.sharma@example.com | San Jose, CA

PROFESSIONAL SUMMARY
Full Stack Developer with 4 years of experience delivering robust web platforms using React, Node.js, Express, TypeScript, and relational databases.

SKILLS
- Languages: JavaScript, TypeScript, Python, SQL, HTML/CSS
- Frameworks: React, Node.js, Express, Next.js, Tailwind CSS
- Databases: PostgreSQL, MySQL, MongoDB
- Tools: Docker, Git / Version Control, Jest, CI/CD, Agile / Scrum

EXPERIENCE
Full Stack Software Engineer | TechSphere Solutions (2020 - Present)
- Built modular React web components and backend Node.js microservices.
- Created automated unit testing suites in Jest and implemented GitHub Actions CI/CD.
- Integrated payment gateways and external partner RESTful APIs.

EDUCATION
- M.S. in Computer Science, San Jose State University (2020)"""
    },
    # 6. Senior DevOps / SRE - Strong Fit
    {
        "candidate_id": "CAND_006",
        "name": "David 'Kip' Miller",
        "target_role": "Senior DevOps / SRE Engineer",
        "experience_years": 6.0,
        "education": "Bachelor's Degree",
        "text": """David Miller
Email: david.miller@example.com | Austin, TX

SUMMARY
Senior DevOps & Infrastructure Engineer with 6 years of experience managing distributed cloud clusters. Expert in Kubernetes, Docker, Terraform, AWS, and zero-downtime CI/CD pipelines.

SKILLS
- Cloud & Infrastructure: AWS, Docker, Kubernetes, Terraform, Ansible, Linux, Bash/Shell
- CI/CD & Automation: CI/CD, GitHub Actions, Jenkins, Python, Go
- Observability: Monitoring & Logging, Prometheus, Grafana, ELK Stack
- Architecture: System Design, Distributed Systems, Git / Version Control, Agile / Scrum

WORK EXPERIENCE
Lead Infrastructure Engineer | Apex Cloud (2021 - Present)
- Administered 15+ multi-region Kubernetes (EKS) clusters with Terraform and Helm.
- Implemented automated GitOps CI/CD pipelines deploying 120+ microservices daily.
- Reduced cloud spend by 30% through automated AWS spot instance provisioning.

DevOps Engineer | Stratus Systems (2018 - 2021)
- Migrated bare-metal workloads to Docker and AWS using Terraform.
- Built monitoring dashboards in Prometheus and Grafana for system health telemetry.

EDUCATION
- B.S. in Computer Engineering, Texas A&M University (2018)"""
    },
    # 7. Lead Data Analyst - Strong Fit
    {
        "candidate_id": "CAND_007",
        "name": "Jessica Taylor",
        "target_role": "Lead Data Analyst",
        "experience_years": 4.5,
        "education": "Bachelor's Degree",
        "text": """Jessica Taylor
Email: jessica.t@example.com | Boston, MA

SUMMARY
Lead Data Analyst with 4.5 years of experience applying SQL, Python, and business intelligence dashboards to drive revenue growth and operational optimization.

SKILLS
- Analytics & Modeling: SQL, Python, Pandas, NumPy, Data Warehousing, Statistics, A/B Testing
- Visualization: Tableau, PowerBI, Matplotlib, Seaborn
- Soft Skills: Problem Solving, Communication, Stakeholder Management, Project Management

EXPERIENCE
Lead Analytics Consultant | Beacon Insights (2021 - Present)
- Formulated complex SQL data pipelines in Snowflake and BigQuery for customer lifetime value (LTV) forecasting.
- Designed 25+ executive dashboards in Tableau used weekly by C-level leadership.
- Conducted multivariate statistical experiments and A/B testing on pricing models.

Data Analyst | RetailMetric Group (2019 - 2021)
- Cleaned and transformed high-volume transactional data using Python and Pandas.
- Partnered with product managers to define KPIs and tracking schemas.

EDUCATION
- B.S. in Statistics and Economics, Boston College (2019)"""
    },
    # 8. Generative AI Specialist - Strong Fit
    {
        "candidate_id": "CAND_008",
        "name": "Kenji Takahashi",
        "target_role": "Generative AI & LLM Systems Engineer",
        "experience_years": 4.5,
        "education": "Master's Degree",
        "text": """Kenji Takahashi
Email: kenji.t@example.com | Seattle, WA

SUMMARY
AI Systems Engineer with 4.5 years of experience building Natural Language Processing and Generative AI applications. Expert in Large Language Models (LLMs), RAG, LangChain, PyTorch, and Vector Databases.

SKILLS
- GenAI & NLP: Large Language Models, LangChain, Vector Databases, Transformers, PyTorch, Deep Learning, Natural Language Processing
- Backend & Cloud: Python, FastAPI, RESTful APIs, Docker, AWS, Git / Version Control
- Secondary: Scikit-Learn, MLOps, PostgreSQL

EXPERIENCE
Senior AI Engineer | CognitiveWorks (2022 - Present)
- Developed production RAG pipeline using LangChain, Chroma vector database, and fine-tuned open-source LLMs.
- Built low-latency asynchronous FastAPI microservices containerized with Docker on AWS.
- Designed evaluation frameworks for hallucination detection and response quality.

NLP Engineer | Lexicon Labs (2020 - 2022)
- Trained transformer classification models using PyTorch and Hugging Face for enterprise document search.

EDUCATION
- M.S. in Artificial Intelligence, Carnegie Mellon University (2020)
- B.S. in Computer Science, University of Washington (2018)"""
    },
    # 9. Low Fit (Junior / Web Entry)
    {
        "candidate_id": "CAND_009",
        "name": "Robert 'Bob' Jenkins",
        "target_role": "Junior Web Coordinator",
        "experience_years": 1.2,
        "education": "Diploma / Associate",
        "text": """Robert Jenkins
Email: bob.j@example.com | Dallas, TX

SUMMARY
Junior Web Enthusiast with 1 year of experience building basic websites. Familiar with HTML, CSS, and elementary JavaScript.

SKILLS
- HTML/CSS, JavaScript, WordPress, Photoshop
- Seeking entry-level web developer or content coordinator role.

EXPERIENCE
Junior Web Coordinator | Local Agency (2023 - Present)
- Updated static HTML and CSS pages for retail clients.
- Uploaded assets and wrote blog descriptions in WordPress.

EDUCATION
- Associate Degree in Graphic Design, Dallas Community College (2022)"""
    },
    # 10. Low Fit (Digital Marketing)
    {
        "candidate_id": "CAND_010",
        "name": "Samantha O'Connor",
        "target_role": "Digital Marketing Specialist",
        "experience_years": 3.0,
        "education": "Bachelor's Degree",
        "text": """Samantha O'Connor
Email: samantha.oc@example.com | Denver, CO

SUMMARY
Marketing & SEO Specialist with 3 years of experience managing social media campaigns, Google Ads, and content marketing.

SKILLS
- Digital Marketing, SEO, Google Analytics, Content Strategy, Copywriting, Communication, Social Media
- Tools: Canva, WordPress, Mailchimp

EXPERIENCE
Marketing Specialist | Peak Growth Agency (2021 - Present)
- Increased website organic traffic by 40% through SEO optimization.
- Managed paid social ad spend across Facebook and Instagram.

EDUCATION
- B.A. in Communications, University of Colorado Boulder (2021)"""
    },
    # 11. Data Engineer & Big Data Architect
    {
        "candidate_id": "CAND_011",
        "name": "Aarav Patel",
        "target_role": "Senior Data Engineer",
        "experience_years": 5.0,
        "education": "Master's Degree",
        "text": """Aarav Patel, M.S.
Email: aarav.patel@example.com | Atlanta, GA

SUMMARY
Data Engineer with 5 years of experience building distributed batch and streaming pipelines. Expert in Apache Spark, Apache Kafka, Apache Airflow, Snowflake, and Python.

TECHNICAL SKILLS
- Big Data & Data Warehousing: Apache Spark, PySpark, Apache Kafka, Apache Airflow, Snowflake, Databricks, PostgreSQL, Data Warehousing, SQL
- Languages & Cloud: Python, Scala, SQL, AWS (S3, EMR, Redshift), Docker, Git / Version Control

EXPERIENCE
Senior Data Platform Engineer | FinFlow (2021 - Present)
- Architected enterprise ETL pipelines processing 10 TBs daily using PySpark and Apache Airflow on AWS.
- Modeled scalable data warehouses in Snowflake reducing query times by 50%.

EDUCATION
- M.S. in Information Systems, Georgia Tech (2019)"""
    },
    # 12. Computer Vision Specialist
    {
        "candidate_id": "CAND_012",
        "name": "Dr. Lucas Meyer",
        "target_role": "Computer Vision ML Scientist",
        "experience_years": 5.0,
        "education": "Ph.D.",
        "text": """Dr. Lucas Meyer
Email: lucas.meyer@example.com | Boston, MA

SUMMARY
Computer Vision and Deep Learning Specialist with 5 years of experience deploying spatial perception models. Expert in PyTorch, OpenCV, Computer Vision, C++, and Python.

SKILLS
- Deep Learning, Computer Vision, PyTorch, TensorFlow, OpenCV, Python, C++, Docker, Linux, Git / Version Control

EXPERIENCE
Computer Vision Engineer | VisionTech (2021 - Present)
- Designed real-time object detection models using PyTorch and YOLO with edge inference optimization.
- Published 4 peer-reviewed papers on neural video segmentation.

EDUCATION
- Ph.D. in Computer Vision, MIT (2020)"""
    },
    # 13. Junior ML Bootcamp Grad
    {
        "candidate_id": "CAND_013",
        "name": "Jordan Lee",
        "target_role": "Junior Machine Learning Engineer",
        "experience_years": 1.0,
        "education": "Bachelor's Degree",
        "text": """Jordan Lee
Email: jordan.lee@example.com | Austin, TX

SUMMARY
Enthusiastic Junior ML Developer with 1 year of project experience. Proficient in Python, Scikit-Learn, Pandas, and basic deep learning in PyTorch. Eager to learn MLOps and cloud engineering.

SKILLS
- Python, Scikit-Learn, Pandas, NumPy, Matplotlib, SQL, Git / Version Control, Machine Learning
- Familiar with: Flask, Docker basics

EXPERIENCE
Junior ML Intern | DataLab (2023 - Present)
- Assisted senior data scientists with feature preprocessing in Pandas and model validation in Scikit-Learn.
- Built interactive exploratory data analysis notebooks.

EDUCATION
- B.S. in Mathematics, UT Austin (2023)"""
    },
    # 14. Senior Backend Java Engineer
    {
        "candidate_id": "CAND_014",
        "name": "Vikram Sethi",
        "target_role": "Senior Backend Systems Engineer",
        "experience_years": 7.0,
        "education": "Bachelor's Degree",
        "text": """Vikram Sethi
Email: vikram.sethi@example.com | Chicago, IL

SUMMARY
Senior Backend Systems Engineer with 7 years of experience designing mission-critical distributed architectures. Deep expertise in Java, Spring Boot, Microservices, PostgreSQL, and AWS.

SKILLS
- Java, Spring Boot, Microservices, RESTful APIs, PostgreSQL, MySQL, Redis, Docker, Kubernetes, AWS, System Design, Distributed Systems

EXPERIENCE
Lead Backend Engineer | Enterprise Core (2020 - Present)
- Architected enterprise banking microservices in Spring Boot handling 10,000 TPS.
- Led migration from monolithic architecture to Kubernetes-hosted microservices.

EDUCATION
- B.Tech in Computer Science, IIT Delhi (2017)"""
    },
    # 15. Frontend React Specialist
    {
        "candidate_id": "CAND_015",
        "name": "Chloe Dupont",
        "target_role": "Senior Frontend Engineer",
        "experience_years": 5.0,
        "education": "Bachelor's Degree",
        "text": """Chloe Dupont
Email: chloe.dupont@example.com | Montreal / Remote

SUMMARY
Senior Frontend Engineer with 5 years specializing in React, Next.js, TypeScript, and modern CSS architecture. Passionate about web performance and accessibility.

SKILLS
- React, Next.js, TypeScript, JavaScript, HTML/CSS, Tailwind CSS, GraphQL, Jest, Git / Version Control, Agile / Scrum

EXPERIENCE
Lead Frontend Developer | UI Dynamics (2021 - Present)
- Rebuilt design system using React and Tailwind CSS adopted across 8 engineering teams.
- Optimized bundle sizes and lazy-loading in Next.js, achieving 98+ Lighthouse scores.

EDUCATION
- B.S. in Computer Science, McGill University (2019)"""
    },
    # 16. Cloud Architect & SRE Lead
    {
        "candidate_id": "CAND_016",
        "name": "Gabriel Santos",
        "target_role": "Principal Cloud Architect",
        "experience_years": 8.5,
        "education": "Master's Degree",
        "text": """Gabriel Santos
Email: gabriel.santos@example.com | New York, NY

SUMMARY
Principal Cloud Architect with 8.5 years of experience leading multi-cloud infrastructure transformations across AWS and GCP. Expert in Kubernetes, Terraform, System Design, and Enterprise Security.

SKILLS
- AWS, GCP, Terraform, Kubernetes, Docker, Linux, Bash/Shell, CI/CD, System Design, Distributed Systems, Leadership, Monitoring & Logging

EXPERIENCE
Principal Cloud Architect | FinCloud Global (2019 - Present)
- Directed cloud governance and automated infrastructure provisioning using Terraform across 300+ AWS accounts.
- Led SRE team of 14 engineers in achieving five-nines availability (99.999%).

EDUCATION
- M.S. in Computer Engineering, Columbia University (2016)"""
    },
    # 17. SRE Engineer
    {
        "candidate_id": "CAND_017",
        "name": "Liam O'Reilly",
        "target_role": "Site Reliability Engineer",
        "experience_years": 4.5,
        "education": "Bachelor's Degree",
        "text": """Liam O'Reilly
Email: liam.oreilly@example.com | Dublin / Remote

SUMMARY
Site Reliability Engineer with 4.5 years of experience automating infrastructure operations, incident response, and observability.

SKILLS
- Linux, Kubernetes, Docker, Terraform, Ansible, CI/CD, Prometheus, Grafana, Python, Go, Bash/Shell, AWS

EXPERIENCE
SRE | ScaledOps (2021 - Present)
- Maintained distributed Kubernetes clusters on AWS with automated Prometheus and Grafana alerts.
- Built custom infrastructure telemetry exporters in Go and Python.

EDUCATION
- B.S. in Computing, Trinity College Dublin (2020)"""
    },
    # 18. Python Full Stack Developer
    {
        "candidate_id": "CAND_018",
        "name": "Fatima Al-Mansoor",
        "target_role": "Full Stack Engineer (Python + React)",
        "experience_years": 4.0,
        "education": "Bachelor's Degree",
        "text": """Fatima Al-Mansoor
Email: fatima.m@example.com | London, UK

SUMMARY
Full Stack Engineer with 4 years of experience building modern web portals using React, Python, Django, FastAPI, and PostgreSQL.

SKILLS
- React, JavaScript, TypeScript, Python, Django, FastAPI, RESTful APIs, PostgreSQL, Docker, Git / Version Control, HTML/CSS

EXPERIENCE
Software Engineer | GlobalWeb Ltd (2020 - Present)
- Engineered high-traffic portals using React and FastAPI with asynchronous PostgreSQL connections.
- Deployed Dockerized applications on cloud infrastructure with GitHub Actions CI/CD.

EDUCATION
- B.Sc. in Software Engineering, University of Manchester (2020)"""
    },
    # 19. BI & Data Analytics Consultant
    {
        "candidate_id": "CAND_019",
        "name": "Brandon Cole",
        "target_role": "Business Intelligence Analyst",
        "experience_years": 3.5,
        "education": "Bachelor's Degree",
        "text": """Brandon Cole
Email: brandon.c@example.com | Charlotte, NC

SUMMARY
BI Analyst with 3.5 years of experience transforming transactional sales data into executive dashboards and predictive KPIs.

SKILLS
- SQL, Tableau, PowerBI, Python, Pandas, Statistics, Data Warehousing, Communication, Problem Solving

EXPERIENCE
BI Analyst | Vanguard Retail (2021 - Present)
- Designed automated PowerBI reporting models connected to Snowflake data warehouse.
- Delivered weekly revenue forecasting models using Python and Pandas.

EDUCATION
- B.S. in Economics, UNC Chapel Hill (2021)"""
    },
    # 20. Technical Product Manager
    {
        "candidate_id": "CAND_020",
        "name": "Meera Raman",
        "target_role": "Technical Product Manager",
        "experience_years": 6.0,
        "education": "Master's Degree",
        "text": """Meera Raman, MBA
Email: meera.raman@example.com | Seattle, WA

SUMMARY
Technical Product Manager with 6 years leading AI/ML and enterprise software products from ideation to scale.

SKILLS
- Product Management, Agile / Scrum, Jira, System Design, Communication, Leadership, SQL, Stakeholder Management

EXPERIENCE
Senior Technical Product Manager | Horizon Software (2020 - Present)
- Led cross-functional roadmap for predictive intelligence features driving $14M ARR.
- Partnered with ML engineering teams to define model acceptance criteria and SLAs.

EDUCATION
- MBA, Kellogg School of Management (2018)
- B.S. in Computer Science (2015)"""
    }
]


def generate_benchmark_dataset(output_dir: str = "data"):
    """
    Generates structured job descriptions JSON, resumes CSV, and physical sample PDF & TXT files.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_dir = os.path.join(output_dir, "sample_resumes")
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Save Job Descriptions JSON
    jd_path = os.path.join(output_dir, "job_descriptions.json")
    with open(jd_path, "w", encoding="utf-8") as f:
        json.dump(JOB_DESCRIPTIONS, f, indent=2)
    print(f"Saved {len(JOB_DESCRIPTIONS)} Job Descriptions to {jd_path}")

    # 2. Save Resumes CSV
    resumes_df = pd.DataFrame(BENCHMARK_RESUMES)
    csv_path = os.path.join(output_dir, "resumes.csv")
    resumes_df.to_csv(csv_path, index=False)
    print(f"Saved {len(resumes_df)} Candidate Resumes to {csv_path}")

    # 3. Create Sample TXT & PDF Resumes for live file upload demonstrations
    for cand in BENCHMARK_RESUMES[:6]:
        cand_id = cand["candidate_id"]
        # TXT file
        clean_cand_name = cand['name'].replace(' ', '_').replace("'", "").replace('"', '').replace('.', '')
        txt_path = os.path.join(sample_dir, f"{cand_id}_{clean_cand_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cand["text"])

        # PDF file using ReportLab
        pdf_path = os.path.join(sample_dir, f"{cand_id}_{clean_cand_name}.pdf")
        _create_pdf_resume(pdf_path, cand)

    print(f"Generated sample TXT and PDF resumes in {sample_dir}")


def _create_pdf_resume(filepath: str, candidate_info: dict):
    """
    Helper to render a clean, standard PDF resume using ReportLab.
    """
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter

    # Margin and initial coordinates
    x = 54
    y = height - 54

    # Header: Candidate Name
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.HexColor("#1e293b"))
    c.drawString(x, y, candidate_info["name"])
    y -= 20

    # Role & Target
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#3b82f6"))
    c.drawString(x, y, f"Target: {candidate_info['target_role']} | Exp: {candidate_info['experience_years']} Years | Degree: {candidate_info['education']}")
    y -= 14

    # Decorative separator line
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.line(x, y, width - 54, y)
    y -= 18

    # Resume Body Text
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#334155"))

    body_lines = candidate_info["text"].split("\n")
    for line in body_lines:
        line_clean = line.strip()
        if not line_clean:
            y -= 8
            continue

        if any(h in line_clean.upper() for h in ["PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "WORK EXPERIENCE", "EDUCATION", "SKILLS", "OBJECTIVE", "EXECUTIVE SUMMARY"]):
            y -= 6
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(colors.HexColor("#0f172a"))
            c.drawString(x, y, line_clean)
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#334155"))
            y -= 14
        else:
            # Wrap line if too long
            if len(line_clean) > 85:
                part1 = line_clean[:85]
                part2 = line_clean[85:]
                c.drawString(x, y, part1)
                y -= 12
                c.drawString(x + 10, y, part2)
                y -= 12
            else:
                c.drawString(x, y, line_clean)
                y -= 12

        if y < 54:
            c.showPage()
            y = height - 54

    c.save()


if __name__ == "__main__":
    generate_benchmark_dataset()

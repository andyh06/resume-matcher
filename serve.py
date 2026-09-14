"""FastAPI app serving the resume-matching TF-IDF pipeline.

The fitted pipeline bundle (pipeline.joblib) is loaded once at import time.
If it's missing or fails to load, the app still starts -- every endpoint
that needs it returns 503 Service Unavailable instead of crashing the
process or returning a bare 500.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

# joblib.load needs this class importable to reconstruct the "cleaner" pipeline
# step. This import looks unused but is required -- without it, unpickling the
# bundle raises AttributeError in any process that hasn't already imported
# pipeline_def (e.g. a fresh container). This is the single most common way
# this kind of deployment fails.
from pipeline_def import ResumeTextCleaner  # noqa: F401

logger = logging.getLogger("uvicorn.error")

# Relative paths break once the working directory differs from this file's
# directory (e.g. inside a Modal container). PIPELINE_PATH can still be
# overridden via env var for deployments that mount the artifact elsewhere.
PIPELINE_PATH = Path(os.environ.get("PIPELINE_PATH", Path(__file__).parent / "pipeline.joblib"))
NEAR_ZERO = 1e-9

# Job postings are marketing copy; the O*NET corpus the pipeline was fit on is
# formal task/skill text. Posting boilerplate ("looking", "plus", "required")
# is therefore near-absent from the corpus, so IDF rates it as ultra-rare and
# ultra-important -- a train/serve distribution mismatch. This list filters
# that boilerplate out of the keyword-gap output without touching the corpus
# or vectorizer.
JOB_POSTING_STOPWORDS = frozenset({
    "looking", "plus", "required", "requirements", "preferred", "nice", "ideal",
    "candidate", "candidates", "seeking", "join", "must", "strong", "excellent",
    "proven", "solid", "experience", "experienced", "years", "year", "ability",
    "able", "work", "working", "team", "teams", "role", "position", "opportunity",
    "company", "help", "great", "good", "well", "etc", "including", "include",
    "includes", "responsibilities", "responsible", "duties", "skills", "skilled",
    "knowledge", "familiarity", "familiar", "understanding", "background",
    "environment", "environments", "based", "using", "use", "used", "related",
    "various", "multiple", "new", "high", "highly", "level", "levels", "self",
    "growing", "growth", "fast", "paced", "dynamic", "passion", "passionate",
    "detail", "oriented", "communication", "written", "verbal", "collaborate",
    "collaborative", "collaboration", "minimum", "bonus", "equivalent", "degree",
    "bachelor", "benefits", "salary", "compensation", "employer", "equal",
    "diversity", "inclusive", "apply", "application", "job", "jobs", "career",
    "hire", "hiring", "employment", "day", "days", "week", "weeks", "time",
    "flexible", "remote", "onsite", "hybrid", "location", "office", "world",
    "innovative", "leading", "leader", "industry", "organization", "mission",
    "culture", "value", "values", "people", "individual", "individuals",
})

# ---------------------------------------------------------------------------
# Load the artifact once at module import. BUNDLE stays None on any failure
# so the app can still start and serve /health with artifact_loaded=False.
# ---------------------------------------------------------------------------
BUNDLE = None
try:
    BUNDLE = joblib.load(PIPELINE_PATH)
    logger.info("Loaded %s", PIPELINE_PATH)
except Exception as exc:  # noqa: BLE001 - any load failure must not crash the app
    logger.warning("Could not load %s: %s", PIPELINE_PATH, exc)
    BUNDLE = None


def get_bundle():
    if BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not loaded ({PIPELINE_PATH} missing or failed to load).",
        )
    return BUNDLE


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Resume Matcher API",
    description=(
        "Matches a resume against a job description using a TF-IDF pipeline "
        "fitted on O*NET occupation data. Returns a match score, keyword gaps "
        "between the resume and job description, and the closest O*NET "
        "occupations to the resume."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="Always 'ok' if the process is up and answering requests.")
    artifact_loaded: bool = Field(..., description="Whether pipeline.joblib was loaded successfully at startup.")


class PipelineStep(BaseModel):
    name: str = Field(..., description="Step name inside the sklearn Pipeline.")
    class_name: str = Field(..., description="Class name of the fitted step, e.g. 'TfidfVectorizer'.")


class InfoResponse(BaseModel):
    steps: List[PipelineStep] = Field(..., description="The pipeline's steps, in order.")
    built_at: str = Field(..., description="ISO 8601 UTC timestamp of when the pipeline was fitted.")
    sklearn_version: str = Field(..., description="scikit-learn version used to fit the pipeline.")
    vocabulary_size: int = Field(..., description="Number of terms learned by the TF-IDF vectorizer.")
    n_occupations: int = Field(..., description="Number of O*NET occupations in the stored matrix.")
    matrix_shape: List[int] = Field(..., description="[n_occupations, vocabulary_size] shape of the stored TF-IDF matrix.")


class MatchRequest(BaseModel):
    job_description: str = Field(
        ...,
        min_length=50,
        max_length=20000,
        description="Full text of the job posting.",
    )
    resume: str = Field(
        ...,
        min_length=50,
        max_length=20000,
        description="Full text of the candidate's resume.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of closest O*NET occupations to return for the resume.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_description": (
                    "We are looking for a Backend Software Engineer to design and maintain "
                    "REST APIs in Python and Django, work with PostgreSQL and Redis, "
                    "containerize services with Docker and Kubernetes, and deploy to AWS. "
                    "Experience with CI/CD pipelines, code review, and unit testing required."
                ),
                "resume": (
                    "Software engineer with 4 years of experience building web applications "
                    "in Python using Flask. Worked with MySQL databases, wrote unit tests, "
                    "and collaborated with teammates on code review. Some exposure to Docker "
                    "and basic AWS deployment. Comfortable with Git and agile workflows."
                ),
                "top_k": 5,
            }
        }
    }


class KeywordWeight(BaseModel):
    term: str = Field(..., description="A vocabulary term (unigram or bigram) from the TF-IDF vectorizer.")
    weight: float = Field(..., description="TF-IDF weight associated with this term.")


class OccupationMatch(BaseModel):
    soc_code: str = Field(..., description="O*NET-SOC occupation code.")
    title: str = Field(..., description="O*NET occupation title.")
    score: float = Field(..., description="Cosine similarity between the resume and this occupation's document, 0-1.")


class MatchResponse(BaseModel):
    match_score: float = Field(..., description="Cosine similarity between resume and job description, scaled 0-100.")
    missing_keywords: List[KeywordWeight] = Field(
        ..., description="Top terms with real weight in the job description but near-absent from the resume."
    )
    matched_keywords: List[KeywordWeight] = Field(
        ..., description="Top terms with weight in both the resume and job description."
    )
    closest_occupations: List[OccupationMatch] = Field(
        ..., description="The resume's top_k closest O*NET occupations by cosine similarity."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_informative_term(term: str) -> bool:
    """Drop an n-gram if every one of its tokens is job-posting boilerplate or
    purely numeric, and drop unigrams under 3 characters."""
    tokens = term.split(" ")

    if len(tokens) == 1 and len(term) < 3:
        return False

    if all(t in JOB_POSTING_STOPWORDS or t.isdigit() for t in tokens):
        return False

    return True


def _ranked_informative(order, mask, feature_names, weights, limit) -> List[KeywordWeight]:
    """Walk indices in `order` (already ranked, highest weight first), keep
    only those passing `mask`, filter boilerplate, then take the top `limit`.
    Filtering happens after ranking so boilerplate terms don't crowd out
    informative ones further down the ranked list."""
    results = []
    for i in order:
        if not mask[i]:
            continue
        term = feature_names[i]
        if not _is_informative_term(term):
            continue
        results.append(KeywordWeight(term=term, weight=round(float(weights[i]), 4)))
        if len(results) >= limit:
            break
    return results


def _keyword_gaps(resume_vec, jd_vec, feature_names) -> Tuple[List[KeywordWeight], List[KeywordWeight]]:
    resume_arr = resume_vec.toarray().ravel()
    jd_arr = jd_vec.toarray().ravel()

    jd_present = jd_arr > NEAR_ZERO

    missing_mask = jd_present & (resume_arr < NEAR_ZERO)
    missing_order = np.argsort(jd_arr)[::-1]
    missing = _ranked_informative(missing_order, missing_mask, feature_names, jd_arr, limit=15)

    matched_mask = jd_present & (resume_arr > NEAR_ZERO)
    combined = (jd_arr + resume_arr) / 2.0
    matched_order = np.argsort(combined)[::-1]
    matched = _ranked_informative(matched_order, matched_mask, feature_names, combined, limit=10)

    return missing, matched


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Cheap check that the server process is up and reports whether the model artifact loaded successfully at startup.",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", artifact_loaded=BUNDLE is not None)


@app.get(
    "/info",
    response_model=InfoResponse,
    summary="Describe the loaded model artifact",
    description="Returns pipeline structure and build metadata read directly from the loaded pipeline.joblib bundle. 503 if the artifact isn't loaded.",
)
def info() -> InfoResponse:
    bundle = get_bundle()
    metadata = bundle["metadata"]
    matrix = bundle["matrix"]

    return InfoResponse(
        steps=[PipelineStep(name=name, class_name=class_name) for name, class_name in metadata["steps"]],
        built_at=metadata["built_at"],
        sklearn_version=metadata["sklearn_version"],
        vocabulary_size=len(bundle["pipeline"].named_steps["tfidf"].vocabulary_),
        n_occupations=matrix.shape[0],
        matrix_shape=list(matrix.shape),
    )


@app.post(
    "/match",
    response_model=MatchResponse,
    summary="Match a resume against a job description",
    description=(
        "Transforms the resume and job description through the fitted TF-IDF pipeline, "
        "scores their similarity, surfaces job-description keywords missing from the resume "
        "and keywords that did land, and returns the resume's closest O*NET occupations."
    ),
)
def match(request: MatchRequest) -> MatchResponse:
    bundle = get_bundle()
    pipeline = bundle["pipeline"]
    matrix = bundle["matrix"]
    documents = bundle["documents"]

    resume_vec = pipeline.transform([request.resume])
    jd_vec = pipeline.transform([request.job_description])

    match_score = float(cosine_similarity(resume_vec, jd_vec)[0, 0]) * 100
    match_score = round(match_score, 1)

    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    missing_keywords, matched_keywords = _keyword_gaps(resume_vec, jd_vec, feature_names)

    occupation_scores = cosine_similarity(resume_vec, matrix).ravel()
    top_idx = np.argsort(occupation_scores)[::-1][: request.top_k]
    closest_occupations = [
        OccupationMatch(
            soc_code=documents.iloc[i]["soc_code"],
            title=documents.iloc[i]["title"],
            score=round(float(occupation_scores[i]), 4),
        )
        for i in top_idx
    ]

    return MatchResponse(
        match_score=match_score,
        missing_keywords=missing_keywords,
        matched_keywords=matched_keywords,
        closest_occupations=closest_occupations,
    )

"""Load pipeline.joblib in a fresh process and verify it actually works:
fitted state, metadata, and real matching against two contrasting fake resumes.

Run this as its own `python verify_pipeline.py` invocation -- never import
build_pipeline's in-memory objects -- because that's what the API and Modal
deployment will do, and a pipeline that only works "in the same process it
was fit in" is the most common way this kind of assignment silently breaks.
"""

import joblib
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Needed so joblib can unpickle the ResumeTextCleaner step.
from pipeline_def import ResumeTextCleaner  # noqa: F401


SOFTWARE_ENGINEER_RESUME = """
Experienced software engineer with 5 years building backend web services in
Python and Django. Designed and maintained REST APIs consumed by mobile and
web clients. Worked extensively with PostgreSQL for data modeling and query
optimization, and containerized services with Docker for consistent local
and production environments. Deployed and operated workloads on AWS,
including EC2, S3, and RDS. Collaborated closely with teammates through
regular code review, writing unit tests, and maintaining CI pipelines to
catch regressions before release. Comfortable debugging production issues,
profiling slow endpoints, and working with teammates to improve reliability
and developer velocity across the codebase.
""".strip()

NURSE_RESUME = """
Registered nurse with 6 years of experience providing direct patient care
in a busy hospital medical-surgical unit. Skilled in medication
administration, monitoring vital signs, and wound care. Responsible for
patient assessment, developing care plans, and coordinating with physicians
and other staff to ensure timely treatment. Experienced with electronic
health record (EHR) charting, documenting patient status, and maintaining
accurate records. Trained new nursing staff on unit protocols and infection
control procedures, and regularly educated patients and families on
post-discharge care instructions.
""".strip()


def print_top_matches(label, resume_text, pipeline, matrix, documents, top_n=5):
    resume_vector = pipeline.transform([resume_text])
    scores = cosine_similarity(resume_vector, matrix).ravel()
    top_idx = np.argsort(scores)[::-1][:top_n]

    print(f"\nTop {top_n} matches for {label}:")
    for rank, idx in enumerate(top_idx, start=1):
        title = documents.iloc[idx]["title"]
        soc = documents.iloc[idx]["soc_code"]
        print(f"  {rank}. {title} ({soc})  score={scores[idx]:.4f}")

    return [documents.iloc[idx]["title"] for idx in top_idx]


def main():
    print("=" * 100)
    print("1. Loading pipeline.joblib")
    bundle = joblib.load("pipeline.joblib")
    print("   Loaded OK.")

    pipeline = bundle["pipeline"]
    matrix = bundle["matrix"]
    documents = bundle["documents"]
    metadata = bundle["metadata"]

    print("\n" + "=" * 100)
    print("2. Learned vocabulary size")
    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    print(f"   loaded vocabulary: {vocab_size} terms")

    print("\n" + "=" * 100)
    print("3. Fitted vs. unfitted proof")
    print(f"   loaded vocabulary:  {vocab_size} terms (fitted TfidfVectorizer inside the loaded pipeline)")
    try:
        fresh_vectorizer = TfidfVectorizer()
        fresh_vectorizer.transform(["some text"])
        print("   fresh vectorizer:   UNEXPECTED - no error raised")
    except NotFittedError as e:
        print(f"   fresh vectorizer:   NotFittedError -> {e}")

    print("\n" + "=" * 100)
    print("4. Metadata")
    for key, value in metadata.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 100)
    print("5. Matching test: software engineer resume")
    se_top = print_top_matches("SOFTWARE ENGINEER resume", SOFTWARE_ENGINEER_RESUME, pipeline, matrix, documents)
    if not any("software developer" in t.lower() for t in se_top):
        print("\n   *** WARNING: no 'Software Developers'-like title in top 5. "
              "Something is wrong with the corpus or vectorizer settings. ***")
    else:
        print("\n   OK: a Software Developers-like title is in the top 5.")

    print("\n" + "=" * 100)
    print("6. Matching test: nurse resume")
    print_top_matches("NURSE resume", NURSE_RESUME, pipeline, matrix, documents)


if __name__ == "__main__":
    main()

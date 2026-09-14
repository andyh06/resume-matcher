"""Fit the resume-matching TF-IDF pipeline on data/corpus.csv and dump a
bundle (pipeline + matrix + row metadata) to pipeline.joblib.
"""

import datetime
import os

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from pipeline_def import ResumeTextCleaner


def main():
    corpus = pd.read_csv("data/corpus.csv", encoding="utf-8")

    pipeline = Pipeline([
        ("cleaner", ResumeTextCleaner()),
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            max_features=200000,
        )),
    ])

    matrix = pipeline.fit_transform(corpus["document"])

    bundle = {
        "pipeline": pipeline,
        "matrix": matrix,
        "documents": corpus[["soc_code", "title", "description"]].reset_index(drop=True),
        "metadata": {
            "steps": [(name, type(step).__name__) for name, step in pipeline.steps],
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
        },
    }

    joblib.dump(bundle, "pipeline.joblib")

    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    n_rows, n_cols = matrix.shape
    density = matrix.nnz / (n_rows * n_cols)
    size_mb = os.path.getsize("pipeline.joblib") / (1024 * 1024)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix density: {density:.6f} ({matrix.nnz} nonzero of {n_rows * n_cols})")
    print(f"pipeline.joblib size: {size_mb:.2f} MB")
    if size_mb > 80:
        print("WARNING: pipeline.joblib exceeds 80 MB")


if __name__ == "__main__":
    main()

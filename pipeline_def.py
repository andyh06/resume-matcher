"""Custom sklearn transformer for cleaning resume/occupation text.

ResumeTextCleaner lowercases text, protects multi-character tech tokens
(C++, C#, .NET, Node.js, ...) from punctuation stripping by rewriting them
to punctuation-free equivalents, then strips remaining punctuation and
collapses whitespace.
"""

import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Module-level constant, not built in __init__: mapping of raw tech token
# (lowercase) -> punctuation-free equivalent. Order in this dict does not
# matter; longest-match-first is enforced at match time in transform().
DEFAULT_TOKEN_MAP = {
    "asp.net": "aspdotnet",
    ".net": "dotnet",
    "c++": "cplusplus",
    "c#": "csharp",
    "f#": "fsharp",
    "objective-c": "objectivec",
    "node.js": "nodejs",
    "next.js": "nextjs",
    "vue.js": "vuejs",
    "react.js": "reactjs",
    "ci/cd": "cicd",
}


def _build_token_pattern(token_map):
    """Compile a regex matching any token key, longest first, with
    boundaries that tolerate adjacent punctuation like "(C#)," or "C++,"
    but avoid matching inside a longer alphanumeric word.
    """
    tokens = sorted(token_map.keys(), key=len, reverse=True)
    alternation = "|".join(re.escape(t) for t in tokens)
    return re.compile(r"(?<![a-z0-9])(?:" + alternation + r")(?![a-z0-9])", re.IGNORECASE)


class ResumeTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, token_map=None, lowercase=True, min_token_length=2):
        self.token_map = token_map
        self.lowercase = lowercase
        self.min_token_length = min_token_length

    def fit(self, X, y=None):
        self.n_documents_seen_ = len(list(X))
        return self

    def transform(self, X):
        check_is_fitted(self, "n_documents_seen_")

        token_map = self.token_map if self.token_map is not None else DEFAULT_TOKEN_MAP
        pattern = _build_token_pattern(token_map)

        cleaned = []
        for doc in X:
            text = doc if isinstance(doc, str) else "" if doc is None else str(doc)

            if self.lowercase:
                text = text.lower()

            text = pattern.sub(lambda m: token_map[m.group(0).lower()], text)

            text = re.sub(r"[^a-z0-9\s]", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()

            if self.min_token_length > 1:
                words = [w for w in text.split(" ") if len(w) >= self.min_token_length]
                text = " ".join(words)

            cleaned.append(text)

        return cleaned


if __name__ == "__main__":
    sample = ["Experienced in C++, C#, Node.js, ASP.NET and CI/CD pipelines"]

    cleaner = ResumeTextCleaner()
    result = cleaner.fit_transform(sample)

    print("Input: ", sample[0])
    print("Output:", result[0])

    expected_tokens = ["cplusplus", "csharp", "nodejs", "aspdotnet", "cicd"]
    missing = [t for t in expected_tokens if t not in result[0]]
    if missing:
        print(f"FAIL: missing tokens {missing}")
    else:
        print(f"PASS: all expected tokens present {expected_tokens}")

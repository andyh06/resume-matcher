"""Modal deployment for the resume-matcher FastAPI app.

`modal deploy modal_serve.py` builds an image containing exactly three
project files (serve.py, pipeline_def.py, pipeline.joblib) baked in via
add_local_file(..., copy=True), and serves serve.py's FastAPI app.
"""

import re
from pathlib import Path

import modal

REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"

# Packages needed to unpickle pipeline.joblib and run serve.py: scikit-learn,
# numpy, and scipy for the fitted pipeline and its sparse TF-IDF matrix,
# pandas for the stored documents DataFrame, joblib to load the bundle, and
# fastapi/pydantic (plus their direct runtime deps) to serve the API.
RUNTIME_PACKAGES = [
    "scikit-learn",
    "numpy",
    "scipy",
    "pandas",
    "joblib",
    "fastapi",
    "pydantic",
    "pydantic_core",
    "starlette",
    "typing_extensions",
    "typing-inspection",
    "annotated-types",
    "annotated-doc",
    "anyio",
    "idna",
    "threadpoolctl",
    "python-dateutil",
    "six",
    "tzdata",
]


def _read_pinned_versions(path: Path) -> dict:
    """Parse requirements.txt into {package_name: exact_version}, refusing
    anything that isn't an exact `==` pin."""
    pinned = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)$", line)
        if not match:
            raise ValueError(f"requirements.txt line is not an exact pin: {raw_line!r}")
        name, version = match.groups()
        pinned[name.lower()] = version
    return pinned


def _build_pip_specs(package_names, pinned_versions) -> list:
    specs = []
    for name in package_names:
        version = pinned_versions.get(name.lower())
        if version is None:
            raise KeyError(f"{name!r} not found in requirements.txt")
        specs.append(f"{name}=={version}")
    return specs


PINNED_VERSIONS = _read_pinned_versions(REQUIREMENTS_PATH)
PIP_SPECS = _build_pip_specs(RUNTIME_PACKAGES, PINNED_VERSIONS)

# Local dev used Python 3.13.1 -- match the minor version in the container so
# the pickled objects in pipeline.joblib unpickle against the same Python ABI.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(*PIP_SPECS)
    .add_local_file("serve.py", "/root/serve.py", copy=True)
    .add_local_file("pipeline_def.py", "/root/pipeline_def.py", copy=True)
    .add_local_file("pipeline.joblib", "/root/pipeline.joblib", copy=True)
    # modal_serve.py itself re-executes inside the container to rehydrate the
    # App/function definitions, so requirements.txt must exist at the same
    # relative path (__file__'s directory) there too, or the read above fails
    # with FileNotFoundError at container startup, not at deploy time.
    .add_local_file("requirements.txt", "/root/requirements.txt", copy=True)
)

app = modal.App("resume-matcher", image=image)


@app.function()
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")
    from serve import app as web_app

    return web_app

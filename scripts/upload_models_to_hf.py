import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "Rinil-Parmar"
MODEL_REPO_NAME = "tpch-query-router-models"

repo_id = f"{HF_USERNAME}/{MODEL_REPO_NAME}"

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

files = [
    (os.path.join(MODELS_DIR, "model_sqlite.joblib"), "model_sqlite.joblib"),
    (os.path.join(MODELS_DIR, "model_duckdb.joblib"), "model_duckdb.joblib"),
    (os.path.join(MODELS_DIR, "model_metadata.json"), "model_metadata.json"),
]

# Validate files
for local_path, _ in files:
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Missing file: {local_path}")

create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

api = HfApi()

for local_path, path_in_repo in files:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Model v2: retrained with larger dataset"
    )
    print(f"✅ Uploaded: {path_in_repo}")

print(f"\n✅ Done: https://huggingface.co/{repo_id}")
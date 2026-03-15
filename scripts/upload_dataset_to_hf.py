from huggingface_hub import HfApi

HF_USERNAME = "Rinil-Parmar"
DATASET_NAME = "tpch-query-routing-dataset"

api = HfApi()

for file in ["master_dataset.csv", "train_dataset.csv", "val_dataset.csv", "test_dataset.csv"]:
    api.upload_file(
        path_or_fileobj=f"data/{file}",
        path_in_repo=file,
        repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
        repo_type="dataset",
        commit_message="Dataset v2: larger query variants and improved features"
    )
    print(f"✅ Uploaded: {file}")

print("✅ Dataset updated successfully")
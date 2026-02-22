from huggingface_hub import HfApi, create_repo

HF_USERNAME = "Rinil-Parmar"  # ← Your HF username
DATASET_NAME = "tpch-query-routing-dataset"

# Create dataset repo
create_repo(
    repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
    repo_type="dataset",
    private=False,
    exist_ok=True
)

# Upload all CSVs
api = HfApi()
for file in ["master_dataset.csv", "train_dataset.csv", "val_dataset.csv", "test_dataset.csv"]:
    api.upload_file(
        path_or_fileobj=f"../data/{file}",
        path_in_repo=file,
        repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
        repo_type="dataset"
    )
    print(f"   ✅ Uploaded: {file}")

print(f"\n✅ Done: https://huggingface.co/datasets/{HF_USERNAME}/{DATASET_NAME}")
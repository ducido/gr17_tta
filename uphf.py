from huggingface_hub import HfApi

repo_id = "ducido/gr00tn1.7_sen_1gpus_20000steps_100bs_merged_libero_20k"
local_folder = "outputs/20260708_233643_sen_merged_libero_1gpus_20000steps_100bs_5000ss/checkpoint-20000"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Initial commit",
    # optional filters
    ignore_patterns=["**/__pycache__/**", "**/*.tmp", "**/.ipynb_checkpoints/**"],
    # allow_patterns=["**/*.pt","**/*.json"]  # alternatively, whitelist
)
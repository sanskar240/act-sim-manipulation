from huggingface_hub import HfApi, login
import os

# --- CONFIG ---
FILE_TO_UPLOAD = "episodes-v2.zip"
REPO_ID = "sanskxr02/act_sim_cube_sort"
# --------------

if not os.path.exists(FILE_TO_UPLOAD):
    print(f"❌ Error: Cannot find {FILE_TO_UPLOAD} in this folder.")
    exit()

print(f"🚀 pushing {FILE_TO_UPLOAD} to {REPO_ID}...")

# 1. Login
token = input("Paste HF Write Token: ").strip()
login(token=token)

api = HfApi()

# 2. Create Repo
api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

# 3. Upload File
api.upload_file(
    path_or_fileobj=FILE_TO_UPLOAD,
    path_in_repo=FILE_TO_UPLOAD, # Keeps name same in cloud
    repo_id=REPO_ID,
    repo_type="dataset"
)

print(f"✅ Done. File is at: https://huggingface.co/datasets/{REPO_ID}/blob/main/{FILE_TO_UPLOAD}")


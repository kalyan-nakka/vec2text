
import argparse
from pathlib import Path
import sys
from huggingface_hub import HfApi, create_repo, upload_folder

# local_folder = Path("path/to/local/folder")
# repo_id = "your-username/your-repo-name"
# repo_type = "model" # or "dataset"
# commit_message = "Uploading local folder"
# ignore_patterns = [".git*", "__pycache__", "*.pyc", ".DS_Store"]
# private = False (For private or public repo)
# exist_ok = True (Don't fail if it already exists)

# This code can be used for any type of upload to hugging face repo


# To upload to a hugging face repo that already exists(model)
def upload_to_huggingface(local_folder:str, repo_id:str,path_in_repo:str ="",  repo_type:str = "model",commit_message:str = "Uploading local folder", ignore_patterns: list[str] | None = None):
    local_path = Path(local_folder) 
    assert local_path.is_dir(), f"Local path {local_path} is not a directory" 
    local_path = str(local_path)
    api = HfApi()
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    url = upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns # or [".git*", "__pycache__", "*.pyc", ".DS_Store"]
    )
    print(f"Upload URL: {url}")
    return url

def create_huggingface_repo(local_folder:str, repo_id:str,path_in_repo:str ="",  repo_type:str = "model",commit_message:str = "Uploading local folder", ignore_patterns: list[str] | None = None, private: bool = False, exist_ok: bool = True):
    local_path = Path(local_folder).expanduser().resolve()
    assert local_path.is_dir(), f"Local path {local_path} is not a directory"
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=exist_ok
    )
    print(f"Repository created: {repo_id}, private: {private}")
    upload_to_huggingface(local_folder, repo_id, path_in_repo, repo_type, commit_message, ignore_patterns)
    return

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local folder to a Hugging Face repository",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--local_folder", type = str, help="Path to the local folder to upload")
    parser.add_argument("--repo_id", type = str, help="Target repo in 'username/repo-name' form (e.g. AusmitM/TestUpload).")
    parser.add_argument("--path_in_repo", type = str, help="Path in the repository where the folder will be uploaded.", default="")
    parser.add_argument("--repo_type", choices=["model", "dataset", "space"], default="model", help="Type of repo.",)
    parser.add_argument("--commit_message", type = str, help="Commit message for the upload.", default="Uploading local folder")
    parser.add_argument("--ignore_patterns", help="Patterns to ignore during upload.", default = None)
    parser.add_argument("--private", type = bool, help="Make the repository private.", default=False)
    parser.add_argument("--no_exist_ok", type = bool, help="Do not exit if the repository already exists.", default=False)
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    create_huggingface_repo(
        local_folder=args.local_folder,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
        ignore_patterns=args.ignore_patterns,
        private=args.private,
        exist_ok=not args.no_exist_ok,
    )
   
if __name__ == "__main__":
    sys.exit(main())

# CLI Template: python3 huggingFaceUpload.py --local_folder ./YourFolder --repo_id AusmitM/NewRepo --commit_message "Your commit message here"
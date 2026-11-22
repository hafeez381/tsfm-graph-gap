"""Off-VM copies: upload to a private Hugging Face repo and verify the sha256 on the Hub."""
import hashlib
import os


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def push(path: str, repo: str, path_in_repo: str | None = None) -> str:
    """Upload `path` to the private model repo `repo` and raise unless the Hub copy has the same sha256.

    Never delete the local file before this returns.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api, name, local = HfApi(), path_in_repo or os.path.basename(path), sha256(path)
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo)
    info = api.get_paths_info(repo, [name])[0]
    # LFS files carry their sha256; small files are stored in git, so download and hash them.
    remote = info.lfs.sha256 if info.lfs else sha256(hf_hub_download(repo, name, force_download=True))
    if remote != local:
        raise RuntimeError(f"sha256 mismatch for {repo}/{name}: local {local}, hub {remote}")
    return local

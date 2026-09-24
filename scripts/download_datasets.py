"""Fetch public lung datasets. Requires network access (HF / Kaggle / MSD).

Kaggle needs ~/.kaggle/kaggle.json (pip install kaggle).
Hugging Face needs `pip install huggingface_hub` (+ `huggingface-cli login` for gated repos).

  python scripts/download_datasets.py msd          # CT tumor, 63 labelled cases (CC-BY-SA 4.0)
  python scripts/download_datasets.py kaggle-ct    # CT slices + lung masks
  python scripts/download_datasets.py kaggle-cxr   # Montgomery + Shenzhen CXR lung masks
  python scripts/download_datasets.py hf --repo <user/dataset>   # any HF dataset mirror
"""
import argparse
import os
import subprocess
import tarfile
import urllib.request

KAGGLE = {
    "kaggle-ct": "kmader/finding-lungs-in-ct-data",
    "kaggle-cxr": "nikhilpandey360/chest-xray-masks-and-labels",
}
# Medical Segmentation Decathlon, Task06_Lung (Simpson et al., 2019)
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["msd", "hf", *KAGGLE])
    ap.add_argument("--out", default="data")
    ap.add_argument("--repo", help="Hugging Face dataset repo id")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.which == "msd":
        tar = os.path.join(a.out, "Task06_Lung.tar")
        if not os.path.exists(tar):
            urllib.request.urlretrieve(MSD_URL, tar)
        tarfile.open(tar).extractall(a.out)
    elif a.which == "hf":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=a.repo, repo_type="dataset", local_dir=os.path.join(a.out, a.repo.split("/")[-1]))
    else:
        subprocess.check_call(["kaggle", "datasets", "download", "-d", KAGGLE[a.which], "-p",
                               os.path.join(a.out, a.which), "--unzip"])


if __name__ == "__main__":
    main()

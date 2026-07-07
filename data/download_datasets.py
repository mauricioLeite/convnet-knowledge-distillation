#!/usr/bin/env python3
"""Download and prepare image-classification datasets for the KD project.

Run from the project root or from this data directory:
    python data/download_datasets.py

All datasets are placed under this shared data root:
    data/
    ├── tiny-imagenet-200/
    ├── flowers-102/
    └── oxford-iiit-pet/
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import warnings
import zipfile
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_file(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"[skip] {destination.name} already exists")
        return

    print(f"[download] {url}")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024

        with destination.open("wb") as output:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {destination.name}: {pct:5.1f}%", end="")

    print()


def extract_zip(archive_path: Path, output_dir: Path, force: bool = False) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        print(f"[skip] {output_dir.name} already extracted")
        return

    print(f"[extract] {archive_path.name}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir.parent)


def _remove_flowers102_partial_files(root: Path) -> None:
    """Remove partial TorchVision downloads that fail MD5 checks."""
    candidates = []
    for base in [root, root / "flowers-102"]:
        candidates.extend([
            base / "102flowers.tgz",
            base / "imagelabels.mat",
            base / "setid.mat",
        ])

    for path in candidates:
        if path.exists():
            print(f"[clean] removing possibly corrupt {path}")
            path.unlink()


def _flowers102_is_ready(root: Path) -> bool:
    from torchvision import datasets

    try:
        for split in ("train", "val", "test"):
            datasets.Flowers102(root=root, split=split, download=False)
        return True
    except Exception:
        return False


def download_flowers102(force: bool = False, retries: int = 3) -> None:
    print("\n== Flowers-102 ==")
    from torchvision import datasets

    root = DATA_ROOT / "flowers-102"

    if force and root.exists():
        print(f"[clean] removing existing {root}")
        shutil.rmtree(root)

    if _flowers102_is_ready(root):
        print(f"[ok] Flowers-102 already ready at {root}")
        return

    for attempt in range(1, retries + 1):
        try:
            print(f"[try] Flowers-102 download attempt {attempt}/{retries}")
            for split in ("train", "val", "test"):
                datasets.Flowers102(root=root, split=split, download=True)
            print(f"[ok] Flowers-102 ready at {root}")
            return
        except RuntimeError as exc:
            message = str(exc)
            if "File not found or corrupted" not in message or attempt == retries:
                raise
            print(f"[retry] Flowers-102 download failed MD5/integrity check: {message}")
            _remove_flowers102_partial_files(root)

def download_oxford_pets() -> None:
    print("\n== Oxford-IIIT Pet ==")
    from torchvision import datasets

    root = DATA_ROOT
    datasets.OxfordIIITPet(root=root, split="trainval", download=True)
    datasets.OxfordIIITPet(root=root, split="test", download=True)
    print(f"[ok] Oxford-IIIT Pet ready at {root / 'oxford-iiit-pet'}")


def download_tiny_imagenet(force: bool = False) -> None:
    print("\n== Tiny ImageNet ==")
    archive_path = DATA_ROOT / "tiny-imagenet-200.zip"
    output_dir = DATA_ROOT / "tiny-imagenet-200"

    download_file(TINY_IMAGENET_URL, archive_path, force=force)
    extract_zip(archive_path, output_dir, force=force)
    print(f"[ok] Tiny ImageNet ready at {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["tiny_imagenet", "flowers102", "oxford_pets"],
        choices=["tiny_imagenet", "flowers102", "oxford_pets"],
        help="Datasets to download. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives and recreate extracted/organized folders.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"dtype\(\): align should be passed.*",
    )

    args = parse_args()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"DATA_ROOT = {DATA_ROOT}")

    actions = {
        "tiny_imagenet": lambda: download_tiny_imagenet(force=args.force),
        "flowers102": lambda: download_flowers102(force=args.force),
        "oxford_pets": download_oxford_pets,
    }

    failures = []
    for dataset_name in args.datasets:
        try:
            actions[dataset_name]()
        except Exception as exc:
            failures.append((dataset_name, exc))
            print(f"[error] {dataset_name} failed: {exc}")

    if failures:
        print("\nSome datasets failed:")
        for dataset_name, exc in failures:
            print(f"  - {dataset_name}: {exc}")
        print("\nYou can rerun only the failed dataset, for example:")
        print("  python data/download_datasets.py --datasets flowers102")
        raise SystemExit(1)

    print("\nDone. You can now run notebooks/01_data_exploration.ipynb.")


if __name__ == "__main__":
    main()

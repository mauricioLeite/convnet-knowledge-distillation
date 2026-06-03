#!/usr/bin/env python3
"""Download and prepare image-classification datasets for the KD project.

Run from the project root or from this data directory:

    python data/download_datasets.py

All datasets are placed under this shared data root:

    data/
    ├── cifar-100/
    ├── tiny-imagenet-200/
    ├── flowers-102/
    └── stanford-cars/

Notes
-----
TorchVision can download CIFAR-100 and Flowers-102 directly.
Tiny ImageNet is downloaded from the CS231n archive.
Stanford Cars is downloaded from the live Academic Torrents/FastAI web seed
and then converted to an ImageFolder-compatible layout:

    stanford-cars/train/<class_name>/*.jpg
    stanford-cars/test/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import warnings
import zipfile
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

STANFORD_CARS_ARCHIVE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz"


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


def extract_tgz(archive_path: Path, output_dir: Path, force: bool = False) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        print(f"[skip] {output_dir.name} already extracted")
        return

    print(f"[extract] {archive_path.name}")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(output_dir.parent)


def extract_tgz_into(archive_path: Path, output_dir: Path, force: bool = False) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        print(f"[skip] {output_dir.name} already extracted")
        return

    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] {archive_path.name}")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(output_dir)


def download_cifar100() -> None:
    print("\n== CIFAR-100 ==")
    from torchvision import datasets

    root = DATA_ROOT / "cifar-100"
    datasets.CIFAR100(root=root, train=True, download=True)
    datasets.CIFAR100(root=root, train=False, download=True)
    print(f"[ok] CIFAR-100 ready at {root}")


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


def _matlab_scalar(value):
    while hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        if len(value) == 0:
            return None
        value = value[0]
    return value


def _matlab_string(value) -> str:
    value = _matlab_scalar(value)
    return str(value)


def _safe_class_name(name: str) -> str:
    return (
        name.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
    )


def _loadmat(path: Path):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError(
            "Preparing Stanford Cars requires scipy. Install it with `pip install scipy`."
        ) from exc
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def _class_names_from_mat(data) -> list[str]:
    return [_safe_class_name(str(name)) for name in data["class_names"].tolist()]


def _load_stanford_cars_metadata(devkit_dir: Path, test_annos_path: Path):
    meta = _loadmat(devkit_dir / "cars_meta.mat")
    train_mat = _loadmat(devkit_dir / "cars_train_annos.mat")
    test_mat = _loadmat(test_annos_path)
    return _class_names_from_mat(meta), train_mat["annotations"], test_mat["annotations"]


def _load_stanford_cars_combined_metadata(annotations_path: Path):
    data = _loadmat(annotations_path)
    train_annos = []
    test_annos = []
    for annotation in data["annotations"]:
        is_test = bool(int(_matlab_scalar(getattr(annotation, "test", 0))))
        if is_test:
            test_annos.append(annotation)
        else:
            train_annos.append(annotation)
    return _class_names_from_mat(data), train_annos, test_annos


def _annotation_items(annotations):
    if getattr(annotations, "shape", None) == ():
        annotations = [annotations.item()]
    for annotation in annotations:
        filename_value = getattr(
            annotation,
            "fname",
            getattr(annotation, "relative_im_path", None),
        )
        class_value = getattr(annotation, "class_", getattr(annotation, "class"))
        filename = _matlab_string(filename_value)
        class_index = int(_matlab_scalar(class_value)) - 1
        yield filename, class_index


def _copy_stanford_split(
    annotations,
    image_dir: Path,
    output_dir: Path,
    class_names: list[str],
    force: bool = False,
) -> int:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for split_index, (filename, class_index) in enumerate(_annotation_items(annotations), start=1):
        class_name = class_names[class_index]
        image_name = Path(filename).name
        src = image_dir / filename
        if not src.exists():
            src = image_dir / image_name
        if not src.exists() and image_name.endswith(".jpg"):
            image_name = f"{int(Path(image_name).stem):05d}.jpg"
            src = image_dir / image_name
        if not src.exists() and image_name.endswith(".jpg"):
            image_name = f"{split_index:05d}.jpg"
            src = image_dir / image_name
        dst = output_dir / class_name / image_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1
    return copied


def _find_stanford_cars_path(root: Path, name: str, is_dir: bool) -> Path:
    path = _find_optional_stanford_cars_path(root, name, is_dir)
    if path is None:
        kind = "directory" if is_dir else "file"
        raise FileNotFoundError(f"Could not find Stanford Cars {kind} {name!r} under {root}")
    return path


def _find_optional_stanford_cars_path(root: Path, name: str, is_dir: bool) -> Path | None:
    matches = [path for path in root.rglob(name) if path.is_dir() == is_dir]
    return matches[0] if matches else None


def download_stanford_cars(force: bool = False) -> None:
    print("\n== Stanford Cars ==")
    root = DATA_ROOT / "stanford-cars"
    raw_dir = root / "raw"
    extracted_dir = raw_dir / "archive"
    archive_path = raw_dir / "stanford-cars.tgz"
    root.mkdir(parents=True, exist_ok=True)

    download_file(STANFORD_CARS_ARCHIVE_URL, archive_path, force=force)
    extract_tgz_into(archive_path, extracted_dir, force=force)

    cars_train_dir = _find_stanford_cars_path(extracted_dir, "cars_train", is_dir=True)
    cars_test_dir = _find_optional_stanford_cars_path(extracted_dir, "cars_test", is_dir=True)
    if cars_test_dir is None:
        cars_test_dir = cars_train_dir
    devkit_dir = _find_optional_stanford_cars_path(extracted_dir, "devkit", is_dir=True)
    test_annos_path = _find_optional_stanford_cars_path(
        extracted_dir,
        "cars_test_annos_withlabels.mat",
        is_dir=False,
    )

    if devkit_dir is not None and test_annos_path is not None:
        class_names, train_annos, test_annos = _load_stanford_cars_metadata(
            devkit_dir,
            test_annos_path,
        )
    else:
        class_names, train_annos, test_annos = _load_stanford_cars_combined_metadata(
            _find_stanford_cars_path(extracted_dir, "cars_annos.mat", is_dir=False)
        )

    train_count = _copy_stanford_split(
        train_annos,
        cars_train_dir,
        root / "train",
        class_names,
        force=force,
    )
    test_count = _copy_stanford_split(
        test_annos,
        cars_test_dir,
        root / "test",
        class_names,
        force=force,
    )

    print(f"[ok] Stanford Cars ready at {root}")
    print(f"     train images: {train_count}")
    print(f"     test images:  {test_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar100", "tiny_imagenet", "flowers102", "stanford_cars", "oxford_pets"],
        choices=["cifar100", "tiny_imagenet", "flowers102", "stanford_cars", "oxford_pets"],
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
        "cifar100": download_cifar100,
        "tiny_imagenet": lambda: download_tiny_imagenet(force=args.force),
        "flowers102": lambda: download_flowers102(force=args.force),
        "stanford_cars": lambda: download_stanford_cars(force=args.force),
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

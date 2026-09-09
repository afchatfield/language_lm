"""Shared download helpers.

Several Phase 0 artefacts are large (the German n-gram archive is 1.7GB), so
downloads stream to disk, resume from a partial file, and skip work that is
already done.
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

CHUNK_SIZE = 1 << 20  # 1 MiB


def download(url: str, destination: Path, force: bool = False) -> Path:
    """Stream a URL to `destination`, resuming an interrupted download.

    Args:
        url: Source URL.
        destination: Final file path. A sibling ``.part`` file holds progress.
        force: Re-download even if the destination already exists.

    Returns:
        The path written.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        print(f"  already downloaded: {destination} ({_human(destination.stat().st_size)})")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    resume_from = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if resume_from and response.status_code == 200:
            # Server ignored the Range header; start over rather than corrupt.
            resume_from = 0
        elif resume_from and response.status_code == 416:
            # Already have the whole thing.
            partial.rename(destination)
            return destination
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"
        written = resume_from
        with partial.open(mode) as fh:
            for chunk in response.iter_content(CHUNK_SIZE):
                fh.write(chunk)
                written += len(chunk)
                _progress(destination.name, written, total)
    print()

    partial.rename(destination)
    return destination


def extract(archive: Path, destination: Path, strip_if_single_root: bool = False) -> Path:
    """Extract a ``.zip``, ``.tar.gz`` or ``.tgz`` archive.

    Args:
        archive: Archive to extract.
        destination: Directory to extract into (created if absent).
        strip_if_single_root: If the archive contains exactly one top-level
            directory, move its contents up so callers get a predictable layout.

    Returns:
        The destination directory.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {destination}")

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            # `filter="data"` refuses absolute paths, symlink escapes, and
            # device nodes: cheap insurance on third-party archives.
            tf.extractall(destination, filter="data")
    else:
        raise ValueError(f"Unsupported archive type: {archive.name}")

    if strip_if_single_root:
        entries = [p for p in destination.iterdir() if p.name != ".gitkeep"]
        if len(entries) == 1 and entries[0].is_dir():
            _flatten(entries[0], destination)

    return destination


def _flatten(inner: Path, destination: Path) -> None:
    for item in list(inner.iterdir()):
        target = destination / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))
    inner.rmdir()


def _progress(name: str, written: int, total: int) -> None:
    if total:
        pct = 100 * written / total
        sys.stdout.write(f"\r  {name}: {_human(written)} / {_human(total)} ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {name}: {_human(written)}")
    sys.stdout.flush()


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}GB"  # pragma: no cover

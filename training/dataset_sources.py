"""Dataset source resolution for local paths and Hugging Face file URLs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from training.backends.registry import TrainingCliError
from training.distributed import is_main_process, is_torchrun_child, wait_for_files


DEFAULT_HF_DATASET_CACHE_DIR = Path("cache/huggingface_datasets")
HF_DATASET_CACHE_CONFIG_KEYS = {"hf_dataset_cache_dir", "huggingface_dataset_cache_dir"}
HF_HOSTS = {"huggingface.co", "www.huggingface.co"}


class DatasetSourceError(TrainingCliError):
    """Raised when a configured dataset source cannot be resolved."""


@dataclass(frozen=True)
class HuggingFaceDatasetFile:
    original_url: str
    download_url: str
    repo_id: str
    revision: str
    file_path: str
    cache_dir: Path
    output_path: Path
    metadata_path: Path
    cache_hit: bool


@dataclass(frozen=True)
class _HuggingFaceUrlParts:
    original_url: str
    download_url: str
    repo_id: str
    revision: str
    file_path: str


def is_huggingface_dataset_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        _parse_huggingface_dataset_url(value)
    except DatasetSourceError:
        return False
    return True


def _looks_like_huggingface_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in HF_HOSTS


def _slug(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-_.")
    slug = slug or "dataset"
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{slug[: max_length - 11].rstrip('-_.')}-{digest}"


def _quote_path(value: str) -> str:
    return "/".join(quote(part, safe="") for part in value.split("/"))


def _parse_huggingface_dataset_url(url: str) -> _HuggingFaceUrlParts:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in HF_HOSTS:
        raise DatasetSourceError(
            "Hugging Face dataset file URLs must use https://huggingface.co/datasets/..."
        )

    segments = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if len(segments) < 5 or segments[0] != "datasets":
        raise DatasetSourceError(
            "Hugging Face dataset file URLs must look like "
            "https://huggingface.co/datasets/<repo>/blob/<revision>/<file>."
        )

    marker_index = None
    marker = None
    for candidate in ("blob", "resolve"):
        if candidate in segments[1:]:
            marker_index = segments.index(candidate)
            marker = candidate
            break
    if marker_index is None or marker is None or marker_index < 2:
        raise DatasetSourceError(
            "Hugging Face dataset file URLs must contain '/blob/<revision>/' or '/resolve/<revision>/'."
        )
    if marker_index + 2 >= len(segments):
        raise DatasetSourceError("Hugging Face dataset file URL must include a revision and file path.")

    repo_id = "/".join(segments[1:marker_index])
    revision = segments[marker_index + 1]
    file_path = "/".join(segments[marker_index + 2 :])
    if not repo_id or not revision or not file_path:
        raise DatasetSourceError("Hugging Face dataset file URL must include repo, revision, and file path.")

    if marker == "resolve":
        download_url = url
    else:
        download_url = (
            "https://huggingface.co/datasets/"
            f"{_quote_path(repo_id)}/resolve/{quote(revision, safe='')}/{_quote_path(file_path)}"
        )
        if parsed.query:
            download_url = f"{download_url}?{parsed.query}"

    return _HuggingFaceUrlParts(
        original_url=url,
        download_url=download_url,
        repo_id=repo_id,
        revision=revision,
        file_path=file_path,
    )


def _configured_cache_dir(*settings: dict[str, Any] | None) -> Path:
    for section in settings:
        if not isinstance(section, dict):
            continue
        for key in HF_DATASET_CACHE_CONFIG_KEYS:
            value = section.get(key)
            if value:
                return Path(str(value))
    return DEFAULT_HF_DATASET_CACHE_DIR


def _cache_paths(parts: _HuggingFaceUrlParts, cache_dir: str | Path | None) -> tuple[Path, Path, Path]:
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_HF_DATASET_CACHE_DIR
    key_payload = {
        "repo_id": parts.repo_id,
        "revision": parts.revision,
        "file_path": parts.file_path,
        "download_url": parts.download_url,
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    label = _slug(f"{parts.repo_id}-{parts.revision}-{Path(parts.file_path).stem}", max_length=80)
    output_dir = root / f"{label}-{key}"
    return output_dir, output_dir / "dataset.jsonl", output_dir / "source.json"


def _download_url_to_path(url: str, output_path: Path) -> None:
    headers = {"User-Agent": "embedder-training-v2/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        with urlopen(request) as response, tmp_path.open("wb") as out_fh:
            shutil.copyfileobj(response, out_fh)
        tmp_path.replace(output_path)
    except (HTTPError, URLError, OSError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise DatasetSourceError(f"Could not download Hugging Face dataset file '{url}': {exc}") from exc


def materialize_huggingface_dataset_file(
    url: str,
    *,
    cache_dir: str | Path | None = None,
) -> HuggingFaceDatasetFile:
    parts = _parse_huggingface_dataset_url(url)
    output_dir, output_path, metadata_path = _cache_paths(parts, cache_dir)

    if output_path.exists():
        print(f"[INFO] Hugging Face dataset cache hit: {url} -> {output_path}", flush=True)
        return HuggingFaceDatasetFile(
            original_url=parts.original_url,
            download_url=parts.download_url,
            repo_id=parts.repo_id,
            revision=parts.revision,
            file_path=parts.file_path,
            cache_dir=output_dir,
            output_path=output_path,
            metadata_path=metadata_path,
            cache_hit=True,
        )

    if is_torchrun_child() and not is_main_process():
        wait_for_files([output_path])
        print(f"[INFO] Hugging Face dataset cache hit: {url} -> {output_path}", flush=True)
        return HuggingFaceDatasetFile(
            original_url=parts.original_url,
            download_url=parts.download_url,
            repo_id=parts.repo_id,
            revision=parts.revision,
            file_path=parts.file_path,
            cache_dir=output_dir,
            output_path=output_path,
            metadata_path=metadata_path,
            cache_hit=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading Hugging Face dataset file: {url} -> {output_path}", flush=True)
    _download_url_to_path(parts.download_url, output_path)
    metadata = {
        "original_url": parts.original_url,
        "download_url": parts.download_url,
        "repo_id": parts.repo_id,
        "revision": parts.revision,
        "file_path": parts.file_path,
        "output_path": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return HuggingFaceDatasetFile(
        original_url=parts.original_url,
        download_url=parts.download_url,
        repo_id=parts.repo_id,
        revision=parts.revision,
        file_path=parts.file_path,
        cache_dir=output_dir,
        output_path=output_path,
        metadata_path=metadata_path,
        cache_hit=False,
    )


def resolve_train_data_entry(
    train_data: str | Path,
    *settings: dict[str, Any] | None,
) -> str | Path:
    value = str(train_data)
    if not _looks_like_huggingface_url(value):
        return train_data
    cache_dir = _configured_cache_dir(*settings)
    return materialize_huggingface_dataset_file(value, cache_dir=cache_dir).cache_dir

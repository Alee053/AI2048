"""Collect immutable benchmark provenance without changing evaluation behavior."""
from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_CANONICALIZATION = "d4-min-packed"
SEARCH_BATCH_SIZE = 32768
SEARCH_TRANSPOSITION_TABLE = True
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: str | Path | None) -> str:
    """Return a SHA-256 digest, or an empty string when an artifact is absent."""
    if path is None:
        return ""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def validate_artifact_sha256(
    path: str | Path | None,
    expected: str | None = None,
    *,
    label: str = "artifact",
) -> str:
    """Require a non-empty file and optionally verify its SHA-256 digest."""
    if path is None:
        raise ValueError(f"{label} path is missing")
    artifact = Path(path)
    if not artifact.is_file():
        raise ValueError(f"{label} artifact is missing: {artifact}")
    try:
        if artifact.stat().st_size == 0:
            raise ValueError(f"{label} artifact is empty: {artifact}")
    except OSError as exc:
        raise ValueError(f"{label} artifact is unreadable: {artifact}") from exc

    digest = sha256_file(artifact)
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} artifact could not be hashed: {artifact}")
    if expected is not None:
        if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
            raise ValueError(f"{label} hash is invalid")
        if digest != expected:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected}, got {digest}"
            )
    return digest


def collect_runtime_provenance(*, model_path: str, effective_config: str | None) -> dict:
    """Return file digests and execution-environment identifiers for one run."""
    model_artifact = Path(model_path).resolve()
    effective_config_path = (
        Path(effective_config).resolve() if effective_config else None
    )
    extension_path = _native_extension_path()
    gpu_name = ""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return {
        "model_sha256": sha256_file(model_artifact),
        "effective_config_path": str(effective_config_path or ""),
        "effective_config_sha256": sha256_file(effective_config_path),
        "uv_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "native_extension_sha256": sha256_file(extension_path),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "sb3_version": _package_version("stable-baselines3"),
        "cuda_runtime": torch.version.cuda or "",
        "gpu_name": gpu_name,
        "compiler": _compiler_identity(),
    }


def _native_extension_path() -> str | None:
    try:
        from twenty_forty_eight_ai.utils import searcher
        return getattr(searcher._impl, "__file__", None)
    except Exception:
        return None


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return ""


def _compiler_identity() -> str:
    cache_path = REPO_ROOT / "cpp_src" / "build" / "CMakeCache.txt"
    try:
        match = re.search(
            r"^CMAKE_CXX_COMPILER:FILEPATH=(.+)$", cache_path.read_text(), re.MULTILINE
        )
        if match:
            compiler = match.group(1)
            output = subprocess.check_output(
                [compiler, "--version"], stderr=subprocess.DEVNULL, text=True,
            )
            return output.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.python_compiler()

"""Build the benchmark reproduction archive and optional internal trace archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def source_files():
    files = []
    for name in (
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "MANIFEST.in",
        "requirements-reproducibility.txt",
        "tools/check_distribution.py",
        "tools/export_public_repository.py",
        "CHANGELOG.md",
        "docs/ci/tests.yml",
        ".gitignore",
        ".gitattributes",
    ):
        if (ROOT / name).is_file():
            files.append(ROOT / name)
    for directory in ("src/qgrn", "src/qgrn_benchmarks", "reproduction"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(
                x in ("__pycache__", "outputs", ".pytest_cache", "validation") for x in rel.parts
            ):
                continue
            if path.suffix in (".pyc", ".gz"):
                continue
            if path.name == "MANIFEST.json":
                continue
            if rel.parts[:3] == ("reproduction", "data", "raw"):
                continue
            files.append(path)
    for name in (
        "installation.md",
        "algorithm.md",
        "api.md",
        "development.md",
        "figures.md",
        "references.md",
        "reproduction.md",
    ):
        path = ROOT / "docs" / name
        if path.is_file():
            files.append(path)
    for path in (ROOT / "examples").glob("qgrn*.py"):
        files.append(path)
    for name in ("test_qgrn_solver.py", "test_qgrn_benchmarks.py"):
        path = ROOT / "tests" / name
        if path.exists():
            files.append(path)
    files.append(Path(__file__).resolve())
    return sorted(set(files))


def build(output, include_raw=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = source_files()
    manifest = dict(
        schema="qgrn_distribution_manifest_v1",
        files={p.relative_to(ROOT).as_posix(): digest(p) for p in files},
    )
    manifest_path = ROOT / "reproduction/MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    target = output / "qgrn-reproduction.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [manifest_path]:
            archive.write(path, path.relative_to(ROOT).as_posix())
    result = dict(
        standard=dict(
            path=target.name,
            bytes=target.stat().st_size,
            sha256=digest(target),
            file_count=len(files) + 1,
        )
    )
    if include_raw:
        raw_files = sorted((ROOT / "reproduction/data/raw").glob("*.json.gz"))
        if len(raw_files) != 200:
            raise ValueError("Expected exactly 200 selected raw records")
        target = output / "qgrn-selected-raw.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in raw_files + [ROOT / "reproduction/data/raw/index.json"]:
                archive.write(path, path.relative_to(ROOT).as_posix())
        result["raw"] = dict(
            path=target.name,
            bytes=target.stat().st_size,
            sha256=digest(target),
            file_count=len(raw_files) + 1,
        )
    (output / "reproduction_archives.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.include_raw), indent=2))

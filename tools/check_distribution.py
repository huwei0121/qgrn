"""Inspect Q-GRN wheel/source contents without executing or extracting them."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def read_members(path: Path) -> dict[str, bytes]:
    """Return regular archive members, normalized to the distribution root."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {
                item.filename: archive.read(item)
                for item in archive.infolist()
                if not item.is_dir()
            }
    with tarfile.open(path, "r:gz") as archive:
        result = {}
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"unexpected archive link: {member.name}")
            if not member.isfile():
                continue
            name = PurePosixPath(member.name)
            normalized = str(name.relative_to(name.parts[0]))
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            result[normalized] = stream.read()
        return result


def check_distribution(path: Path) -> dict[str, object]:
    members = read_members(path)
    wheel = path.suffix == ".whl"
    prefix = "qgrn/" if wheel else "src/qgrn/"
    required = {prefix + "__init__.py", prefix + "py.typed"}
    if not wheel:
        required |= {
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "docs/api.md",
            "tests/test_qgrn_solver.py",
        }
    if not required <= members.keys():
        raise ValueError(f"missing required files: {sorted(required - members.keys())}")
    allowed_files = {
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "MANIFEST.in",
        "PKG-INFO",
        "setup.cfg",
        "requirements-reproducibility.txt",
        "docs/api.md",
        "docs/algorithm.md",
        "docs/development.md",
        "docs/installation.md",
        "docs/references.md",
        "examples/qgrn_gate_control.py",
        "tests/test_qgrn_solver.py",
        "tools/check_distribution.py",
    }
    code = {}
    for name, raw in members.items():
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in name:
            raise ValueError(f"unsafe member path: {name}")
        allowed = name.startswith(prefix)
        if wheel:
            allowed |= ".dist-info/" in name and pure.parts[0].startswith("qgrn-")
        else:
            allowed |= name in allowed_files or name.startswith("src/qgrn.egg-info/")
        if not allowed:
            raise ValueError(f"unexpected member: {name}")
        if name.endswith((".py", ".md", ".toml", ".txt", ".cfg")):
            content = raw.decode("utf8")
            if re.search(r"[A-Za-z]:[/\\](Users|Desktop|Program Files|xwechat_files)", content):
                raise ValueError(f"private machine path in {name}")
        if name.startswith(prefix) and name.endswith(".py"):
            content = raw.decode("utf8")
            tree = ast.parse(content, filename=name)
            forbidden = {"quantum_paper_benchmarks", "qgrn_benchmarks", "orbit_dfo"}
            for node in ast.walk(tree):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported = [node.module or ""]
                if any(module.split(".")[0] in forbidden for module in imported):
                    raise ValueError(f"historical/benchmark dependency in {name}")
            if re.search(r"(?:20\d{6}|tuned_unified|development_round)", name):
                raise ValueError(f"research-stage identifier in module path: {name}")
            code[name] = content
    return {
        "artifact": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "files": len(members),
        "python_modules": len(code),
        "status": "passed",
    }


def check_reproduction(path: Path) -> dict[str, object]:
    """Check the benchmark archive against its byte-level file manifest."""
    manifest_name = "reproduction/MANIFEST.json"
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("duplicate reproduction archive members")
        manifest = json.loads(archive.read(manifest_name))
        expected = set(manifest["files"]) | {manifest_name}
        if set(names) != expected:
            raise ValueError("archive members do not match the file manifest")
        required = {
            "src/qgrn/solver.py",
            "src/qgrn_benchmarks/GPL-3.0.txt",
            "src/qgrn_benchmarks/DYNAMO_LICENSE.txt",
            "reproduction/reproduce.py",
            "reproduction/configs/selected.json",
            "reproduction/THIRD_PARTY_NOTICES.md",
        }
        if not required <= expected:
            raise ValueError(f"missing reproduction files: {sorted(required - expected)}")
        for entry in entries:
            name = entry.filename
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise ValueError(f"unsafe reproduction path: {name}")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"unexpected reproduction link: {name}")
            if pure.parts[0] in {"reports", "results", ".git", ".runtime"}:
                raise ValueError(f"unrelated research file: {name}")
            if pure.parts[:3] in {
                ("reproduction", "data", "raw"),
            } or pure.parts[:2] == ("reproduction", "validation"):
                raise ValueError(f"internal trace or validation file: {name}")
            raw = archive.read(entry)
            if name != manifest_name and hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
                raise ValueError(f"file checksum differs: {name}")
            if pure.suffix in {".py", ".jl", ".md", ".toml", ".json", ".txt", ".yml"}:
                # Preserve upstream license bytes while scanning ASCII path patterns.
                content = raw.decode("utf-8-sig", errors="replace")
                if re.search(r"[A-Za-z]:[/\\](Users|Desktop|Program Files|xwechat_files)", content):
                    raise ValueError(f"private machine path in {name}")
        config = json.loads(archive.read("reproduction/configs/selected.json"))
        tasks = config["tasks"]
        run_count = sum(len(task["seeds"]) * len(task["methods"]) for task in tasks)
        if len(tasks) != 9 or run_count != 200:
            raise ValueError("unexpected benchmark problem or run count")
        for task in tasks:
            if len(task["seeds"]) != 10 or len(set(task["seeds"])) != 10:
                raise ValueError(f"invalid paired seeds for {task['id']}")
            for seed in task["seeds"]:
                if f"reproduction/data/inputs/{task['id']}__{seed}.json" not in expected:
                    raise ValueError(f"missing paired input: {task['id']}/{seed}")
                for method in task["methods"]:
                    name = f"reproduction/data/curves/{task['id']}__{method}__{seed}.json"
                    if name not in expected:
                        raise ValueError(f"missing reference curve: {name}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return {
        "artifact": path.name,
        "sha256": hasher.hexdigest(),
        "bytes": path.stat().st_size,
        "files": len(names),
        "problems": len(tasks),
        "runs": run_count,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reproduction", action="store_true", help="Also inspect the reproduction ZIP"
    )
    args = parser.parse_args()
    archives = sorted(args.directory.glob("qgrn-*.whl"))
    archives += sorted(args.directory.glob("qgrn-*.tar.gz"))
    if not archives:
        parser.error("no Q-GRN wheel or source distribution found")
    results = [check_distribution(path) for path in archives]
    if args.reproduction:
        results.append(check_reproduction(args.directory / "qgrn-reproduction.zip"))
    encoded = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf8")
    print(encoded)


if __name__ == "__main__":
    main()

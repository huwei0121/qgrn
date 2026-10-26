"""Export the solver and benchmark reproduction files into a new directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from build_reproduction_bundle import ROOT, digest, source_files


def export(destination: Path) -> dict:
    """Copy the distribution file set without research history or internal traces."""
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("The export directory must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    files = source_files()
    for source in files:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "schema": "qgrn_distribution_manifest_v1",
        "files": {path.relative_to(ROOT).as_posix(): digest(path) for path in files},
    }
    (destination / "reproduction/MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"directory": str(destination), "files": len(files) + 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.destination), indent=2))


if __name__ == "__main__":
    main()

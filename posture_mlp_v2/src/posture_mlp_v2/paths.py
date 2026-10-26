"""Locate data projects independently of the installed Python package."""
from pathlib import Path


def find_project_root(anchor=None):
    starts = [Path(anchor).resolve()] if anchor is not None else []
    starts.append(Path.cwd().resolve())
    for start in starts:
        for parent in (start, *start.parents):
            for candidate in (parent, parent / "posture_mlp_v2"):
                if (candidate / "pyproject.toml").is_file() and (candidate / "inputs").is_dir():
                    return candidate
    raise ValueError("Cannot locate the data project; pass --project-root or --output-dir explicitly")

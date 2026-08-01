from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def environment_text() -> str:
    import cv2
    import sklearn

    lines = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"opencv={cv2.__version__}",
        f"sklearn={sklearn.__version__}",
    ]
    return "\n".join(lines) + "\n"

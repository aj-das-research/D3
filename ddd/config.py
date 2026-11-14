"""Load the shared run configuration (paper Appendix D.9)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paper.json"


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read a JSON config, optionally layered over the paper defaults."""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        config = _merge(config, json.loads(Path(path).read_text(encoding="utf-8")))
    if overrides:
        config = _merge(config, overrides)
    return config


def resolve_path(path: str) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p

"""User settings, stored in settings.json next to the app (not committed)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = ROOT / "settings.json"

DEFAULTS = {
    "model": None,  # None = use the GPU-based recommendation
    "num_ctx": 8192,
    "library_path": str(Path.home() / "Ollama Library"),
    "tools_default": True,  # library tools on for new chats
    "custom_models": [],  # Hugging Face models the user added (hf.co/user/repo:quant)
}

MAX_INSTRUCTIONS = 4000  # characters, per chat (instructions live in each chat's thread file)


def load() -> dict:
    data = dict(DEFAULTS)
    try:
        stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        data.update({k: v for k, v in stored.items() if k in DEFAULTS})
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        logging.getLogger("app").warning("settings.json unreadable; using defaults")
    return data


def save(data: dict) -> None:
    clean = {k: data.get(k, v) for k, v in DEFAULTS.items()}
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_FILE)

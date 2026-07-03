import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union
import yaml

REQUIRED_FIELDS = ("run_id", "seed", "model", "training")


def load_config(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file and validate required top-level fields are present."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    missing = [field for field in REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise ValueError(f"Config missing required field(s): {missing}")

    return cfg


def save_run_manifest(
    cfg: Dict[str, Any],
    results: Dict[str, Any],
    artifacts: Dict[str, Any],
    config_path: Union[str, Path],
) -> Dict[str, Any]:
    """Merge cfg + timestamp + results + artifacts into a manifest dict, dump as JSON, and return it."""
    manifest = {
        "run_id": cfg["run_id"],
        "run_name": cfg.get("run_name", cfg["run_id"]),
        "timestamp": datetime.now().isoformat(),
        **cfg,
        "results": results,
        "artifacts": artifacts,
    }

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest

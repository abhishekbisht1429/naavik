import os
from typing import Any

import yaml


def write_session_config(
        session_dir: str,
        config: dict[str, Any],
        filename: str = "config.yaml",
) -> None:
    os.makedirs(session_dir, exist_ok=True)
    config_dump_path = os.path.join(session_dir, filename)
    with open(config_dump_path, "w") as config_file:
        yaml.safe_dump(config, config_file)

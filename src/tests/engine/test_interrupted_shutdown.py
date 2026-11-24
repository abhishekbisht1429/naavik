"""A second Ctrl-C while the engine is stopping its workers stops them now:
no worker is ever left running after the engine has gone."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

from tests.hpc.test_end_to_end import COMPONENTS, FRAMEWORK_SRC, session

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc"),
    pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="one-rank gloo DDP"),
]

ENTRY = """
from naavik.engine import Configurator, TrainingEngine

if __name__ == "__main__":
    with TrainingEngine(Configurator(["--config", "my_project/config.yaml",
                                      "--process_timeout_on_join", "60"])) as engine:
        engine.start_session()
"""


def _children(pid: int) -> list[int]:
    text = Path(f"/proc/{pid}/task/{pid}/children").read_text()
    return [int(item) for item in text.split()]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/status")
    return not (status.exists() and "State:\tZ" in status.read_text())


def test_a_second_interrupt_while_stopping_terminates_the_workers(tmp_path):
    package = tmp_path / "my_project"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "components.py").write_text(COMPONENTS.read_text())
    marker = tmp_path / "markers"
    config = dict(session(max_iterations=5), integration_rank_delay={
        "rank": 0, "delay": 120, "marker_dir": str(marker)})
    (package / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "entry.py").write_text(ENTRY)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(FRAMEWORK_SRC), os.environ.get("PYTHONPATH", "")]))
    parent = subprocess.Popen([sys.executable, "entry.py"], cwd=tmp_path, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 120
        while not (marker / "rank_0.delaying").exists():       # the worker is busy now
            assert parent.poll() is None, parent.stdout.read()[-3000:]
            assert time.monotonic() < deadline
            time.sleep(0.1)
        workers = _children(parent.pid)
        assert workers
        parent.send_signal(signal.SIGINT)        # the graceful stop waits for the worker
        time.sleep(1.0)
        parent.send_signal(signal.SIGINT)        # and this one says: now
        output, _ = parent.communicate(timeout=60)
        assert "Interrupted while stopping: terminating" in output
        time.sleep(0.5)
        assert [pid for pid in workers if _alive(pid)] == []
    finally:
        if parent.poll() is None:
            parent.kill()
        for pid in _children(parent.pid) if parent.poll() is None else []:
            os.kill(pid, signal.SIGKILL)

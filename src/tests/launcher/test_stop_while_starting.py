"""A stop (the scheduler's advance signal, or a cancel) that reaches a job
while its task is still starting, before the task installs its own SIGINT
handler, stops the task: it is not lost, leaving the task to run on."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from naavik.launcher.scheduler.common import task_body

#: A task slow to install its handler, as a real one is while it imports.
SLOW_START = ("import signal, sys, time\n"
              "print('starting', flush=True)\n"
              "time.sleep(3)\n"
              "signal.signal(signal.SIGINT, lambda *a: sys.exit(3))\n"
              "print('ready', flush=True)\n"
              "time.sleep(60)\n")


def _job(tmp_path):
    batch_dir = tmp_path / "batch"
    (batch_dir / "groups" / "g").mkdir(parents=True)
    (batch_dir / "groups" / "g" / "tasks.tsv").write_text(
        "index\ttask\tvariant\tconfig\tconfig_hash\n0\tg/e/000\t[]\ttasks/g/e/000/config.yaml\th\n")
    script = tmp_path / "job.sh"
    script.write_text("\n".join(["#!/bin/bash", *task_body(
        str(batch_dir), "g", [sys.executable, "-c", SLOW_START], "TF_INDEX")]) + "\n")
    return subprocess.Popen(["bash", str(script)], env={**os.environ, "TF_INDEX": "0"},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_a_stop_while_the_task_is_starting_stops_it(tmp_path):
    job = _job(tmp_path)
    try:
        assert job.stdout.readline().strip() == "starting"      # no handler of its own yet
        started = time.monotonic()
        job.send_signal(signal.SIGUSR1)
        assert job.wait(timeout=30) == 75
        assert time.monotonic() - started < 10                  # not the task's 60 s
        output = job.stdout.read()
        assert "ready" not in output                            # it never got past starting
        assert "received the scheduler advance signal" in output
    finally:
        if job.poll() is None:
            job.kill()

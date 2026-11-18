"""`naavik batches --all-targets`: the batches on every target of the target
file, one connection each; a target that cannot be read is one `?` row and
never hides the others."""

from __future__ import annotations

import json
import shutil

import pytest
import yaml

from training_framework.cli import main as naavik


def receipt(batch_id, target, batch_dir):
    return {"batch_id": batch_id, "batch_dir": str(batch_dir), "status": "submitted",
            "nodes": {}, "profile": target, "created": "2026-11-17T09:00:00+00:00",
            "task_count": 2, "scheduler": "host", "target": {"transport": "local"}}


def local_target(root):
    return {"local": True, "scheduler": "host", "host": {},
            "roots": {key: str(root / key) for key in ("batches", "envs", "data")}}


@pytest.fixture
def targets(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    data = {"default": "workstation", "targets": {
        "workstation": local_target(tmp_path / "ws"),
        "lab-box": local_target(tmp_path / "lab"),
        "far": {key: value for key, value in {**local_target(tmp_path / "far"),
                                              "ssh_host": "naavik-test-host.invalid"}.items()
                if key != "local"},
        "broken": {**local_target(tmp_path / "broken"), "colour": "red"},
    }}
    for name in ("workstation", "lab-box"):
        for index in range(2):
            batch_id = f"{name}-batch-{index}"
            batch_dir = tmp_path / ("ws" if name == "workstation" else "lab") / "batches" / batch_id
            batch_dir.mkdir(parents=True)
            (batch_dir / "receipt.json").write_text(json.dumps(receipt(batch_id, name, batch_dir)))
    # One of them was launched from here too: listed once.
    receipts = tmp_path / "state" / "training-framework" / "submissions"
    receipts.mkdir(parents=True)
    known = tmp_path / "ws" / "batches" / "workstation-batch-0"
    (receipts / "workstation-batch-0.json").write_text(
        json.dumps(receipt("workstation-batch-0", "workstation", known)))
    path = tmp_path / "targets.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.skipif(shutil.which("ssh") is None, reason="needs ssh")
def test_every_target_is_listed_and_the_unreachable_is_one_row(targets, capsys):
    assert naavik(["batches", "--all-hosts", "--targets", str(targets)]) == 1
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    batches = [line.split()[0] for line in lines[1:] if not line.startswith("?")]
    assert sorted(batches) == ["lab-box-batch-0", "lab-box-batch-1",
                               "workstation-batch-0", "workstation-batch-1"]
    assert lines[0].split()[3] == "from"
    unreachable = [line for line in lines if line.startswith("?")]
    assert len(unreachable) == 2
    assert any("far" in line and "(cannot list:" in line and "naavik-test-host.invalid" in line
               for line in unreachable)
    assert any("broken" in line and "colour" in line for line in unreachable)
    assert "error: cannot list host 'far':" in captured.err
    assert "error: cannot list host 'broken':" in captured.err


@pytest.mark.skipif(shutil.which("ssh") is None, reason="needs ssh")
def test_json_names_the_unreachable_targets(targets, capsys):
    naavik(["batches", "--all-hosts", "--targets", str(targets), "--json"])
    entries = json.loads(capsys.readouterr().out)
    assert sorted(entry["target"] for entry in entries if "unreachable" in entry) == ["broken", "far"]
    assert len([entry for entry in entries if "batch_id" in entry]) == 4


def test_all_targets_and_one_target_conflict(targets, capsys):
    assert naavik(["batches", "--all-hosts", "--target", "workstation",
                   "--targets", str(targets)]) == 2
    assert "--all-hosts lists every host" in capsys.readouterr().err


def test_reachable_targets_alone_exit_zero(tmp_path, targets, capsys):
    data = yaml.safe_load(targets.read_text())
    del data["targets"]["far"], data["targets"]["broken"]
    targets.write_text(yaml.safe_dump(data))
    assert naavik(["batches", "--all-hosts", "--targets", str(targets)]) == 0
    assert "?" not in capsys.readouterr().out



@pytest.mark.skipif(shutil.which("ssh") is None, reason="needs ssh")
def test_a_live_listing_puts_the_reason_where_the_state_goes(targets, capsys):
    naavik(["batches", "--all-hosts", "--live", "--targets", str(targets)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[-1] == "now"
    far = next(line for line in lines if line.startswith("?") and " far " in line)
    assert far.index("(cannot list:") == lines[0].index("now")      # in the state column

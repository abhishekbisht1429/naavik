"""The engine's resume and extend flags name a session or a checkpoint:
`--resume-session SESSION_DIR` / `--extend-session SESSION_DIR` continue from
the session's latest complete checkpoint (said in the log);
`--resume-session-from-checkpoint` / `--extend-session-from-checkpoint` take
the checkpoint. A checkpoint given where a session is expected is refused,
naming the flag to use."""

from __future__ import annotations

import json

import pytest

from naavik.engine import Configurator
from naavik.session.checkpoint_format import (
    CheckpointFormatError,
    FORMAT_VERSION,
    latest_checkpoint,
)


def _checkpoint(directory, iteration):
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(
        {"format_version": FORMAT_VERSION, "iteration": iteration}))
    return directory


@pytest.fixture
def session_dir(tmp_path):
    root = tmp_path / "runs" / "session_20261124_010000"
    _checkpoint(root / "checkpoints" / "20261124_010100_1", 10)
    _checkpoint(root / "checkpoints" / "20261124_010200_2", 30)
    _checkpoint(root / "checkpoints" / "20261124_010300_3", 20)
    (root / "checkpoints" / "20261124_010400_4.tmp").mkdir()           # an interrupted save
    return root


def test_the_latest_checkpoint_is_the_highest_complete_iteration(session_dir):
    path, iteration = latest_checkpoint(session_dir)
    assert path.endswith("20261124_010200_2") and iteration == 30


def test_a_session_without_a_complete_checkpoint_is_refused(tmp_path):
    empty = tmp_path / "session_x"
    (empty / "checkpoints" / "a.tmp").mkdir(parents=True)
    with pytest.raises(CheckpointFormatError, match="holds no complete checkpoint"):
        latest_checkpoint(empty)
    with pytest.raises(CheckpointFormatError, match="no such directory"):
        latest_checkpoint(tmp_path / "nowhere")


def test_resume_session_takes_the_latest_checkpoint_and_says_so(session_dir, capsys):
    configurator = Configurator(["--resume-session", str(session_dir)])
    assert configurator.mode == "resume"
    assert configurator.checkpoint_path.endswith("20261124_010200_2")
    assert (f"resuming session {session_dir} from its latest checkpoint: "
            f"{configurator.checkpoint_path} (iteration 30)") in capsys.readouterr().out


def test_extend_session_takes_the_latest_checkpoint_and_says_so(session_dir, capsys):
    configurator = Configurator(["--extend-session", str(session_dir),
                                 "--override", "session_config.max_iterations=50"])
    assert configurator.mode == "extend"
    assert configurator.checkpoint_path.endswith("20261124_010200_2")
    assert configurator.extension_overrides == ("session_config.max_iterations=50",)
    assert "extending session" in capsys.readouterr().out


def test_the_checkpoint_flags_take_the_checkpoint_given(session_dir, capsys):
    checkpoint = str(session_dir / "checkpoints" / "20261124_010100_1")
    assert Configurator(["--resume-session-from-checkpoint", checkpoint]).checkpoint_path \
        == checkpoint
    assert f"resuming from checkpoint {checkpoint}" in capsys.readouterr().out
    extended = Configurator(["--extend-session-from-checkpoint", checkpoint,
                             "--override", "session_config.max_iterations=50"])
    assert extended.checkpoint_path == checkpoint


@pytest.mark.parametrize("flag, other", [("--resume-session", "--resume-session-from-checkpoint"),
                                         ("--extend-session", "--extend-session-from-checkpoint")])
def test_a_checkpoint_given_for_a_session_names_the_checkpoint_flag(session_dir, capsys, flag,
                                                                    other):
    checkpoint = str(session_dir / "checkpoints" / "20261124_010100_1")
    argv = [flag, checkpoint] + (["--override", "x=1"] if "extend" in flag else [])
    with pytest.raises(SystemExit):
        Configurator(argv)
    assert other in capsys.readouterr().err


def test_a_single_file_checkpoint_given_for_a_session_is_refused_too(tmp_path, capsys):
    legacy = tmp_path / "session.pt"
    legacy.write_bytes(b"")
    with pytest.raises(SystemExit):
        Configurator(["--resume-session", str(legacy)])
    assert "use --resume-session-from-checkpoint" in capsys.readouterr().err

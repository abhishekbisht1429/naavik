"""The hosts file (`hosts.yaml`) in its current names, and the earlier names
(`targets.yaml` / `hpc.yaml`, `profiles:`, `roots`, `env`, ...) still read,
with one warning, to the same host."""

from __future__ import annotations

import copy
import dataclasses

import pytest
import yaml

from tests.hpc.batch_repo import PROFILE, compile_repo, make_repo
from naavik.launcher.compile.render import plan_files
from naavik.launcher.config.schema import parse_profiles
from naavik.launcher.errors import PlanError

IDENTITY = PROFILE["profiles"]["cluster"]["env"]["site_python_identity"]

#: PROFILE (earlier names) written in the current names.
CURRENT = {
    "default_host": "cluster",
    "hosts": {"cluster": {
        "connection": {"ssh": "cluster"},
        "scheduler": {"kind": "slurm", "sbatch_options": {"account": "lab", "qos": "normal"},
                      "partitions": {"gpu_jobs": "gpu", "cpu_jobs": "general"},
                      "max_array_size": 1000},
        "directories": {"missions": "/scratch/me/batches",
                        "python_environments": "/scratch/me/envs",
                        "datasets": "/scratch/me/data"},
        "shell_commands_before": {"everything": ["module load python/3.12"],
                                  "experiments": ["module load cuda"]},
        "python_environment": {"interpreter_from": "site", "site_interpreter": IDENTITY},
    }},
}


def same_host(a, b):
    return dataclasses.replace(a, renamed=()) == dataclasses.replace(b, renamed=())


def test_the_current_names_describe_the_same_host_as_the_earlier_ones():
    current = parse_profiles(CURRENT, None, "hosts.yaml")
    earlier = parse_profiles(PROFILE, None, "hpc.yaml")
    assert same_host(current, earlier)
    assert current.renamed == ()
    assert "roots -> directories" in "; ".join(earlier.renamed)
    assert "env -> python_environment" in "; ".join(earlier.renamed)


def test_a_mission_compiles_the_same_from_either_names(tmp_path):
    root = make_repo(tmp_path)
    current = compile_repo(root, profile_data=CURRENT)
    earlier = compile_repo(root, profile_data=PROFILE)
    assert plan_files(current.plan, current.manifest.raw) == plan_files(earlier.plan, earlier.manifest.raw)


def test_every_setting_in_its_current_name():
    data = copy.deepcopy(CURRENT)
    host = data["hosts"]["cluster"]
    host.update({
        "environment_build": {"on": "login_node",
                              "allocation": {"walltime": "00:30:00", "cpus": 2, "memory": "8G"}},
        "data_preparation": {"on": "login_node",
                             "allocation": {"walltime": "04:00:00", "cpus": 4, "memory": "16G"}},
        "local_to_host_paths": {"/data/me": "/scratch/shared"},
        "confirm_uploads_over": "5G",
        "run_code_from": "working_tree",
        "working_tree_untracked_limit": "10M",
        "dataset_delivery": {"default": "copy_from_here", "imagenet": "already_on_host"},
    })
    host["python_environment"].update({"kind": "venv", "conda_packages": [],
                                       "bootstrap_interpreter": "/opt/python/bin/python3"})
    profile = parse_profiles(data, None, "hosts.yaml")
    assert (profile.setup_location, profile.prepare_location) == ("login", "login")
    assert profile.setup_resources.walltime == "00:30:00"
    assert profile.preparation_resources.walltime == "04:00:00"
    assert profile.path_map == (("/data/me", "/scratch/shared"),)
    assert profile.upload_limit == 5 * 1024 ** 3 and profile.snapshot_limit == 10 * 1024 ** 2
    assert profile.worktree == "snapshot"
    assert (profile.data.mode("imagenet"), profile.data.mode("cifar10")) == ("present", "upload")
    assert profile.env.bootstrap_python == "/opt/python/bin/python3"


def test_a_builtin_scheduler_on_this_machine():
    data = {"hosts": {"here": {
        "connection": "local",
        "scheduler": {"kind": "builtin", "gpus": [1, 0], "when_gpus_busy": "use_anyway"},
        "directories": {"missions": "/m", "python_environments": "/e", "datasets": "/d"},
        "python_environment": {"use_existing": "/opt/env"},
    }}}
    profile = parse_profiles(data, "here", "hosts.yaml")
    assert profile.local and profile.scheduler == "host"          # its id in plans and receipts
    assert profile.scheduler_settings.gpus == (1, 0)
    assert profile.scheduler_settings.busy_gpus == "use"


@pytest.mark.parametrize("change, message", [
    ({"connection": "ssh"}, r"connection must be `local` \(this machine\) or \{ssh: "),
    ({"connection": {"ssh": ""}}, "connection must be `local`"),
    ({"scheduler": "slurm"}, None),                                  # earlier form: still read
    ({"scheduler": {"kind": "pbs"}}, "scheduler.kind must be one of builtin, slurm"),
    ({"directories": {"missions": "relative"}}, "directories"),
    ({"run_code_from": "stash"}, "run_code_from must be one of commit, working_tree"),
    ({"environment_build": {"on": "laptop"}}, "environment_build.on must be one of compute_node, login_node"),
    ({"dataset_delivery": {"x": "fetch"}}, "must be one of download_on_host, copy_from_here, already_on_host"),
])
def test_a_wrong_setting_is_named_as_written(change, message):
    data = copy.deepcopy(CURRENT)
    data["hosts"]["cluster"].update(change)
    if message is None:
        data["hosts"]["cluster"]["slurm"] = {"max_array_size": 10}
        assert parse_profiles(data, None, "hosts.yaml").scheduler == "slurm"
        return
    with pytest.raises(PlanError, match=message):
        parse_profiles(data, None, "hosts.yaml")


@pytest.mark.parametrize("both, message", [
    ({"roots": {}}, "roots and directories are the same setting"),
    ({"env": {}}, "env and python_environment are the same setting"),
    ({"path_map": {}, "local_to_host_paths": {}}, "path_map and local_to_host_paths are the same setting"),
])
def test_an_earlier_name_beside_its_current_name_is_refused(both, message):
    data = copy.deepcopy(CURRENT)
    data["hosts"]["cluster"].update(both)
    with pytest.raises(PlanError, match=message):
        parse_profiles(data, None, "hosts.yaml")


def test_a_hosts_file_with_both_hosts_and_profiles_is_refused():
    data = dict(CURRENT, profiles=PROFILE["profiles"])
    with pytest.raises(PlanError, match="has both 'profiles' and 'hosts'"):
        parse_profiles(data, None, "hosts.yaml")


def test_data_preparation_has_its_own_allocation(tmp_path):
    from tests.launcher.test_data_requirements import compile_with, port, requirement

    data = copy.deepcopy(CURRENT)
    data["hosts"]["cluster"].update({
        "dataset_delivery": {"default": "download_on_host"},
        "environment_build": {"allocation": {"walltime": "00:30:00", "cpus": 2, "memory": "8G"}},
        "data_preparation": {"allocation": {"walltime": "06:00:00", "cpus": 8, "memory": "32G"}},
    })
    plan = compile_with(make_repo(tmp_path), port([requirement()]),
                        parse_profiles(data, None, "hosts.yaml")).plan
    (fetch,) = [node for node in plan.nodes if node.kind == "prepare"]
    (setup,) = [node for node in plan.nodes if node.kind == "setup"]
    assert (fetch.resources.walltime, setup.resources.walltime) == ("06:00:00", "00:30:00")


# The file on disk ------------------------------------------------------------------------------


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    (home / "training-framework").mkdir(parents=True)
    return home / "training-framework"


def validate(root):
    from naavik.cli import main as naavik_cli

    return naavik_cli(["validate", str(root / "batch.yaml")])


def test_hosts_yaml_is_the_file_read(tmp_path, config_home, capsys):
    (config_home / "hosts.yaml").write_text(yaml.safe_dump(CURRENT))
    (config_home / "targets.yaml").write_text("not: [valid")
    assert validate(make_repo(tmp_path)) == 0
    err = capsys.readouterr().err
    assert f"{config_home / 'targets.yaml'} is ignored" in err and "warning:" not in err.split("\n")[0]


def test_earlier_names_are_read_with_one_warning_naming_the_fix(tmp_path, config_home, capsys):
    (config_home / "targets.yaml").write_text(yaml.safe_dump(PROFILE))
    assert validate(make_repo(tmp_path)) == 0
    err = capsys.readouterr().err
    warnings = [line for line in err.splitlines() if line.startswith("warning:")
                and "earlier names" in line and "targets.yaml uses" in line]
    assert len(warnings) == 1
    assert "roots -> directories" in warnings[0]
    assert warnings[0].endswith("rename them by hand (old -> new)")


def test_the_warning_names_each_renamed_setting():
    renamed = "; ".join(parse_profiles(PROFILE, None, "hpc.yaml").renamed)
    assert "directories.batches -> directories.missions" in renamed
    assert "python_environment.site_python_identity -> python_environment.site_interpreter" in renamed


def test_an_earlier_name_beside_its_current_name_inside_a_section_is_refused():
    data = copy.deepcopy(PROFILE)
    data["profiles"]["cluster"]["roots"]["missions"] = "/scratch/me/missions"
    with pytest.raises(PlanError, match="batches and missions are the same setting"):
        parse_profiles(data, None, "hpc.yaml")


def test_earlier_setup_resources_still_serve_data_preparation():
    # Before the rename, data steps borrowed the environment build's resources.
    data = copy.deepcopy(PROFILE)
    data["profiles"]["cluster"]["setup_resources"] = {"walltime": "03:00:00", "cpus": 2, "memory": "8G"}
    profile = parse_profiles(data, None, "hpc.yaml")
    assert profile.preparation_resources == profile.setup_resources
    assert profile.preparation_resources.walltime == "03:00:00"

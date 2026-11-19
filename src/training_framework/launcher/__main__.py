"""The launcher's command line: `naavik <command>` (`training_framework.cli`),
also `python -m training_framework.hpc <command>` and
`python -m training_framework.launcher <command>`."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from training_framework.launcher.compile.content import HashCache, content_summary, default_cache_path
from training_framework.launcher.config.schema import parse_profiles
from training_framework.launcher.errors import HpcError
from training_framework.launcher.ports.repository import GitRepository
from training_framework.launcher.ports.ssh import SSH_HINT, TransportError


HOSTS_FILE = "hosts.yaml"
#: The hosts file's earlier names, newest first; still read when
#: `hosts.yaml` is absent.
EARLIER_HOSTS_FILES = ("targets.yaml", "hpc.yaml")
#: Kept for code that imports the earlier names.
TARGETS_FILE, LEGACY_TARGETS_FILE = EARLIER_HOSTS_FILES


def _config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "training-framework"


def default_profiles_path() -> Path:
    """The hosts file: `hosts.yaml`, else an earlier name (`targets.yaml`,
    then `hpc.yaml`) if only that exists (with a note on stderr), else
    `hosts.yaml` (to create)."""
    directory = _config_dir()
    current = directory / HOSTS_FILE
    earlier = [directory / name for name in EARLIER_HOSTS_FILES if (directory / name).exists()]
    if current.exists():
        for path in earlier:
            print(f"note: {path} is ignored; hosts are read from {current}", file=sys.stderr)
        return current
    if earlier:
        print(f"note: reading hosts from {earlier[0]}; rename it to {current} "
              f"({earlier[0].name} is an earlier name)", file=sys.stderr)
        for path in earlier[1:]:
            print(f"note: {path} is ignored; hosts are read from {earlier[0]}", file=sys.stderr)
        return earlier[0]
    return current


class HashProgress:
    """Reports hashing of an upload source on stderr once it takes longer
    than `delay` seconds: a line rewritten in place on a terminal, a line
    every `log_interval` seconds otherwise, and a closing summary."""

    def __init__(self, stream=None, delay: float = 1.0, log_interval: float = 15.0,
                 clock=time.monotonic):
        self.stream = stream or sys.stderr
        self.delay = delay
        self.log_interval = log_interval
        self.clock = clock
        self._root = None

    def __call__(self, root: str, done: int, total: int, size: int,
                 hashed: int | None = None) -> None:
        now = self.clock()
        if root != self._root:
            self._root, self._started, self._last, self._shown = root, now, None, False
        finished = done == total
        if now - self._started < self.delay and not (finished and self._shown):
            return
        tty = self.stream.isatty()
        interval = 0.2 if tty else self.log_interval
        if not finished and self._last is not None and now - self._last < interval:
            return
        self._last, self._shown = now, True
        # A run over cached files only compares timestamps; say so, or it
        # looks like the whole dataset is hashed again.
        verb = "hashing" if hashed is None else "checking"
        line = (f"[hpc] {verb} {root} (upload source): {done:,}/{total:,} files, "
                f"{_size(size)}")
        if hashed is not None:
            line += f" ({hashed:,} hashed, {done - hashed:,} unchanged since the last run)"
        if finished:
            line += f", done in {now - self._started:.0f}s"
        if tty:
            self.stream.write("\r\033[K" + line + ("\n" if finished else ""))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()


class WorkstationFacts:
    """`LocalFacts` backed by this machine's filesystem and hash cache."""

    def __init__(self, repository_root: Path, cache_path: Path | None = None,
                 progress=None):
        self.repository_root = str(repository_root)
        self.home = os.path.expanduser("~")
        self._cache_path = cache_path or default_cache_path()
        self._cache = None
        self._progress = progress if progress is not None else HashProgress()
        self._summaries = {}

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def content(self, path: str):
        if path in self._summaries:       # several sources can name one dataset
            return self._summaries[path]
        if self._cache is None:
            self._cache = HashCache(self._cache_path)
        start = self._cache.hashed

        def report(root, done, total, size):
            self._progress(root, done, total, size, self._cache.hashed - start)

        summary = content_summary(path, self._cache, report)
        self._summaries[path] = summary.key, summary.files, summary.size
        return self._summaries[path]


#: Command names kept as aliases, and the command they run.
#: Earlier command names -> current ones (both work).
COMMAND_ALIASES = {"submit": "launch", "batches": "missions",
                   "probe-python": "show-site-interpreter"}


def _parser(prog: str = "python -m training_framework.hpc") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, **options):
        # Flags are spelled out: an abbreviation (`--target` for `--targets`)
        # could silently mean another flag.
        aliases = [old for old, new in COMMAND_ALIASES.items() if new == name]
        return commands.add_parser(name, aliases=aliases, allow_abbrev=False, **options)

    def host_options(command):
        command.add_argument("--host", "--target", "--profile", dest="profile",
                             help="the host (default: the hosts file's default_host); "
                                  "--target and --profile are earlier names")
        command.add_argument("--hosts-file", "--targets", "--profiles", dest="profiles", type=Path,
                             help="the hosts file (default: $XDG_CONFIG_HOME/training-framework/"
                                  "hosts.yaml); --targets and --profiles are earlier names")

    def receipts(command):
        command.add_argument("--receipts", type=Path, help="receipt directory (default: $XDG_STATE_HOME/training-framework/submissions)")

    for name, help_text in (
            ("validate", "compile the mission file and report every offline problem"),
            ("render", "write a preview of the mission directory"),
            ("launch", "start a mission: stage it on its host and start its jobs"),
    ):
        command = add(name, help=help_text)
        command.add_argument("batch", metavar="MISSION_FILE", help="path to the committed mission file")
        host_options(command)
        command.add_argument("--job", action="append", default=[], metavar="JOB",
                             help="only this job's experiments (repeatable)")
        command.add_argument("--recipe", action="append", default=[], metavar="JOB/RECIPE",
                             help="only this recipe's experiments (repeatable)")
        if name == "validate":
            command.add_argument("--suggest-data", action="store_true",
                                 help="print data_sources entries for uncovered local paths")
        elif name == "render":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--fresh-env", action="store_true",
                                 help="build a managed environment for this mission only")
            command.add_argument("--yes", action="store_true",
                                 help="do not ask (also accepts uploads above confirm_uploads_over)")
    run = add(
        "run", help="run one config as a mission of one experiment, and follow it",
        description="Run CONFIG as a mission of one experiment, built from the project file "
                    "(naavik-project.yaml at the repository root): checked, staged, and "
                    "recorded like any mission. Follows the experiment's log until it ends and "
                    "exits with its status. Ctrl-C cancels it; a second Ctrl-C stops following.")
    run.add_argument("config", help="the training config (a file in the repository)")
    run.add_argument("--override", action="append", default=[], metavar="PATH=VALUE",
                     help="set PATH (in the session) or component:PATH (inside that "
                          "component's entry) to VALUE, read as YAML; repeatable")
    run.add_argument("--note", help="why this run: kept in its receipt and run record")
    run.add_argument("--gpus", type=int, help="GPUs for the experiment (default: quick_run_defaults)")
    run.add_argument("--walltime", help="walltime of the experiment, [D-]HH:MM:SS (default: quick_run_defaults)")
    run.add_argument("--detach", action="store_true", help="return once launched; do not follow")
    run.add_argument("--fresh-env", action="store_true",
                     help="build a managed environment for this run only")
    run.add_argument("--yes", action="store_true", help="do not ask, even above confirm_uploads_over")
    host_options(run)
    status = add("status", help="show the state of every experiment of a mission")
    status.add_argument("batch_id", metavar="MISSION", help="a mission id, or the path to its receipt")
    receipts(status)
    host_options(status)
    listing = add(
        "missions", help="list every mission you launched, past and present",
        description="List every mission launched from this machine (from its receipts). "
                    "--live adds what each mission is doing now; --experiments shows every "
                    "experiment.")
    listing.add_argument("--live", action="store_true",
                         help="connect to the host(s) and show what each mission is doing now")
    listing.add_argument("--experiments", "--tasks", dest="tasks", action="store_true",
                         help="show every experiment of each mission (implies --live); "
                              "--tasks is its earlier name")
    listing.add_argument("--remote", action="store_true",
                         help="also list missions launched from other machines (needs a host)")
    listing.add_argument("--all-hosts", action="store_true",
                         help="list the missions on every host of the hosts file (one "
                              "connection each; a host that cannot be reached is one ? row)")
    listing.add_argument("--active", action="store_true",
                         help="only missions with queued or running experiments (implies --live)")
    listing.add_argument("--since", metavar="AGE", help="only missions launched within AGE: 30m, 12h, 7d, 2w")
    listing.add_argument("--name", help="only missions whose id contains NAME")
    listing.add_argument("--limit", type=int, help="only the newest LIMIT missions")
    listing.add_argument("--json", action="store_true", help="print JSON instead of a table")
    receipts(listing)
    host_options(listing)
    resume = add(
        "resume", help="continue a mission's unfinished experiments from their checkpoints",
        description="Resubmit a mission's interrupted experiments (and those that ended "
                    "without a record) from their latest checkpoints, as a new attempt in the "
                    "same mission directory, with the same environment.")
    resume.add_argument("batch_id", metavar="MISSION", help="a mission id, or the path to its receipt")
    resume.add_argument("--experiment", "--task", dest="task", action="append", metavar="NAME",
                        help="resume this experiment (repeatable; --task is its earlier name); "
                             "default: every interrupted experiment and every one that ended "
                             "without a record")
    resume.add_argument("--include-failed", action="store_true",
                        help="also resume failed experiments (fix the cause first)")
    resume.add_argument("--restart", action="store_true",
                        help="run experiments that have no checkpoint again from their config "
                             "(default: skip them)")
    resume.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help="resume experiment NAME from this checkpoint instead of its latest")
    resume.add_argument("--walltime", help="walltime of the resumed experiments ([D-]HH:MM:SS)")
    resume.add_argument("--cpus", type=int, help="CPUs per resumed experiment")
    resume.add_argument("--memory", help="memory per resumed experiment (e.g. 64G)")
    resume.add_argument("--max-parallel", type=int, help="most resumed experiments running at once")
    resume.add_argument("--yes", action="store_true", help="do not ask")
    resume.add_argument("--dry-run", action="store_true",
                        help="show what would be resumed and the job scripts; submit nothing")
    receipts(resume)
    host_options(resume)
    fetch = add(
        "fetch", help="download a mission's results (records, logs, run files, checkpoints)",
        description="Download the results of a mission's experiments to this machine: "
                    "records, configs, run directories (TensorBoard events, outputs), job "
                    "logs, and the latest complete checkpoint of each experiment. Running it "
                    "again copies only what is new.")
    fetch.add_argument("batch_id", metavar="MISSION", help="a mission id, or the path to its receipt")
    fetch.add_argument("--dest", type=Path, default=None,
                       help="where to put it: <dest>/<mission-id>/ (default: ./hpc-results; "
                            "a mission that ran on this machine is copied only with --dest)")
    fetch.add_argument("--experiment", "--task", dest="task", action="append", metavar="NAME",
                       help="fetch this experiment (repeatable; --task is its earlier name)")
    fetch.add_argument("--state", choices=("done", "finished", "all"), default="done",
                       help="done: finished successfully (default); finished: any final state; "
                            "all: also experiments still queued or running")
    fetch.add_argument("--checkpoints", choices=("latest", "all", "none"), default="latest",
                       help="which checkpoints of each experiment (default: latest complete one)")
    fetch.add_argument("--no-logs", action="store_true", help="skip the job logs")
    fetch.add_argument("--include", action="append", default=[], metavar="GLOB",
                       help="also fetch files matching GLOB, relative to the mission directory")
    fetch.add_argument("--yes", action="store_true", help="do not ask, even for large downloads")
    fetch.add_argument("--dry-run", action="store_true", help="show what would be fetched")
    receipts(fetch)
    host_options(fetch)
    cancel = add(
        "cancel", help="stop a mission's queued and running jobs, or some of its experiments",
        description="Cancel every job of a mission that is still queued or running, or (with "
                    "--experiment) only those experiments. Running experiments get the "
                    "scheduler's stop signal first; the receipt records what was cancelled.")
    cancel.add_argument("batch_id", metavar="MISSION", help="a mission id, or the path to its receipt")
    cancel.add_argument("--experiment", "--task", dest="task", action="append", default=[],
                        metavar="NAME", help="cancel only this experiment (repeatable)")
    cancel.add_argument("--yes", action="store_true", help="do not ask")
    receipts(cancel)
    host_options(cancel)
    logs = add(
        "logs", help="show (or follow) the log of an experiment or step of a mission",
        description="Print the job log of an experiment (<job>/<recipe>/<NNN>) or step "
                    "(setup, prepare-...) of a mission, from its host; its latest attempt "
                    "unless --attempt is given.")
    logs.add_argument("batch_id", metavar="MISSION", help="a mission id, or the path to its receipt")
    logs.add_argument("task", metavar="EXPERIMENT",
                      help="an experiment name, or a step (setup, prepare-...)")
    logs.add_argument("--attempt", type=int, help="this attempt's log (default: the latest)")
    logs.add_argument("--follow", "-f", action="store_true",
                      help="keep printing what is appended (Ctrl-C stops following, not the "
                           "experiment)")
    receipts(logs)
    host_options(logs)
    probe = add("show-site-interpreter",
                help="print the site interpreter's identity on the host, for "
                     "python_environment.site_interpreter")
    host_options(probe)
    return parser


def _profile(args):
    path = args.profiles or default_profiles_path()
    if not path.exists():
        if args.profiles is None:
            raise HpcError(f"no hosts file: {path} does not exist (nor do its earlier names "
                           f"{', '.join(EARLIER_HOSTS_FILES)})")
        raise HpcError(f"hosts file {path} does not exist")
    profile = parse_profiles(yaml.safe_load(path.read_text()), args.profile, str(path))
    if profile.renamed:
        print(f"warning: {path} uses earlier names ({'; '.join(profile.renamed)}); "
              "rename them by hand (old -> new)", file=sys.stderr)
    return profile


def _compile(args, profile, fresh_env=False, fetch_here=False):
    # Imported here so `--help` stays fast.
    from training_framework.launcher.compile.compiler import Selection, compile_plan

    batch = Path(args.batch).resolve()
    repository = GitRepository(batch.parent)
    manifest_path = batch.relative_to(repository.root.resolve()).as_posix()
    compiled = compile_plan(
        manifest_path=manifest_path,
        repository=repository,
        profile=profile,
        facts=WorkstationFacts(repository.root),
        clock=datetime.now(timezone.utc),
        fresh_env=fresh_env,
        code_facts=_code_facts(repository, profile, fetch_here=fetch_here),
        selection=Selection(tuple(args.job), tuple(args.recipe)),
        checkpoint_facts=_checkpoint_facts(profile),
    )
    _warn_earlier_names(compiled, repository)
    _print_code_notes(compiled)
    return compiled, repository


def _code_facts(repository, profile, fetch_here=False):
    """The compiler's `code_facts` port: every task config checked with the
    batch's own code before anything is staged; None (with a warning) when
    the check is turned off. `fetch_here` (launch, run): data the target's
    profile uploads is first fetched on this machine when it is missing."""
    from training_framework.launcher.services.evaluation import ENV_SWITCH, enabled, evaluate_batch

    if not enabled():
        print(f"warning: the experiment configs were not checked ({ENV_SWITCH}=0)", file=sys.stderr)
        return None

    def port(sha, manifest, managed, tasks):
        _progress(f"checking {len(tasks)} experiment config(s) with the code of {sha[:12]}")
        facts = evaluate_batch(sha, manifest, managed, tasks, repository, profile)
        if fetch_here and not facts.problems and profile.data is not None:
            _fetch_uploads_here(sha, manifest, tasks, facts, repository, profile)
        return facts

    return port


def _checkpoint_facts(profile):
    """The compiler's `checkpoint_facts` port: another mission's complete
    checkpoints, for a recipe's `from_mission` (asked only when one has it;
    on the host it reads, never writes)."""
    from training_framework.launcher.services.source_mission import read_source_mission

    def port(mission_id):
        _progress(f"listing the checkpoints of {mission_id}")
        return read_source_mission(mission_id, profile, require_tools=_require_tools)

    return port


def _fetch_uploads_here(sha, manifest, tasks, facts, repository, profile) -> None:
    """Fetch, on this machine, every missing requirement the target uploads
    (once per location), unless a hand-written data source covers it."""
    from training_framework.launcher import framework
    from training_framework.launcher.config import address as addresses
    from training_framework.launcher.services.evaluation import fetch_here

    covered = set()
    for item in manifest.data_sources:
        target, _ = addresses.parse_address(item.get("address") if isinstance(item, dict) else None, "")
        covered.add(target)
    done = set()
    for name, config in tasks:
        for requirement in facts.requirements.get(name, ()):
            spec = requirement.requirement
            address = addresses.Address(requirement.component, (spec["path"],))
            if (profile.data.mode(spec["name"]) != "upload" or not spec["fetchable"]
                    or address in covered or requirement.location in done
                    or _ready_here(requirement)):
                continue
            done.add(requirement.location)
            entry = dict(framework.component_entry_of(config["sessions"][0], requirement.component))
            fetch_here(sha, manifest, requirement, entry, repository, profile, progress=_progress)


def _ready_here(requirement) -> bool:
    """Whether this machine has the requirement's data, marked ready."""
    from training_framework.data import DataRequirement, requirement_key
    from training_framework.launcher.remote.lease import read_marker
    from training_framework.launcher.remote.runtime import StdlibFilesystem

    key = requirement_key(DataRequirement(**requirement.requirement))
    return read_marker(StdlibFilesystem(), requirement.location) == key


def _warn_earlier_names(compiled, repository) -> None:
    """One warning for mission and project files written with earlier names
    (their paths as they are from here, to open and edit)."""
    if not compiled.renamed:
        return
    files = sorted({item.split(": ", 1)[0] for item in compiled.renamed})
    here = []
    for name in files:
        path = Path(repository.root, name).resolve()
        # Relative when it is under the current directory, else absolute.
        here.append(str(path.relative_to(Path.cwd().resolve()))
                    if path.is_relative_to(Path.cwd().resolve()) else str(path))
    verb = "uses" if len(here) == 1 else "use"
    print(f"warning: {', '.join(here)} {verb} earlier names ({'; '.join(compiled.renamed)}); "
          "rename them by hand (old -> new)", file=sys.stderr)


def _print_code_notes(compiled) -> None:
    """What the code check could not check, once."""
    for note in compiled.code_facts.notes if compiled.code_facts is not None else ():
        print(f"note: {note}")


def _cluster(profile):
    from training_framework.launcher.ports.local import LocalProcessProbe
    from training_framework.launcher.ports.targets import ports_for, target_of_profile
    from training_framework.launcher.scheduler import get_scheduler
    from training_framework.launcher.services.artifacts import WorkstationProbe

    # A terminal shows the transfer's own live line; elsewhere (an IDE
    # console, a log) a timestamped line every 30s, or a long upload looks hung.
    ports = ports_for(target_of_profile(profile), progress=_progress)
    executor = ports.executor
    scheduler = get_scheduler(profile.scheduler)
    return {
        "executor": executor,
        "filesystem": ports.filesystem,
        "transfer": ports.transfer,
        "submitter": scheduler.submitter(executor),
        "probe": WorkstationProbe(scheduler.owner_probe(executor), LocalProcessProbe()),
    }


def _size(value) -> str:
    value = float(value or 0)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _review(assume_yes: bool, only_over_limit: bool = False):
    """Print the batch's summary; ask before submitting (`only_over_limit`:
    only when the uploads exceed the profile's upload_limit)."""
    def review(plan, summary) -> bool:
        print(f"mission {plan.batch_id}: {summary.tasks} experiment(s) in {summary.groups} job(s)")
        print(f"environment: {'ready' if summary.environment_ready else 'will be built'}")
        for source, local, size in summary.uploads:
            print(f"upload {source}: {local} ({_size(size)})")
        if summary.uploads:
            limit = f" (limit {_size(summary.upload_limit)})" if summary.upload_limit else ""
            print(f"total upload: {_size(summary.upload_bytes)}{limit}")
        if assume_yes or (only_over_limit and not summary.over_limit):
            return True
        if not sys.stdin.isatty():
            print("error: not interactive; pass --yes to submit", file=sys.stderr)
            return False
        prompt = ("Upload exceeds upload_limit. Submit anyway? [y/N] " if summary.over_limit
                  else "Submit? [y/N] ")
        return input(prompt).strip().lower() in ("y", "yes")
    return review


def _progress(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _require_tools(*names: str) -> None:
    """Fail before any slow work (hashing, staging) when a tool is missing."""
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise HpcError(
            f"{' and '.join(missing)} not found on this machine's PATH; submit uses "
            "ssh to reach the cluster and rsync to copy the source and data "
            "(install them, e.g. `apt install openssh-client rsync` or "
            "`conda install -c conda-forge rsync`)"
        )


def _submit(args, profile) -> int:
    from training_framework.launcher.ports.targets import target_of_profile

    _require_tools(*target_of_profile(profile).tools(transfer=True))
    compiled, repository = _compile(args, profile, fresh_env=args.fresh_env, fetch_here=True)
    label = Path(args.batch).resolve().relative_to(repository.root.resolve()).as_posix()
    status, _ = _launch(profile, compiled, repository, manifest_label=label,
                        review=_review(args.yes))
    return status


def _launch(profile, compiled, repository, *, manifest_label: str, review):
    """Stage and submit a compiled batch; (exit status, receipt or None)."""
    from training_framework.launcher.remote.runtime import current_process_owner
    from training_framework.launcher.services.evaluation import enabled as code_check_enabled
    from training_framework.launcher.services.staging import SourceStager
    from training_framework.launcher.services.submit import submit
    from training_framework.launcher.ports.targets import target_of_profile

    plan = compiled.plan
    for note in plan.notes:
        print(f"note: {note}")
    ports = _cluster(profile)
    from training_framework.launcher.scheduler import get_scheduler

    bundle = get_scheduler(profile.scheduler)
    if bundle.target_check is not None:
        problems = bundle.target_check(ports["executor"], profile.scheduler_settings)
        if problems:
            raise HpcError("the host cannot run this mission:\n  " + "\n  ".join(problems))
    manifest = compiled.manifest
    snapshot_info = {}
    if plan.base_sha:
        # Kept before anything is staged: the commit that runs stays reachable,
        # whatever happens to the launch.
        ref = repository.keep_snapshot(plan.sha, plan.batch_id)
        snapshot_info = {"base_sha": plan.base_sha, "snapshot": True, "snapshot_ref": ref}
        print(f"snapshot {plan.sha[:12]} kept as {ref} (git diff {plan.base_sha[:12]} {ref})")
    stager = SourceStager(repository, ports["executor"], ports["transfer"],
                          manifest.source_strategy, manifest.source_url)
    try:
        receipt = submit(
            plan, manifest.raw, stager=stager, filesystem=ports["filesystem"],
            executor=ports["executor"], submitter=ports["submitter"],
            transfer=ports["transfer"], probe=ports["probe"], owner=current_process_owner(),
            upload_limit=profile.upload_limit, review=review, progress=_progress,
            receipt_info={
                **target_of_profile(profile).receipt_fields(),
                **snapshot_info,
                "code_check": "done" if code_check_enabled() else "skipped",
                "repository": str(repository.root),
                "manifest": manifest_label,
                **({"description": plan.description} if plan.description else {}),
                **({"selection": plan.selection} if plan.selection else {}),
                "submitted_from": socket.gethostname(),
            },
        )
    except HpcError as error:
        receipt = getattr(error, "receipt", None)
        if receipt is not None:
            print(f"{receipt.status}: {plan.batch_dir}")
            print(f"receipt: {receipt.path}")
        raise
    except KeyboardInterrupt as interrupt:
        receipt = getattr(interrupt, "receipt", None)
        print("\ninterrupted", file=sys.stderr)
        if receipt is not None:
            print(f"{receipt.status}: {plan.batch_dir}")
            if receipt.data.get("error"):
                print(f"  {receipt.data['error']}")
            print(f"receipt: {receipt.path}")
        return 130, receipt
    print(f"{receipt.status}: {plan.batch_dir}")
    for node_id, job_ref in receipt.submitted_jobs():
        print(f"  {node_id}: job {job_ref}")
    print(f"receipt: {receipt.path}")
    return (0 if receipt.status == "submitted" else 1), receipt


def _load_receipt(args) -> tuple[Path, dict]:
    from training_framework.launcher.services.receipt import default_receipts_dir
    from training_framework.launcher.services.status import receipt_problem

    path = Path(args.batch_id)
    if not path.is_file():
        path = Path(args.receipts or default_receipts_dir()) / f"{args.batch_id}.json"
    if not path.is_file():
        raise HpcError(f"no receipt for {args.batch_id} (looked at {path})")
    try:
        receipt = json.loads(path.read_text())
    except ValueError as error:
        raise HpcError(f"{path} is not valid JSON: {error}") from None
    if problem := receipt_problem(receipt):
        raise HpcError(f"{path} is not a valid receipt ({problem})")
    return path, receipt


def _receipt_target(args, receipt: dict):
    """(target, scheduler, profile or None) for a receipt: where its batch
    went. A profile is read only for older receipts that do not record it,
    or when named; a named profile must point at the receipt's target."""
    from training_framework.launcher.ports.targets import target_of_profile, target_of_receipt
    from training_framework.launcher.scheduler import LEGACY_SCHEDULER

    target, scheduler = target_of_receipt(receipt), receipt.get("scheduler") or LEGACY_SCHEDULER
    profile = None
    if target is None or args.profile or args.profiles:
        profile = _profile(args)
        named = target_of_profile(profile)
        if target is None:
            target, scheduler = named, profile.scheduler
        elif (named, profile.scheduler) != (target, scheduler):
            raise HpcError(
                f"{receipt['batch_id']} was launched on {target} ({scheduler}), but host "
                f"'{profile.name}' is {named} ({profile.scheduler}); drop --host "
                "to use the receipt's cluster")
    return target, scheduler, profile


def _status(args) -> int:
    from training_framework.launcher.ports.targets import ports_for
    from training_framework.launcher.scheduler import get_scheduler
    from training_framework.launcher.services.status import (
        fetch_batches,
        format_rows,
        missing_files,
        task_rows,
    )

    _, receipt = _load_receipt(args)
    # The batch went where its receipt says (see _receipt_target).
    host, scheduler, _ = _receipt_target(args, receipt)
    _require_tools(*host.tools(transfer=False))
    executor = ports_for(host).executor
    bundle = get_scheduler(scheduler)
    refs = [node["job_ref"] for node in receipt["nodes"].values() if node.get("job_ref")]
    live = bundle.job_states(executor)(refs) if bundle.job_states and refs else {}
    files = fetch_batches(executor, [receipt["batch_dir"]]).get(receipt["batch_dir"])
    print(f"mission {receipt['batch_id']}: {receipt['status']} | {host}:{receipt['batch_dir']}")
    if receipt.get("description"):
        print(f"note: {receipt['description']}")
    if problem := missing_files(receipt, files):
        raise HpcError(f"{problem} on {host}")
    print(format_rows(task_rows(receipt, files, live)))
    return 0


def _batches(args) -> int:
    from training_framework.launcher.ports.targets import ports_for, target_of_profile
    from training_framework.launcher.scheduler import get_scheduler
    from training_framework.launcher.services import batches as listing
    from training_framework.launcher.services.receipt import default_receipts_dir

    live = args.live or args.tasks or args.active
    since = None
    if args.since:
        try:
            since = listing.parse_age(args.since)
        except ValueError as error:
            raise HpcError(str(error)) from None
    # Receipts name their cluster, so the profile is read only when it is
    # needed: for --remote, when one is named explicitly, or (lazily) for a
    # live lookup of a receipt written before receipts recorded the cluster.
    # A plain listing never reads it.
    if args.all_hosts and (args.profile or args.remote):
        raise HpcError("--all-hosts lists every host; drop --host and --remote")
    explicit = bool(args.profile or args.profiles)
    profile = _profile(args) if args.remote or (explicit and not args.all_hosts) else None
    receipts_dir = str(args.receipts or default_receipts_dir())
    views = listing.load_receipts(receipts_dir)
    unreachable_targets = []
    if args.all_hosts:
        unreachable_targets = _list_every_target(args, views)
        views = listing.sort_views(views)

    def connect(target, scheduler: str):
        _require_tools(*target.tools(transfer=False))
        executor = ports_for(target).executor
        bundle = get_scheduler(scheduler)
        return executor, (lambda: bundle.user_job_states(executor)) if bundle.user_job_states else None

    if args.remote:
        if profile is None:
            raise HpcError("--remote needs a host (it lists the host's directories.missions)")
        known = {view.batch_id for view in views}
        _require_tools(*target_of_profile(profile).tools(transfer=False))
        executor = ports_for(target_of_profile(profile)).executor
        for receipt in listing.remote_receipts(executor, profile.roots["batches"]):
            if receipt["batch_id"] not in known:
                views.append(listing.BatchView(receipt, "cluster"))
        views = listing.sort_views(views)
    # Cheap filters first, so --live only reads the batches that are shown.
    views = listing.select(views, since=since, name=args.name,
                           limit=None if args.active else args.limit)
    if live:
        def fallback():
            selected = profile or _profile(args)
            return target_of_profile(selected), selected.scheduler

        listing.add_live_state(views, connect, fallback)
        if args.active:
            views = listing.select(views, active=True, limit=args.limit)
    if live and not views:
        # Nothing to look up, but a live listing should still say whether the
        # cluster can be reached (e.g. after a submit failed before its receipt).
        try:
            selected = profile or _profile(args)
        except HpcError:
            selected = None
        if selected is not None:
            _require_tools(*target_of_profile(selected).tools(transfer=False))
            ports_for(target_of_profile(selected)).executor.run(["true"])   # raises TransportError
    if args.json:
        print(json.dumps(listing.as_json(views, tasks=args.tasks)
                         + [{"target": target.name, "unreachable": target.reason}
                            for target in unreachable_targets], indent=2))
    else:
        print(listing.format_views(views, live=live, tasks=args.tasks,
                                   origins=args.remote or args.all_hosts,
                                   unreachable=unreachable_targets))
        if not views and not args.all_hosts:
            print(f"note: no missions were launched from this machine (receipts in "
                  f"{receipts_dir}){'' if args.remote else '; `missions --remote` also lists the missions on the host'}",
                  file=sys.stderr)
    unreachable = sorted({view.unreachable for view in views if view.unreachable})
    for host in unreachable:
        problem = next(view.problem for view in views if view.unreachable == host)
        reason = problem.removeprefix(f"cannot read {host}: ")
        print(f"error: cannot reach {host} over SSH: {reason}", file=sys.stderr)
        print(f"hint: {SSH_HINT.format(host=host)}", file=sys.stderr)
    for target in unreachable_targets:
        print(f"error: cannot list host '{target.name}': {target.reason}", file=sys.stderr)
    return 1 if unreachable or unreachable_targets else 0


def _list_every_target(args, views) -> list:
    """Add the batches found on every target of the target file to `views`
    (those without a receipt here); the targets that could not be read."""
    from training_framework.launcher.config.schema import parse_profiles, target_names
    from training_framework.launcher.ports.targets import ports_for, target_of_profile
    from training_framework.launcher.services import batches as listing

    path = args.profiles or default_profiles_path()
    if not path.exists():
        raise HpcError(f"no hosts file: {path} does not exist")
    raw = yaml.safe_load(path.read_text())
    names = target_names(raw)
    if not names:
        raise HpcError(f"{path} lists no hosts")
    known = {view.batch_id for view in views}
    missing = []
    for name in names:
        try:
            profile = parse_profiles(raw, name, str(path))
            target = target_of_profile(profile)
            _require_tools(*target.tools(transfer=False))
            receipts = listing.remote_receipts(ports_for(target).executor, profile.roots["batches"])
        except HpcError as error:          # transport errors too: one row, the others go on
            missing.append(listing.UnreachableTarget(name, str(error).replace("\n", " ")))
            continue
        for receipt in receipts:
            if receipt["batch_id"] not in known:
                known.add(receipt["batch_id"])
                views.append(listing.BatchView(receipt, "cluster"))
    return missing


def _resume(args) -> int:
    from training_framework.launcher.ports.targets import ports_for
    from training_framework.launcher.scheduler import get_scheduler
    from training_framework.launcher.services import resume as resuming
    from training_framework.launcher.services.receipt import Receipt
    from training_framework.launcher.services.status import (
        fetch_batches,
        missing_files,
        task_rows,
    )

    path, data = _load_receipt(args)
    if not data.get("staged") or data.get("status") != "submitted":
        raise HpcError(f"{data['batch_id']} was not submitted ({data.get('status')}); "
                       "only a launched mission can be resumed")
    for attempt in data.get("attempts", []):
        if attempt.get("status") == "needs reconciliation":
            raise HpcError(f"attempt {attempt['attempt']} needs reconciliation: "
                           f"{attempt.get('error')}; resolve it before resuming again")
    # The receipt names the cluster; the plan holds the scheduler settings the
    # batch was rendered with, so the current profile does not matter.
    host, scheduler, _ = _receipt_target(args, data)
    changes = resuming.ResourceChanges(args.walltime, args.cpus, args.memory, args.max_parallel)
    changes.check()
    _require_tools(*host.tools(transfer=False))
    ports = ports_for(host)
    executor, filesystem = ports.executor, ports.filesystem
    bundle = get_scheduler(scheduler)
    batch_dir = data["batch_dir"]

    live = bundle.user_job_states(executor) if bundle.user_job_states else {}
    if running := resuming.active_jobs(data, live):
        raise HpcError(f"{data['batch_id']} still has jobs in the queue ({', '.join(running[:5])}"
                       f"{' ...' if len(running) > 5 else ''}); resume after they end, or cancel them")
    files = fetch_batches(executor, [batch_dir]).get(batch_dir)
    if problem := missing_files(data, files):
        raise HpcError(f"{problem} on {host}")
    rows = task_rows(data, files, live)
    candidates = resuming.select_tasks(data, rows, resuming.task_index(files.tables),
                                       args.task, args.include_failed)
    if not candidates:
        raise HpcError(f"{data['batch_id']} has no interrupted experiments and none that ended "
                       "without a record; name them with --experiment (or --include-failed)")
    explicit = resuming.explicit_checkpoints(filesystem, batch_dir, args.checkpoint)
    if unknown := sorted(set(explicit) - {c.task for c in candidates}):
        raise HpcError(f"--checkpoint names {', '.join(unknown)}, which this command does not "
                       "resume; name it with --experiment too, or fix the experiment name")
    checkpoints, configs = resuming.find_checkpoints(executor, batch_dir,
                                                     [c.task for c in candidates])
    resuming.choose_checkpoints(candidates, checkpoints, configs, explicit)
    restarted = resuming.restart_without_checkpoint(candidates) if args.restart else []
    chosen = [c for c in candidates if c.reason is None]
    skipped = [c for c in candidates if c.reason is not None]
    for candidate in skipped:
        print(f"skip {candidate.task} ({candidate.state}): {candidate.reason}")
    if not chosen:
        hint = ("" if args.restart or not any(c.no_checkpoint for c in skipped) else
                "; --restart runs experiments without a checkpoint again from their config")
        raise HpcError(f"nothing can be resumed (see above){hint}")
    plan = resuming.read_plan(filesystem, batch_dir)
    dialect = resuming.plan_dialect(plan)
    attempt = resuming.build_attempt(plan, data, chosen, dialect, changes)
    attempt.skipped = skipped
    _progress("checking the mission's source, environment, and task entry point on the host")
    resuming.check_cluster(plan, filesystem, executor, restart=bool(restarted))

    print(f"resume {data['batch_id']} as attempt {attempt.attempt}: {len(chosen)} experiment(s)")
    for candidate in chosen:
        if candidate.restarts:
            print(f"  {candidate.task:<30} {candidate.state:<24} restart from its config "
                  "(no checkpoint)")
            continue
        total = "?" if candidate.max_iterations is None else candidate.max_iterations
        relative = candidate.checkpoint.removeprefix(batch_dir + "/")
        print(f"  {candidate.task:<30} {candidate.state:<24} from iteration "
              f"{candidate.iteration}/{total}  {relative}")
    for group in attempt.groups:
        r = group.node.resources
        print(f"  job {group.group}: walltime {r.walltime}, {r.gpus} GPU(s), {r.cpus} CPU(s), "
              f"{r.memory}" + (f", at most {group.node.max_parallel} at once"
                               if group.node.max_parallel else ""))
    if args.dry_run:
        for group in attempt.groups:
            print(f"\n# {group.script_path}\n{group.script}# {group.table_path}\n{group.table}")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("error: not interactive; pass --yes to resume", file=sys.stderr)
            return 1
        if input("Resume? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not resumed")
            return 1
    receipt = Receipt(str(path), data)
    receipt.mirror_to(filesystem, f"{batch_dir}/receipt.json")
    try:
        entry = resuming.submit_attempt(receipt, attempt, filesystem, bundle.submitter(executor),
                                        submitted_from=socket.gethostname(), progress=_progress)
    except KeyboardInterrupt:
        print(f"\ninterrupted; receipt: {path}", file=sys.stderr)
        return 130
    except HpcError:
        print(f"receipt: {path}")
        raise
    print(f"resumed: attempt {entry['attempt']} of {data['batch_id']}")
    print(f"receipt: {path}")
    return 0


def _run(args) -> int:
    """`naavik run`: see the `run` parser's description."""
    from training_framework.launcher.compile.compiler import compile_plan
    from training_framework.launcher.config.schema import PROJECT_FILE, parse_project
    from training_framework.launcher.ports.targets import target_of_profile
    from training_framework.launcher.services import quick_run

    config = Path(args.config).resolve()
    if not config.is_file():
        raise HpcError(f"{args.config} does not exist")
    # The project is the repository holding the config.
    repository = GitRepository(config.parent)
    root = repository.root.resolve()
    project_file = root / PROJECT_FILE
    if not project_file.is_file():
        raise HpcError(f"naavik run needs a project file: {project_file} does not exist "
                       "(it holds the code, python_packages, and quick_run_defaults of the project)")
    try:
        project = parse_project(yaml.safe_load(project_file.read_text()), PROJECT_FILE)
    except yaml.YAMLError as error:
        raise HpcError(f"{PROJECT_FILE} is not valid YAML ({error})") from None
    if args.profile is None and project.run_defaults.target:
        args.profile = project.run_defaults.target
    profile = _profile(args)
    _require_tools(*target_of_profile(profile).tools(transfer=True))
    relative = config.relative_to(root).as_posix()
    manifest = quick_run.run_manifest(
        config=relative, project=PROJECT_FILE, run_defaults=project.run_defaults,
        overrides=args.override, note=args.note, gpus=args.gpus, walltime=args.walltime)
    label = f"naavik run {relative}"
    compiled = compile_plan(
        manifest_path=label, repository=repository, profile=profile,
        facts=WorkstationFacts(repository.root), clock=datetime.now(timezone.utc),
        fresh_env=args.fresh_env, manifest_data=quick_run.manifest_bytes(manifest),
        code_facts=_code_facts(repository, profile, fetch_here=True))
    _warn_earlier_names(compiled, repository)
    _print_code_notes(compiled)
    status, receipt = _launch(profile, compiled, repository, manifest_label=label,
                              review=_review(args.yes, only_over_limit=True))
    if status != 0 or receipt is None or args.detach:
        if status == 0 and receipt is not None:
            print(f"follow: naavik logs {receipt.data['batch_id']} "
                  f"{compiled.plan.tasks[0].name} --follow")
        return status
    return _follow(receipt.path, receipt.data, compiled.plan.tasks[0].name)


def _follow(receipt_path, receipt: dict, task: str) -> int:
    """Follow `task` of a launched batch to its end (see services.follow)."""
    from training_framework.launcher.ports.targets import ports_for, target_of_receipt
    from training_framework.launcher.scheduler import get_scheduler
    from training_framework.launcher.services import control
    from training_framework.launcher.services.follow import Follower
    from training_framework.launcher.services.status import fetch_batches, task_rows

    target = target_of_receipt(receipt)
    executor = ports_for(target).executor
    bundle = get_scheduler(receipt["scheduler"])
    refs = [node["job_ref"] for node in receipt["nodes"].values() if node.get("job_ref")]
    batch_dir = receipt["batch_dir"]
    tables = {}

    def rows():
        # The queue first, then the files: a task gone from the queue has
        # already written its record.
        live = bundle.job_states(executor)(refs) if bundle.job_states and refs else {}
        files = fetch_batches(executor, [batch_dir]).get(batch_dir)
        if files is not None:
            tables.update(files.tables)
        # Re-read: a cancel is recorded in the receipt.
        return task_rows(json.loads(Path(receipt_path).read_text()), files, live)

    def log():
        if not tables:
            return None
        return f"{batch_dir}/{control.log_path(receipt, tables, task)}"

    def read(path, offset):
        result = executor.run(["sh", "-c", 'test -f "$1" && tail -c "+$2" "$1" || true',
                               "sh", path, str(offset + 1)])
        return result.stdout.encode()

    def cancel():
        _cancel(argparse.Namespace(batch_id=str(receipt_path), receipts=None, profile=None,
                                    profiles=None, task=[task], yes=True))

    def write(chunk: bytes):
        sys.stdout.write(chunk.decode(errors="replace"))
        sys.stdout.flush()

    print(f"following {task} (Ctrl-C cancels it)", flush=True)
    outcome = Follower(task=task, rows=rows, log=log, read=read, cancel=cancel, write=write,
                       say=lambda message: print(message, file=sys.stderr, flush=True)).run()
    if outcome.state is None:
        print(f"follow again: naavik logs {receipt['batch_id']} {task} --follow")
    else:
        print(f"{task}: {outcome.state}")
    return outcome.status


def _batch_files(args, receipt):
    """(target, bundle, ports, files) of a launched batch, from where its
    receipt says it went."""
    from training_framework.launcher.ports.targets import ports_for
    from training_framework.launcher.scheduler import get_scheduler
    from training_framework.launcher.services.status import fetch_batches, missing_files

    if not receipt.get("staged"):
        raise HpcError(f"{receipt['batch_id']} never reached its host ({receipt.get('status')})")
    host, scheduler, _ = _receipt_target(args, receipt)
    _require_tools(*host.tools(transfer=False))
    ports = ports_for(host)
    files = fetch_batches(ports.executor, [receipt["batch_dir"]]).get(receipt["batch_dir"])
    if problem := missing_files(receipt, files):
        raise HpcError(f"{problem} on {host}")
    return host, get_scheduler(scheduler), ports, files


def _cancel(args) -> int:
    from training_framework.launcher.services import control
    from training_framework.launcher.services.receipt import Receipt

    path, data = _load_receipt(args)
    host, bundle, ports, files = _batch_files(args, data)
    refs = [node["job_ref"] for node in data["nodes"].values() if node.get("job_ref")]
    live = bundle.job_states(ports.executor)(refs) if bundle.job_states and refs else {}
    chosen, notes = control.to_cancel(data, files.tables, args.task, live)
    for note in notes:
        print(f"skip {note}")
    if not chosen:
        print(f"nothing to cancel in {data['batch_id']}")
        return 0
    print(f"cancel in {data['batch_id']} on {host}:")
    for item in chosen:
        print(f"  {item.label:<30} job {item.job_ref}")
    if not args.yes:
        if not sys.stdin.isatty():
            print("error: not interactive; pass --yes to cancel", file=sys.stderr)
            return 1
        if input("Cancel? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not cancelled")
            return 1
    receipt = Receipt(str(path), data)
    receipt.mirror_to(ports.filesystem, f"{data['batch_dir']}/receipt.json")
    submitter = bundle.submitter(ports.executor)
    failed = 0
    for item in chosen:
        result = submitter.cancel(item.job_ref)
        control.record_cancel(receipt, item, result.ok, result.detail)
        print(f"{'cancelled' if result.ok else 'could not cancel'} {item.label}: {result.detail}")
        failed += not result.ok
    return 1 if failed else 0


def _logs(args) -> int:
    from training_framework.launcher.services import control

    _, data = _load_receipt(args)
    host, _, ports, files = _batch_files(args, data)
    relative = control.log_path(data, files.tables, args.task, args.attempt)
    path = f"{data['batch_dir']}/{relative}"
    if args.follow:
        try:
            return ports.executor.stream(["tail", "-n", "+1", "-F", path])
        except KeyboardInterrupt:
            return 0                     # stops following; the task goes on
    result = ports.executor.run(["cat", path])
    if result.status != 0:
        raise HpcError(f"no log at {host}:{path} yet ({result.stderr.strip()})")
    sys.stdout.write(result.stdout)
    return 0


def _fetch(args) -> int:
    from training_framework.launcher.ports.targets import ports_for
    from training_framework.launcher.scheduler import get_scheduler
    from training_framework.launcher.services import fetch as fetching
    from training_framework.launcher.services.resume import task_index
    from training_framework.launcher.services.status import fetch_batches, missing_files, task_rows

    _, receipt = _load_receipt(args)
    if not receipt.get("staged"):
        raise HpcError(f"{receipt['batch_id']} never reached the cluster ({receipt.get('status')}); "
                       "there is nothing to fetch")
    host, scheduler, _ = _receipt_target(args, receipt)
    if host.is_local and args.dest is None:
        # Its results are already on this machine; copying needs a reason.
        print(f"{receipt['batch_id']} ran on this machine; its results are in "
              f"{receipt['batch_dir']} (pass --dest to copy them elsewhere)")
        return 0
    _require_tools(*host.tools(transfer=True))
    ports = ports_for(host, progress=_progress)
    executor = ports.executor
    bundle = get_scheduler(scheduler)
    batch_dir = receipt["batch_dir"]
    live = bundle.user_job_states(executor) if bundle.user_job_states else {}
    files = fetch_batches(executor, [batch_dir]).get(batch_dir)
    if problem := missing_files(receipt, files):
        raise HpcError(f"{problem} on {host}")
    rows = task_rows(receipt, files, live)
    tasks, skipped = fetching.select(receipt, rows, task_index(files.tables), args.task, args.state)
    for name, reason in skipped:
        print(f"skip {name}: {reason}")
    if not tasks:
        raise HpcError("no experiment to fetch (see above)")
    listing = fetching.plan_files(executor, batch_dir, tasks, args.checkpoints, args.include)
    if not args.no_logs:
        fetching.add_logs(executor, batch_dir, receipt, tasks, listing)
    local_root = (args.dest or Path("hpc-results")) / receipt["batch_id"]
    present, present_bytes = fetching.already_here(listing, str(local_root))
    total = sum(listing.values())
    print(f"fetch {receipt['batch_id']}: {len(tasks)} experiment(s) from {host} into {local_root}")
    for task in tasks:
        checkpoint = (", ".join(f"iteration {it}" for _, it in task.checkpoints)
                      if task.checkpoints else "no checkpoint")
        print(f"  {task.task:<30} {task.state:<24} {checkpoint}")
    print(f"{len(listing)} file(s), {_size(total)}; already here: {present} file(s), "
          f"{_size(present_bytes)}")
    if args.dry_run:
        return 0
    to_fetch = total - present_bytes
    if to_fetch > fetching.CONFIRM_BYTES and not args.yes:
        if not sys.stdin.isatty():
            print(f"error: {_size(to_fetch)} to download; pass --yes to confirm", file=sys.stderr)
            return 1
        if input(f"Download {_size(to_fetch)}? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not fetched")
            return 1
    transfer = ports.transfer
    summary = transfer.fetch(batch_dir, sorted(listing), str(local_root))
    verified = {}
    for task in tasks:
        for relative, _ in task.checkpoints:
            problems = fetching.verify_checkpoint(str(local_root / relative))
            verified[relative] = problems
            for problem in problems:
                print(f"warning: checkpoint {relative}: {problem}", file=sys.stderr)
    index = fetching.write_index(str(local_root), receipt, str(host), tasks, verified, summary)
    bad = sum(1 for problems in verified.values() if problems)
    print(f"fetched {summary.fetched} of {summary.files} file(s) ({_size(summary.bytes)}); "
          f"{summary.files - summary.fetched} were already here")
    if verified:
        print(f"checkpoints: {len(verified) - bad} verified" + (f", {bad} with problems" if bad else ""))
    print(f"results: {local_root}  (index: {index})")
    print(f"view: tensorboard --logdir {local_root / 'tasks'}")
    return 1 if bad else 0


def _probe_python(profile) -> int:
    from training_framework.launcher.compile.compiler import (
        SET_VARIABLES,
        UNSET_VARIABLES,
        bootstrap_interpreter,
    )
    from training_framework.launcher.compile.plan import ExecutionContext
    from training_framework.launcher.services.artifacts import identity_block, probe_identity

    if profile.env.python_source != "site":
        raise HpcError("probe-python reports a 'site' interpreter; uv and conda "
                       "identities come from python_packages.python_version")
    interpreter, _ = bootstrap_interpreter(profile.env, None, ".")
    context = ExecutionContext(
        phase="setup", interpreter=interpreter, environment=None, working_directory=".",
        pythonpath=(), set_variables=SET_VARIABLES, unset_variables=UNSET_VARIABLES,
        hooks=tuple(profile.hooks["common"]) + tuple(profile.hooks["setup"]),
    )
    print(identity_block(probe_identity(_cluster(profile)["executor"], context)))
    return 0


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    """`prog`: the name shown in usage lines (`naavik` passes its own)."""
    args = (_parser(prog) if prog else _parser()).parse_args(argv)
    # Dispatch on one name whichever was typed (`submit` is `launch`).
    args.command = COMMAND_ALIASES.get(args.command, args.command)
    try:
        if args.command == "missions":
            return _batches(args)
        if args.command == "status":
            return _status(args)
        if args.command == "resume":
            return _resume(args)
        if args.command == "fetch":
            return _fetch(args)
        if args.command == "cancel":
            return _cancel(args)
        if args.command == "logs":
            return _logs(args)
        if args.command == "run":
            return _run(args)
        profile = _profile(args)
        if args.command == "show-site-interpreter":
            from training_framework.launcher.ports.targets import target_of_profile

            _require_tools(*target_of_profile(profile).tools(transfer=False))
        if args.command == "show-site-interpreter":
            return _probe_python(profile)
        if args.command == "launch":
            return _submit(args, profile)
        compiled, _ = _compile(args, profile)
    except HpcError as error:  # includes PlanError, with every problem
        print(f"error: {error}", file=sys.stderr)
        if isinstance(error, TransportError):
            print(f"hint: {SSH_HINT.format(host=error.host)}", file=sys.stderr)
        return 2
    plan = compiled.plan
    for note in plan.notes:
        print(f"note: {note}")
    if args.command == "validate":
        print(f"ok: {len(plan.tasks)} experiment(s) in {sum(n.kind == 'task_set' for n in plan.nodes)} "
              f"job(s); commit {plan.sha[:12]}; environment "
              f"{plan.environment['key'] or plan.environment['location']}")
        if args.suggest_data and compiled.suggestions:
            print(yaml.safe_dump({"data_sources": compiled.suggestions}, sort_keys=False))
        return 0
    from training_framework.launcher.compile.render import write_plan

    try:
        written = write_plan(plan, compiled.manifest.raw, args.output)
    except HpcError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(written)} file(s) to {args.output} (preview of {plan.batch_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The `host` scheduler's runner: what Slurm does, for one job, on a machine
without a scheduler (building block of the `host` scheduler).

Standard library only, Python 3.8 syntax: the launcher bundles it with
`runtime` and `lease` into `host_runner.py` in the batch directory, run by
the target's `python3`. Imports from this package must stay on single lines;
the bundler removes them.

One runner process per job node, started detached by the submitter:

    python3 host_runner.py run SCRIPT [--after REF ...]

It reads the job's `#TFHOST key=value` lines from SCRIPT, waits for the jobs
it depends on, then runs the script once per element (a task set's tasks,
or once for setup/prepare), at most `max-parallel` at a time, each with its
share of the GPU pool. An element gets USR1 `signal-lead` seconds before its
walltime (the task wrapper turns that into a cooperative stop) and SIGKILL
at the walltime; without a walltime it runs until it ends. CPUs and memory
are not limited.

The GPU pool (`gpu-pool`) is `all` or a list of indices; `gpu-limit` caps
the GPUs held at once by all of this user's runners on the host (default:
the pool's size). Which GPUs an element gets is decided when it starts:
free ones of the pool, skipping those held by other runners and, with
`busy-gpus=wait`, those someone else is using. While an element waits, the
job file says why (`waiting`); a request the pool can never satisfy fails
the job at once, with the reason in each element's log.
The job's state is a JSON file, rewritten on every change:

    <state dir>/jobs/<ref>.json

where the state directory is per user and per host (like the scheduler's own
queue): `$XDG_STATE_HOME/naavik/host/<hostname>` (or, where only that exists,
`training-framework/host/...`, its earlier name: jobs started before the
rename are read in either). The job's
reference is `h<pid>.<start time>` of the runner, checkable on this host.

`python3 host_runner.py states [REF ...]` prints the live states (JSON), and
`python3 host_runner.py cancel REF [INDEX ...]` asks a runner to stop.
"""

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time

from naavik.launcher.remote.runtime import StdlibFilesystem, process_start_time
from naavik.launcher.remote.lease import LOCK_SUFFIX, Lease

POLL = 0.5
DIRECTIVE = "#TFHOST "
#: Element states. Final ones are dropped from the live view, like jobs that
#: left a scheduler's queue.
PENDING, RUNNING = "pending", "running"
COMPLETED, FAILED, TIMEOUT, CANCELLED = "completed", "failed", "timeout", "cancelled"
FINAL = (COMPLETED, FAILED, TIMEOUT, CANCELLED)


#: The state directory's name, then its earlier one (standard library only:
#: this runs on the host, without naavik; `naavik.paths` keeps the same rule).
STATE_NAMES = ("naavik", "training-framework")


def state_dirs():
    """Every place a runner's state may be, current name first."""
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state")
    return [os.path.join(base, name, "host", socket.gethostname()) for name in STATE_NAMES]


def state_dir():
    """Where a new runner keeps its state: under `naavik`, or under the
    earlier name when only that one exists on this host."""
    current, earlier = (os.path.dirname(os.path.dirname(path)) for path in state_dirs())
    if os.path.isdir(earlier) and not os.path.exists(current):
        return state_dirs()[1]
    return state_dirs()[0]


def job_dir(ref):
    """The state directory holding runner `ref`'s job file (where its
    runner reads a cancel request), else where a new runner writes."""
    for directory in state_dirs():
        if os.path.exists(_job_file(directory, ref)):
            return directory
    return state_dir()


def job_ref(pid):
    start = process_start_time(pid)
    return None if start is None else "h{}.{}".format(pid, start)


def parse_ref(ref):
    """(pid, start time) of a reference `h<pid>.<start>`, or None."""
    if not ref.startswith("h") or "." not in ref:
        return None
    pid, _, start = ref[1:].partition(".")
    return (int(pid), start) if pid.isdigit() else None


def ref_alive(ref):
    parsed = parse_ref(ref)
    return parsed is not None and process_start_time(parsed[0]) == parsed[1]


def read_directives(script):
    values = {}
    with open(script) as handle:
        for line in handle:
            if line.startswith(DIRECTIVE):
                key, _, value = line[len(DIRECTIVE):].rstrip("\n").partition("=")
                values[key] = value
    return values


def _job_file(directory, ref):
    return os.path.join(directory, "jobs", ref + ".json")


def read_job(directory, ref):
    data = StdlibFilesystem().read(_job_file(directory, ref))
    return json.loads(data.decode()) if data else None


# GPUs ----------------------------------------------------------------------


def _nvidia_smi(query):
    result = subprocess.run(["nvidia-smi", "--query-" + query, "--format=csv,noheader"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi failed: " + result.stderr.strip())
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_indices():
    """{UUID: nvidia-smi index} of every GPU of the machine."""
    found = {}
    for line in _nvidia_smi("gpu=index,uuid"):
        index, _, uuid = line.partition(",")
        found[uuid.strip()] = int(index)
    return found


def gpu_uuids(pool):
    """The UUIDs of the pool's GPUs (nvidia-smi indices), in pool order;
    `pool` None: every GPU of the machine, in index order."""
    by_index = {index: uuid for uuid, index in gpu_indices().items()}
    if pool is None:
        return [by_index[index] for index in sorted(by_index)]
    missing = [index for index in pool if index not in by_index]
    if missing:
        raise RuntimeError("GPU(s) {} of the pool do not exist here (have {})".format(
            missing, sorted(by_index)))
    return [by_index[index] for index in pool]


#: Memory in use (MiB) from which a GPU counts as busy: above what an idle
#: GPU shows, far below what any training process holds.
BUSY_MEMORY_MIB = 256


def busy_uuids():
    """GPUs someone else is using: any compute process `nvidia-smi` lists, or
    memory in use. Inside a container other containers' processes are not
    listed, but the memory they hold is."""
    busy = {line.split(",")[0].strip() for line in _nvidia_smi("compute-apps=gpu_uuid")}
    for line in _nvidia_smi("gpu=uuid,memory.used"):
        uuid, _, used = line.partition(",")
        amount = used.strip().split(" ")[0]
        if amount.isdigit() and int(amount) >= BUSY_MEMORY_MIB:
            busy.add(uuid.strip())
    return busy


SLOT_PREFIX = "gpu-"


class Slots:
    """Host-wide GPU slots: one lease per GPU, held by the runner using it,
    so jobs of every batch on this host share the pool without overlap; and
    at most `limit` held at once by all of this user's runners, counted
    under one lock so two runners never both take the last ones."""

    def __init__(self, directory, uuids, wait_busy, owner, limit=None, indices=None):
        self.directory = directory
        self.uuids = uuids
        self.wait_busy = wait_busy
        self.owner = owner
        #: The most GPUs held at once (never more than the pool has).
        self.limit = len(uuids) if limit is None else min(limit, len(uuids))
        self.index_of = indices or {}
        self.held = {}
        #: Why the last `take` returned None (for the job file).
        self.reason = None

    def take(self, count):
        """`count` free GPUs (their leases now held), or None (`reason`
        says why)."""
        if count == 0:
            return []
        with self._locked():
            busy = busy_uuids() if self.wait_busy else set()
            held = self._held_by_runners() | set(self.held)
            free = [uuid for uuid in self.uuids if uuid not in busy and uuid not in held]
            if len(held) + count > self.limit or len(free) < count:
                self.reason = self._why(count, free, held, busy)
                return None
            taken = []
            for uuid in free:
                if len(taken) == count:
                    break
                lease = Lease(StdlibFilesystem(), _ProcessOwners(),
                              os.path.join(self.directory, "slots", SLOT_PREFIX + uuid),
                              self.owner)
                if lease.try_acquire():
                    self.held[uuid] = lease
                    taken.append(uuid)
            if len(taken) < count:
                self.give_back(taken)
                self.reason = self._why(count, [], held, busy)
                return None
        self.reason = None
        return taken

    def _locked(self):
        """The lock every runner of this user takes to count and take slots."""
        path = os.path.join(self.directory, "slots", "take.lock")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return _FileLock(path)

    def _held_by_runners(self):
        """UUIDs of the GPUs whose lease a live runner holds (this one too)."""
        held = set()
        try:
            names = os.listdir(os.path.join(self.directory, "slots"))
        except FileNotFoundError:
            return held
        for name in names:
            if not (name.startswith(SLOT_PREFIX) and name.endswith(LOCK_SUFFIX)):
                continue
            data = StdlibFilesystem().read(
                os.path.join(self.directory, "slots", name, "owner.json"))
            if data is None:
                continue            # being created: counted by its own runner's lock
            if _ProcessOwners().is_alive(json.loads(data.decode())):
                held.add(name[len(SLOT_PREFIX):-len(LOCK_SUFFIX)])
        return held

    def _why(self, count, free, held, busy):
        room = max(self.limit - len(held), 0)
        parts = ["limit {}".format(self.limit)]
        if held:
            parts.append("{} held by your experiments".format(len(held)))
        outside = sorted(self.index_of.get(uuid, uuid) for uuid in busy if uuid in self.uuids
                         and uuid not in held)
        if outside:
            parts.append("GPU(s) {} busy outside naavik".format(
                ", ".join(str(index) for index in outside)))
        return "waiting for GPUs: needs {}; {} free now ({})".format(
            count, min(len(free), room), "; ".join(parts))

    def give_back(self, uuids):
        for uuid in uuids:
            lease = self.held.pop(uuid, None)
            if lease is not None:
                lease.release()


class _FileLock:
    """An exclusive `flock` on a file, held for a `with` block."""

    def __init__(self, path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.handle = open(self.path, "a")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
        return False


class _ProcessOwners:
    """Slot owners are runners on this host."""

    def is_alive(self, owner):
        start = process_start_time(owner.get("pid"))
        return start is not None and start == owner.get("start_time")


# The runner ------------------------------------------------------------------


class Runner:
    def __init__(self, script, after, directory=None):
        self.script = script
        self.after = list(after)
        self.directory = directory or state_dir()
        self.ref = job_ref(os.getpid())
        self.directives = read_directives(script)
        d = self.directives
        self.elements = [int(item) for item in d["elements"].split(",")]
        self.max_parallel = int(d.get("max-parallel") or 0) or len(self.elements)
        self.gpus = int(d.get("gpus", "0"))
        self.walltime = int(d["walltime"]) if d.get("walltime") else None
        self.lead = int(d["signal-lead"]) if d.get("signal-lead") else None
        pool = d.get("gpu-pool", "")
        #: The pool's indices, or None for every GPU of the machine (`all`).
        self.pool = None if pool == "all" else [int(item) for item in pool.split(",") if item]
        #: The most GPUs held at once by this user's runners (None: the pool's size).
        self.gpu_limit = int(d["gpu-limit"]) if d.get("gpu-limit") else None
        self.state = {
            "ref": self.ref, "job_name": d["job-name"], "script": script,
            "after": self.after, "elements": {str(i): PENDING for i in self.elements},
            "exit": {}, "final": False, "error": None,
            # element -> why it has not started yet
            "waiting": {},
        }
        self.running = {}        # index -> (process, gpus, started, signalled)
        self.stopping = None     # when a cancel of the whole job arrived
        self.slots = None

    # State --------------------------------------------------------------

    def save(self):
        StdlibFilesystem().write(_job_file(self.directory, self.ref),
                                 json.dumps(self.state, sort_keys=True).encode())

    def set(self, index, value, exit_status=None):
        self.state["elements"][str(index)] = value
        if exit_status is not None:
            self.state["exit"][str(index)] = exit_status
        self.save()

    def set_waiting(self, reasons):
        """Why the elements not started yet wait ({index: reason})."""
        reasons = {str(index): reason for index, reason in reasons.items()}
        if reasons != self.state["waiting"]:
            self.state["waiting"] = reasons
            self.save()

    def finish(self, error=None):
        for index, value in self.state["elements"].items():
            if value not in FINAL:
                self.state["elements"][index] = CANCELLED
        self.state["final"], self.state["error"] = True, error
        self.state["waiting"] = {}
        self.save()

    def fail(self, message):
        """Fail every element without running it, the reason in its log
        (where `naavik logs` and `naavik run` show it)."""
        for index in self.elements:
            path = self.directives["log"].format(job=self.ref, index=index)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as log:
                log.write("[host runner] {}\n".format(message))
            self.state["elements"][str(index)] = FAILED
        self.finish(message)

    # Steps ----------------------------------------------------------------

    def dependencies_ok(self):
        """Wait for every job this one runs after; False when one did not
        complete entirely (this job is then cancelled, like Slurm's
        `--kill-on-invalid-dep`)."""
        for ref in self.after:
            self.set_waiting({index: "waiting for job {} to finish first".format(ref)
                              for index in self.elements})
            while True:
                if self.stopping is not None:
                    return False
                data = read_job(self.directory, ref)
                if data and data.get("final"):
                    if any(value != COMPLETED for value in data["elements"].values()):
                        return False
                    break
                if not ref_alive(ref):
                    return False              # died without finishing
                time.sleep(POLL)
        return True

    def cancel_requests(self):
        path = _job_file(self.directory, self.ref) + ".cancel"
        data = StdlibFilesystem().read(path)
        return {int(item) for item in data.decode().split()} if data else set()

    def start(self, index, uuids):
        directory = os.path.dirname(self.directives["log"].format(job=self.ref, index=index))
        os.makedirs(directory, exist_ok=True)
        log = open(self.directives["log"].format(job=self.ref, index=index), "ab")
        env = dict(os.environ, TF_HOST_TASK_INDEX=str(index), TF_HOST_JOB=self.ref)
        # Only its own GPUs: none at all for an element that asked for none.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids or [])
        process = subprocess.Popen(["bash", self.script], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, env=env,
                                   cwd=self.directives.get("chdir") or None,
                                   start_new_session=True)
        log.close()
        self.running[index] = (process, uuids or [], time.time(), False)
        self.set(index, RUNNING)

    def stop(self, index, kill=False):
        process = self.running[index][0]
        try:
            if kill:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.send_signal(signal.SIGUSR1)   # the wrapper's advance signal
        except (ProcessLookupError, PermissionError):
            pass

    def cancel(self):
        """SIGTERM: stop the whole job, cooperatively first (USR1), killing
        what is still running after the signal lead."""
        if self.stopping is None:
            self.stopping = time.time()
            for index in list(self.running):
                self.stop(index)

    def run(self):
        self.save()
        if not self.dependencies_ok():
            self.finish("a job it runs after did not complete")
            return 1
        self.set_waiting({})
        try:
            uuids = gpu_uuids(self.pool) if self.gpus else []
            indices = gpu_indices() if self.gpus else {}
        except (OSError, RuntimeError) as error:
            self.finish("cannot use the GPU pool: {}".format(error))
            return 1
        self.slots = Slots(self.directory, uuids,
                           self.directives.get("busy-gpus", "wait") == "wait",
                           {"host": socket.gethostname(), "pid": os.getpid(),
                            "start_time": process_start_time(os.getpid())},
                           limit=self.gpu_limit, indices=indices)
        if self.gpus > self.slots.limit:
            if self.pool is not None:
                written = "[{}]".format(", ".join(str(index) for index in self.pool))
            else:
                written = "all" if self.gpu_limit is None else str(self.gpu_limit)
            self.fail("requests {} GPU(s) per experiment, but at most {} may be in use at "
                      "once on this host (scheduler.gpus: {}; the machine has {})".format(
                          self.gpus, self.slots.limit, written, len(indices)))
            return 1
        waiting = list(self.elements)
        try:
            while waiting or self.running:
                cancelled = self.cancel_requests()
                for index in [i for i in waiting if i in cancelled]:
                    waiting.remove(index)
                    self.set(index, CANCELLED)
                for index in [i for i in self.running if i in cancelled]:
                    self.stop(index)
                if self.stopping is not None:
                    for index in waiting:
                        self.set(index, CANCELLED)
                    waiting = []
                    if time.time() - self.stopping >= (self.lead or 0):
                        for index in list(self.running):
                            self.stop(index, kill=True)
                reason = None
                while waiting and len(self.running) < self.max_parallel:
                    uuids = self.slots.take(self.gpus) if self.gpus else None
                    if self.gpus and uuids is None:
                        reason = self.slots.reason    # no free GPU yet
                        break
                    self.start(waiting.pop(0), uuids)
                if waiting and reason is None:
                    reason = ("waiting: at most {} experiment(s) of this job run at "
                              "once".format(self.max_parallel))
                self.set_waiting({index: reason for index in waiting})
                self.watch()
                time.sleep(POLL)
        finally:
            for index in list(self.running):
                self.stop(index, kill=True)
            self.slots.give_back(list(self.slots.held))
        self.finish()
        return 0

    def watch(self):
        now = time.time()
        for index, (process, uuids, started, signalled) in list(self.running.items()):
            status = process.poll()
            if status is not None:
                del self.running[index]
                self.slots.give_back(uuids)
                timed_out = self.walltime is not None and now - started >= self.walltime
                if status == 0:
                    value = COMPLETED
                elif timed_out:
                    value = TIMEOUT
                elif self.stopping is not None or index in self.cancel_requests():
                    value = CANCELLED
                else:
                    value = FAILED
                self.set(index, value, status)
                continue
            if self.walltime is None:
                continue
            if now - started >= self.walltime:
                self.stop(index, kill=True)
            elif (not signalled and self.lead is not None
                  and now - started >= self.walltime - self.lead):
                self.stop(index)
                self.running[index] = (process, uuids, started, True)


# Commands ------------------------------------------------------------------------


def live_states(refs=None):
    """{ref: state, ref_index: state} of the non-final elements of live
    runners (all of them, or `refs`): what is still queued or running."""
    found = {}
    for state in reversed(state_dirs()):         # the current name wins a clash
        try:
            for name in os.listdir(os.path.join(state, "jobs")):
                if name.endswith(".json"):
                    found[name[:-len(".json")]] = state
        except FileNotFoundError:
            continue
    wanted = None if refs is None else set(refs)
    states = {}
    for ref in sorted(found):
        if wanted is not None and ref not in wanted:
            continue
        data = read_job(found[ref], ref)
        if not data or data.get("final") or not ref_alive(ref):
            continue
        values = [value for value in data["elements"].values() if value not in FINAL]
        if not values:
            continue
        states[ref] = RUNNING if RUNNING in values else PENDING
        for index, value in data["elements"].items():
            if value not in FINAL:
                states["{}_{}".format(ref, index)] = value
    return states


def live_details(refs=None):
    """{ref_index: why it waits} of the pending elements of live runners
    that say (jobs written before runners recorded it say nothing)."""
    states = live_states(refs)
    details = {}
    for ref in {key.split("_", 1)[0] for key in states}:
        data = read_job(job_dir(ref), ref) or {}
        for index, reason in (data.get("waiting") or {}).items():
            key = "{}_{}".format(ref, index)
            if reason and states.get(key) == PENDING:
                details[key] = reason
    return details


def request_cancel(ref, indices):
    """Ask a runner to stop: everything (SIGTERM), or only `indices`."""
    if not ref_alive(ref):
        return "not running"
    if indices:
        path = _job_file(job_dir(ref), ref) + ".cancel"
        existing = StdlibFilesystem().read(path)
        known = existing.decode().split() if existing else []
        StdlibFilesystem().write(path, " ".join(known + [str(i) for i in indices]).encode())
        return "cancelling {} element(s)".format(len(indices))
    os.kill(parse_ref(ref)[0], signal.SIGTERM)
    return "cancelling"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv else ""
    if command == "run":
        script, after = argv[0], []
        if "--after" in argv:
            after = argv[argv.index("--after") + 1:]
        runner = Runner(script, after)
        signal.signal(signal.SIGTERM, lambda signum, frame: runner.cancel())
        return runner.run()
    if command == "states":
        # The launcher passes this program itself (`python3 -c`), so it reads
        # the shape it prints, whatever runner wrote the job files.
        print(json.dumps({"states": live_states(argv or None),
                          "details": live_details(argv or None)}, sort_keys=True))
        return 0
    if command == "cancel":
        print(request_cancel(argv[0], [int(i) for i in argv[1:]]))
        return 0
    sys.stderr.write("usage: host_runner.py run SCRIPT [--after REF ...] | states [REF ...] | "
                     "cancel REF [INDEX ...]\n")
    return 2

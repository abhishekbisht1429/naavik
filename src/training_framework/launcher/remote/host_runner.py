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
at the walltime. The job's state is a JSON file, rewritten on every change:

    <state dir>/jobs/<ref>.json

where the state directory is per user and per host (like the scheduler's own
queue): `$XDG_STATE_HOME/training-framework/host/<hostname>`. The job's
reference is `h<pid>.<start time>` of the runner, checkable on this host.

`python3 host_runner.py states [REF ...]` prints the live states (JSON), and
`python3 host_runner.py cancel REF [INDEX ...]` asks a runner to stop.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time

from training_framework.launcher.remote.runtime import StdlibFilesystem, process_start_time
from training_framework.launcher.remote.lease import Lease

POLL = 0.5
DIRECTIVE = "#TFHOST "
#: Element states. Final ones are dropped from the live view, like jobs that
#: left a scheduler's queue.
PENDING, RUNNING = "pending", "running"
COMPLETED, FAILED, TIMEOUT, CANCELLED = "completed", "failed", "timeout", "cancelled"
FINAL = (COMPLETED, FAILED, TIMEOUT, CANCELLED)


def state_dir():
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state")
    return os.path.join(base, "training-framework", "host", socket.gethostname())


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


def gpu_uuids(pool):
    """The UUIDs of the pool's GPUs (nvidia-smi indices), in pool order."""
    by_index = {}
    for line in _nvidia_smi("gpu=index,uuid"):
        index, _, uuid = line.partition(",")
        by_index[int(index)] = uuid.strip()
    missing = [index for index in pool if index not in by_index]
    if missing:
        raise RuntimeError("GPU(s) {} of the pool do not exist here (have {})".format(
            missing, sorted(by_index)))
    return [by_index[index] for index in pool]


def busy_uuids():
    """GPUs with any compute process (anyone's)."""
    return {line.split(",")[0].strip() for line in _nvidia_smi("compute-apps=gpu_uuid")}


class Slots:
    """Host-wide GPU slots: one lease per GPU, held by the runner using it,
    so jobs of every batch on this host share the pool without overlap."""

    def __init__(self, directory, uuids, wait_busy, owner):
        self.directory = directory
        self.uuids = uuids
        self.wait_busy = wait_busy
        self.owner = owner
        self.held = {}

    def take(self, count):
        """`count` free GPUs (their leases now held), or None."""
        if count == 0:
            return []
        busy = busy_uuids() if self.wait_busy else set()
        taken = []
        for uuid in self.uuids:
            if len(taken) == count:
                break
            if uuid in busy or uuid in self.held:
                continue
            lease = Lease(StdlibFilesystem(), _ProcessOwners(),
                          os.path.join(self.directory, "slots", "gpu-" + uuid), self.owner)
            if lease.try_acquire():
                self.held[uuid] = lease
                taken.append(uuid)
        if len(taken) < count:
            self.give_back(taken)
            return None
        return taken

    def give_back(self, uuids):
        for uuid in uuids:
            lease = self.held.pop(uuid, None)
            if lease is not None:
                lease.release()


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
        self.pool = [int(item) for item in d.get("gpu-pool", "").split(",") if item]
        self.state = {
            "ref": self.ref, "job_name": d["job-name"], "script": script,
            "after": self.after, "elements": {str(i): PENDING for i in self.elements},
            "exit": {}, "final": False, "error": None,
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

    def finish(self, error=None):
        for index, value in self.state["elements"].items():
            if value not in FINAL:
                self.state["elements"][index] = CANCELLED
        self.state["final"], self.state["error"] = True, error
        self.save()

    # Steps ----------------------------------------------------------------

    def dependencies_ok(self):
        """Wait for every job this one runs after; False when one did not
        complete entirely (this job is then cancelled, like Slurm's
        `--kill-on-invalid-dep`)."""
        for ref in self.after:
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
        if uuids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
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
        try:
            uuids = gpu_uuids(self.pool) if self.gpus else []
        except (OSError, RuntimeError) as error:
            self.finish("cannot use the GPU pool: {}".format(error))
            return 1
        self.slots = Slots(self.directory, uuids,
                           self.directives.get("busy-gpus", "wait") == "wait",
                           {"host": socket.gethostname(), "pid": os.getpid(),
                            "start_time": process_start_time(os.getpid())})
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
                while waiting and len(self.running) < self.max_parallel:
                    uuids = self.slots.take(self.gpus) if self.gpus else None
                    if self.gpus and uuids is None:
                        break                         # no free GPU yet
                    self.start(waiting.pop(0), uuids)
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
    directory = os.path.join(state_dir(), "jobs")
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return {}
    wanted = None if refs is None else set(refs)
    states = {}
    for name in names:
        if not name.endswith(".json"):
            continue
        ref = name[:-len(".json")]
        if wanted is not None and ref not in wanted:
            continue
        data = read_job(state_dir(), ref)
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


def request_cancel(ref, indices):
    """Ask a runner to stop: everything (SIGTERM), or only `indices`."""
    if not ref_alive(ref):
        return "not running"
    if indices:
        path = _job_file(state_dir(), ref) + ".cancel"
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
        print(json.dumps(live_states(argv or None), sort_keys=True))
        return 0
    if command == "cancel":
        print(request_cancel(argv[0], [int(i) for i in argv[1:]]))
        return 0
    sys.stderr.write("usage: host_runner.py run SCRIPT [--after REF ...] | states [REF ...] | "
                     "cancel REF [INDEX ...]\n")
    return 2

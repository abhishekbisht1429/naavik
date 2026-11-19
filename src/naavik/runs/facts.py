"""What a run record says about the code, the environment, and a worker's
device. Each function gathers what it can and reports what it cannot in the
result, never by raising: recording must not stop a run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import socket
from pathlib import Path

#: `TRAINING_FRAMEWORK_RUNS_SNAPSHOT=0`: never make snapshot commits for
#: runs of a dirty working tree (the record then says the tree was dirty).
ENV_SNAPSHOT = "TRAINING_FRAMEWORK_RUNS_SNAPSHOT"
#: Where a run's snapshot commit is kept in the repository.
RUN_REFS = "refs/naavik/runs"
DEFAULT_SNAPSHOT_LIMIT = 50 * 1024 ** 2


def environment_hash() -> str:
    """Hash of the installed distributions (what `pip freeze` would list)."""
    entries = sorted({
        f"{(dist.metadata['Name'] or '').lower()}=={dist.version}"
        for dist in importlib.metadata.distributions()
    })
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def environment_facts() -> dict:
    facts = {"python": platform.python_version(), "packages_sha256": environment_hash()}
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda"] = torch.version.cuda
    except ImportError:                   # recorded as missing, not an error
        facts["torch"] = None
    facts["framework"] = framework_facts()
    return facts


def framework_facts() -> dict:
    """The framework this process imported: where from, the installed
    distribution's version only when that installation is the imported
    code (an older install beside a source checkout says nothing about
    it), and the checkout's commit when it is one (never snapshotted)."""
    import naavik

    location = Path(naavik.__file__).resolve().parent
    facts = {"location": str(location), "version": None}
    from naavik.distribution import distribution_of

    distribution = distribution_of(location)
    if distribution is not None:
        facts["version"] = distribution.version
    from naavik.launcher.errors import HpcError
    from naavik.launcher.ports.repository import GitRepository

    try:
        repository = GitRepository(location)
        facts.update(sha=repository.resolve("HEAD"), dirty=not repository.is_clean())
    except (HpcError, OSError):
        pass
    return facts


def package_directory(package: str | None) -> Path | None:
    """Where a (components) package lives, without importing it."""
    if not package:
        return None
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
        return None
    return Path(spec.origin).parent


def code_identity(start: Path, run_id: str, *, snapshot: bool | None = None,
                  limit: int = DEFAULT_SNAPSHOT_LIMIT) -> dict | None:
    """The code a run runs: the Git repository holding `start`, at HEAD when
    clean, else the working tree as a snapshot commit kept as
    `refs/naavik/runs/<run-id>`. None outside a repository; `error` when the
    tree is dirty but no snapshot could be made (the run goes on)."""
    from naavik.launcher.errors import HpcError
    from naavik.launcher.ports.repository import GitRepository

    try:
        repository = GitRepository(Path(start))
    except (HpcError, OSError):
        return None
    identity = {"repository": str(repository.root)}
    try:
        head = repository.resolve("HEAD")
    except HpcError as error:             # a repository without commits
        return {**identity, "sha": None, "error": str(error)}
    identity.update(sha=head, base_sha=None, snapshot=False)
    if repository.is_clean():
        return identity
    if snapshot is None:
        snapshot = os.environ.get(ENV_SNAPSHOT, "1") != "0"
    if not snapshot:
        return {**identity, "dirty": True,
                "error": f"the working tree has changes; {ENV_SNAPSHOT}=0 keeps no snapshot"}
    try:
        taken = repository.snapshot(limit)
        ref = repository.keep_ref(f"{RUN_REFS}/{run_id}", taken.sha)
    except HpcError as error:
        return {**identity, "dirty": True, "error": f"no snapshot of the working tree: {error}"}
    return {**identity, "sha": taken.sha, "base_sha": head, "snapshot": True, "ref": ref,
            "modified": taken.modified, "untracked": taken.untracked}


def device_facts(device, rank: int, world_size: int | None = None,
                 backend: str | None = None) -> dict:
    """A worker's device, as it sees it (CUDA indices are within its
    visible devices)."""
    facts = {"host": socket.gethostname(), "pid": os.getpid(), "device": str(device),
             "world_size": world_size, "backend": backend,
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if getattr(device, "type", None) == "cuda":
        try:
            import torch

            index = device.index if device.index is not None else torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            uuid = str(getattr(properties, "uuid", "")) or None
            # As nvidia-smi and CUDA_VISIBLE_DEVICES write it (`GPU-...`).
            if uuid and not uuid.startswith("GPU-"):
                uuid = f"GPU-{uuid}"
            facts.update(index=index, name=properties.name, uuid=uuid,
                         total_memory=properties.total_memory)
        except Exception as error:        # recorded, never raised
            facts["error"] = f"{type(error).__name__}: {error}"
    return facts

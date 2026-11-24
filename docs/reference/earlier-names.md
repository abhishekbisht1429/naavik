# Earlier names

[← Docs](../README.md)

Files written with these names are still read, with one warning per file
listing what to rename. Rename them by hand; an earlier name beside its
current one is an error. The current keys are described in [the hosts
file](hosts-file.md) and [mission files](mission-files.md).

## Files and commands

| Earlier | Current |
|---|---|
| `hpc.yaml`, `targets.yaml` | `hosts.yaml` |
| manifest, experiment file | mission file |
| batch | mission |
| group (in a mission file) | job |
| experiment (in a mission file) | recipe: "experiment" now means one run, the former task |
| task | experiment |
| `naavik batches`, `submit`, `probe-python` | `naavik missions`, `launch`, `show-site-interpreter` |
| `--target`, `--profile` / `--targets`, `--profiles` | `--host` / `--hosts-file` |
| `--task` (resume, fetch, cancel), `missions --tasks` | `--experiment`, `missions --experiments` |

## The hosts file

| Earlier | Current |
|---|---|
| `default` | `default_host` |
| `targets` / `profiles` | `hosts` |
| `ssh_host: X` / `local: true` | `connection: {ssh: X}` / `connection: local` |
| `scheduler: slurm` + `slurm: {...}` | `scheduler: {kind: slurm, ...}` |
| `scheduler: host` + `host: {...}` | `scheduler: {kind: builtin, ...}` |
| `slurm.defaults`, `slurm.partitions {gpu, cpu}` | `sbatch_options`, `partitions {gpu_jobs, cpu_jobs}` |
| `host.busy_gpus: wait\|use` | `when_gpus_busy: wait\|use_anyway` |
| `scheduler.gpus` left out: no GPUs | left out: `all`; write `gpus: []` for none |
| `roots {batches, envs, data}` | `directories {missions, python_environments, datasets}` |
| `hooks {common, setup, prepare, task}` | `shell_commands_before {everything, environment_build, data_preparation, experiments}` |
| `env {existing, python_source, site_python_identity, backend, conda_packages, conda_run, bootstrap_python}` | `python_environment {use_existing, interpreter_from, site_interpreter, kind, conda_packages, activate_with_conda_run, bootstrap_interpreter}` |
| `setup_location: job\|login` + `setup_resources` | `environment_build {on: compute_node\|login_node, allocation}` |
| `prepare_location: job\|login` | `data_preparation {on: compute_node\|login_node, allocation}` |
| `path_map` | `local_to_host_paths` |
| `upload_limit` | `confirm_uploads_over` |
| `worktree: commit\|snapshot`, `snapshot_limit` | `run_code_from: commit\|working_tree`, `working_tree_untracked_limit` |
| `data_modes: fetch\|upload\|present` | `dataset_delivery: download_on_host\|copy_from_here\|already_on_host` |

## Mission and project files

| Earlier | Current |
|---|---|
| `version`, `description`, `project` | `format_version`, `note`, `project_file` |
| a config file `sessions: [entry]` (one entry) | the entry's keys at the top of the file; a file is one session (several entries are refused) |
| `--override sessions[0].a.b=V` | `--override a.b=V` (the earlier form still works on an earlier-form file) |
| `base.session_index` | not needed: `0` is noted, anything else refused |
| `source {strategy: bundle\|fetch, ref, url}` | `code_transfer {method: git_bundle\|git_clone, commit, clone_url}` |
| `code {framework: dependency\|checkout, paths}` | `code {framework_from: installed\|this_repository, python_paths}` |
| `environment {requirements, python}` | `python_packages {requirements_file, python_version}` |
| `training {config, session, engine_options}` | `base {config_file, session_index, engine_options}` |
| `data: {sources: [...]}` | `data_sources: [...]` |
| source `target`, `upload`, `cluster`, `prepare` | `address`, `copy_from_here`, `already_on_host`, `prepare_on_host` |
| `version`, `inputs`, `resources` (in a source) | `content_version`, `rerun_when_changed`, `allocation` |
| `groups {G: {resources, max_parallel, stop_grace, experiments}}` | `jobs {J: {per_experiment, max_parallel_experiments, stop_grace_period, recipes}}` |
| experiment `config`, `matrix` | recipe `config_file`, `vary` |
| `run_defaults {resources, max_parallel, target}` | `quick_run_defaults {per_experiment, max_parallel_experiments, host}` |

## Components and session files

Each is read with a `DeprecationWarning`. A checkpoint that stored one of
the binding keys restores without one.

| Earlier | Current | What still works |
|---|---|---|
| top-level `component_bindings`, `aliases` | `role_bindings` | read; only one of the three may be configured |
| nested `component_bindings: {consumer: {role: target}}` | that component's own `dependency_bindings` | read; wiring one component in both places is an error |
| `dependencies_role_bindings` (in a component's entry) | `dependency_bindings` | read; one entry may not use both; an extension that writes the same wiring under the new name changes nothing |
| `fine_tuned_model` | `backbone_with_head` | builds and restores: same roles, same keys (without `pool`), same parameter names and saved state. A session saved as `fine_tuned_model` resumes under that name (the name is in every instance's checkpoint entry); moving it to `backbone_with_head` means a new session, or [importing](import-components.md) its parts into one |
| `Session.component_bindings`, `Session.component_aliases` | `Session.role_bindings` | deprecated properties |
| labelled `import_components` entries (with `resource`, `role` or `suffix`) | the keyed form | accepted; see [labelled imports](import-components.md#deprecated-labelled-imports) |

## The framework's own names

The framework was called `training-framework` (imported as
`training_framework`). Everything under the earlier name still works, with
one `FutureWarning`:

| Earlier | Current | What still works |
|---|---|---|
| package `training_framework` | `naavik` | imports, `components_package` strings, legacy single-file checkpoints, and `python -m training_framework...` commands recorded in earlier missions' job scripts (no warning there) |
| distribution `training-framework` | `naavik` | requirements may pin either name, once; an installed `training-framework` is found |
| `TRAINING_FRAMEWORK_CODE_CHECK`, `_RUNS`, `_RUNS_SNAPSHOT` | `NAAVIK_CODE_CHECK`, `NAAVIK_RUNS`, `NAAVIK_RUNS_SNAPSHOT` | read when the new one is not set; both set differently is an error |
| directories `~/.config/training-framework`, `~/.local/state/...`, `~/.local/share/...`, `~/.cache/...` | `.../naavik` | an earlier one is used where it is when no `naavik` one exists (nothing is moved); jobs of the `builtin` scheduler are found under either |
| job scripts running `training_framework.hpc.task` (execution protocol 1) | `naavik.launcher.task` (protocol 2) | a mission keeps the commands its plan recorded, so missions launched before the rename resume |

# Analysis Sessions

[← Docs](../README.md) · [Project README](../../README.md)

An analysis session drives a model recovered from a training checkpoint through
the same component lifecycle a training session uses, with its own registry and
defaults. This page covers writing an analysis step, configuring the source
checkpoint, and running the result.

It assumes you have read [Resources, hooks, and steps](01-resources-hooks-steps.md)
and [Configuration](04-configuration.md).

## Writing an analysis step

Analysis sessions use the same resource, hook, step, context, and iteration
lifecycle as training sessions. Their effective registry combines shared
components with components registered using `session_type="analysis"`:

```python
from naavik.components import Step, requires_resource, step


@step("report", session_type="analysis")
@requires_resource("trained_model")
class ReportStep(Step):
    def __init__(self, config):
        self.output_path = config["output_path"]

    def run(self, session):
        model = self.get_dependency("trained_model").model
        # Analyze the model and write the configured report.
        ...
```

The built-in [generic steps](../reference/generic-steps.md) work in analysis
sessions too (see [Building an iteration](03-building-an-iteration.md) for how
they fit together): `load_batch` names the batch, and `forward`
calls `trained_model.model` without gradients, so a custom step can start
from the model's outputs:

```yaml
load_batch: {fields: [inputs, targets]}
forward: {args: [inputs], outputs: logits}
report: {output_path: ./report.json}   # a step that consumes the output
```

The `ReportStep` above accesses the model directly. To consume `logits`
instead, declare `@reads("logits")` and accept `logits` in its `run` method.

## Configuring the source checkpoint

An analysis configuration is a session file like a training one. The shared
`trained_model` resource and analysis logger are default roots. Because the
trained-model resource has no default checkpoint, configure its checkpoint
path in the component mapping:

```yaml
session_type: analysis

session_config:
  rng_seed: 42
  sessions_dir: ./analysis-runs
  max_iterations: 1
  components_package: my_project.analysis_components
  device: cpu
  show_execution_graph: true

trained_model:
  model_checkpoint_path: ./sessions/session_.../checkpoints/<checkpoint-name>

report:
  output_path: ./analysis-runs/report.json
```

To analyse every checkpoint of a training mission instead of one, a mission
file's recipe can take them from it: one experiment per checkpoint, its path
written at `trained_model.model_checkpoint_path`, compared in summaries by
the training settings behind it. See
[`from_mission`](08-missions.md#analysing-another-missions-checkpoints-from_mission).

`max_iterations: 1` runs a single iteration. To go through the whole analysis
dataset once, give `epochs: 1` in its place. The session then runs as many
iterations as the analysis `data_manager` has batches
([details](04-configuration.md#run-length-in-epochs)).

## Running an analysis

For direct execution, construct the concrete analysis subclass:

```python
from naavik.session import AnalysisSession


session = AnalysisSession(analysis_config)
```

Run it like any config:

```bash
naavik run my_project/analysis.yaml
```

To analyse every checkpoint of a mission's experiments at once, a mission
file's recipe takes them with
[`from_mission`](08-missions.md#analysing-another-missions-checkpoints-from_mission).

The configured `trained_model.model_checkpoint_path` must identify a framework
`TrainingSession` checkpoint, not a standalone model state dictionary. The
source session must expose a model through the `model` resource role, directly
or through a role binding, and that resource must
provide `to(device)` and `eval()`. During analysis setup, `trained_model` loads
the source session on CPU, moves the recovered model -- and every part it was
wired to, whether or not it holds the part as a submodule -- to the analysis
device,
places it in evaluation mode, and exposes it as `.model` -- a component that
declares `@requires_resource("trained_model")` reads it with
`self.get_dependency("trained_model").model`. Gradients remain available to
custom analysis steps for attribution, while the built-in `forward` step
defaults to `no_grad: true`; set it to `false` when its output needs gradients.
Analysis does not activate the training
checkpointer by default.

`layer_inspector` is an available opt-in building block for analysis `Step`s
that need per-layer activations — it automates layer discovery and
forward-hook lifecycle management, leaving interpretation of the captured
tensors (heatmaps or anything else) to the `Step`. See [`layer_inspector` in the
built-in component reference](../reference/builtin-components.md#layer_inspector)
for its configuration and API.

`trained_model` rebuilds only the model and what it was wired to, so a
component of the training run that no longer builds does not block analysis.
A single-file checkpoint from before 0.5.0 is read with unrestricted Python
deserialization; load one only from a trusted source.

---

**Reference:** the analysis built-ins — `trained_model`, the analysis
`data_manager`, and `layer_inspector` — are documented in the
[built-in component reference](../reference/builtin-components.md#analysis-built-ins).

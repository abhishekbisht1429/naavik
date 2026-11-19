"""Built-in component implementations registered by the component package."""

from naavik.components.builtin.checkpointing import Checkpointer
from naavik.components.builtin.analysis import AnalysisLogger
from naavik.components.builtin.computation import (
    AnalysisForward,
    Compute,
    Forward,
    LoadBatch,
)
from naavik.components.builtin.data import (
    AnalysisDataManager,
    DataManager,
)
from naavik.components.builtin.diagram import ModelDiagram
from naavik.components.builtin.distributed import DDPResource
from naavik.components.builtin.fine_tuning import (
    BackboneWithHead,
    FineTunedModel,
    ModulePart,
)
from naavik.components.builtin.layer_inspection import (
    LayerCapture,
    LayerInspector,
)
from naavik.components.builtin.model import TrainedModel
from naavik.components.builtin.observability import (
    Logger,
    Tensorboard,
    TensorboardSummaries,
    Timer,
)
from naavik.components.builtin.optimization import (
    Backward,
    ClipGradients,
    ForwardContext,
    FreezeGradients,
    GradientProcessor,
    OptimizerResource,
    OptimizerStep,
)
from naavik.components.builtin.profiling import Profiler
from naavik.components.builtin.self_supervised import (
    DINOLoss,
    EMATeacher,
    EMAUpdate,
    MultiView,
    ProjectionHead,
)
from naavik.components.builtin.transformer import (
    AttentionPooledEncoder,
    PatchTransformer,
    PooledPatchTransformer,
)

__all__ = [
    "AnalysisDataManager",
    "AnalysisForward",
    "AnalysisLogger",
    "AttentionPooledEncoder",
    "BackboneWithHead",
    "Backward",
    "Checkpointer",
    "ClipGradients",
    "Compute",
    "DataManager",
    "DDPResource",
    "DINOLoss",
    "EMATeacher",
    "EMAUpdate",
    "FineTunedModel",
    "Forward",
    "ForwardContext",
    "FreezeGradients",
    "GradientProcessor",
    "LayerCapture",
    "LayerInspector",
    "LoadBatch",
    "Logger",
    "ModelDiagram",
    "ModulePart",
    "MultiView",
    "OptimizerResource",
    "OptimizerStep",
    "PatchTransformer",
    "PooledPatchTransformer",
    "Profiler",
    "ProjectionHead",
    "Tensorboard",
    "TensorboardSummaries",
    "Timer",
    "TrainedModel",
]

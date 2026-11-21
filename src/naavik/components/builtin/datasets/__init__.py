"""Image datasets from torchvision, as resources for the ``dataset`` role.

Optional: needs ``pip install naavik[vision]``, and is not
imported with the other built-ins. Import it from a module of your
``components_package`` so that every process registers it::

    import naavik.components.builtin.datasets  # noqa: F401
"""

try:
    import torchvision  # noqa: F401
except ImportError as error:
    raise ImportError(
        "naavik.components.builtin.datasets needs torchvision; "
        "install it with `pip install naavik[vision]`"
    ) from error

from naavik.components.builtin.datasets.base import (
    TorchvisionDataset,
    TorchvisionDatasetConfig,
)
from naavik.components.builtin.datasets.cifar10 import CIFAR10
from naavik.components.builtin.datasets.flowers102 import (
    Flowers102,
)
from naavik.components.builtin.datasets.imagenet import ImageNet
from naavik.components.builtin.datasets.inaturalist import (
    INaturalist,
)
from naavik.components.builtin.datasets.normalization import (
    NormalizeImages,
)
from naavik.components.builtin.datasets.stanford_cars import (
    StanfordCars,
)

__all__ = [
    "CIFAR10",
    "Flowers102",
    "ImageNet",
    "INaturalist",
    "NormalizeImages",
    "StanfordCars",
    "TorchvisionDataset",
    "TorchvisionDatasetConfig",
]

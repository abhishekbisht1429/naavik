from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from torchvision import datasets
from torchvision.transforms import v2

from training_framework.components import resource
from training_framework.components.builtin.datasets.base import (
    TorchvisionDataset,
    TorchvisionDatasetConfig,
    missing_files,
)
from training_framework.data import DataRequirement

_NATIVE_SIZE = 32


@dataclass
class CIFAR10Config(TorchvisionDatasetConfig):
    SPLITS: ClassVar[tuple[str, ...]] = ("train", "test")


@resource("cifar10")
class CIFAR10(TorchvisionDataset):
    """CIFAR-10: 32x32 images of 10 classes (50k train, 10k test).

    The ``train`` preset pads and crops at the native 32 px and flips; both
    presets resize afterwards when `image_size` asks for larger images.
    """

    config_schema = CIFAR10Config
    IMAGE_SIZE = _NATIVE_SIZE

    # Both splits are one download into `root`.
    @classmethod
    def data_requirements(cls, config) -> tuple[DataRequirement, ...]:
        return (DataRequirement("cifar10", "root", "1", size_hint=170_498_071),)

    @classmethod
    def fetch_data(cls, config, requirement, path) -> None:
        datasets.CIFAR10(root=path, train=True, download=True)

    @classmethod
    def verify_data(cls, config, requirement, path) -> list[str]:
        return missing_files(path, [f"cifar-10-batches-py/{name}" for name in (
            "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5",
            "test_batch", "batches.meta")])

    def _build_source(self) -> Any:
        return datasets.CIFAR10(
            root=self._cfg.root,
            train=self._cfg.split == "train",
            download=self._cfg.download,
        )

    def train_transform(self, size: int) -> list[Callable[[Any], Any]]:
        return [
            v2.RandomCrop(_NATIVE_SIZE, padding=4),
            v2.RandomHorizontalFlip(),
            *_resized(size),
        ]

    def eval_transform(self, size: int) -> list[Callable[[Any], Any]]:
        return _resized(size)


def _resized(size: int) -> list[Callable[[Any], Any]]:
    return [] if size == _NATIVE_SIZE else [v2.Resize((size, size))]

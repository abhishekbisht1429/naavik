from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from torchvision import datasets

from training_framework.components import resource
from training_framework.components.builtin.datasets.base import (
    TorchvisionDataset,
    TorchvisionDatasetConfig,
    missing_files,
)
from training_framework.data import DataRequirement


@dataclass
class Flowers102Config(TorchvisionDatasetConfig):
    SPLITS: ClassVar[tuple[str, ...]] = ("train", "val", "test")


@resource("flowers102")
class Flowers102(TorchvisionDataset):
    """Oxford Flowers-102: 102 flower classes (1020 train, 1020 val, 6149
    test). Reading it needs scipy."""

    config_schema = Flowers102Config

    # Every split is one download into `root`.
    @classmethod
    def data_requirements(cls, config) -> tuple[DataRequirement, ...]:
        return (DataRequirement("flowers102", "root", "1", size_hint=344_862_509),)

    @classmethod
    def fetch_data(cls, config, requirement, path) -> None:
        datasets.Flowers102(root=path, split="train", download=True)

    @classmethod
    def verify_data(cls, config, requirement, path) -> list[str]:
        return missing_files(path, ["flowers-102/jpg/", "flowers-102/imagelabels.mat",
                                    "flowers-102/setid.mat"])

    def _build_source(self) -> Any:
        return datasets.Flowers102(
            root=self._cfg.root,
            split=self._cfg.split,
            download=self._cfg.download,
        )

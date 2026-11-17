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
class INaturalistConfig(TorchvisionDatasetConfig):
    #: torchvision's versions; each is one split, read from `root/<split>`.
    SPLITS: ClassVar[tuple[str, ...]] = (
        "2017",
        "2018",
        "2019",
        "2021_train",
        "2021_train_mini",
        "2021_valid",
    )


@resource("inaturalist")
class INaturalist(TorchvisionDataset):
    """iNaturalist, labelled by species (torchvision's ``full`` category)."""

    config_schema = INaturalistConfig

    # Each version (split) is its own download, into `root/<split>`.
    @classmethod
    def data_requirements(cls, config) -> tuple[DataRequirement, ...]:
        return (DataRequirement("inaturalist", "root", "1", key={"split": config.split},
                                subpath=config.split),)

    @classmethod
    def fetch_data(cls, config, requirement, path) -> None:
        datasets.INaturalist(root=path, version=config.split, target_type="full",
                             download=True)

    @classmethod
    def verify_data(cls, config, requirement, path) -> list[str]:
        return missing_files(path, [f"{config.split}/"])

    def _build_source(self) -> Any:
        return datasets.INaturalist(
            root=self._cfg.root,
            version=self._cfg.split,
            target_type="full",
            download=self._cfg.download,
        )

    def _find_labels(self) -> list[str]:
        return list(self._get_source().all_categories)

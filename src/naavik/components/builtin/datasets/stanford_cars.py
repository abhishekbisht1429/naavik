from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from torchvision import datasets

from naavik.components import resource
from naavik.components.builtin.datasets.base import (
    TorchvisionDataset,
    TorchvisionDatasetConfig,
    missing_files,
)
from naavik.data import DataError, DataRequirement


@dataclass
class StanfordCarsConfig(TorchvisionDatasetConfig):
    SPLITS: ClassVar[tuple[str, ...]] = ("train", "test")


@resource("stanford_cars")
class StanfordCars(TorchvisionDataset):
    """Stanford Cars: 196 car models (8144 train, 8041 test). Reading it
    needs scipy; torchvision can no longer download it, so place the files
    under `root` by hand."""

    config_schema = StanfordCarsConfig

    # torchvision can no longer download it: placed by hand, only verified.
    @classmethod
    def data_requirements(cls, config) -> tuple[DataRequirement, ...]:
        return (DataRequirement("stanford_cars", "root", "1", key={"split": config.split},
                                fetchable=False),)

    @classmethod
    def fetch_data(cls, config, requirement, path) -> None:
        raise DataError("Stanford Cars cannot be downloaded any more; place it under root")

    @classmethod
    def verify_data(cls, config, requirement, path) -> list[str]:
        if config.split == "train":
            return missing_files(path, ["stanford_cars/cars_train/",
                                        "stanford_cars/devkit/cars_train_annos.mat"])
        return missing_files(path, ["stanford_cars/cars_test/",
                                    "stanford_cars/cars_test_annos_withlabels.mat"])

    def _build_source(self) -> Any:
        return datasets.StanfordCars(
            root=self._cfg.root,
            split=self._cfg.split,
            download=self._cfg.download,
        )

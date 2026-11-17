"""The built-in datasets declare their data (nothing is downloaded here)."""

from __future__ import annotations

import pytest

from training_framework.components.builtin.datasets.cifar10 import CIFAR10
from training_framework.components.builtin.datasets.flowers102 import Flowers102
from training_framework.components.builtin.datasets.imagenet import ImageNet
from training_framework.components.builtin.datasets.inaturalist import INaturalist
from training_framework.components.builtin.datasets.stanford_cars import StanfordCars
from training_framework.data import DataError, ensure, requirement_key, requirements_of


def declared(cls, split, root="/data/x"):
    (item,) = requirements_of(cls, {"root": root, "split": split, "transform": "eval"})
    return item


@pytest.mark.parametrize("cls, splits", [(CIFAR10, ("train", "test")),
                                         (Flowers102, ("train", "val", "test"))])
def test_one_download_serves_every_split(cls, splits):
    items = [declared(cls, split) for split in splits]
    assert len({requirement_key(item.requirement) for item in items}) == 1
    assert {item.location for item in items} == {"/data/x"}
    assert all(item.requirement.fetchable for item in items)


def test_inaturalist_versions_are_separate_downloads_in_their_own_folders():
    a, b = declared(INaturalist, "2021_valid"), declared(INaturalist, "2021_train_mini")
    assert requirement_key(a.requirement) != requirement_key(b.requirement)
    assert (a.location, b.location) == ("/data/x/2021_valid", "/data/x/2021_train_mini")


@pytest.mark.parametrize("cls, split", [(ImageNet, "val"), (StanfordCars, "test")])
def test_data_obtained_by_hand_is_not_fetchable(cls, split, tmp_path):
    item = declared(cls, split, root=str(tmp_path))
    assert not item.requirement.fetchable
    with pytest.raises(DataError, match="cannot be fetched"):
        ensure(cls, item, notify=lambda _: None)


def test_verification_names_the_missing_files(tmp_path):
    item = declared(CIFAR10, "train", root=str(tmp_path))
    problems = CIFAR10.verify_data(item.config, item.requirement, item.path)
    assert "cifar-10-batches-py/data_batch_1 is missing" in problems and len(problems) == 7
    batches = tmp_path / "cifar-10-batches-py"
    batches.mkdir()
    for name in ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
                 "data_batch_5", "test_batch", "batches.meta"):
        (batches / name).write_text("")
    assert CIFAR10.verify_data(item.config, item.requirement, item.path) == []


def test_an_empty_split_folder_is_not_imagenet(tmp_path):
    (tmp_path / "val").mkdir()
    item = declared(ImageNet, "val", root=str(tmp_path))
    assert ImageNet.verify_data(item.config, item.requirement, item.path) == ["val/ is missing or empty"]
    (tmp_path / "val" / "n01440764").mkdir()
    assert ensure(ImageNet, item, notify=lambda _: None) == "present"

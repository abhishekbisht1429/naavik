"""Datasets for the data tests: their fetch writes a file and counts calls
(in `<path>/../fetches`), slowly, so concurrent runs overlap."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from training_framework.data import DataRequirement


@dataclass
class FakeConfig:
    root: str
    version: str = "1"
    fetch_seconds: float = 0.0
    broken: bool = False


def _count(path):
    with open(os.path.join(os.path.dirname(path.rstrip("/")), "fetches"), "a") as handle:
        handle.write(f"{os.getpid()}\n")


class Fetchable:
    config_schema = FakeConfig

    @classmethod
    def data_requirements(cls, config):
        return (DataRequirement("fake", "root", config.version),)

    @classmethod
    def fetch_data(cls, config, requirement, path):
        _count(path)
        time.sleep(config.fetch_seconds)
        os.makedirs(path, exist_ok=True)
        if not config.broken:
            with open(os.path.join(path, "data.bin"), "w") as handle:
                handle.write("x")

    @classmethod
    def verify_data(cls, config, requirement, path):
        return [] if os.path.exists(os.path.join(path, "data.bin")) else ["data.bin is missing"]


class ByHand(Fetchable):
    @classmethod
    def data_requirements(cls, config):
        return (DataRequirement("by_hand", "root", "1", fetchable=False),)


class NotDeclared:
    config_schema = FakeConfig

"""A components package whose resources declare data (for the code facts
tests)."""

from __future__ import annotations

from training_framework.components import Resource, resource
from training_framework.data import DataRequirement
from tests.data.fake_datasets import FakeConfig, Fetchable


@resource("fake_data")
class FakeData(Fetchable, Resource):
    config_schema = FakeConfig

    def __init__(self, config=None):
        super().__init__(config)

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@resource("wrongly_declared_data")
class WronglyDeclared(FakeData):
    @classmethod
    def data_requirements(cls, config):
        return [DataRequirement("fake", "root", "1")]          # a list, not a tuple

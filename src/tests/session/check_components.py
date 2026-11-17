"""Components for `test_check_session_config.py`: a resource that counts its
constructions, so a test can prove checking builds nothing."""

from dataclasses import dataclass

from training_framework.components import Resource, requires_resource, resource

CONSTRUCTED: list[str] = []


@dataclass
class CountedConfig:
    size: int = 1

    def __post_init__(self):
        if not isinstance(self.size, int) or self.size < 1:
            raise ValueError(f"size must be a positive integer; got {self.size!r}")


@resource("counted")
class Counted(Resource):
    config_schema = CountedConfig

    def __init__(self, config=None):
        CONSTRUCTED.append("counted")
        super().__init__(config)

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@resource("needs_counted")
@requires_resource("counted")
class NeedsCounted(Resource):
    def __init__(self, config=None):
        CONSTRUCTED.append("needs_counted")
        super().__init__(config)

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

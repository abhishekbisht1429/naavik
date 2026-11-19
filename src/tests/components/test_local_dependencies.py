"""Local dependencies: what a component declares as a part of itself, wired
only in its own `dependency_bindings`, and the deprecations that keep the
earlier forms working -- a session-wide binding of such a dependency, the
`role()` declarations it replaced, and the `dependencies_role_bindings` key.
"""

from __future__ import annotations

import warnings

import pytest

from tests.test_utils import make_config, resource_named
from naavik.components import (
    DependencyDeclaration,
    Resource,
    RoleBindings,
    RoleDeclaration,
    Step,
    dependency_declaration,
    local_dependencies,
    parse_component_config,
    requires_resource,
    requires_step,
    resource,
    role,
    role_registry,
)
from naavik.components.builtin import Checkpointer
from naavik.components.config import parse_session_config
from naavik.session import TrainingSession


# -- components --------------------------------------------------------------------
# At module scope, so a checkpointed session can rebuild them.


class _Plain(Resource):
    def setup(self, session):
        pass

    def teardown(self, session):
        pass


class LdPart(_Plain):
    """Fills a local dependency."""


class LdOtherPart(_Plain):
    """Another implementation of it."""


@requires_resource("ld_part", local=True, description="the part ld_whole is built from")
class LdWhole(_Plain):
    """Takes `ld_part` as a part of itself."""


@requires_resource("ld_part")
class LdSharer(_Plain):
    """Takes the same name as an ordinary role."""


def _register():
    resource("ld_part_impl", overwrite=True)(LdPart)
    resource("ld_other_part", overwrite=True)(LdOtherPart)
    resource("ld_whole", overwrite=True)(LdWhole)
    resource("ld_sharer", overwrite=True)(LdSharer)


@pytest.fixture(autouse=True)
def _components():
    _register()


def _config(tmp_path, **entries):
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config.update(entries)
    return config


def _built(tmp_path, **entries):
    """Build and check a session, returning it and its DeprecationWarnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = TrainingSession(_config(tmp_path, **entries))
        session.check_role_bindings()
    return session, [
        str(warning.message) for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]


def _part_of(session, consumer="ld_whole"):
    return resource_named(session, consumer).get_dependency("ld_part")


def _class(name, base=_Plain):
    return type(name, (base,), {})


# -- declaring --------------------------------------------------------------------------


def test_a_dependency_is_global_with_no_description_unless_declared_otherwise():
    plain = requires_resource("ld_thing")(_class("Plain"))
    whole = requires_resource(
        "ld_thing", local=True, description="what it is for",
    )(_class("Whole"))

    assert dependency_declaration(plain, "ld_thing") == DependencyDeclaration(
        "ld_thing", Resource,
    )
    assert dependency_declaration(whole, "ld_thing") == DependencyDeclaration(
        "ld_thing", Resource, local=True, description="what it is for",
    )
    assert local_dependencies(plain) == frozenset()
    assert local_dependencies(whole) == {"ld_thing"}
    # The public list is what it always was.
    assert whole.required_resources == ["ld_thing"]


def test_declarations_are_inherited_and_a_subclass_may_redeclare_one():
    parent = requires_resource("ld_a", local=True)(_class("Parent"))
    inherited = _class("Inherited", parent)
    redeclared = requires_resource("ld_a")(_class("Redeclared", parent))

    assert local_dependencies(inherited) == {"ld_a"}
    assert local_dependencies(redeclared) == frozenset()
    assert local_dependencies(parent) == {"ld_a"}


def test_one_class_cannot_declare_a_dependency_two_ways():
    cls = requires_resource("ld_a", local=True)(_class("Twice"))

    with pytest.raises(ValueError, match="declares Resource 'ld_a' twice"):
        requires_resource("ld_a")(cls)


@pytest.mark.parametrize(
    ("options", "match"),
    (
        ({"local": "yes"}, r"local=\.\.\.\) must be true or false"),
        ({"local": 1}, r"local=\.\.\.\) must be true or false"),
        ({"description": ""}, r"description=\.\.\.\) must be a non-empty string"),
        ({"description": 3}, r"description=\.\.\.\) must be a non-empty string"),
    ),
)
def test_decorator_options_are_checked_where_they_are_written(options, match):
    with pytest.raises(TypeError, match=match):
        requires_resource("ld_a", **options)


def test_a_name_listed_by_hand_reads_as_a_global_dependency():
    cls = requires_resource("ld_a", local=True)(_class("ByHand"))
    cls.required_resources = ["ld_b"]

    assert dependency_declaration(cls, "ld_b") == DependencyDeclaration("ld_b", Resource)
    # A declaration no longer listed is not reported.
    assert dependency_declaration(cls, "ld_a") == DependencyDeclaration("ld_a", Resource)
    assert local_dependencies(cls) == frozenset()


def test_a_step_declaring_its_own_step_still_drops_its_parents():
    parent = requires_step("ld_first", local=True)(_class("ParentStep", Step))
    child = requires_step("ld_second")(_class("ChildStep", parent))

    assert child.required_steps == ["ld_second"]
    assert local_dependencies(child) == frozenset()
    assert not dependency_declaration(child, "ld_first", Step).local
    assert local_dependencies(_class("Plain", parent)) == {"ld_first"}


# -- deprecated role declarations ---------------------------------------------------


def test_reading_a_deprecated_role_warns_but_asking_for_it_does_not():
    role("ld_old_role", Resource, description="old", deprecated="use the new way")
    registry = role_registry()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "ld_old_role" in registry
        assert "ld_old_role" in list(registry)
        assert len(registry) >= 1
    with pytest.warns(DeprecationWarning, match="use the new way"):
        assert registry["ld_old_role"].description == "old"
    with pytest.warns(DeprecationWarning, match="use the new way"):
        registry.get("ld_old_role")
    with pytest.warns(DeprecationWarning, match="use the new way"):
        role("ld_old_role", Resource, overwrite=True)


def test_a_role_declaration_keeps_its_positional_form():
    declaration = role("ld_new_role", Resource, description="a thing")

    assert declaration == RoleDeclaration("ld_new_role", Resource, "a thing")
    assert declaration.deprecated is None
    with pytest.raises(TypeError, match="deprecated must be a non-empty string"):
        role("ld_bad_role", Resource, deprecated="")


def test_the_role_registry_follows_later_declarations():
    registry = role_registry("training")
    role("ld_later_role", Resource, session_type="training")

    assert "ld_later_role" in registry


def test_built_in_roles_that_became_local_dependencies_are_deprecated():
    from naavik.components.builtin.fine_tuning import (
        BackboneWithHead,
        FineTunedModel,
        ModulePart,
    )

    registry = role_registry("training")
    assert "source" in registry
    with pytest.warns(DeprecationWarning, match="local dependency of module_part"):
        registry["source"]
    assert local_dependencies(ModulePart) == {"source"}
    assert local_dependencies(BackboneWithHead) == {"backbone", "head"}
    assert local_dependencies(FineTunedModel) == {"backbone", "head"}


# -- the binding rule -------------------------------------------------------------------


def test_scope_findings_name_the_consumers_a_session_wide_binding_reaches():
    bindings = RoleBindings(
        {"ld_part": "ld_part_impl"},
        dependency_bindings={"ld_whole#wired": {"ld_part": "ld_other_part"}},
    )

    (finding,) = bindings.scope_findings({
        "ld_whole": LdWhole,
        "ld_whole#wired": LdWhole,     # wires the name itself: not reached
        "ld_sharer": LdSharer,         # an ordinary role there
    })

    assert (finding.role, finding.target, finding.consumers) == (
        "ld_part", "ld_part_impl", ("ld_whole",),
    )


def test_a_binding_of_a_role_no_one_takes_locally_has_no_finding():
    bindings = RoleBindings({"ld_part": "ld_part_impl"})

    assert bindings.scope_findings({"ld_sharer": LdSharer}) == []


# -- in a session -------------------------------------------------------------------


def test_a_session_wide_binding_of_a_local_dependency_still_applies_and_warns(tmp_path):
    session, warned = _built(
        tmp_path,
        role_bindings={"ld_part": "ld_other_part"},
        ld_whole={},
    )

    assert isinstance(_part_of(session), LdOtherPart)
    (message,) = warned
    assert "'ld_part' is bound for the whole session (top-level role_bindings)" in message
    assert "ld_whole takes it as a local dependency" in message
    assert "ld_whole: {dependency_bindings: {ld_part: ld_other_part}}" in message
    assert "ld_part -> ld_other_part (deprecated: local to ld_whole)" in (
        session.execution_graph()
    )


def test_the_wiring_the_warning_suggests_is_the_same_wiring_without_it(tmp_path):
    session, warned = _built(
        tmp_path,
        ld_whole={"dependency_bindings": {"ld_part": "ld_other_part"}},
    )

    assert warned == []
    assert isinstance(_part_of(session), LdOtherPart)
    assert "(deprecated" not in session.execution_graph()


def test_a_name_both_local_and_shared_warns_only_for_the_local_consumer(tmp_path):
    session, warned = _built(
        tmp_path,
        role_bindings={"ld_part": "ld_other_part"},
        ld_whole={},
        ld_sharer={},
    )

    assert _part_of(session) is _part_of(session, "ld_sharer")
    (message,) = warned
    assert "ld_whole takes it" in message and "ld_sharer" not in message


def test_the_warning_is_given_once(tmp_path):
    session, warned = _built(
        tmp_path, role_bindings={"ld_part": "ld_other_part"}, ld_whole={},
    )
    assert len(warned) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        session.check_role_bindings()


def test_a_restored_session_does_not_warn(tmp_path):
    session, _ = _built(
        tmp_path,
        role_bindings={"ld_part": "ld_other_part"},
        ld_whole={"dependencies_role_bindings": {}},
    )
    path = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        restored = Checkpointer.load_checkpoint(path)
        restored.check_role_bindings()
    assert isinstance(_part_of(restored), LdOtherPart)


def test_a_missing_local_dependency_says_what_it_is_and_where_to_wire_it(tmp_path):
    resource("ld_lonely", overwrite=True)(
        requires_resource("ld_missing", local=True, description="the lonely part")(
            _class("Lonely")
        )
    )

    with pytest.raises(RuntimeError) as raised:
        TrainingSession(_config(tmp_path, ld_lonely={}))

    message = str(raised.value)
    assert message.startswith(
        "Local dependency 'ld_missing' (Resource: the lonely part) of Lonely "
        "has no implementation registered."
    )
    assert "ld_lonely: {dependency_bindings: {ld_missing: <implementation_name>}}" in message


# -- the `dependency_bindings` key --------------------------------------------------


def test_the_former_key_still_wires_and_warns(tmp_path):
    session, warned = _built(
        tmp_path,
        ld_whole={"dependencies_role_bindings": {"ld_part": "ld_other_part"}},
    )

    assert isinstance(_part_of(session), LdOtherPart)
    assert warned == [
        "'ld_whole': 'dependencies_role_bindings' is deprecated; rename it to "
        "'dependency_bindings', which takes the same wiring"
    ]


def test_an_entry_may_not_use_both_keys(tmp_path):
    with pytest.raises(ValueError, match="wired under both 'dependency_bindings'"):
        TrainingSession(_config(tmp_path, ld_whole={
            "dependency_bindings": {"ld_part": "ld_part_impl"},
            "dependencies_role_bindings": {"ld_part": "ld_part_impl"},
        }))


def test_a_checkpoint_written_with_the_former_key_restores_without_warning(tmp_path):
    session, _ = _built(
        tmp_path,
        ld_whole={"dependencies_role_bindings": {"ld_part": "ld_other_part"}},
    )
    path = Checkpointer.save_checkpoint(session, tmp_path / "saved")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        restored = Checkpointer.load_checkpoint(path)
    assert isinstance(_part_of(restored), LdOtherPart)
    # The configuration is kept as written.
    assert "dependencies_role_bindings" in restored.full_config["ld_whole"]


def test_renaming_the_key_alone_is_no_change_to_compare():
    wiring = {"ld_part": "ld_other_part"}
    former = parse_session_config({"ld_whole": {"dependencies_role_bindings": wiring}})
    current = parse_session_config({"ld_whole": {"dependency_bindings": wiring}})

    assert former.comparable() == current.comparable()
    assert former.dependency_bindings == current.dependency_bindings


def test_a_config_schema_may_not_declare_the_wiring_key():
    from dataclasses import dataclass

    @dataclass
    class Schema:
        dependency_bindings: dict | None = None

    cls = _class("Clashing")
    cls.config_schema = Schema

    with pytest.raises(TypeError, match="'dependency_bindings', which is reserved"):
        parse_component_config(cls, {})


def test_an_imports_role_reaching_a_local_dependency_warns(tmp_path, monkeypatch):
    from tests.components.builtin import test_fine_tuning as ft
    from tests.test_utils import stub_process_group

    stub_process_group(monkeypatch)
    ft._register()
    path, _ = ft.pretrain_checkpoint(tmp_path / "pre")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = TrainingSession(ft.fine_tuning_config(tmp_path / "run", path))
        session.check_role_bindings()
    messages = [str(warning.message) for warning in caught]

    (from_import,) = [m for m in messages if "'source' is bound" in m]
    assert "(import_components.pretrained.role)" in from_import
    assert "module_part takes it as a local dependency" in from_import
    assert "drop the import's `role`" in from_import
    assert resource_named(session, "module_part").module is (
        resource_named(session, "ft_pretrain_model").encoder
    )


def _wired_under_former_key(tmp_path):
    session, _ = _built(
        tmp_path,
        ld_whole={"dependencies_role_bindings": {"ld_part": "ld_other_part"}},
    )
    return session


@pytest.mark.parametrize("written, override", [
    ("dependencies_role_bindings", "dependency_bindings"),
    ("dependency_bindings", "dependencies_role_bindings"),
])
def test_an_extension_may_write_the_same_wiring_under_either_name(tmp_path, written, override):
    session, _ = _built(tmp_path, ld_whole={written: {"ld_part": "ld_other_part"}})

    session.apply_extension_overrides([f"ld_whole.{override}.ld_part=ld_other_part"])

    # The entry keeps the name it was written with.
    assert list(session.full_config["ld_whole"]) == [written]


def test_an_extension_changing_wiring_under_the_new_name_is_still_refused(tmp_path):
    session = _wired_under_former_key(tmp_path)

    with pytest.raises(
            ValueError,
            match=r"does not allow changes to: ld_whole\.dependency_bindings\.ld_part",
    ):
        session.apply_extension_overrides(["ld_whole.dependency_bindings.ld_part=ld_part_impl"])

"""JSON serialization roundtrip tests for every vistiq.core.Configuration subclass."""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import vistiq
from vistiq.core import Configuration
from vistiq.matrix.types import default_matrix_annotations


def discover_configs() -> list[type[Configuration]]:
    """Return all concrete ``vistiq.core.Configuration`` subclasses."""
    classes: list[type[Configuration]] = []
    seen: set[type[Configuration]] = set()

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        vistiq.__path__, prefix="vistiq."
    ):
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != modname:
                continue
            if obj is Configuration or obj in seen:
                continue
            try:
                if not issubclass(obj, Configuration):
                    continue
            except TypeError:
                continue
            classes.append(obj)
            seen.add(obj)

    return sorted(classes, key=lambda cls: f"{cls.__module__}.{cls.__qualname__}")


def config_factories() -> dict[str, Callable[[], Configuration]]:
    """Factories for configs that cannot be built with a zero-arg constructor."""
    from vistiq.analysis.overlap import BoxOverlapCalculatorConfig
    from vistiq.core import TilerConfig
    from vistiq.io import FileListConfig
    from vistiq.preprocess.preprocess import ResizeConfig, UpsampleConfig
    from vistiq.segment.label import BasicSegmenterConfig, SeriesSegmenterConfig

    return {
        "vistiq.analysis.overlap.OverlapCalculatorConfig": BoxOverlapCalculatorConfig,
        "vistiq.core.TilerConfig": lambda: TilerConfig(factor=(2, 2)),
        "vistiq.io.FileListConfig": lambda: FileListConfig(paths=["."]),
        "vistiq.preprocess.preprocess.ResizeConfig": lambda: ResizeConfig(width=128),
        "vistiq.preprocess.preprocess.UpsampleConfig": lambda: UpsampleConfig(width=128),
        "vistiq.segment.label.SeriesSegmenterConfig": lambda: SeriesSegmenterConfig(
            segmenters=[BasicSegmenterConfig()]
        ),
    }


def assert_json_roundtrip(config: Configuration) -> None:
    """Configs must JSON-serialize and stabilize after one validate/dump cycle."""
    cls = type(config)
    payload = config.model_dump(mode="json")
    json.dumps(payload)
    restored = cls.model_validate(payload)
    payload2 = restored.model_dump(mode="json")
    json.dumps(payload2)
    restored2 = cls.model_validate(payload2)
    assert restored2.model_dump(mode="json") == payload2


CONFIGURATION_CLASSES = discover_configs()
CONFIGURATION_IDS = [f"{cls.__module__}.{cls.__qualname__}" for cls in CONFIGURATION_CLASSES]
CONFIGURATION_FACTORY_MAP = config_factories()


@pytest.mark.parametrize("config_cls", CONFIGURATION_CLASSES, ids=CONFIGURATION_IDS)
def test_config_json(config_cls: type[Configuration]) -> None:
    """Each Configuration subclass round-trips through JSON mode."""
    key = f"{config_cls.__module__}.{config_cls.__qualname__}"
    factory = CONFIGURATION_FACTORY_MAP.get(key, config_cls)
    config = factory()
    assert_json_roundtrip(config)


class TestSerializationHelpers:
    def test_callable_roundtrip(self) -> None:
        from vistiq.core import (
            deserialize_callable,
            serialize_callable,
        )

        path = serialize_callable(default_matrix_annotations)
        assert path == "vistiq.matrix.types.default_matrix_annotations"
        assert deserialize_callable(path) is default_matrix_annotations

    def test_index_roundtrip(self) -> None:
        from vistiq.core import resolve_index_tuple, serialize_index_tuple

        original = np.s_[:, 0:10, 2]
        restored = resolve_index_tuple(serialize_index_tuple(original))
        assert restored == original

    def test_legacy_classname(self) -> None:
        from vistiq.matrix.ops import MatrixFormatterConfig

        restored = MatrixFormatterConfig.model_validate(
            {
                "classname": "Configurable",
                "annotation_factory": "vistiq.matrix.types.default_matrix_annotations",
            }
        )
        from vistiq.core import Configurable

        assert restored.classname is Configurable

    def test_preprocessor_dtype(self) -> None:
        from vistiq.preprocess.preprocess import RescaleConfig

        config = RescaleConfig(dtype=np.uint8)
        assert_json_roundtrip(config)

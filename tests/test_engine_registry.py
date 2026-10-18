import pytest
from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry, get_engine, list_engines


class DummyEngine(BaseVideoEngine):
    engine_name = "dummy"

    def compile(self, asset_plans, orientation="horizontal", fps=24, **kwargs):
        return {"engine": "dummy", "plans": len(asset_plans)}

    def render(self, compiled_data, output_path, width=1280, height=720, fps=24, progress_callback=None, **kwargs):
        return output_path

    def validate_support(self, asset_plans):
        return True, None


def test_registry_registration_and_get():
    EngineRegistry.register("dummy", DummyEngine)
    engine = get_engine("dummy")
    assert isinstance(engine, DummyEngine)
    assert engine.engine_name == "dummy"
    assert "dummy" in list_engines()


def test_registry_case_insensitivity():
    EngineRegistry.register("dummy", DummyEngine)
    engine = get_engine("DUMMY")
    assert isinstance(engine, DummyEngine)


def test_registry_unknown_engine():
    with pytest.raises(ValueError) as excinfo:
        get_engine("unknown_engine_xyz")
    assert "Unknown video engine: 'unknown_engine_xyz'" in str(excinfo.value)

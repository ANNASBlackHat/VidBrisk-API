import pytest
from pipeline.engines.base import BaseVideoEngine
from pipeline.engines.registry import EngineRegistry, get_engine, list_engines
import pipeline.engines.remotion  # Ensure Remotion is registered as default
import pipeline.engines.hyperframes  # Ensure HyperFrames is registered as default
import pipeline.engines.native_fast  # Ensure NativeFast is registered as default


@pytest.fixture(autouse=True)
def clean_registry():
    EngineRegistry.reset()
    yield
    EngineRegistry.reset()


class DummyEngine(BaseVideoEngine):
    engine_name = "dummy"

    def compile(self, job_or_plan, fps=30, **kwargs):
        return {"engine": "dummy"}

    def render(self, compiled_data, output_path, width=1280, height=720, fps=24, progress_callback=None, **kwargs):
        return output_path

    def validate_support(self, job_or_plan):
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


def test_registry_invalid_engine_name():
    with pytest.raises(ValueError) as excinfo:
        get_engine(None)
    assert "Invalid engine name" in str(excinfo.value)


def test_registry_hyperframes_active():
    engine = get_engine("hyperframes")
    assert engine.engine_name == "hyperframes"


def test_registry_native_fast_active():
    engine = get_engine("native_fast")
    assert engine.engine_name == "native_fast"


def test_registry_future_scheduled_engine(monkeypatch):
    monkeypatch.setitem(EngineRegistry._future_engines, "future_preview", "Phase 9")
    with pytest.raises(NotImplementedError) as excinfo:
        get_engine("future_preview")
    assert "Phase 9" in str(excinfo.value)

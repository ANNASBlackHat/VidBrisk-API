import pytest
from pipeline.models import Beat
from pipeline.cinematic.models import ShotScale, SceneDirectorSheet
from pipeline.cinematic.director import direct_scene


def test_direct_scene_inductive_progression():
    """Verify that a 4-beat intro narrative follows classical inductive reveal: ELS -> MS -> CU -> ECU/Reset."""
    beats = [
        Beat(id="b1", text="Deep beneath the Atlantic, a forgotten world exists.", visual_intent="Wide underwater trench"),
        Beat(id="b2", text="Submersibles navigate through narrow rocky canyons.", visual_intent="Submarine gliding past rock wall"),
        Beat(id="b3", text="The pilot spots a metallic artifact half-buried in the silt.", visual_intent="Close up on mechanical artifact"),
        Beat(id="b4", text="It bears the stamped serial number of the lost expedition.", visual_intent="Extreme macro of serial numbers on metal"),
    ]
    sheet = direct_scene(beats, genre="deep_sea_documentary", scene_id="scene_intro")
    assert isinstance(sheet, SceneDirectorSheet)
    assert sheet.scene_id == "scene_intro"
    assert sheet.beat_shot_scales["b1"] in (ShotScale.ELS, ShotScale.LS)
    assert sheet.beat_shot_scales["b2"] in (ShotScale.MS, ShotScale.MCU)
    assert sheet.beat_shot_scales["b3"] in (ShotScale.CU, ShotScale.MCU)
    assert sheet.beat_shot_scales["b4"] in (ShotScale.ECU, ShotScale.INS)
    assert sheet.target_color_temp_k > 5500  # Deep sea target is cool/cyan


def test_direct_scene_assigns_inserts_to_stats():
    """Verify that stats and numbers are assigned insert cutaways (INS)."""
    beats = [
        Beat(id="b1", text="The descent begins.", visual_intent="Submarine diving"),
        Beat(id="b2", text="At 3,800 meters, the pressure exceeds 380 atmospheres.", beat_type="stat", visual_intent="Depth gauge spinning"),
    ]
    sheet = direct_scene(beats, genre="deep_sea_documentary")
    assert sheet.beat_shot_scales["b2"] == ShotScale.INS


def test_direct_scene_handles_single_beat():
    """Verify that a single beat does not crash and gets a sensible establishing shot."""
    beats = [Beat(id="b1", text="Only darkness remains.", visual_intent="Dark ocean")]
    sheet = direct_scene(beats)
    assert sheet.beat_shot_scales["b1"] in (ShotScale.ELS, ShotScale.LS, ShotScale.MS)

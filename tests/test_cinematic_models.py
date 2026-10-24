import pytest
from pipeline.cinematic.models import (
    ShotScale,
    MotionVector,
    SceneDirectorSheet,
    CinematicCandidateMetadata,
)


def test_shot_scale_enums():
    assert ShotScale.ELS.value == "extreme_long_shot"
    assert ShotScale.LS.value == "long_shot"
    assert ShotScale.MS.value == "medium_shot"
    assert ShotScale.MCU.value == "medium_close_up"
    assert ShotScale.CU.value == "close_up"
    assert ShotScale.ECU.value == "extreme_close_up"
    assert ShotScale.INS.value == "insert_cutaway"


def test_motion_vector_enums():
    assert MotionVector.STATIC.value == "static"
    assert MotionVector.PAN_LEFT.value == "pan_left"
    assert MotionVector.PAN_RIGHT.value == "pan_right"
    assert MotionVector.TILT_UP.value == "tilt_up"
    assert MotionVector.TILT_DOWN.value == "tilt_down"
    assert MotionVector.ZOOM_IN.value == "zoom_in"
    assert MotionVector.ZOOM_OUT.value == "zoom_out"


def test_scene_director_sheet_instantiation():
    sheet = SceneDirectorSheet(
        scene_id="scene_01",
        tone="eerie abyssal tension",
        target_color_temp_k=6500,
        target_asl=3.5,
        visual_motif="descending sonar ping",
        beat_shot_scales={"b1": ShotScale.ELS, "b2": ShotScale.MS, "b3": ShotScale.CU},
        rhythm_pattern="accelerando",
    )
    assert sheet.scene_id == "scene_01"
    assert sheet.beat_shot_scales["b1"] == ShotScale.ELS
    assert sheet.beat_shot_scales["b3"] == ShotScale.CU
    assert sheet.rhythm_pattern == "accelerando"


def test_cinematic_candidate_metadata():
    meta = CinematicCandidateMetadata(
        chunk_id="chk_001",
        shot_scale=ShotScale.CU,
        motion_vector=MotionVector.PAN_LEFT,
        motion_speed=6.2,
        dominant_color_temp_k=6200,
        eye_trace_center=(0.45, 0.55),
    )
    assert meta.shot_scale == ShotScale.CU
    assert meta.motion_vector == MotionVector.PAN_LEFT
    assert meta.eye_trace_center == (0.45, 0.55)

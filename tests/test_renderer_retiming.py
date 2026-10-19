import unittest
from unittest.mock import MagicMock, patch
from pipeline.renderer.engine import VideoRenderer


def test_get_filter_complex_with_pts_factor():
    renderer = VideoRenderer()
    # Default without pts factor
    fc = renderer._get_filter_complex(1280, 720, "blur_bg", pts_factor=1.0)
    assert "[0:v]split=2" in fc

    # With slowmo pts factor
    fc_slow = renderer._get_filter_complex(1280, 720, "blur_bg", pts_factor=1.45)
    assert "setpts=1.4500*PTS" in fc_slow
    assert "split=2" in fc_slow


def test_trim_scale_video_clip_omits_stream_loop_when_retiming():
    renderer = VideoRenderer()
    renderer._run_cmd = MagicMock()
    renderer._get_media_duration = MagicMock(return_value=3.0)

    props = {"retime_strategy": "slowmo"}
    renderer._trim_scale_video_clip(
        output_path="test_out.mp4",
        video_path="test_in.mp4",
        start_ts=0.0,
        duration=4.5,
        width=1280,
        height=720,
        fps=24,
        fit_mode="blur_bg",
        props=props,
    )

    args = renderer._run_cmd.call_args[0][0]
    # Verify -stream_loop is NOT present
    assert "-stream_loop" not in args
    # Verify setpts is in filter complex
    filter_arg = args[args.index("-filter_complex") + 1]
    assert "setpts=" in filter_arg

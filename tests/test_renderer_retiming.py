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
    # 3.6s on 3.0s is factor 1.20x, within 1.35x ceiling -> no stream_loop needed
    renderer._trim_scale_video_clip(
        output_path="test_out.mp4",
        video_path="test_in.mp4",
        start_ts=0.0,
        duration=3.6,
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


def test_renderer_clamps_pts_factor_and_uses_stream_loop_on_extreme_deficit():
    import re
    renderer = VideoRenderer()
    renderer._run_cmd = MagicMock()
    # Source is 1.0s, requested duration is 6.0s (a 6.0x deficit)
    renderer._get_media_duration = MagicMock(return_value=1.0)

    props = {"retime_strategy": "slowmo"}
    renderer._trim_scale_video_clip(
        output_path="test_out.mp4",
        video_path="test_in.mp4",
        start_ts=0.0,
        duration=6.0,
        width=1280,
        height=720,
        fps=24,
        fit_mode="blur_bg",
        props=props,
    )

    args = renderer._run_cmd.call_args[0][0]
    # Verify setpts factor does not exceed 1.35x
    filter_arg = args[args.index("-filter_complex") + 1]
    m = re.search(r"setpts=([0-9.]+)\*PTS", filter_arg)
    assert m is not None
    pts = float(m.group(1))
    assert pts <= 1.35
    # Verify -stream_loop -1 is present because 6.0s > 1.0s * 1.35
    assert "-stream_loop" in args
    assert args[args.index("-stream_loop") + 1] == "-1"


import pytest
from pipeline.renderer.engine import VideoRenderer, RenderEngineMismatchError


def test_motion_timeline_raises_mismatch_error():
    renderer = VideoRenderer(debug=False)
    
    # Timeline with motion assetType
    motion_timeline = {
        "total_duration": 5.0,
        "tracks": [
            {
                "type": "video",
                "items": [
                    {
                        "id": "item_1",
                        "trackStart": 0.0,
                        "trackEnd": 5.0,
                        "assetType": "motion",
                        "componentId": "DataAnimations/StatCard",
                        "props": {"value": "90%"},
                    }
                ],
            }
        ],
    }

    with pytest.raises(RenderEngineMismatchError) as exc_info:
        renderer.render_timeline(
            timeline=motion_timeline,
            output_path="/tmp/test_motion_out.mp4",
        )

    assert "This timeline contains motion components that require the Remotion renderer" in str(exc_info.value)


def test_motion_component_id_raises_mismatch_error():
    renderer = VideoRenderer(debug=False)
    
    # Timeline with componentId set even if assetType is unspecified
    motion_timeline = {
        "total_duration": 4.0,
        "tracks": [
            {
                "type": "video",
                "items": [
                    {
                        "id": "item_2",
                        "trackStart": 0.0,
                        "trackEnd": 4.0,
                        "componentId": "TextAnimations/QuoteCard",
                    }
                ],
            }
        ],
    }

    with pytest.raises(RenderEngineMismatchError) as exc_info:
        renderer.render_timeline(
            timeline=motion_timeline,
            output_path="/tmp/test_motion_out2.mp4",
        )

    assert "Remotion renderer (headless Chromium + frame-accurate interpolation)" in str(exc_info.value)


def test_split_screen_raises_mismatch_error():
    renderer = VideoRenderer(debug=False)
    split_timeline = {
        "total_duration": 4.0,
        "tracks": [
            {
                "type": "video",
                "items": [
                    {
                        "id": "clip_left",
                        "trackStart": 0.0,
                        "trackEnd": 4.0,
                        "assetType": "video",
                        "layout": "split-left",
                        "zIndex": 0,
                    },
                    {
                        "id": "clip_right",
                        "trackStart": 0.0,
                        "trackEnd": 4.0,
                        "assetType": "video",
                        "layout": "split-right",
                        "zIndex": 1,
                    },
                ],
            }
        ],
    }

    with pytest.raises(RenderEngineMismatchError) as exc_info:
        renderer.render_timeline(
            timeline=split_timeline,
            output_path="/tmp/test_split_out.mp4",
        )

    assert "multi-layer visual compositions" in str(exc_info.value)


def test_overlay_layer_role_raises_mismatch_error():
    renderer = VideoRenderer(debug=False)
    overlay_timeline = {
        "total_duration": 3.0,
        "tracks": [
            {
                "type": "video",
                "items": [
                    {
                        "id": "bg_item",
                        "trackStart": 0.0,
                        "trackEnd": 3.0,
                        "assetType": "video",
                        "layerRole": "background",
                        "layout": "full",
                    },
                    {
                        "id": "overlay_item",
                        "trackStart": 0.0,
                        "trackEnd": 3.0,
                        "layerRole": "overlay",
                        "layout": "overlay-lower-third",
                        "zIndex": 1,
                    },
                ],
            }
        ],
    }

    with pytest.raises(RenderEngineMismatchError) as exc_info:
        renderer.render_timeline(
            timeline=overlay_timeline,
            output_path="/tmp/test_overlay_out.mp4",
        )

    assert "multi-layer visual compositions" in str(exc_info.value)


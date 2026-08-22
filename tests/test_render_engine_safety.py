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

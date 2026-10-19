"""Unit tests for OmniTTSProvider voice reference resolution and prompt normalization."""

import os
from unittest.mock import MagicMock, patch
from pipeline.tts.omni import (
    DEFAULT_VOICE_INSTRUCT,
    OmniTTSProvider,
    normalize_instruct,
    resolve_voice_reference,
)


def test_normalize_instruct_freeform_mapping():
    prompt = "calm, articulate British male narrator, slow pace, documentary tone"
    normalized = normalize_instruct(prompt)
    assert "male" in normalized
    assert "british accent" in normalized
    assert "low pitch" in normalized


def test_normalize_instruct_already_valid():
    prompt = "female, young adult, moderate pitch, american accent"
    normalized = normalize_instruct(prompt)
    assert normalized == "female, young adult, moderate pitch, american accent"


def test_normalize_instruct_empty():
    assert normalize_instruct("") == DEFAULT_VOICE_INSTRUCT
    assert normalize_instruct(None) == DEFAULT_VOICE_INSTRUCT


def test_resolve_voice_reference_known_file(tmp_path):
    audio_file = tmp_path / "test_sample.mp3"
    audio_file.write_bytes(b"mock audio bytes")
    txt_file = tmp_path / "test_sample.txt"
    txt_file.write_text("Reference speech transcript text.")

    ref_audio, ref_text, instruct = resolve_voice_reference(str(audio_file))
    assert ref_audio == str(audio_file)
    assert ref_text == "Reference speech transcript text."
    assert instruct is None


def test_resolve_voice_reference_prompt():
    prompt = "female, young adult, high pitch, british accent"
    ref_audio, ref_text, instruct = resolve_voice_reference(prompt)
    assert ref_audio is None
    assert ref_text is None
    assert instruct == prompt


def test_omni_tts_provider_synthesize_mocked(tmp_path):
    provider = OmniTTSProvider(device="cpu")
    mock_model = MagicMock()
    # Mock model.generate returning numpy array list
    import numpy as np

    sample_audio = np.zeros(24000, dtype=np.float32)
    mock_model.generate.return_value = [sample_audio]
    provider._model = mock_model

    out_file = tmp_path / "omni_test.wav"
    result = provider.synthesize(
        text="Testing mocked omni synthesis.",
        voice="male, young adult, low pitch, american accent",
        output_path=str(out_file),
    )

    assert os.path.exists(result.audio_path)
    assert result.sample_rate == 24000
    assert result.duration_sec > 0
    mock_model.generate.assert_called_once()


def test_omni_tts_shared_model_cache():
    from pipeline.tts.omni import _SHARED_OMNI_MODELS
    _SHARED_OMNI_MODELS.clear()

    mock_omni_class = MagicMock()
    mock_model_instance = MagicMock()
    mock_omni_class.from_pretrained.return_value = mock_model_instance

    with patch.dict("sys.modules", {"omnivoice": MagicMock(OmniVoice=mock_omni_class)}):
        p1 = OmniTTSProvider(device="cpu")
        m1 = p1._init_model()

        p2 = OmniTTSProvider(device="cpu")
        m2 = p2._init_model()

        assert m1 is m2
        assert m1 is mock_model_instance
        mock_omni_class.from_pretrained.assert_called_once()
        assert ("k2-fsa/OmniVoice", "cpu") in _SHARED_OMNI_MODELS


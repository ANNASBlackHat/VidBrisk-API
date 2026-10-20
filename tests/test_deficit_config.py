from pipeline.config import Settings, get_settings


def test_deficit_strategy_setting_defaults_to_balanced_splice():
    s = Settings()
    assert s.DEFICIT_STRATEGY == "balanced_splice"


def test_deficit_strategy_custom_setting():
    s = Settings(DEFICIT_STRATEGY="slowmo")
    assert s.DEFICIT_STRATEGY == "slowmo"


def test_retiming_ceiling_settings():
    s = Settings()
    assert hasattr(s, "MAX_SLOWMO_FACTOR")
    assert s.MAX_SLOWMO_FACTOR == 1.35
    assert hasattr(s, "MIN_SPEEDUP_FACTOR")
    assert s.MIN_SPEEDUP_FACTOR == 0.85


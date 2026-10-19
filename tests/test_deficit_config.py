from pipeline.config import Settings, get_settings


def test_deficit_strategy_setting_defaults_to_balanced_splice():
    s = Settings()
    assert s.DEFICIT_STRATEGY == "balanced_splice"


def test_deficit_strategy_custom_setting():
    s = Settings(DEFICIT_STRATEGY="slowmo")
    assert s.DEFICIT_STRATEGY == "slowmo"

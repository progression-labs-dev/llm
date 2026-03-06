"""Tests for LLM configuration module."""

from progression_labs.llm.config import (
    LLMSettings,
    configure,
    get_settings,
    reset_settings,
    set_settings,
)


class TestGetSettings:
    """Tests for get_settings() singleton."""

    def setup_method(self):
        reset_settings()

    def teardown_method(self):
        reset_settings()

    def test_returns_llm_settings(self):
        settings = get_settings()
        assert isinstance(settings, LLMSettings)

    def test_returns_same_instance(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_defaults(self):
        settings = get_settings()
        assert settings.default_model == "gpt-4o"
        assert settings.default_timeout == 60.0
        assert settings.default_max_retries == 3


class TestConfigure:
    """Tests for configure()."""

    def setup_method(self):
        reset_settings()

    def teardown_method(self):
        reset_settings()

    def test_configure_updates_settings(self):
        configure(default_model="claude-sonnet-4-20250514", default_timeout=30.0)
        settings = get_settings()
        assert settings.default_model == "claude-sonnet-4-20250514"
        assert settings.default_timeout == 30.0

    def test_configure_ignores_none_values(self):
        configure(default_model=None)
        settings = get_settings()
        assert settings.default_model == "gpt-4o"  # unchanged


class TestResetSettings:
    """Tests for reset_settings()."""

    def teardown_method(self):
        reset_settings()

    def test_reset_clears_singleton(self):
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2


class TestSetSettings:
    """Tests for set_settings() dependency injection."""

    def setup_method(self):
        reset_settings()

    def teardown_method(self):
        reset_settings()

    def test_set_settings_injects_custom(self):
        custom = LLMSettings(default_model="gpt-4o-mini", default_timeout=10.0)
        set_settings(custom)
        settings = get_settings()
        assert settings is custom
        assert settings.default_model == "gpt-4o-mini"
        assert settings.default_timeout == 10.0

"""Tests for metrics bridge service."""

import os
from unittest.mock import MagicMock, patch

import pytest

from progression_labs.llm import (
    MetricsBridge,
    collect_metrics_once,
    init_metrics_bridge,
    stop_metrics_bridge,
)


class TestMetricsBridge:
    """Tests for MetricsBridge class."""

    def test_init_defaults(self):
        """Test MetricsBridge initialization with default values."""
        bridge = MetricsBridge()
        assert bridge.poll_interval == 60.0
        assert bridge.lookback_window == 300.0
        assert bridge._running is False
        assert bridge._task is None

    def test_init_custom_values(self):
        """Test MetricsBridge initialization with custom values."""
        bridge = MetricsBridge(poll_interval=30.0, lookback_window=600.0)
        assert bridge.poll_interval == 30.0
        assert bridge.lookback_window == 600.0

    def test_setup_langfuse_requires_credentials(self):
        """Test that _setup_langfuse requires credentials."""
        from progression_labs.llm.config import LLMSettings

        bridge = MetricsBridge()
        # Mock settings with no credentials
        mock_settings = LLMSettings(
            langfuse_public_key=None,
            langfuse_secret_key=None,
        )
        with (
            patch("progression_labs.llm.metrics.get_settings", return_value=mock_settings),
            pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"),
        ):
            bridge._setup_langfuse()

    def test_setup_langfuse_requires_secret_key(self):
        """Test that _setup_langfuse requires secret key."""
        from progression_labs.llm.config import LLMSettings

        bridge = MetricsBridge()
        # Mock settings with only public key
        mock_settings = LLMSettings(
            langfuse_public_key="pk-test",
            langfuse_secret_key=None,
        )
        with (
            patch("progression_labs.llm.metrics.get_settings", return_value=mock_settings),
            pytest.raises(ValueError, match="LANGFUSE_SECRET_KEY"),
        ):
            bridge._setup_langfuse()

    def test_parse_metrics_empty_data(self):
        """Test _parse_metrics with empty data."""
        bridge = MetricsBridge()
        bridge._parse_metrics({})
        assert bridge._metric_values == {
            "latency_p50": {},
            "latency_p95": {},
            "latency_p99": {},
            "input_tokens": {},
            "output_tokens": {},
            "total_cost": {},
            "request_count": {},
            "error_count": {},
        }

    def test_parse_metrics_with_data(self):
        """Test _parse_metrics with sample Langfuse data."""
        bridge = MetricsBridge()
        sample_data = {
            "data": [
                {
                    "model": "gpt-4o",
                    "inputTokens": 1000,
                    "outputTokens": 500,
                    "totalCost": 0.05,
                    "countTraces": 10,
                    "latencyP50": 250.0,
                    "latencyP95": 800.0,
                    "latencyP99": 1200.0,
                },
                {
                    "model": "claude-sonnet-4-20250514",
                    "inputTokens": 2000,
                    "outputTokens": 1000,
                    "totalCost": 0.10,
                    "countTraces": 20,
                    "latencyP50": 300.0,
                    "latencyP95": 900.0,
                    "latencyP99": 1500.0,
                },
            ]
        }

        bridge._parse_metrics(sample_data)

        assert bridge._metric_values["input_tokens"]["gpt-4o"] == 1000
        assert bridge._metric_values["input_tokens"]["claude-sonnet-4-20250514"] == 2000
        assert bridge._metric_values["output_tokens"]["gpt-4o"] == 500
        assert bridge._metric_values["output_tokens"]["claude-sonnet-4-20250514"] == 1000
        assert bridge._metric_values["total_cost"]["gpt-4o"] == 0.05
        assert bridge._metric_values["total_cost"]["claude-sonnet-4-20250514"] == 0.10
        assert bridge._metric_values["request_count"]["gpt-4o"] == 10
        assert bridge._metric_values["request_count"]["claude-sonnet-4-20250514"] == 20
        assert bridge._metric_values["latency_p50"]["gpt-4o"] == 250.0
        assert bridge._metric_values["latency_p95"]["gpt-4o"] == 800.0
        assert bridge._metric_values["latency_p99"]["gpt-4o"] == 1200.0

    def test_parse_metrics_aggregates_by_model(self):
        """Test that _parse_metrics aggregates values for same model."""
        bridge = MetricsBridge()
        sample_data = {
            "data": [
                {
                    "model": "gpt-4o",
                    "inputTokens": 500,
                    "outputTokens": 250,
                    "totalCost": 0.025,
                    "countTraces": 5,
                },
                {
                    "model": "gpt-4o",
                    "inputTokens": 500,
                    "outputTokens": 250,
                    "totalCost": 0.025,
                    "countTraces": 5,
                },
            ]
        }

        bridge._parse_metrics(sample_data)

        assert bridge._metric_values["input_tokens"]["gpt-4o"] == 1000
        assert bridge._metric_values["output_tokens"]["gpt-4o"] == 500
        assert bridge._metric_values["total_cost"]["gpt-4o"] == 0.05
        assert bridge._metric_values["request_count"]["gpt-4o"] == 10

    def test_parse_metrics_uses_count_observations_fallback(self):
        """Test that _parse_metrics uses countObservations when countTraces is missing."""
        bridge = MetricsBridge()
        sample_data = {
            "data": [
                {
                    "model": "gpt-4o",
                    "countObservations": 15,
                },
            ]
        }

        bridge._parse_metrics(sample_data)
        assert bridge._metric_values["request_count"]["gpt-4o"] == 15

    def test_get_metric_observations_empty(self):
        """Test _get_metric_observations with no data."""
        bridge = MetricsBridge()
        bridge._metric_values = {"latency_p50": {}}
        observations = bridge._get_metric_observations("latency_p50")
        assert observations == []

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Test that stop() works even if bridge was never started."""
        bridge = MetricsBridge()
        await bridge.stop()  # Should not raise


class TestInitMetricsBridge:
    """Tests for init_metrics_bridge function."""

    def setup_method(self):
        """Reset global bridge before each test."""
        import progression_labs.llm.metrics

        progression_labs.llm.metrics._bridge = None

    def teardown_method(self):
        """Clean up global bridge after each test."""
        import progression_labs.llm.metrics

        progression_labs.llm.metrics._bridge = None

    @patch("progression_labs.llm.metrics.MetricsBridge.start")
    def test_init_metrics_bridge_creates_instance(self, mock_start):
        """Test that init_metrics_bridge creates a MetricsBridge instance."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
            },
        ):
            bridge = init_metrics_bridge(poll_interval=30.0, lookback_window=120.0)
            assert bridge is not None
            assert bridge.poll_interval == 30.0
            assert bridge.lookback_window == 120.0
            mock_start.assert_called_once()

    @patch("progression_labs.llm.metrics.MetricsBridge.start")
    def test_init_metrics_bridge_returns_existing(self, mock_start):
        """Test that init_metrics_bridge returns existing instance if already initialized."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
            },
        ):
            bridge1 = init_metrics_bridge()
            bridge2 = init_metrics_bridge()
            assert bridge1 is bridge2
            # start should only be called once
            assert mock_start.call_count == 1


class TestCollectMetricsOnce:
    """Tests for collect_metrics_once function."""

    @pytest.mark.asyncio
    async def test_collect_metrics_once_lifecycle(self):
        """Test that collect_metrics_once sets up and tears down properly."""
        with (
            patch.dict(
                os.environ,
                {
                    "LANGFUSE_PUBLIC_KEY": "pk-test",
                    "LANGFUSE_SECRET_KEY": "sk-test",
                },
            ),
            patch("progression_labs.llm.metrics.MetricsBridge._setup_otel") as mock_setup_otel,
            patch("progression_labs.llm.metrics.MetricsBridge._setup_langfuse") as mock_setup_langfuse,
            patch("progression_labs.llm.metrics.MetricsBridge._collect_once") as mock_collect,
        ):
            # Reset settings cache for test
            import progression_labs.llm.config

            old_settings = progression_labs.llm.config._settings
            progression_labs.llm.config._settings = None

            try:
                await collect_metrics_once()
                mock_setup_langfuse.assert_called_once()
                mock_setup_otel.assert_called_once()
                mock_collect.assert_called_once()
            finally:
                progression_labs.llm.config._settings = old_settings


class TestStopMetricsBridge:
    """Tests for stop_metrics_bridge function."""

    def setup_method(self):
        """Reset global bridge before each test."""
        import progression_labs.llm.metrics

        progression_labs.llm.metrics._bridge = None

    def teardown_method(self):
        """Clean up global bridge after each test."""
        import progression_labs.llm.metrics

        progression_labs.llm.metrics._bridge = None

    @pytest.mark.asyncio
    async def test_stop_metrics_bridge_no_bridge(self):
        """Test that stop_metrics_bridge works when no bridge exists."""
        await stop_metrics_bridge()  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_metrics_bridge_stops_bridge(self):
        """Test that stop_metrics_bridge stops the running bridge."""
        import progression_labs.llm.metrics

        mock_bridge = MagicMock()
        mock_bridge.stop = MagicMock(return_value=None)

        # Make stop an async function
        async def async_stop():
            pass

        mock_bridge.stop = async_stop
        progression_labs.llm.metrics._bridge = mock_bridge

        await stop_metrics_bridge()
        assert progression_labs.llm.metrics._bridge is None


class TestLazyImports:
    """Tests for lazy imports from __init__.py."""

    def test_metrics_bridge_import(self):
        """Test that MetricsBridge can be imported from progression_labs.llm."""
        from progression_labs.llm import MetricsBridge

        assert MetricsBridge is not None

    def test_init_metrics_bridge_import(self):
        """Test that init_metrics_bridge can be imported from progression_labs.llm."""
        from progression_labs.llm import init_metrics_bridge

        assert init_metrics_bridge is not None

    def test_collect_metrics_once_import(self):
        """Test that collect_metrics_once can be imported from progression_labs.llm."""
        from progression_labs.llm import collect_metrics_once

        assert collect_metrics_once is not None

    def test_stop_metrics_bridge_import(self):
        """Test that stop_metrics_bridge can be imported from progression_labs.llm."""
        from progression_labs.llm import stop_metrics_bridge

        assert stop_metrics_bridge is not None


class TestConfigFields:
    """Tests for OTLP configuration fields."""

    def test_otlp_config_defaults(self):
        """Test that OTLP config fields have correct defaults."""
        from progression_labs.llm.config import LLMSettings

        settings = LLMSettings()
        assert settings.otlp_endpoint == "http://localhost:4317"
        assert settings.is_otlp_insecure is True
        assert settings.service_name == "progression-labs-llm"
        assert settings.service_environment == "development"

    def test_otlp_config_from_env(self):
        """Test that OTLP config fields can be set from environment."""
        from progression_labs.llm.config import LLMSettings

        with patch.dict(
            os.environ,
            {
                "OTLP_ENDPOINT": "http://signoz:4317",
                "IS_OTLP_INSECURE": "false",
                "SERVICE_NAME": "my-app",
                "SERVICE_ENVIRONMENT": "production",
            },
        ):
            settings = LLMSettings()
            assert settings.otlp_endpoint == "http://signoz:4317"
            assert settings.is_otlp_insecure is False
            assert settings.service_name == "my-app"
            assert settings.service_environment == "production"

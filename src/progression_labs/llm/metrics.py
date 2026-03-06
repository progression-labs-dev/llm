"""
Metrics bridge service for polling Langfuse metrics and exporting to OTLP.

This module provides a bridge that periodically fetches aggregated LLM metrics
from Langfuse's Metrics V2 API and pushes them to an OTLP-compatible collector
(e.g., SigNoz, Jaeger, Prometheus via OTLP receiver).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from progression_labs.llm.config import LLMSettings, get_settings

if TYPE_CHECKING:
    import httpx
    from opentelemetry.metrics import Meter
    from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider

logger = logging.getLogger(__name__)

# Global bridge instance
_bridge: MetricsBridge | None = None


class MetricsBridge:
    """
    Manages polling Langfuse metrics and exporting to OTLP.

    The bridge periodically queries Langfuse's Metrics V2 API for aggregated
    LLM metrics and reports them as OpenTelemetry observable gauges.

    Supports dependency injection of settings for testing flexibility.
    """

    def __init__(
        self,
        poll_interval: float = 60.0,
        lookback_window: float = 300.0,
        settings: LLMSettings | None = None,
    ) -> None:
        """
        Initialize the metrics bridge.

        Args:
            poll_interval: How often to poll Langfuse metrics (in seconds)
            lookback_window: Time window to query for metrics (in seconds)
            settings: Optional custom LLMSettings instance (uses global if not provided)
        """
        self.poll_interval = poll_interval
        self.lookback_window = lookback_window
        self._settings = settings
        self._task: asyncio.Task | None = None
        self._running = False
        self._meter: Meter | None = None
        self._meter_provider: SdkMeterProvider | None = None
        self._langfuse_client: httpx.Client | None = None

        # Latest metric values (updated by polling)
        self._metric_values: dict[str, dict[str, float]] = {}

    def _get_settings(self) -> LLMSettings:
        """Get settings, using injected settings or falling back to global."""
        if self._settings is not None:
            return self._settings
        return get_settings()

    def _setup_otel(self) -> None:
        """Set up OpenTelemetry meter and exporter."""
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        settings = self._get_settings()

        resource = Resource.create(
            {
                "service.name": settings.service_name,
                "deployment.environment": settings.service_environment,
            }
        )

        exporter = OTLPMetricExporter(
            endpoint=settings.otlp_endpoint,
            insecure=settings.is_otlp_insecure,
        )

        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=int(self.poll_interval * 1000),
        )

        self._meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        self._meter = self._meter_provider.get_meter("progression-labs-llm-metrics")

        # Register observable gauges for all metrics
        self._register_gauges()

    def _register_gauges(self) -> None:
        """Register observable gauges for all LLM metrics."""
        if self._meter is None:
            return

        # Latency percentiles
        self._meter.create_observable_gauge(
            name="llm.latency.p50",
            callbacks=[lambda _: self._get_metric_observations("latency_p50")],
            description="LLM request latency (50th percentile) by model",
            unit="ms",
        )
        self._meter.create_observable_gauge(
            name="llm.latency.p95",
            callbacks=[lambda _: self._get_metric_observations("latency_p95")],
            description="LLM request latency (95th percentile) by model",
            unit="ms",
        )
        self._meter.create_observable_gauge(
            name="llm.latency.p99",
            callbacks=[lambda _: self._get_metric_observations("latency_p99")],
            description="LLM request latency (99th percentile) by model",
            unit="ms",
        )

        # Token usage
        self._meter.create_observable_gauge(
            name="llm.tokens.input",
            callbacks=[lambda _: self._get_metric_observations("input_tokens")],
            description="Total input tokens by model",
            unit="tokens",
        )
        self._meter.create_observable_gauge(
            name="llm.tokens.output",
            callbacks=[lambda _: self._get_metric_observations("output_tokens")],
            description="Total output tokens by model",
            unit="tokens",
        )

        # Cost tracking
        self._meter.create_observable_gauge(
            name="llm.cost.total",
            callbacks=[lambda _: self._get_metric_observations("total_cost")],
            description="Total cost by model",
            unit="USD",
        )

        # Request volume
        self._meter.create_observable_gauge(
            name="llm.requests.count",
            callbacks=[lambda _: self._get_metric_observations("request_count")],
            description="Request count by model",
            unit="requests",
        )

        # Error rate
        self._meter.create_observable_gauge(
            name="llm.errors.count",
            callbacks=[lambda _: self._get_metric_observations("error_count")],
            description="Error count by model",
            unit="errors",
        )

    def _get_metric_observations(self, metric_name: str) -> list:
        """Get observations for a specific metric."""
        from opentelemetry.metrics import Observation

        observations = []
        if metric_name in self._metric_values:
            for model, value in self._metric_values[metric_name].items():
                observations.append(Observation(value, {"model": model}))
        return observations

    def _setup_langfuse(self) -> None:
        """Set up Langfuse client for API queries."""
        import httpx

        settings = self._get_settings()
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables required"
            )

        self._langfuse_client = httpx.Client(
            base_url=settings.langfuse_host,
            auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
            timeout=30.0,
        )

    async def _fetch_langfuse_metrics(self) -> dict:
        """Fetch aggregated metrics from Langfuse Metrics V2 API."""
        if self._langfuse_client is None:
            return {}

        now = datetime.now(UTC)
        from_time = now - timedelta(seconds=self.lookback_window)

        # Build the query parameters for daily metrics endpoint
        params = {
            "fromTimestamp": from_time.isoformat(),
            "toTimestamp": now.isoformat(),
        }

        try:
            # Run blocking HTTP call in thread pool
            response = await asyncio.to_thread(
                self._langfuse_client.get,
                "/api/public/metrics/daily",
                params=params,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            logger.exception("Failed to fetch Langfuse metrics")
            return {}

    def _parse_metrics(self, data: dict) -> None:
        """Parse Langfuse metrics response and update internal state."""
        # Initialize metric containers
        self._metric_values = {
            "latency_p50": {},
            "latency_p95": {},
            "latency_p99": {},
            "input_tokens": {},
            "output_tokens": {},
            "total_cost": {},
            "request_count": {},
            "error_count": {},
        }

        # Parse the daily metrics response
        # The Langfuse Metrics V2 API returns data grouped by date with totals
        daily_data = data.get("data", [])

        for day_metrics in daily_data:
            # Aggregate by model if model breakdown is available
            model = day_metrics.get("model", "unknown")

            # Update token counts
            if "inputTokens" in day_metrics:
                self._metric_values["input_tokens"][model] = (
                    self._metric_values["input_tokens"].get(model, 0) + day_metrics["inputTokens"]
                )
            if "outputTokens" in day_metrics:
                self._metric_values["output_tokens"][model] = (
                    self._metric_values["output_tokens"].get(model, 0) + day_metrics["outputTokens"]
                )

            # Update cost
            if "totalCost" in day_metrics:
                self._metric_values["total_cost"][model] = (
                    self._metric_values["total_cost"].get(model, 0) + day_metrics["totalCost"]
                )

            # Update request count
            if "countTraces" in day_metrics:
                self._metric_values["request_count"][model] = (
                    self._metric_values["request_count"].get(model, 0) + day_metrics["countTraces"]
                )
            elif "countObservations" in day_metrics:
                self._metric_values["request_count"][model] = (
                    self._metric_values["request_count"].get(model, 0)
                    + day_metrics["countObservations"]
                )

            # Update latency percentiles if available
            if "latencyP50" in day_metrics:
                self._metric_values["latency_p50"][model] = day_metrics["latencyP50"]
            if "latencyP95" in day_metrics:
                self._metric_values["latency_p95"][model] = day_metrics["latencyP95"]
            if "latencyP99" in day_metrics:
                self._metric_values["latency_p99"][model] = day_metrics["latencyP99"]

        # Try to fetch error counts from observations with level=ERROR
        # This would require a separate API call in production
        # For now, we'll set error_count to 0 as a placeholder
        for model in self._metric_values["request_count"]:
            if model not in self._metric_values["error_count"]:
                self._metric_values["error_count"][model] = 0

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._collect_once()
            except Exception as e:
                logger.exception("Error collecting metrics: %s", e)

            await asyncio.sleep(self.poll_interval)

    async def _collect_once(self) -> None:
        """Perform a single metrics collection cycle."""
        data = await self._fetch_langfuse_metrics()
        self._parse_metrics(data)
        logger.debug("Collected metrics: %s", self._metric_values)

    def start(self) -> None:
        """Start the metrics polling loop."""
        if self._running:
            return

        self._setup_langfuse()
        self._setup_otel()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Metrics bridge started (poll_interval=%ss, lookback_window=%ss)",
            self.poll_interval,
            self.lookback_window,
        )

    async def stop(self) -> None:
        """Stop the metrics polling loop and cleanup resources."""
        self._running = False

        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        # Shutdown meter provider
        if self._meter_provider is not None:
            self._meter_provider.shutdown()
            self._meter_provider = None

        # Close Langfuse HTTP client
        if self._langfuse_client is not None:
            self._langfuse_client.close()
            self._langfuse_client = None

        logger.info("Metrics bridge stopped")


def init_metrics_bridge(
    poll_interval: float = 60.0,
    lookback_window: float = 300.0,
    settings: LLMSettings | None = None,
) -> MetricsBridge:
    """
    Initialize and start the metrics bridge.

    Creates a background task that periodically polls Langfuse for
    aggregated LLM metrics and exports them to OTLP.

    Args:
        poll_interval: How often to poll Langfuse metrics (in seconds)
        lookback_window: Time window to query for metrics (in seconds)
        settings: Optional custom LLMSettings instance (uses global if not provided)

    Returns:
        The initialized MetricsBridge instance

    Requires environment variables:
        LANGFUSE_PUBLIC_KEY: Your Langfuse public key
        LANGFUSE_SECRET_KEY: Your Langfuse secret key
        OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4317)

    Example:
        >>> bridge = init_metrics_bridge(poll_interval=60.0)
        >>> # ... runs in background ...
        >>> await stop_metrics_bridge()
    """
    global _bridge

    if _bridge is not None:
        logger.warning("Metrics bridge already initialized, returning existing instance")
        return _bridge

    _bridge = MetricsBridge(
        poll_interval=poll_interval,
        lookback_window=lookback_window,
        settings=settings,
    )
    _bridge.start()
    return _bridge


def reset_metrics_bridge() -> None:
    """
    Reset the global metrics bridge instance without stopping it.

    Useful for testing to clear the global state. Note: This does NOT
    stop a running bridge - use stop_metrics_bridge() first if needed.

    Example:
        >>> await stop_metrics_bridge()
        >>> reset_metrics_bridge()
    """
    global _bridge
    _bridge = None


async def collect_metrics_once() -> None:
    """
    Perform a single metrics collection cycle.

    Use this for cron-job style execution where you want to collect
    metrics on-demand rather than running a continuous polling loop.

    Requires environment variables:
        LANGFUSE_PUBLIC_KEY: Your Langfuse public key
        LANGFUSE_SECRET_KEY: Your Langfuse secret key
        OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4317)

    Example:
        >>> # In a cron job or scheduled task:
        >>> await collect_metrics_once()
    """
    bridge = MetricsBridge(poll_interval=60.0, lookback_window=300.0)
    bridge._setup_langfuse()
    bridge._setup_otel()

    try:
        await bridge._collect_once()
        # Force a metric export
        if bridge._meter_provider is not None:
            bridge._meter_provider.force_flush()
    finally:
        # Cleanup
        if bridge._meter_provider is not None:
            bridge._meter_provider.shutdown()
        if bridge._langfuse_client is not None:
            bridge._langfuse_client.close()


async def stop_metrics_bridge() -> None:
    """
    Stop the global metrics bridge.

    Stops the polling loop, flushes pending metrics, and releases resources.

    Example:
        >>> bridge = init_metrics_bridge()
        >>> # ... application runs ...
        >>> await stop_metrics_bridge()
    """
    global _bridge

    if _bridge is not None:
        await _bridge.stop()
        _bridge = None

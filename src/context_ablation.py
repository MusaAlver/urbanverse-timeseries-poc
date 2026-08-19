
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ContextSetting:
    key: str
    label: str
    points: int
    duration: str

CONTEXT_SETTINGS = (
    ContextSetting("8h", "8 hours", 96, "8 h"),
    ContextSetting("24h", "24 hours", 288, "24 h"),
    ContextSetting("7d", "7 days", 2016, "7 days"),
)

FORECAST_HORIZON_POINTS = 24  # fixed 2-hour forecast at 5-minute resolution

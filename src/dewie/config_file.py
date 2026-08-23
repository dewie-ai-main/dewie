# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
dewie/config_file.py — Load and expose dewie.yml operator config.

Usage:
    from dewie.config_file import cfg

    workers   = cfg.enrichment.workers
    auto_restart = cfg.enrichment.auto_restart
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


def _default_config_path() -> Path:
    # Honour DEWIE_DATA_DIR so Docker deployments persist config on the volume.
    data_dir = os.environ.get("DEWIE_DATA_DIR", "")
    if data_dir:
        return Path(data_dir) / "dewie.yml"
    return Path(__file__).resolve().parents[2] / "dewie.yml"

_CONFIG_PATH = _default_config_path()


class EnrichmentConfig(BaseModel):
    workers: int = Field(default=3, ge=1, le=8)
    batch_size: int = Field(default=5, ge=1, le=100, description="Docs per worker per 30s tick")
    auto_restart: bool = True
    max_restarts_per_hour: int = Field(default=3, ge=0)


class IngestConfig(BaseModel):
    rss_enabled: bool = True
    wiki_enabled: bool = False
    reddit_enabled: bool = False
    blocked_sources: list[str] = Field(
        default_factory=list,
        description="Source hostnames that should never be ingested.",
    )
    low_quality_sources: list[str] = Field(
        default_factory=list,
        description="Source hostnames considered low quality.",
    )


class MonitoringConfig(BaseModel):
    timezone: str = Field(default="America/Los_Angeles", description="Timezone for quiet-hours and monitoring timestamps")
    quiet_start_hour: int = Field(default=23, ge=0, le=23)
    quiet_end_hour: int = Field(default=7, ge=0, le=23)
    alert_dedup_seconds: int = Field(default=1800, ge=0)

    @model_validator(mode="after")
    def quiet_hours_differ(self) -> MonitoringConfig:
        if self.quiet_start_hour == self.quiet_end_hour:
            raise ValueError(
                f"quiet_start_hour and quiet_end_hour cannot be equal ({self.quiet_start_hour}); "
                "that would suppress alerts for the entire day."
            )
        return self


class WatcherConfig(BaseModel):
    enabled: bool = False
    watch_dir: str = "data/bodies"


class YoutubeConfig(BaseModel):
    enabled: bool = False
    channels: list[str] = Field(default_factory=list)
    videos_per_channel: int = Field(default=20, ge=1, le=200)
    schedule_hours: int = Field(default=24, ge=1, le=168)


class DewieConfig(BaseModel):
    enrichment: EnrichmentConfig = EnrichmentConfig()
    ingest: IngestConfig = IngestConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    watcher: WatcherConfig = WatcherConfig()
    youtube: YoutubeConfig = YoutubeConfig()


def _load(path: Path = _CONFIG_PATH) -> DewieConfig:
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        return DewieConfig.model_validate(raw)
    return DewieConfig()


# Module-level singleton — loaded once at import time
cfg: DewieConfig = _load()

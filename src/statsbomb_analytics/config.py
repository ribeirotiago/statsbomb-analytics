"""Central project configuration (paths and data source)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings read from environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    statsbomb_data_dir: Path = PROJECT_ROOT.parent / "statsbomb-open-data" / "data"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"


def get_settings() -> Settings:
    return Settings()

"""Settings read from environment variables (prefix STAC_EO_MCP_)."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CatalogName(StrEnum):
    PLANETARY_COMPUTER = "planetary-computer"
    EARTH_SEARCH = "earth-search"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STAC_EO_MCP_", extra="ignore")

    catalog: CatalogName = Field(
        default=CatalogName.PLANETARY_COMPUTER,
        description="Catalog used when a tool call does not name one.",
    )
    output_dir: Path = Field(
        default=Path.home() / "stac-eo-mcp-output",
        description="Folder where PNG/GeoTIFF outputs are written.",
    )
    max_area_km2: float = Field(
        default=100.0, gt=0, description="Largest bbox accepted for raster reads."
    )
    max_scenes: int = Field(
        default=24, ge=1, le=200, description="Most scenes read by time_series."
    )
    max_search_limit: int = Field(
        default=50, ge=1, le=500, description="Most scenes returned by search_scenes."
    )
    log_level: str = Field(default="INFO", description="Python logging level (to stderr).")


@lru_cache
def get_settings() -> Settings:
    return Settings()

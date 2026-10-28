from pathlib import Path

import pytest
from pydantic import ValidationError

from stac_eo_mcp.config import CatalogName, Settings


def test_defaults(monkeypatch):
    for var in ("CATALOG", "OUTPUT_DIR", "MAX_AREA_KM2", "MAX_SCENES"):
        monkeypatch.delenv(f"STAC_EO_MCP_{var}", raising=False)
    s = Settings()
    assert s.catalog is CatalogName.PLANETARY_COMPUTER
    assert s.output_dir == Path.home() / "stac-eo-mcp-output"
    assert s.max_area_km2 == 100
    assert s.max_scenes == 24


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("STAC_EO_MCP_CATALOG", "earth-search")
    monkeypatch.setenv("STAC_EO_MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("STAC_EO_MCP_MAX_AREA_KM2", "250")
    s = Settings()
    assert s.catalog is CatalogName.EARTH_SEARCH
    assert s.output_dir == tmp_path
    assert s.max_area_km2 == 250


def test_rejects_unknown_catalog(monkeypatch):
    monkeypatch.setenv("STAC_EO_MCP_CATALOG", "nasa")
    with pytest.raises(ValidationError):
        Settings()

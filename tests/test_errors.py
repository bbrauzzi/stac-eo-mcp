from stac_eo_mcp.errors import NoScenesError, StacMcpError


def test_str_includes_hint():
    err = NoScenesError("No scenes with cloud cover < 20%.", hint="Widen the date range.")
    assert str(err) == "No scenes with cloud cover < 20%. Hint: Widen the date range."
    assert isinstance(err, StacMcpError)


def test_str_without_hint():
    assert str(StacMcpError("boom")) == "boom"

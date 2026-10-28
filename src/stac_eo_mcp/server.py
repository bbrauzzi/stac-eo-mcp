"""MCP server entry point. Tools are added in later steps."""

import logging
import sys


def main() -> None:
    # stdout is reserved for the stdio protocol: logs go to stderr only.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.getLogger(__name__).info("stac-eo-mcp scaffold: no tools yet")

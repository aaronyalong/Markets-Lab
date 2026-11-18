"""Smoke test: the package and its subpackages are installed and importable."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    ["markets_lab", "markets_lab.collectors", "markets_lab.analysis"],
)
def test_package_imports(module: str) -> None:
    assert importlib.import_module(module) is not None

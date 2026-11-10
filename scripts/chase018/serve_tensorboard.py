#!/usr/bin/env python3
"""Serve TensorBoard without the Open3D plugin, which imports a broken TensorFlow."""
from __future__ import annotations

import pkg_resources

_iter_entry_points = pkg_resources.iter_entry_points


def _skip_open3d(group, name=None):
    for entry in _iter_entry_points(group, name):
        module = getattr(entry, "module_name", "") or ""
        if module.startswith("open3d"):
            continue
        yield entry


pkg_resources.iter_entry_points = _skip_open3d

import tensorboard.default as tb_default  # noqa: E402

_get_plugins = tb_default.get_plugins


def _unique_plugins():
    seen: set[str] = set()
    plugins = []
    for plugin in _get_plugins():
        name = getattr(plugin, "plugin_name", None)
        if name in seen:
            continue
        if name:
            seen.add(name)
        plugins.append(plugin)
    return plugins


tb_default.get_plugins = _unique_plugins

from tensorboard.main import run_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_main())

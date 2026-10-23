"""Scenario parameters: every value carries its source and date.

A scenario is a YAML file (config/*.yaml).  Each leaf parameter is a mapping
with keys value, source, date and optional range and note.  `Params` gives
attribute access to values (p.electorate.E_full) and keeps the metadata for
the output files, so every CSV can say which values produced it.
"""
from __future__ import annotations

import copy
import datetime as _dt
import os
from typing import Any

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "config")
DATA_DIR = os.path.join(ROOT, "data")


def _is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node


class Section:
    """Attribute access to the values of one YAML section."""

    def __init__(self, name: str, node: dict):
        self._name = name
        self._node = node

    def __getattr__(self, key: str) -> Any:
        node = self.__dict__["_node"]
        if key not in node:
            raise AttributeError(f"{self.__dict__['_name']}.{key} is not in the scenario")
        leaf = node[key]
        return leaf["value"] if _is_leaf(leaf) else leaf

    def meta(self, key: str) -> dict:
        return dict(self._node[key])

    def keys(self):
        return list(self._node.keys())


class Params:
    def __init__(self, tree: dict, path: str | None = None):
        self.tree = tree
        self.path = path

    @classmethod
    def load(cls, name_or_path: str = "baseline_2025") -> "Params":
        path = name_or_path
        if not os.path.exists(path):
            path = os.path.join(CONFIG_DIR, name_or_path + (".yaml" if not name_or_path.endswith(".yaml") else ""))
        with open(path, encoding="utf-8") as f:
            tree = yaml.safe_load(f)
        return cls(tree, path)

    def __getattr__(self, key: str) -> Any:
        tree = self.__dict__["tree"]
        if key in tree and isinstance(tree[key], dict):
            return Section(key, tree[key])
        raise AttributeError(key)

    def override(self, **changes) -> "Params":
        """Return a copy with values replaced: override(**{'electorate.R': 1.95e6})."""
        tree = copy.deepcopy(self.tree)
        for dotted, value in changes.items():
            sec, key = dotted.split(".", 1)
            leaf = tree.setdefault(sec, {}).setdefault(key, {"source": "override", "date": str(_dt.date.today())})
            if not isinstance(leaf, dict):
                leaf = {"source": "override", "date": str(_dt.date.today())}
                tree[sec][key] = leaf
            leaf["value"] = value
            leaf["source"] = f"override ({leaf.get('source', '')})"
        return Params(tree, self.path)

    def provenance(self) -> list[dict]:
        """Flat list of (parameter, value, source, date) for output metadata."""
        out = []
        for sec, node in self.tree.items():
            if not isinstance(node, dict):
                continue
            for key, leaf in node.items():
                if _is_leaf(leaf):
                    out.append(dict(parameter=f"{sec}.{key}", value=leaf["value"],
                                    source=leaf.get("source", ""), date=str(leaf.get("date", "")),
                                    range=leaf.get("range", "")))
        return out

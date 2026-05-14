"""Probe-point grammar parser — Python mirror of crates/rocket-surgeon-probes/src/grammar.rs.

Grammar (EBNF): protocol/probe-grammar.ebnf
Five-level hierarchical namespace: model:rank:layer:component:event
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ComponentOrWild",
    "ComponentSeg",
    "IndexedSeg",
    "NameOrWild",
    "NamedSeg",
    "NumOrWild",
    "ParseError",
    "ProbePoint",
    "Wildcard",
]

_U32_MAX = 0xFFFF_FFFF


# ---------------------------------------------------------------------------
# AST types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Wildcard:
    """Matches any value in a probe-point field."""

    def __repr__(self) -> str:
        return "Wildcard()"


@dataclass(frozen=True, slots=True)
class NamedSeg:
    name: str


@dataclass(frozen=True, slots=True)
class IndexedSeg:
    name: str
    index: int


ComponentSeg = NamedSeg | IndexedSeg

NameOrWild = Wildcard | str
NumOrWild = Wildcard | int
ComponentOrWild = Wildcard | tuple[ComponentSeg, ...]


class ParseError(Exception):
    def __init__(self, message: str, offset: int) -> None:
        super().__init__(message)
        self.offset = offset


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_DIGITS_RE = re.compile(r"[0-9]+")


class _Parser:
    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0

    @property
    def _remaining(self) -> str:
        return self._text[self._pos :]

    def _error(self, msg: str) -> ParseError:
        return ParseError(f"{msg} at offset {self._pos}", self._pos)

    def _expect_char(self, ch: str) -> None:
        if self._pos >= len(self._text) or self._text[self._pos] != ch:
            raise self._error(f"expected '{ch}'")
        self._pos += 1

    def _at_end(self) -> bool:
        return self._pos >= len(self._text)

    # -- atoms --

    def _identifier(self) -> str:
        m = _IDENT_RE.match(self._remaining)
        if not m:
            raise self._error("expected identifier")
        self._pos += m.end()
        return m.group()

    def _non_neg_integer(self) -> int:
        m = _DIGITS_RE.match(self._remaining)
        if not m:
            raise self._error("expected non-negative integer")
        self._pos += m.end()
        val = int(m.group())
        if val > _U32_MAX:
            raise self._error("integer exceeds u32 range")
        return val

    # -- field parsers --

    def _name_or_wild(self) -> NameOrWild:
        if not self._at_end() and self._text[self._pos] == "*":
            self._pos += 1
            return Wildcard()
        return self._identifier()

    def _num_or_wild(self) -> NumOrWild:
        if not self._at_end() and self._text[self._pos] == "*":
            self._pos += 1
            return Wildcard()
        return self._non_neg_integer()

    def _component_seg(self) -> ComponentSeg:
        name = self._identifier()
        if not self._at_end() and self._text[self._pos] == "[":
            self._pos += 1
            idx = self._non_neg_integer()
            self._expect_char("]")
            return IndexedSeg(name, idx)
        return NamedSeg(name)

    def _component_or_wild(self) -> ComponentOrWild:
        if not self._at_end() and self._text[self._pos] == "*":
            self._pos += 1
            return Wildcard()
        segs: list[ComponentSeg] = [self._component_seg()]
        while not self._at_end() and self._text[self._pos] == ".":
            self._pos += 1
            segs.append(self._component_seg())
        return tuple(segs)

    def parse(self) -> ProbePoint:
        model = self._name_or_wild()
        self._expect_char(":")
        rank = self._num_or_wild()
        self._expect_char(":")
        layer = self._num_or_wild()
        self._expect_char(":")
        component = self._component_or_wild()
        self._expect_char(":")
        event = self._name_or_wild()
        if not self._at_end():
            raise self._error("unexpected trailing input")
        return ProbePoint(
            model=model,
            rank=rank,
            layer=layer,
            component=component,
            event=event,
        )


# ---------------------------------------------------------------------------
# ProbePoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbePoint:
    model: NameOrWild
    rank: NumOrWild
    layer: NumOrWild
    component: ComponentOrWild
    event: NameOrWild

    @classmethod
    def parse(cls, text: str) -> ProbePoint:
        return _Parser(text).parse()

    def matches(self, other: ProbePoint) -> bool:
        return (
            _name_matches(self.model, other.model)
            and _num_matches(self.rank, other.rank)
            and _num_matches(self.layer, other.layer)
            and _component_matches(self.component, other.component)
            and _name_matches(self.event, other.event)
        )

    def __str__(self) -> str:
        return (
            f"{_fmt_name(self.model)}:{_fmt_num(self.rank)}:{_fmt_num(self.layer)}"
            f":{_fmt_component(self.component)}:{_fmt_name(self.event)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": _name_to_dict(self.model),
            "rank": _num_to_dict(self.rank),
            "layer": _num_to_dict(self.layer),
            "component": _component_to_dict(self.component),
            "event": _name_to_dict(self.event),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProbePoint:
        return cls(
            model=_name_from_dict(d["model"]),
            rank=_num_from_dict(d["rank"]),
            layer=_num_from_dict(d["layer"]),
            component=_component_from_dict(d["component"]),
            event=_name_from_dict(d["event"]),
        )


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _name_matches(pattern: NameOrWild, target: NameOrWild) -> bool:
    if isinstance(pattern, Wildcard) or isinstance(target, Wildcard):
        return True
    return pattern == target


def _num_matches(pattern: NumOrWild, target: NumOrWild) -> bool:
    if isinstance(pattern, Wildcard) or isinstance(target, Wildcard):
        return True
    return pattern == target


def _component_matches(pattern: ComponentOrWild, target: ComponentOrWild) -> bool:
    if isinstance(pattern, Wildcard) or isinstance(target, Wildcard):
        return True
    return pattern == target


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _fmt_name(v: NameOrWild) -> str:
    return "*" if isinstance(v, Wildcard) else v


def _fmt_num(v: NumOrWild) -> str:
    return "*" if isinstance(v, Wildcard) else str(v)


def _fmt_seg(seg: ComponentSeg) -> str:
    if isinstance(seg, IndexedSeg):
        return f"{seg.name}[{seg.index}]"
    return seg.name


def _fmt_component(v: ComponentOrWild) -> str:
    if isinstance(v, Wildcard):
        return "*"
    return ".".join(_fmt_seg(s) for s in v)


# ---------------------------------------------------------------------------
# Serde helpers (mirror Rust's default serde for enums)
# ---------------------------------------------------------------------------


def _name_to_dict(v: NameOrWild) -> dict[str, str] | str:
    if isinstance(v, Wildcard):
        return "Wildcard"
    return {"Name": v}


def _num_to_dict(v: NumOrWild) -> dict[str, int] | str:
    if isinstance(v, Wildcard):
        return "Wildcard"
    return {"Num": v}


def _seg_to_dict(seg: ComponentSeg) -> dict[str, Any]:
    if isinstance(seg, IndexedSeg):
        return {"Indexed": {"name": seg.name, "index": seg.index}}
    return {"Named": seg.name}


def _component_to_dict(v: ComponentOrWild) -> dict[str, Any] | str:
    if isinstance(v, Wildcard):
        return "Wildcard"
    return {"Path": [_seg_to_dict(s) for s in v]}


def _name_from_dict(d: dict[str, str] | str) -> NameOrWild:
    if d == "Wildcard":
        return Wildcard()
    assert isinstance(d, dict)
    return d["Name"]


def _num_from_dict(d: dict[str, int] | str) -> NumOrWild:
    if d == "Wildcard":
        return Wildcard()
    assert isinstance(d, dict)
    return d["Num"]


def _seg_from_dict(d: dict[str, Any]) -> ComponentSeg:
    if "Indexed" in d:
        inner = d["Indexed"]
        return IndexedSeg(inner["name"], inner["index"])
    return NamedSeg(d["Named"])


def _component_from_dict(d: dict[str, Any] | str) -> ComponentOrWild:
    if d == "Wildcard":
        return Wildcard()
    assert isinstance(d, dict)
    return tuple(_seg_from_dict(s) for s in d["Path"])

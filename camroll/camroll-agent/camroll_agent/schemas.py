"""Native tool schemas and strict, dependency-free argument validation."""
from __future__ import annotations

from datetime import date
import re
from typing import Any

MAX_IMAGES_PER_CALL = 6
PURPOSE_DESC = "Optional one-sentence purpose of this action. Do not provide private reasoning."
KINDS = ["both", "events", "images", "captions"]
_ID = r"^(img_|ev_)[A-Za-z0-9_-]+$"
_IMAGE_ID = r"^img_[A-Za-z0-9_-]+$"


def thought_in_args() -> bool:
    """Legacy import compatibility; reasoning is never required."""
    return False


def _parameters(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": {
        "purpose": {"type": "string", "maxLength": 300, "description": PURPOSE_DESC},
        # Accept old callers without requiring or asking for hidden reasoning.
        "thought": {"type": "string", "maxLength": 300, "description": "Deprecated optional action purpose."},
        **properties,
    }, "required": required, "additionalProperties": False}


_COMMON = {
    "kind": {"type": "string", "enum": KINDS},
    "date_from": {"type": ["string", "null"], "format": "date", "description": "Inclusive YYYY-MM-DD, or null."},
    "date_to": {"type": ["string", "null"], "format": "date", "description": "Inclusive YYYY-MM-DD, or null."},
}
_FILTERS = {
    "location": {"type": ["string", "null"], "maxLength": 300,
                 "description": "Trusted administrative location only; city metadata cannot prove a named venue."},
    "person": {"type": ["string", "null"], "maxLength": 300,
               "description": "Exact registered name, alias or person_id."},
    "persons": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 300},
                "maxItems": 8, "uniqueItems": True,
                "description": "Require all these named people in the same photo. Unknown matches are excluded."},
}
_QUERY = {
    "query": {"type": "string", "minLength": 1, "maxLength": 20000},
    "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
    **_COMMON,
}
_SPECS = [
    ("search", "Semantic WeMM search over photos and events; trusted date/place/person filters apply before ranking.", {**_QUERY, **_FILTERS}, ["query"]),
    ("grep", "Chinese keyword search in indexed captions, visible text and events. Zero matches only means indexed text did not match; it cannot establish visual absence.", _QUERY, ["query"]),
    ("list_by_date", "List records by known dates, location or person metadata; unknown dates cannot satisfy date filters.", {
        **_COMMON,
        **_FILTERS,
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "offset": {"type": "integer", "minimum": 0, "maximum": 1000000},
    }, []),
    ("get", "Fetch full event or photo metadata by a returned stable ID, never a disk path.", {
        "id": {"type": "string", "pattern": _ID},
    }, ["id"]),
    ("view_image", "Inspect 1 to 6 known photo IDs with Qwen for visual facts or comparison. Returns per-photo evidence, never an inferred identity.", {
        "image_ids": {"type": "array", "items": {"type": "string", "pattern": _IMAGE_ID},
                      "minItems": 1, "maxItems": MAX_IMAGES_PER_CALL, "uniqueItems": True},
        "prompt": {"type": "string", "minLength": 1, "maxLength": 4000},
        "target_person": {"type": ["string", "null"], "maxLength": 300,
                          "description": "Registered name/alias/person_id for a matched face plus context crop. Fails explicitly if unknown or ambiguous."},
        "face_id": {"type": ["string", "null"], "pattern": r"^face_[A-Za-z0-9_-]+$",
                    "description": "Explicit face_id returned by query_photo_people; use one photo when selecting this."},
    }, ["image_ids", "prompt"]),
    ("query_photo_people", "Read stored named-reference face matches and explicit family relationships for 1–6 photo IDs. Unknown/unprocessed is not proof of absence; scores are similarities, not accuracy.", {
        "image_ids": {"type": "array", "items": {"type": "string", "pattern": _IMAGE_ID},
                      "minItems": 1, "maxItems": MAX_IMAGES_PER_CALL, "uniqueItems": True},
        "person": _FILTERS["person"],
    }, ["image_ids"]),
]
TOOLS = [{"type": "function", "function": {"name": name, "description": desc,
          "parameters": _parameters(properties, required)}} for name, desc, properties, required in _SPECS]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


class ToolValidationError(ValueError):
    def __init__(self, field: str, reason: str, allowed: Any = None):
        super().__init__(f"{field}: {reason}")
        self.detail = {"code": "invalid_arguments", "field": field, "reason": reason}
        if allowed is not None:
            self.detail["allowed"] = allowed


def validate_schema(value: Any, schema: dict, field: str = "arguments") -> None:
    """Validate the schema features used by our tools and evidence responses."""
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    checks = {
        "null": lambda x: x is None,
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
    }
    if types and not any(checks[t](value) for t in types):
        raise ToolValidationError(field, "incorrect JSON type", types)
    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(field, "value not allowed", schema["enum"])
    if value is None:
        return
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ToolValidationError(f"{field}.{key}", "required field is missing")
        for key, item in value.items():
            if key not in props and schema.get("additionalProperties") is False:
                raise ToolValidationError(f"{field}.{key}", "unknown field", sorted(props))
            if key in props:
                validate_schema(item, props[key], f"{field}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            raise ToolValidationError(field, "incorrect array length", {"min": schema.get("minItems"), "max": schema.get("maxItems")})
        if schema.get("uniqueItems") and len({repr(x) for x in value}) != len(value):
            raise ToolValidationError(field, "duplicate items are not allowed")
        for i, item in enumerate(value):
            validate_schema(item, schema.get("items", {}), f"{field}[{i}]")
    elif isinstance(value, str):
        if len(value.strip()) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise ToolValidationError(field, "incorrect string length")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise ToolValidationError(field, "invalid ID; use a returned ID, never a path", schema["pattern"])
        if schema.get("format") == "date":
            try:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                    raise ValueError("format")
                date.fromisoformat(value)
            except ValueError:
                raise ToolValidationError(field, "invalid date", "YYYY-MM-DD") from None
    elif isinstance(value, (int, float)):
        if value < schema.get("minimum", -float("inf")) or value > schema.get("maximum", float("inf")):
            raise ToolValidationError(field, "out of range", {"min": schema.get("minimum"), "max": schema.get("maximum")})


def validate_arguments(name: str, arguments: Any) -> dict:
    if name not in SCHEMAS:
        raise ToolValidationError("function.name", "unknown tool", TOOL_NAMES)
    validate_schema(arguments, SCHEMAS[name])
    if arguments.get("date_from") and arguments.get("date_to") and arguments["date_from"] > arguments["date_to"]:
        raise ToolValidationError("arguments.date_to", "must be on or after date_from")
    return {k: v for k, v in arguments.items() if k not in ("thought", "purpose")}

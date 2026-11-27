"""Provenance metadata for extracted tables, and removal of URL credentials."""

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from zarrextract._errors import ZarrExtractError

PROVENANCE_VERSION = "1"
PREFIX = "zarrextract."
VERSION_KEY = f"{PREFIX}provenance_version"
SOURCE_KEY = f"{PREFIX}source"
VARIABLE_KEY = f"{PREFIX}variable"
SELECTIONS_KEY = f"{PREFIX}selections"
PACKAGE_VERSION_KEY = f"{PREFIX}version"
EXTRACTED_AT_KEY = f"{PREFIX}extracted_at"

_FIELD_KEYS = ("units", "long_name")

# fsspec chains protocols with "::". The lookahead skips the "::" inside an
# IPv6 host such as http://[::1]/.
_CHAIN = re.compile(r"::(?=[A-Za-z][A-Za-z0-9+.\-]*(?:::|://))")
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>]+")
_TRAILING = ".,;:)]}"


def _sanitize_part(part: str) -> str:
    scheme, separator, _ = part.partition("://")
    if not separator:
        return part
    try:
        split = urlsplit(part)
    except ValueError:
        return f"{scheme}://<removed>"
    return f"{scheme}://{split.netloc.rpartition('@')[2]}{split.path}"


def sanitize_store(store: str) -> str:
    """Remove the user name, password, query and fragment from a store URL.

    Chained fsspec URLs (``simplecache::s3://...``) are cleaned part by part.
    A string that is not a URL, such as a local path, is returned as it is.
    """
    if "://" not in store:
        return store
    return "::".join(_sanitize_part(part) for part in _CHAIN.split(store))


def _redact_match(match: re.Match[str]) -> str:
    run = match.group()
    body = run.rstrip(_TRAILING)
    return sanitize_store(body) + run[len(body) :]


def redact_urls(text: str) -> str:
    """Apply ``sanitize_store`` to every URL inside a piece of text."""
    return _URL.sub(_redact_match, text)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value))
    if isinstance(value, date):  # datetime is a subclass of date
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        # JSON has no infinity. Open-ended ranges such as (-inf, 10) are allowed.
        # NaN never gets here: extract() rejects it.
        return str(value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def selections_json(selections: Mapping[str, Any]) -> str:
    """Encode selections as JSON, as given: dates as ISO strings, pairs as lists."""
    return json.dumps(
        {name: _jsonable(value) for name, value in selections.items()},
        ensure_ascii=False,
        allow_nan=False,
    )


def provenance_metadata(
    store: str, variable: str, selections: Mapping[str, Any]
) -> dict[str, str]:
    """Return the schema metadata for a table that was just extracted."""
    now = datetime.now(UTC).replace(microsecond=0)
    return {
        VERSION_KEY: PROVENANCE_VERSION,
        SOURCE_KEY: sanitize_store(store),
        VARIABLE_KEY: variable,
        SELECTIONS_KEY: selections_json(selections),
        PACKAGE_VERSION_KEY: version("zarrextract"),
        EXTRACTED_AT_KEY: now.isoformat().replace("+00:00", "Z"),
    }


def field_metadata(
    attrs: Mapping[str, Any], *, decoded_time: bool
) -> dict[str, str] | None:
    """Return the ``units`` and ``long_name`` of an array for its Arrow field.

    The units of a decoded time coordinate (``"days since ..."``) describe the
    stored encoding, not the timestamps, so they are left out.
    """
    metadata: dict[str, str] = {}
    units = attrs.get("units")
    if isinstance(units, str) and units and not decoded_time:
        metadata["units"] = units
    long_name = attrs.get("long_name")
    if isinstance(long_name, str) and long_name:
        metadata["long_name"] = long_name
    return metadata or None


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where an extracted table came from.

    Read it with ``provenance()`` from a table, or from a file written by
    ``write()``.

    Attributes:
        provenance_version: The layout of the metadata, currently ``"1"``.
        source: The store, without any user name, password, query or fragment.
        variable: The variable that was extracted.
        selections: The selections as given: numbers, strings (dates as ISO
            strings) and ``[start, stop]`` lists. An infinite end is the
            string ``"inf"`` or ``"-inf"``.
        version: The version of zarrextract that extracted the table.
        extracted_at: When the table was extracted, in UTC, to the second.
        columns: The ``units`` and ``long_name`` of each column that has them.
    """

    provenance_version: str
    source: str
    variable: str
    selections: dict[str, Any]
    version: str
    extracted_at: datetime
    columns: dict[str, dict[str, str]]


def _text(value: bytes | str) -> str:
    return value if isinstance(value, str) else value.decode("utf-8", "replace")


def _build(
    meta: Mapping[str, Any],
    columns: dict[str, dict[str, str]],
    label: str,
) -> Provenance | None:
    if not meta:
        return None

    def malformed(problem: str) -> ZarrExtractError:
        return ZarrExtractError(f"Malformed provenance in {label}: {problem}.")

    found = meta.get(VERSION_KEY)
    if found is None:
        raise malformed(f"{VERSION_KEY} is missing")
    if found != PROVENANCE_VERSION:
        raise ZarrExtractError(
            f"Unsupported provenance version {found!r} in {label}; upgrade zarrextract."
        )
    strings: dict[str, str] = {}
    for key in (SOURCE_KEY, VARIABLE_KEY, PACKAGE_VERSION_KEY, EXTRACTED_AT_KEY):
        value = meta.get(key)
        if not isinstance(value, str):
            raise malformed(f"{key} is missing")
        strings[key] = value
    selections = meta.get(SELECTIONS_KEY)
    if isinstance(selections, str):
        try:
            selections = json.loads(selections)
        except ValueError:
            raise malformed(f"{SELECTIONS_KEY} is not valid JSON") from None
    if not isinstance(selections, dict):
        raise malformed(f"{SELECTIONS_KEY} is missing or is not a JSON object")
    try:
        extracted_at = datetime.fromisoformat(strings[EXTRACTED_AT_KEY])
    except ValueError:
        raise malformed(f"{EXTRACTED_AT_KEY} is not an ISO 8601 time") from None
    if extracted_at.tzinfo is None:
        extracted_at = extracted_at.replace(tzinfo=UTC)
    return Provenance(
        provenance_version=found,
        source=strings[SOURCE_KEY],
        variable=strings[VARIABLE_KEY],
        selections=selections,
        version=strings[PACKAGE_VERSION_KEY],
        extracted_at=extracted_at,
        columns=columns,
    )


def _from_schema(schema: pa.Schema, label: str) -> Provenance | None:
    meta = {
        _text(key): _text(value)
        for key, value in (schema.metadata or {}).items()
        if _text(key).startswith(PREFIX)
    }
    columns: dict[str, dict[str, str]] = {}
    for field in schema:
        entries = {
            _text(key): _text(value)
            for key, value in (field.metadata or {}).items()
            if _text(key) in _FIELD_KEYS
        }
        if entries:
            columns[field.name] = entries
    return _build(meta, columns, label)


def sidecar_path(path: Path) -> Path:
    """Return the metadata file that goes with a CSV file: ``<name>.csv.json``."""
    return path.with_name(path.name + ".json")


def sidecar_payload(schema: pa.Schema) -> dict[str, Any]:
    """Describe a table's columns and schema metadata for the CSV sidecar."""
    metadata: dict[str, Any] = {}
    for key, value in (schema.metadata or {}).items():
        name, text = _text(key), _text(value)
        if name == SELECTIONS_KEY:
            try:
                metadata[name] = json.loads(text)
            except ValueError:
                metadata[name] = text
        else:
            metadata[name] = text
    columns: list[dict[str, Any]] = []
    for field in schema:
        entry: dict[str, Any] = {"name": field.name, "type": str(field.type)}
        if field.metadata:
            entry["metadata"] = {
                _text(key): _text(value) for key, value in field.metadata.items()
            }
        columns.append(entry)
    return {"columns": columns, "metadata": metadata}


def _from_sidecar(path: Path) -> Provenance | None:
    sidecar = sidecar_path(path)
    try:
        text = sidecar.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    where = f"{str(sidecar)!r}"
    try:
        payload = json.loads(text)
    except ValueError:
        raise ZarrExtractError(
            f"Malformed provenance in {where}: the file is not valid JSON."
        ) from None
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ZarrExtractError(
            f"Malformed provenance in {where}: it has no 'metadata' object."
        )
    meta = {
        key: value
        for key, value in payload["metadata"].items()
        if key.startswith(PREFIX)
    }
    columns: dict[str, dict[str, str]] = {}
    for column in payload.get("columns") or []:
        field_meta = column.get("metadata") if isinstance(column, dict) else None
        if isinstance(field_meta, dict):
            entries = {
                key: value
                for key, value in field_meta.items()
                if key in _FIELD_KEYS and isinstance(value, str)
            }
            if entries:
                columns[str(column.get("name"))] = entries
    return _build(meta, columns, where)


def provenance(source: pa.Table | str | os.PathLike[str]) -> Provenance | None:
    """Read the provenance of a table, or of a file written by ``write()``.

    For a file, only the schema (or the ``.csv.json`` file next to a CSV) is
    read, not the data. The suffix selects the format: ``.parquet``, ``.arrow``,
    ``.feather`` or ``.csv``.

    Returns:
        The provenance, or ``None`` for a table that did not come from
        ``extract()`` and for a CSV file without a ``.csv.json`` file.

    Raises:
        ZarrExtractError: The provenance is malformed, was written in a layout
            this version does not know, or the file has an unknown suffix.
    """
    if isinstance(source, pa.Table):
        return _from_schema(source.schema, "the table")
    text = os.fspath(source)
    path = Path(text)
    label = repr(sanitize_store(text))
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _from_schema(pq.read_schema(path), label)
    if suffix in (".arrow", ".feather"):
        with pa.OSFile(str(path), "rb") as file:
            return _from_schema(pa.ipc.open_file(file).schema, label)
    if suffix == ".csv":
        return _from_sidecar(path)
    raise ZarrExtractError(
        f"Cannot read provenance from {label}: unknown suffix {path.suffix!r}. "
        "Use .parquet, .csv, .arrow or .feather."
    )

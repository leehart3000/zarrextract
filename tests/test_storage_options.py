"""Check that storage_options reaches the fsspec filesystem, offline."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fsspec
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.registry import register_implementation

import zarrextract
from helpers import ZarrFormat, make_store
from zarrextract import describe, extract

received: list[dict[str, Any]] = []


class RecordingFileSystem(MemoryFileSystem):  # type: ignore[misc]
    """An in-memory filesystem that records the options it is created with."""

    protocol = "recordingmem"

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        path = path.removeprefix("recordingmem://")
        return str(super()._strip_protocol(path))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        received.append(dict(kwargs))
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True)
def recording_filesystem() -> Iterator[None]:
    received.clear()
    register_implementation("recordingmem", RecordingFileSystem, clobber=True)
    RecordingFileSystem.clear_instance_cache()
    yield
    RecordingFileSystem.clear_instance_cache()
    MemoryFileSystem.store.clear()
    MemoryFileSystem.pseudo_dirs[:] = [""]


@pytest.fixture(params=[2, 3])
def memory_store(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    """Copy a small local store into the memory filesystem and return its URL."""
    zarr_format: ZarrFormat = request.param
    local = make_store(tmp_path / "s.zarr", zarr_format)
    fsspec.filesystem("memory").put(str(local), "/s.zarr", recursive=True)
    return "recordingmem://s.zarr"


def test_extract_passes_storage_options(memory_store: str) -> None:
    table = extract(memory_store, "temp", storage_options={"marker": 1}, time=0)

    assert table.num_rows == 8
    assert any(options.get("marker") == 1 for options in received)


def test_describe_passes_storage_options(memory_store: str) -> None:
    result = describe(memory_store, storage_options={"marker": 2})

    assert "temp" in result
    assert any(options.get("marker") == 2 for options in received)


def test_storage_options_is_not_a_dimension(memory_store: str) -> None:
    with pytest.raises(zarrextract.UnknownDimensionError):
        extract(memory_store, "temp", nosuchdim=1)
    assert not any("nosuchdim" in options for options in received)


def test_storage_options_default_is_none(memory_store: str) -> None:
    assert extract(memory_store, "temp", time=0).num_rows == 8
    assert all("marker" not in options for options in received)

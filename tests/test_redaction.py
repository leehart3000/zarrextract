"""Credentials in store URLs must not appear in errors, warnings or tracebacks."""

import traceback
import warnings
from pathlib import Path
from typing import Any

import pytest
import zarr

import zarrextract as zx
from helpers import make_ome_image, make_store, ome_root
from test_describe import break_listing
from zarrextract._ome import OmeFinder
from zarrextract._provenance import redact_urls

PASSWORD = "hunter2"
TOKEN = "tok3n"
URL = f"https://user:{PASSWORD}@example.invalid/s.zarr?token={TOKEN}"
CLEAN_URL = "https://example.invalid/s.zarr"
LEAKS = (PASSWORD, TOKEN, "token=")


def assert_no_credentials(error: BaseException) -> None:
    """The message, the traceback and the exception chain hold no credentials."""
    assert error.__cause__ is None
    assert error.__context__ is None
    text = str(error) + "".join(traceback.format_exception(error))
    for leak in LEAKS:
        assert leak not in text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"Cannot open {URL!r}: boom", f"Cannot open {CLEAN_URL!r}: boom"),
        (
            f"404, message='Not Found', url='{URL}'",
            f"404, message='Not Found', url='{CLEAN_URL}'",
        ),
        (f"see {URL}).", f"see {CLEAN_URL})."),
        (f"{URL} and s3://k:{PASSWORD}@b/x?y=1", f"{CLEAN_URL} and s3://b/x"),
        (f"(KeyError: {URL})", f"(KeyError: {CLEAN_URL})"),
        ("no url in 'A/1/1' here", "no url in 'A/1/1' here"),
        ("s3://bucket/key.zarr", "s3://bucket/key.zarr"),
        ("C:\\data\\s.zarr", "C:\\data\\s.zarr"),
    ],
)
def test_redact_urls(text: str, expected: str) -> None:
    assert redact_urls(text) == expected


def test_a_store_that_cannot_be_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError(f"cannot read {URL}/zarr.json")

    monkeypatch.setattr(zarr, "open_group", fail)

    for call in (lambda: zx.extract(URL, "temp"), lambda: zx.describe(URL)):
        with pytest.raises(zx.StoreOpenError) as info:
            call()

        assert str(info.value) == (
            f"Cannot open {CLEAN_URL!r} as a Zarr group: cannot read {CLEAN_URL}"
        )
        assert_no_credentials(info.value)


def test_a_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise ImportError(f"Install s3fs to read {URL}")

    monkeypatch.setattr(zarr, "open_group", fail)

    with pytest.raises(
        zx.StoreOpenError, match="a required package is missing"
    ) as info:
        zx.extract(URL, "temp")

    assert_no_credentials(info.value)
    assert f"Install s3fs to read {CLEAN_URL}" in str(info.value)


def test_a_client_error_while_opening(monkeypatch: pytest.MonkeyPatch) -> None:
    aiohttp = pytest.importorskip("aiohttp")
    yarl = pytest.importorskip("yarl")
    url = yarl.URL(URL)
    error = aiohttp.ClientResponseError(
        request_info=aiohttp.RequestInfo(url, "GET", {}, url),
        history=(),
        status=409,
        message="Conflict",
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(zarr, "open_group", fail)

    with pytest.raises(zx.StoreOpenError, match="the server or the connection") as info:
        zx.extract(URL, "temp")

    assert f"url='{CLEAN_URL}'" in str(info.value)
    assert_no_credentials(info.value)


def test_a_client_error_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aiohttp = pytest.importorskip("aiohttp")
    store = make_store(tmp_path / "s.zarr", 3)

    def fail(self: zarr.Array[Any], *args: object) -> object:
        raise aiohttp.ClientConnectionError(f"reset while reading {URL}")

    monkeypatch.setattr(zarr.Array, "__getitem__", fail)

    with pytest.raises(zx.StoreOpenError, match="reset while reading") as info:
        zx.extract(store, "temp", time=0)

    assert f"reset while reading {CLEAN_URL}" in str(info.value)
    assert_no_credentials(info.value)


def test_a_store_that_cannot_be_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(tmp_path / "s.zarr", 3)
    break_listing(monkeypatch, ValueError(f"409 Conflict for {URL}"))

    with pytest.raises(zx.StoreOpenError, match="cannot be listed") as info:
        zx.describe(store)

    assert f"409 Conflict for {CLEAN_URL}" in str(info.value)
    assert_no_credentials(info.value)


def leaky_open_array(self: OmeFinder, path: str) -> object:
    raise KeyError(f"nothing at {URL}")


def test_ome_warning_has_no_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_ome_image(ome_root(tmp_path / "s.zarr", "0.5"), "0.5")
    monkeypatch.setattr(OmeFinder, "_open_array", leaky_open_array)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        zx.describe(tmp_path / "s.zarr")

    messages = [str(w.message) for w in record if w.category is zx.OmeMetadataWarning]
    assert len(messages) == 1
    assert "which cannot be opened (KeyError: " in messages[0]
    assert f"nothing at {CLEAN_URL}" in messages[0]
    for leak in LEAKS:
        assert leak not in messages[0]


def test_ome_error_has_no_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_ome_image(ome_root(tmp_path / "s.zarr", "0.5"), "0.5")
    break_listing(monkeypatch, ValueError("409 Conflict"))
    monkeypatch.setattr(OmeFinder, "_open_array", leaky_open_array)

    with pytest.raises(zx.OmeMetadataError, match="Invalid OME-Zarr metadata") as info:
        zx.describe(tmp_path / "s.zarr")

    assert f"nothing at {CLEAN_URL}" in str(info.value)
    assert_no_credentials(info.value)

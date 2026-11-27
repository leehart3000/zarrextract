"""Write tables to Parquet, Arrow IPC and CSV files."""

import contextlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from zarrextract._errors import OutputExistsError, UnsupportedFormatError
from zarrextract._provenance import sanitize_store, sidecar_path, sidecar_payload

_SUFFIXES = ".parquet, .csv, .arrow or .feather"
_KNOWN = {".parquet", ".csv", ".arrow", ".feather"}


def _write_parquet(table: pa.Table, file: BinaryIO) -> None:
    pq.write_table(table, file, compression="zstd")


def _write_ipc(table: pa.Table, file: BinaryIO) -> None:
    with pa.ipc.new_file(file, table.schema) as writer:
        writer.write_table(table)


def _write_csv(table: pa.Table, file: BinaryIO) -> None:
    pa_csv.write_csv(table, file)


def _write_sidecar(table: pa.Table, file: BinaryIO) -> None:
    text = json.dumps(
        sidecar_payload(table.schema), indent=2, sort_keys=True, ensure_ascii=False
    )
    file.write(f"{text}\n".encode())


def _write_temporary(
    final: Path, writer: Callable[[BinaryIO], None], overwrite: bool
) -> Path:
    """Write a file next to ``final`` under a temporary name, and return its path.

    The file is created with the permissions the process would give any new file,
    or with those of the file it is going to replace. If ``writer`` fails, the
    temporary file is removed.
    """
    temporary = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as file:
            writer(file)
        if overwrite and final.exists():
            with contextlib.suppress(OSError):
                shutil.copymode(final, temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _create(temporary: Path, final: Path, text: str) -> None:
    """Move ``temporary`` to ``final``, which must not exist."""
    try:
        os.link(temporary, final)  # fails if final exists, even if it is new
    except FileExistsError:
        raise OutputExistsError(
            f"Cannot write {text!r}: {str(final)!r} already exists. "
            "Pass overwrite=True to replace it."
        ) from None
    except OSError:
        # No hard links on this filesystem. The check in write() has to do.
        os.replace(temporary, final)
    else:
        temporary.unlink()


def write(
    table: pa.Table, path: str | os.PathLike[str], *, overwrite: bool = False
) -> tuple[Path, ...]:
    """Write a table to a local file, in the format given by the suffix.

    * ``.parquet`` is Parquet, compressed with zstd.
    * ``.arrow`` and ``.feather`` are the Arrow IPC file format (Feather
      version 2), uncompressed.
    * ``.csv`` is CSV. CSV cannot hold metadata, so a second file, the
      ``.csv.json`` file next to it, holds the schema metadata (including the
      provenance from ``extract()``) and the Arrow type, units and long name of
      each column. ``series.csv`` gets ``series.csv.json``.

    The suffix is matched without regard to case. Parquet and Arrow files keep
    the schema metadata and the field metadata of the table, so ``units`` and
    the provenance survive a round trip. Read the provenance back with
    ``provenance()``. Parquet has no unit of seconds, so a ``timestamp[s]``
    column is stored in milliseconds and reads back as ``timestamp[ms]``, with
    the same values.

    A time column in the 360_day or all_leap calendar holds ISO 8601 strings
    and has the calendar in its field metadata, which every format keeps (CSV
    in the ``.csv.json`` file). A CSV reader may turn such strings into
    timestamps, so read the column as a string.

    The files are first written under temporary names in the same directory,
    and then moved into place. If writing fails, a file that is already there
    is not changed, and the temporary files are removed.

    Args:
        table: The table to write.
        path: A local path. URLs are not supported.
        overwrite: Replace the file (and the ``.csv.json`` file) if it exists.

    Returns:
        The path, and for ``.csv`` also its ``.csv.json`` file, in that order.

    Raises:
        UnsupportedFormatError: The path is a URL, or has no suffix or one
            that is not listed above.
        OutputExistsError: A file exists and ``overwrite`` is not set.
        TypeError: ``table`` is not a ``pyarrow.Table``.
    """
    if not isinstance(table, pa.Table):
        raise TypeError(f"write() takes a pyarrow.Table, not {type(table).__name__}.")
    text = os.fspath(path)
    if "://" in text:
        raise UnsupportedFormatError(
            f"Cannot write {sanitize_store(text)!r}: write() takes local paths only."
        )
    target = Path(text)
    suffix = target.suffix.lower()
    if not suffix:
        raise UnsupportedFormatError(
            f"Cannot write {text!r}: the path has no suffix. Use {_SUFFIXES}."
        )
    if suffix not in _KNOWN:
        raise UnsupportedFormatError(
            f"Cannot write {text!r}: unknown suffix {target.suffix!r}. Use {_SUFFIXES}."
        )

    jobs: list[tuple[Path, Callable[[BinaryIO], None]]]
    if suffix == ".parquet":
        jobs = [(target, partial(_write_parquet, table))]
    elif suffix == ".csv":
        jobs = [
            (target, partial(_write_csv, table)),
            (sidecar_path(target), partial(_write_sidecar, table)),
        ]
    else:
        jobs = [(target, partial(_write_ipc, table))]

    if not overwrite:
        if target.exists():
            raise OutputExistsError(
                f"Cannot write {text!r}: the file already exists. "
                "Pass overwrite=True to replace it."
            )
        if suffix == ".csv" and jobs[1][0].exists():
            raise OutputExistsError(
                f"Cannot write {text!r}: its metadata file {str(jobs[1][0])!r} "
                "already exists. Pass overwrite=True to replace it."
            )

    # Write every file to a temporary one first, so that a failure while writing
    # never touches a file that is already there. Only then move them into place,
    # the sidecar before the CSV.
    temporaries: list[tuple[Path, Path]] = []
    created: list[Path] = []
    try:
        for final, writer in jobs:
            temporaries.append((_write_temporary(final, writer, overwrite), final))
        for temporary, final in reversed(temporaries):
            if overwrite:
                os.replace(temporary, final)
            else:
                _create(temporary, final, text)
                created.append(final)
    except BaseException:
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)
        for final in created:  # made by this call, so not anyone's old file
            final.unlink(missing_ok=True)
        raise
    return tuple(final for final, _ in jobs)

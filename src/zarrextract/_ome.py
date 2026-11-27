"""Read OME-Zarr (OME-NGFF) metadata from the groups of a store."""

import dataclasses
import os
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import zarr
import zarr.errors

from zarrextract._errors import OmeMetadataError, OmeMetadataWarning
from zarrextract._provenance import redact_urls

SUPPORTED_VERSIONS = ("0.4", "0.5")

# Attribute keys that make a group an OME-Zarr group of some kind.
_KIND_KEYS = ("multiscales", "plate", "well", "labels", "bioformats2raw.layout")

# How many entries a __str__ shows in a list before cutting it short.
_SHOWN = 10


def _join_items(items: Sequence[str]) -> str:
    if len(items) <= _SHOWN:
        return ", ".join(items)
    return ", ".join(items[:_SHOWN]) + f", ... ({len(items) - _SHOWN} more)"


def _count(items: Sequence[object], word: str) -> str:
    return f"{len(items)} {word}" + ("" if len(items) == 1 else "s")


def _tuple_text(values: Sequence[object]) -> str:
    return "(" + ", ".join(str(v) for v in values) + ")"


def _version_text(version: str | None) -> str:
    return "version not given" if version is None else f"version {version}"


@dataclass(frozen=True, slots=True)
class OmeAxis:
    """One axis of an OME-Zarr image.

    Attributes:
        name: The axis name, such as ``"x"``.
        type: The axis type, such as ``"space"``, ``"time"`` or ``"channel"``,
            or ``None`` if the metadata does not give one.
        unit: The unit, such as ``"micrometer"``, or ``None``.
    """

    name: str
    type: str | None
    unit: str | None

    def __str__(self) -> str:
        details = [part for part in (self.type, self.unit) if part is not None]
        return f"{self.name} ({', '.join(details)})" if details else self.name


@dataclass(frozen=True, slots=True)
class OmeLevel:
    """One resolution level (one entry of ``"datasets"``) of a multiscale image.

    Attributes:
        path: The dataset path as written in the metadata, such as ``"0"``.
        name: The array's path from the store, such as ``"0/0"``. This is the
            name ``extract()`` takes.
        shape: The length of each dimension, from the array's metadata.
        chunks: The chunk length along each dimension.
        dtype: The data type, as a string.
        scale: The level's scale transform, one value per axis, or ``None``
            if the metadata was not read (an unsupported version).
        translation: The level's translation transform, or ``None`` if the
            metadata has none.
    """

    path: str
    name: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str
    scale: tuple[float, ...] | None
    translation: tuple[float, ...] | None

    def __str__(self) -> str:
        lines = [
            f"{self.name}: shape {_tuple_text(self.shape)}, {self.dtype}, "
            f"chunks {_tuple_text(self.chunks)}"
        ]
        if self.scale is not None:
            lines.append(f"  scale {_tuple_text(self.scale)}")
        if self.translation is not None:
            lines.append(f"  translation {_tuple_text(self.translation)}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class OmeMultiscale:
    """One entry of ``"multiscales"``: the axes and the resolution levels.

    Attributes:
        name: The entry's ``"name"``, or ``None``.
        axes: The axes, in the order of the array dimensions. Empty for an
            unsupported version.
        levels: The resolution levels, from the highest resolution down.
        scale: The scale that applies to every level after the level's own
            transforms, or ``None``. It is not combined into the levels.
        translation: The translation that applies after the levels' own
            transforms, or ``None``.
    """

    name: str | None
    axes: tuple[OmeAxis, ...]
    levels: tuple[OmeLevel, ...]
    scale: tuple[float, ...] | None
    translation: tuple[float, ...] | None

    def _title(self) -> str:
        return "(no name)" if self.name is None else repr(self.name)

    def _lines(self, title: str) -> list[str]:
        lines = [title]
        if self.axes:
            lines.append("  axes: " + ", ".join(str(axis) for axis in self.axes))
        else:
            lines.append("  axes: (not read)")
        transforms = []
        if self.scale is not None:
            transforms.append(f"scale {_tuple_text(self.scale)}")
        if self.translation is not None:
            transforms.append(f"translation {_tuple_text(self.translation)}")
        if transforms:
            lines.append("  after the level transforms: " + ", ".join(transforms))
        lines.append("  levels:")
        for level in self.levels:
            lines.extend("    " + line for line in str(level).splitlines())
        return lines

    def __str__(self) -> str:
        return "\n".join(self._lines(f"multiscales {self._title()}"))


@dataclass(frozen=True, slots=True)
class OmeImage:
    """An OME-Zarr image: a group with ``"multiscales"`` metadata.

    Attributes:
        name: The group's path from the store, ``""`` for the root.
        version: The OME-NGFF version, or ``None`` if the metadata does not
            give one.
        supported: False if the version is not 0.4 or 0.5. Then only the
            dataset paths are read: there are no axes, transforms or channels.
        multiscales: Every entry of ``"multiscales"``, in order.
        channel_labels: The channel labels from ``"omero"`` metadata, with
            ``None`` for a channel without a label. Empty if there is none.
        is_label: True for a label image (one with ``"image-label"``
            metadata).
        labels: The label paths listed by the ``labels`` group beside the
            resolution levels, relative to that group. Empty if there is none.
    """

    name: str
    version: str | None
    supported: bool
    multiscales: tuple[OmeMultiscale, ...]
    channel_labels: tuple[str | None, ...]
    is_label: bool
    labels: tuple[str, ...]

    @property
    def levels(self) -> tuple[OmeLevel, ...]:
        """The resolution levels of every multiscales entry, in order."""
        return tuple(level for entry in self.multiscales for level in entry.levels)

    def __str__(self) -> str:
        kind = "label image" if self.is_label else "image"
        status = _version_text(self.version)
        if not self.supported:
            status += ", not supported: only the dataset paths were read"
        lines = [f"OME-Zarr {kind} {self.name!r} ({status})"]
        count = len(self.multiscales)
        for index, entry in enumerate(self.multiscales, start=1):
            title = f"multiscales {index} of {count} {entry._title()}"
            lines.extend("  " + line for line in entry._lines(title))
        if not self.supported:
            channels = "(not read)"
        elif self.channel_labels:
            channels = _join_items(
                ["(no label)" if c is None else c for c in self.channel_labels]
            )
        else:
            channels = "(none in the metadata)"
        lines.append(f"  channels: {channels}")
        lines.append(f"  labels: {_join_items(self.labels) or '(none)'}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class OmeLabels:
    """The ``labels`` group of an OME-Zarr image.

    Attributes:
        name: The group's path from the store, such as ``"0/labels"``.
        version: The OME-NGFF version, or ``None``.
        labels: The label image paths, relative to this group.
    """

    name: str
    version: str | None
    labels: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"OME-Zarr labels group {self.name!r} ({_version_text(self.version)}): "
            f"{_count(self.labels, 'label')}: {_join_items(self.labels) or '(none)'}"
        )


@dataclass(frozen=True, slots=True)
class OmePlate:
    """An OME-Zarr high-content screening plate.

    Attributes:
        name: The group's path from the store, ``""`` for the root.
        version: The OME-NGFF version, or ``None``.
        rows: The row names, such as ``"A"``.
        columns: The column names, such as ``"1"``.
        wells: The well paths, relative to the plate, such as ``"A/1"``.
    """

    name: str
    version: str | None
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    wells: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"OME-Zarr plate {self.name!r} ({_version_text(self.version)}): "
            f"{_count(self.rows, 'row')} x {_count(self.columns, 'column')}, "
            f"{_count(self.wells, 'well')}: {_join_items(self.wells) or '(none)'}"
        )


@dataclass(frozen=True, slots=True)
class OmeWell:
    """One well of an OME-Zarr plate.

    Attributes:
        name: The group's path from the store, such as ``"A/1"``.
        version: The OME-NGFF version, or ``None``.
        images: The image paths in the well, relative to it, such as ``"0"``.
    """

    name: str
    version: str | None
    images: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"OME-Zarr well {self.name!r} ({_version_text(self.version)}): "
            f"{_count(self.images, 'image')}: {_join_items(self.images) or '(none)'}"
        )


ImagesFrom = Literal["plate", "series", "numbered"]

_IMAGES_FROM_TEXT: dict[ImagesFrom, str] = {
    "plate": "from the plate",
    "series": "from OME/series",
    "numbered": "numbered",
}


@dataclass(frozen=True, slots=True)
class OmeContainer:
    """A ``bioformats2raw`` container: a group holding one or more images.

    Attributes:
        name: The group's path from the store, ``""`` for the root.
        version: The OME-NGFF version, or ``None``.
        layout: The ``"bioformats2raw.layout"`` value, 3.
        images: The image group paths from the store.
        images_from: Where the image paths come from: ``"plate"`` (the
            container's plate metadata), ``"series"`` (the ``series`` list of
            the ``OME`` group) or ``"numbered"`` (groups ``0``, ``1``, ...).
    """

    name: str
    version: str | None
    layout: int
    images: tuple[str, ...]
    images_from: ImagesFrom

    def __str__(self) -> str:
        return (
            f"OME-Zarr bioformats2raw container {self.name!r} "
            f"(layout {self.layout}, {_version_text(self.version)}): "
            f"{_count(self.images, 'image')}, {_IMAGES_FROM_TEXT[self.images_from]}: "
            f"{_join_items(self.images) or '(none)'}"
        )


OmeGroup = OmeContainer | OmeImage | OmeLabels | OmePlate | OmeWell

# Entries of the same group are sorted in this order.
_KIND_ORDER: dict[type, int] = {
    OmeContainer: 0,
    OmePlate: 1,
    OmeWell: 2,
    OmeImage: 3,
    OmeLabels: 4,
}


class _Malformed(Exception):
    """OME metadata of one group is not valid. The argument is the problem."""

    def __init__(self, problem: str, *, fatal: bool = False) -> None:
        super().__init__(problem)
        self.problem = problem
        self.fatal = fatal
        """True if no arrays can be found without this metadata."""


def join_path(*parts: str) -> str:
    """Join group and array paths, skipping empty parts."""
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _parent_paths(path: str) -> list[str]:
    """The ancestors of ``path``, nearest first, ending with the root ``""``."""
    parts = path.split("/") if path else []
    return ["/".join(parts[:i]) for i in range(len(parts) - 1, -1, -1)]


@dataclass(frozen=True, slots=True)
class _Metadata:
    """The OME part of a group's attributes."""

    form: Literal["0.4", "0.5"]  # keys at the top level (0.4) or under "ome"
    keys: Mapping[str, Any]
    version: str | None  # as written in this group, before inheritance


def _string_version(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _Malformed('"version" must be a string')
    return value


def _v04_version(keys: Mapping[str, Any]) -> str | None:
    """The version of a v0.4 group: in multiscales, image-label, plate or well."""
    candidates: list[object] = []
    multiscales = keys.get("multiscales")
    if isinstance(multiscales, list):
        candidates.extend(
            entry.get("version") for entry in multiscales if isinstance(entry, Mapping)
        )
    for key in ("image-label", "plate", "well"):
        value = keys.get(key)
        if isinstance(value, Mapping):
            candidates.append(value.get("version"))
    for candidate in candidates:
        if candidate is not None:
            return _string_version(candidate)
    return None


def read_metadata(attrs: Mapping[str, Any]) -> _Metadata | None:
    """Pick the OME metadata from a group's attributes, or ``None`` if none.

    v0.5 keeps it under ``"ome"``, with the version in ``ome.version``. v0.4
    keeps the keys at the top level, with the version inside them.
    """
    if "ome" in attrs:
        ome = attrs["ome"]
        if not isinstance(ome, Mapping):
            raise _Malformed('"ome" must be an object')
        if not any(key in ome for key in _KIND_KEYS):
            return None
        return _Metadata("0.5", ome, _string_version(ome.get("version")))
    if not any(key in attrs for key in _KIND_KEYS):
        return None
    return _Metadata("0.4", attrs, _v04_version(attrs))


def _ome_version_only(attrs: Mapping[str, Any]) -> str | None:
    """``ome.version`` of a group, whatever else it holds, or ``None``."""
    ome = attrs.get("ome")
    if isinstance(ome, Mapping):
        version = ome.get("version")
        if isinstance(version, str):
            return version
    return None


def _numbers(
    transform: Mapping[str, Any], key: str, prefix: str, n_axes: int
) -> tuple[float, ...]:
    values = transform.get(key)
    if not isinstance(values, list) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in values
    ):
        raise _Malformed(f'{prefix}"{key}" must be a list of numbers')
    if len(values) != n_axes:
        raise _Malformed(f'{prefix}"{key}" has {len(values)} values for {n_axes} axes')
    return tuple(float(v) for v in values)


def _transforms(
    value: object, prefix: str, n_axes: int
) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
    """Read one scale, optionally followed by one translation."""
    rule = (
        f'{prefix}"coordinateTransformations" must be one scale, '
        "optionally followed by one translation"
    )
    if not isinstance(value, list) or not all(isinstance(t, Mapping) for t in value):
        raise _Malformed(rule)
    types = [t.get("type") for t in value]
    for kind in types:
        if kind not in ("scale", "translation"):
            raise _Malformed(f"{prefix}unknown transform type {kind!r}")
    if types not in (["scale"], ["scale", "translation"]):
        raise _Malformed(rule)
    scale = _numbers(value[0], "scale", prefix, n_axes)
    translation = (
        _numbers(value[1], "translation", prefix, n_axes) if len(value) == 2 else None
    )
    return scale, translation


def _axes(entry: Mapping[str, Any], prefix: str) -> tuple[OmeAxis, ...]:
    raw = entry.get("axes")
    if not isinstance(raw, list):
        raise _Malformed(f'{prefix}"axes" must be a list')
    axes: list[OmeAxis] = []
    for index, axis in enumerate(raw):
        if not isinstance(axis, Mapping) or not isinstance(axis.get("name"), str):
            raise _Malformed(f'{prefix}axis {index} has no "name"')
        name: str = axis["name"]
        details: list[str | None] = []
        for key in ("type", "unit"):
            value = axis.get(key)
            if value is not None and not isinstance(value, str):
                raise _Malformed(
                    f'{prefix}axis {name!r} has a "{key}" that is not a string'
                )
            details.append(value)
        if any(existing.name == name for existing in axes):
            raise _Malformed(f"{prefix}duplicate axis name {name!r}")
        axes.append(OmeAxis(name=name, type=details[0], unit=details[1]))
    return tuple(axes)


OpenArray = Callable[[str], "zarr.Array[Any]"]


def _multiscale(
    entry: object,
    index: int,
    group_path: str,
    supported: bool,
    open_array: OpenArray,
    found: dict[str, "zarr.Array[Any]"],
) -> OmeMultiscale:
    """Read one multiscales entry, adding the arrays it names to ``found``."""
    prefix = f"multiscales {index}: "
    if not isinstance(entry, Mapping):
        raise _Malformed(f"multiscales {index} must be an object")
    name = entry.get("name")
    if name is not None and not isinstance(name, str):
        raise _Malformed(f'{prefix}"name" must be a string')
    axes = _axes(entry, prefix) if supported else ()
    datasets = entry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise _Malformed(f'{prefix}"datasets" must be a non-empty list')
    levels: list[OmeLevel] = []
    for number, dataset in enumerate(datasets):
        if not isinstance(dataset, Mapping) or not isinstance(dataset.get("path"), str):
            raise _Malformed(f'{prefix}dataset {number} has no "path"')
        path: str = dataset["path"]
        scale: tuple[float, ...] | None = None
        translation: tuple[float, ...] | None = None
        if supported:
            scale, translation = _transforms(
                dataset.get("coordinateTransformations"),
                f"multiscales {index}, dataset {path!r}: ",
                len(axes),
            )
        array_name = join_path(group_path, path)
        cannot_open: str | None = None
        try:
            array = open_array(array_name)
        except (KeyError, FileNotFoundError, zarr.errors.BaseZarrError) as exc:
            cannot_open = f"{type(exc).__name__}: {exc}"
        if cannot_open is not None:
            # Raised outside the handler and without a cause: the text of the
            # original error can hold a URL with credentials. _bad() removes them.
            raise _Malformed(
                f"{prefix}dataset {path!r} names the array {array_name!r}, which "
                f"cannot be opened ({cannot_open})"
            )
        if supported and array.ndim != len(axes):
            raise _Malformed(
                f"{prefix}array {array_name!r} has {array.ndim} dimensions "
                f"for {len(axes)} axes"
            )
        found[array_name] = array
        levels.append(
            OmeLevel(
                path=path,
                name=array_name,
                shape=tuple(array.shape),
                chunks=tuple(array.chunks),
                dtype=str(array.dtype),
                scale=scale,
                translation=translation,
            )
        )
    entry_scale: tuple[float, ...] | None = None
    entry_translation: tuple[float, ...] | None = None
    if supported and "coordinateTransformations" in entry:
        entry_scale, entry_translation = _transforms(
            entry["coordinateTransformations"], prefix, len(axes)
        )
    return OmeMultiscale(
        name=name,
        axes=axes,
        levels=tuple(levels),
        scale=entry_scale,
        translation=entry_translation,
    )


def _channel_labels(
    meta: _Metadata, attrs: Mapping[str, Any]
) -> tuple[str | None, ...]:
    omero = meta.keys.get("omero")
    if omero is None:
        omero = attrs.get("omero")  # some v0.5 stores keep it at the top level
    if omero is None:
        return ()
    if not isinstance(omero, Mapping):
        raise _Malformed('"omero" must be an object')
    channels = omero.get("channels", [])
    if not isinstance(channels, list) or not all(
        isinstance(c, Mapping) for c in channels
    ):
        raise _Malformed('"omero.channels" must be a list')
    return tuple(
        c["label"] if isinstance(c.get("label"), str) else None for c in channels
    )


def _paths(value: object, key: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) and isinstance(item.get(field), str) for item in value
    ):
        raise _Malformed(f'"{key}" must be a list of objects with a "{field}"')
    return tuple(str(item[field]) for item in value)


def _object(keys: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = keys[key]
    if not isinstance(value, Mapping):
        raise _Malformed(f'"{key}" must be an object')
    return value


# How a group was reached: from the root or a listing ("any"), or through a
# link in another group's metadata.
Role = Literal["any", "image", "label", "well", "labels"]

# For a group reached through a link: the key it needs, and what it is.
_ROLE_NEEDS: dict[str, tuple[str, str]] = {
    "image": ("multiscales", "an image"),
    "label": ("multiscales", "a label image"),
    "well": ("well", "a well"),
}


@dataclass(slots=True)
class _Parsed:
    """Everything read from one group, before any link is followed."""

    version: str | None
    plate: OmePlate | None = None
    well: OmeWell | None = None
    labels: OmeLabels | None = None
    image: OmeImage | None = None
    image_arrays: dict[str, "zarr.Array[Any]"] = dataclasses.field(default_factory=dict)
    container: ImagesFrom | None = None
    series: tuple[str, ...] = ()


class OmeFinder:
    """Find and describe the OME groups of a store by following its metadata.

    ``open_node(path)`` returns the group or array at a path from the store,
    or ``None`` if there is none. In ``strict`` mode, used when the store
    cannot be listed, malformed metadata raises ``OmeMetadataError`` only if
    it leaves nothing to describe: that of the root group, or the ``series``
    list of a container. Any other malformed or missing group gives an
    ``OmeMetadataWarning``, and that group is left out.
    """

    def __init__(
        self,
        open_node: Callable[[str], "zarr.Group | zarr.Array[Any] | None"],
        *,
        strict: bool,
    ) -> None:
        self._open_node = open_node
        self._strict = strict
        self._visited: set[str] = set()
        self._entries: list[OmeGroup] = []
        self._well_images: dict[str, tuple[str, ...]] = {}
        self._labels: dict[str, tuple[str, ...]] = {}
        self.arrays: dict[str, zarr.Array[Any]] = {}
        """The arrays of every image level found, by name."""

    @property
    def entries(self) -> tuple[OmeGroup, ...]:
        """The OME groups found, sorted by name."""
        return tuple(
            sorted(self._entries, key=lambda e: (e.name, _KIND_ORDER[type(e)]))
        )

    def run(self, listed_groups: Iterable[str]) -> None:
        """Visit the root, then every listed group, following the metadata."""
        self.visit("", "any")
        for path in sorted(listed_groups):
            self.visit(path, "any")

    def _group(self, path: str) -> zarr.Group | None:
        node = self._open_node(path)
        return node if isinstance(node, zarr.Group) else None

    def _open_array(self, path: str) -> "zarr.Array[Any]":
        node = self._open_node(path)
        if not isinstance(node, zarr.Array):
            raise KeyError(path if node is None else f"{path!r} is a group")
        return node

    def _bad(self, where: str, problem: str, *, fatal: bool) -> None:
        message = redact_urls(
            f"Invalid OME-Zarr metadata in group {where!r}: {problem}."
        )
        if self._strict and fatal:
            raise OmeMetadataError(
                f"{message} The store cannot be listed, so its arrays can only "
                "be found from this metadata. extract() still works if you know "
                "the array's path."
            )
        warnings.warn(
            message,
            OmeMetadataWarning,
            skip_file_prefixes=(os.path.dirname(__file__),),
        )

    def _inherited_version(self, path: str) -> str | None:
        """The ``ome.version`` of the nearest ancestor that has one."""
        for parent in _parent_paths(path):
            group = self._group(parent)
            if group is not None:
                version = _ome_version_only(dict(group.attrs))
                if version is not None:
                    return version
        return None

    def visit(self, path: str, role: Role, referrer: str = "") -> None:
        """Describe the group at ``path``, then the groups its metadata names."""
        if path in self._visited:
            return
        self._visited.add(path)
        group = self._group(path)
        needs = _ROLE_NEEDS.get(role)
        if group is None:
            if needs is not None:
                self._bad(
                    path,
                    f"the {referrer} lists this group as {needs[1]}, "
                    "but the group is missing",
                    fatal=False,
                )
            return
        attrs = dict(group.attrs)
        bad: _Malformed | None = None
        try:
            meta = read_metadata(attrs)
            if needs is not None and (meta is None or needs[0] not in meta.keys):
                raise _Malformed(
                    f"the {referrer} lists this group as {needs[1]}, "
                    f'but it has no "{needs[0]}"'
                )
            if meta is None:
                return
            parsed = self._parse(path, attrs, meta)
        except _Malformed as error:
            bad = error
        if bad is not None:
            # Outside the handler, so the error raised for a fatal problem does
            # not get this one as its context.
            self._bad(path, bad.problem, fatal=path == "" or bad.fatal)
            return
        self._follow(path, role, parsed)

    def _parse(self, path: str, attrs: Mapping[str, Any], meta: _Metadata) -> _Parsed:
        """Read and check all the OME metadata of one group."""
        keys = meta.keys
        version = meta.version
        if version is None and meta.form == "0.5":
            version = self._inherited_version(path)
        supported = version is None or version in SUPPORTED_VERSIONS
        parsed = _Parsed(version)

        if "plate" in keys:
            if meta.form == "0.5" and version is None:
                raise _Malformed('"ome.version" is required for a plate')
            plate = _object(keys, "plate")
            parsed.plate = OmePlate(
                name=path,
                version=version,
                rows=_paths(plate.get("rows"), "rows", "name"),
                columns=_paths(plate.get("columns"), "columns", "name"),
                wells=_paths(plate.get("wells"), "wells", "path"),
            )
        if "well" in keys:
            well = _object(keys, "well")
            parsed.well = OmeWell(
                name=path,
                version=version,
                images=_paths(well.get("images"), "images", "path"),
            )
        if "labels" in keys:
            raw = keys["labels"]
            if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
                raise _Malformed('"labels" must be a list of paths')
            parsed.labels = OmeLabels(name=path, version=version, labels=tuple(raw))
        if "multiscales" in keys:
            parsed.image = self._image(path, attrs, meta, version, supported, parsed)
        if "bioformats2raw.layout" in keys:
            layout = keys["bioformats2raw.layout"]
            if layout != 3 or isinstance(layout, bool):
                raise _Malformed(
                    f'"bioformats2raw.layout" is {layout!r}, only 3 is known'
                )
            if parsed.plate is not None:
                parsed.container = "plate"
            else:
                series = self._series(path)
                if series is None:
                    parsed.container = "numbered"
                else:
                    parsed.container = "series"
                    parsed.series = series
        return parsed

    def _image(
        self,
        path: str,
        attrs: Mapping[str, Any],
        meta: _Metadata,
        version: str | None,
        supported: bool,
        parsed: _Parsed,
    ) -> OmeImage:
        keys = meta.keys
        is_label = "image-label" in keys
        if is_label:
            _object(keys, "image-label")
            if meta.form == "0.5" and version is None:
                raise _Malformed('"ome.version" is required for a label image')
        multiscales = keys["multiscales"]
        if not isinstance(multiscales, list) or not multiscales:
            raise _Malformed('"multiscales" must be a non-empty list')
        entries = tuple(
            _multiscale(
                entry, index, path, supported, self._open_array, parsed.image_arrays
            )
            for index, entry in enumerate(multiscales)
        )
        return OmeImage(
            name=path,
            version=version,
            supported=supported,
            multiscales=entries,
            channel_labels=_channel_labels(meta, attrs) if supported else (),
            is_label=is_label,
            labels=(),
        )

    def _series(self, path: str) -> tuple[str, ...] | None:
        """The ``series`` list of the container's ``OME`` group, if any."""
        group = self._group(join_path(path, "OME"))
        if group is None:
            return None
        attrs = dict(group.attrs)
        ome = attrs.get("ome")
        if isinstance(ome, Mapping) and "series" in ome:
            value = ome["series"]
        elif "series" in attrs:
            value = attrs["series"]
        else:
            return None
        if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
            raise _Malformed(
                '"OME" group: "series" must be a list of paths', fatal=True
            )
        return tuple(value)

    def _follow(self, path: str, role: Role, parsed: _Parsed) -> None:
        """Record a parsed group, visiting the groups its metadata names."""
        entries: list[OmeGroup] = []
        if parsed.plate is not None:
            entries.append(parsed.plate)
            for well in parsed.plate.wells:
                self.visit(join_path(path, well), "well", "plate")
        if parsed.well is not None:
            entries.append(parsed.well)
            self._well_images[path] = parsed.well.images
            for image_path in parsed.well.images:
                self.visit(join_path(path, image_path), "image", "well")
        if parsed.labels is not None:
            entries.append(parsed.labels)
            self._labels[path] = parsed.labels.labels
            for label in parsed.labels.labels:
                self.visit(join_path(path, label), "label", "labels group")
        if parsed.image is not None:
            image = parsed.image
            self.arrays.update(parsed.image_arrays)
            if not image.is_label and role != "label":
                labels_path = join_path(path, "labels")
                self.visit(labels_path, "labels")
                image = dataclasses.replace(
                    image, labels=self._labels.get(labels_path, ())
                )
            entries.append(image)
        if parsed.container is not None:
            entries.append(
                OmeContainer(
                    name=path,
                    version=parsed.version,
                    layout=3,
                    images=self._container_images(path, parsed),
                    images_from=parsed.container,
                )
            )
        self._entries.extend(entries)

    def _container_images(self, path: str, parsed: _Parsed) -> tuple[str, ...]:
        """Find, and visit, the images of a bioformats2raw container."""
        if parsed.container == "plate":
            # The wells were visited first, so their image lists are known.
            return tuple(
                join_path(path, well, image)
                for well in (parsed.plate.wells if parsed.plate else ())
                for image in self._well_images.get(join_path(path, well), ())
            )
        if parsed.container == "series":
            images = [join_path(path, p) for p in parsed.series]
        else:
            images = []
            while self._group(join_path(path, str(len(images)))) is not None:
                images.append(join_path(path, str(len(images))))
        for image_path in images:
            self.visit(image_path, "image", "bioformats2raw container")
        return tuple(images)

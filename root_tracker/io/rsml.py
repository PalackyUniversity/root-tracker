"""Read-only RSML views backed by an exact, immutable source snapshot.

Unknown XML is never regenerated when saving: original bytes are authoritative.
Image references are metadata only and are never opened by this module.
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from xml.parsers import expat

MAX_BYTES = 64 * 1024 * 1024
MAX_ELEMENTS = 1_000_000
MAX_DEPTH = 256


@dataclass(frozen=True)
class RSMLRoot:
    id: str
    plant_id: str
    parent_id: str | None
    points: tuple[tuple[float, ...], ...]
    # Index disambiguates roots whose source IDs are absent or repeated.
    parent_index: int | None = None
    plant_index: int = 0

    @property
    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.points, self.points[1:]))


@dataclass(frozen=True)
class RSMLDocument:
    source_bytes: bytes
    roots: tuple[RSMLRoot, ...]
    unit: str
    resolution: float
    image_name: str
    plant_ids: tuple[str, ...] = ()


def _validate_xml(raw: bytes) -> None:
    """Bound XML complexity before building a tree; reject all DTDs/entities."""
    parser = expat.ParserCreate()
    depth = count = 0

    def start(name, attributes):
        nonlocal depth, count
        depth += 1
        count += 1
        if depth > MAX_DEPTH or count > MAX_ELEMENTS:
            raise ValueError('RSML document exceeds XML complexity limits')

    def end(name):
        nonlocal depth
        depth -= 1

    def forbidden(*args):
        raise ValueError('RSML DTDs and entity declarations are not supported')

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.StartDoctypeDeclHandler = forbidden
    parser.EntityDeclHandler = forbidden
    parser.ExternalEntityRefHandler = forbidden
    parser.Parse(raw, True)


def read_rsml(path: str | Path) -> RSMLDocument:
    """Load RSML without processing external resources or losing extensions."""
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    return parse_rsml(raw)


def parse_rsml(raw: bytes) -> RSMLDocument:
    """Parse an immutable source snapshot with the same limits as file imports."""
    if len(raw) > MAX_BYTES:
        raise ValueError(f'RSML file exceeds {MAX_BYTES} bytes')
    try:
        _validate_xml(raw)
        tree = ET.fromstring(raw)
    except (expat.ExpatError, ET.ParseError) as exc:
        raise ValueError(f'Invalid RSML XML: {exc}') from exc
    ns = tree.tag.split('}')[0] + '}' if tree.tag.startswith('{') else ''
    if tree.tag != ns + 'rsml':
        raise ValueError('Expected an rsml document')
    scene = tree.find(ns + 'scene')
    if scene is None:
        raise ValueError('RSML document has no scene')

    def metadata(name, default):
        return tree.findtext(f'{ns}metadata/{ns}{name}', default=default)

    unit = metadata('unit', 'pixel')
    try:
        resolution = float(metadata('resolution', '1'))
    except ValueError as exc:
        raise ValueError('RSML resolution must be a positive finite number') from exc
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('RSML resolution must be a positive finite number')
    image_name = tree.findtext(f'{ns}metadata/{ns}image/{ns}name', default='')
    roots = []
    plants = scene.findall(ns + 'plant')
    for plant_index, plant in enumerate(plants):
        plant_id = plant.get('id', '')

        def visit(element, parent_index=None):
            points = []
            for point in element.findall(f'{ns}geometry/{ns}polyline/{ns}point'):
                try:
                    axes = ('x', 'y', 'z') if 'z' in point.attrib else ('x', 'y')
                    coords = tuple(float(point.attrib[axis]) for axis in axes)
                    if not all(math.isfinite(value) for value in coords):
                        raise ValueError('nonfinite point')
                    if points and len(coords) != len(points[0]):
                        raise ValueError('mixed 2D and 3D points')
                except (KeyError, ValueError) as exc:
                    raise ValueError(f'Invalid polyline in root {element.get("id", "")!r}: {exc}') from exc
                points.append(coords)
            index = len(roots)
            roots.append(RSMLRoot(element.get('id', ''), plant_id,
                                 roots[parent_index].id if parent_index is not None else None,
                                 tuple(points), parent_index, plant_index))
            for child in element.findall(ns + 'root'):
                visit(child, index)

        for element in plant.findall(ns + 'root'):
            visit(element)
    return RSMLDocument(raw, tuple(roots), unit, resolution, image_name,
                        tuple(plant.get('id', '') for plant in plants))


def save_rsml(document: RSMLDocument, path: str | Path) -> Path:
    """Atomically save the source snapshot, preserving even unknown XML bytes."""
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.rsml-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(document.source_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination

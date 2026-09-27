"""Read embedded image metadata without decoding pixels, and compare a series.

ExifTool, when installed, includes manufacturer notes, XMP and IPTC. Pillow
provides standard EXIF and container metadata on machines without ExifTool.
"""
import base64
import binascii
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

from PIL import ExifTags, Image

from .photo_settings import photo_settings


@dataclass
class MetadataRow:
    key: str
    values: list[str | None]
    same: bool


def compare_metadata(metadata):
    metadata = [photo_settings(item) for item in metadata]
    rows = []
    warnings = [[] for _ in metadata]
    keys = sorted({key for item in metadata for key in item})
    for key in keys:
        values = [item.get(key) for item in metadata]
        counts = Counter(values)
        same = len(counts) == 1
        rows.append(MetadataRow(key, values, same))
        if same:
            continue
        maximum = max(counts.values())
        leaders = [value for value, count in counts.items() if count == maximum]
        for index, value in enumerate(values):
            if len(leaders) > 1 or value != leaders[0]:
                warnings[index].append(key)
    return rows, warnings


def _text(value):
    if isinstance(value, str) and value.startswith('base64:'):
        try:
            value = base64.b64decode(value[7:], validate=True)
        except (ValueError, binascii.Error):
            pass
    if isinstance(value, bytes):
        # Binary profiles/maker notes aren't text; avoid massive UI payloads.
        import hashlib
        return f'{len(value)} bytes; SHA-256 {hashlib.sha256(value).hexdigest()}'
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _pillow_metadata(path):
    try:
        with Image.open(path) as image:
            values = {'File:ImageWidth': str(image.width),
                      'File:ImageHeight': str(image.height),
                      'File:Format': str(image.format), 'File:Mode': image.mode}
            for key, value in image.info.items():
                if key != 'exif':
                    values[f'Container:{key}'] = _text(value)
            exif = image.getexif()
            for key, value in exif.items():
                if key not in (34665, 34853):
                    values[f'EXIF:{ExifTags.TAGS.get(key, str(key))}'] = _text(value)
            for ifd, group, names in ((34665, 'EXIF', ExifTags.TAGS),
                                      (34853, 'GPS', ExifTags.GPSTAGS)):
                for key, value in exif.get_ifd(ifd).items():
                    values[f'{group}:{names.get(key, str(key))}'] = _text(value)
            return values, ''
    except Exception as error:
        return {}, str(error)


def read_metadata(paths):
    """Return path -> (flat display values, read error), including failed files."""
    executable = shutil.which('exiftool')
    if not executable:
        return {path: _pillow_metadata(path) for path in paths}
    results = {}
    for start in range(0, len(paths), 32):
        batch = paths[start:start + 32]
        try:
            process = subprocess.run(
                [executable, '-json', '-G1', '-s', '-a', '-struct',
                 *[str(Path(path).resolve()) for path in batch]],
                capture_output=True, text=True, timeout=30, check=False,
            )
            records = json.loads(process.stdout)
            records = {record['SourceFile']: record for record in records}
            for path in batch:
                record = records.get(str(Path(path).resolve()), {})
                error = record.get('ExifTool:Error', '')
                if not record:
                    error = 'No metadata returned for this file'
                values = {key: _text(value) for key, value in record.items()
                          if key != 'SourceFile' and not key.startswith('ExifTool:')}
                results[path] = values, error
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as error:
            for path in batch:
                values, fallback_error = _pillow_metadata(path)
                results[path] = values, fallback_error or f'Limited metadata: ExifTool failed ({error})'
    return results

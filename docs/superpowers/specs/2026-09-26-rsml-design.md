# Per-image RSML replacement and export

Status: implemented; incorporates the user's authoritative-replacement requirements.

## User workflow

Each image context menu contains Replace with RSML and Export RSML. Neither
operation is in File. Replacement makes the selected frame's imported roots the
source for preview, measurements, summaries, CSV, and subsequent RSML export.
A purple RSML label replaces its ordinary completion glyph across workflow steps.

## Authority and persistence

Original XML and a frozen image background live in an atomic per-image NPZ file
in `.root_tracker_rsml/` beside the photographs. Cache clearing, settings changes,
masking, and reprocessing do not remove or overwrite the replacement. Moving an
image to/from Aside moves its replacement; deleting the image removes it. The
original photograph is never overwritten.

Processing skips replaced frames. Imported roots never seed automatic tracking
on other frames. A replacement's stored coordinate frame is retained even when
later crop settings differ. Native-only processing and CSV definitions remain
unchanged. The parser retains unknown XML and rejects malformed/unbounded input.
Replacement geometry must be valid 2D polylines within the selected image bounds;
standalone inspection also supports 3D documents.

## Measurements

Total length sums polyline segment lengths; count is the number of root axes.
Main length/depth come from the longest explicit primary axis, with depth as its
vertical extent. Values are in pixels. Source plant IDs are retained alongside
tracker-compatible ordinal IDs. Growth rates update on both sides of replaced
frames; standard 1-based plant IDs match tracker slots, while arbitrary IDs only
match unambiguous identical IDs in adjacent RSML frames. Areas, green areas, and
pixel-difference growth measurements unavailable from polylines are blank, never
retained from superseded tracking data. CSVs with replacements identify sources.

## Export

Per-image export writes only the clicked frame. Native geometry uses assigned
skeleton paths and a matching processed image. Replaced geometry re-exports the
original XML bytes and frozen background. Safe image filenames in the XML are
honored; external paths are never followed, and a preview image is supplied when
needed. CLI batch export remains opt-in. Old native caches must be retracked when
export geometry is absent; export verifies cache group and image identities.

## Verification

Automated coverage includes lossless foreign XML, geometry/path handling,
cache identity, replacement persistence and invalidation, measured statistics,
adjacent growth rates, native-only regression behavior, per-image menu targeting,
distinct badges, and failure safety. Independent review checked the authoritative
replacement path; the primary-axis depth definition was made explicit afterward.

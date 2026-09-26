# Interactive crops and segmentation implementation plan

**Goal:** Replace margin/rotation fields with editable crop boxes in Load and Preprocess; expose blue/green selection and previews.

**Approved design:** Load crops/straightens the specimen; Preprocess refines the root-analysis crop. Both support arbitrary rotation and 15° snap/release behavior. Preserve Apply/Discard, group drafts, preview navigation, and existing automatic outputs.

**Architecture:** Shared normalized rotated rectangles and affine extraction, reused by pipeline and viewer. Keep automatic cropping as the default, preserving legacy configuration. A dedicated editor controller owns source previews and color masks; graphics handles only edit geometry. Existing worker processing remains asynchronous.

**Tech stack:** Python, NumPy/OpenCV, PySide6, unittest. No new dependencies.

## Tasks
- [x] Geometry and pipeline: `preprocessing/roi.py`, `preprocessing/colors.py`, config, cropper, pipeline. Tests cover normalized ROI validation, arbitrary and cardinal rotations, coordinate transforms, masks, and unchanged automatic output.
- [x] Interactive editor: `gui/crop_overlay.py`, `gui/color_range.py`, viewer integration. Tests cover 15° latch release/rearm, resize/move, zoom-independent handles, color picking/ranges and segmentation.
- [x] Workflow wiring: `gui/roi_editor.py`, SettingsPanel and MainWindow. Tests cover Load controls, Preprocess editing source, Apply/Discard/drafts, update-all, cancellation, mask invalidation and worker locking.
- [x] Verification: full test suite, offscreen GUI screenshots and mouse interaction, six real groups / 23 images exact default comparison against fresh pre-change baseline, and a non-cardinal crop run.

## Review focus
- Crops must never compound when reopening or expanding a region.
- Old tracking masks must not be reused after a coordinate-system change.
- Barcode reads and overlays remain in original photograph coordinates.
- Snap suppression is gesture-local and resets after another snap or mouse release.
- Empty color matches must not create empty arrays or crash the application.

No commit requested; preserve all existing uncommitted work.

## Verification results
- 88 tests passed with `QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -q`.
- Fresh baseline against the pre-feature source snapshot: six groups / 23 images, exact equality of all preprocessing arrays, plant coordinates, annotated outputs, root assignments and statistics. No dataset pipeline caches read or written by the benchmark.
- Additional manual crops with 3.5° source adjustment and 3.25° analysis adjustment completed preprocessing, alignment and tracking on the same 23 images. All images retained six plant centers and valid matching image/process/edge dimensions. This is a functionality check, not an equivalence claim for deliberately changed crops; measurements can change with crop geometry and interpolation.
- Offscreen rendering at 1200×800 checked both stages, expanded color controls and handle visibility. Qt mouse tests verify real crop dragging; geometry tests cover rotated corner resizing and snap release/rearming, including skipped mouse-event angles.
- Independent review identified fast-crossing snap recapture and stale masks in other groups after shared crop changes. Both now have passing regressions.
- Defaults and untouched color ranges preserve existing configuration hashes; manual crop coordinates participate in cache invalidation.

Results: `docs/performance-data/interactive-crops-validation.json`.

## Preview interaction follow-up
- Draft analysis crops render without Apply, including configured crops on unprocessed groups. Apply still commits processing settings and runs the full pipeline.
- Segmentation and sampling use the currently displayed processed crop unless crop editing is explicitly active. Range edits show segmentation immediately.
- Pixel-exact scrollbar restoration replaces center-point rounding; fixed scrollbar space prevents deferred viewport resizing.
- Sampling has a crosshair, explicit active button text, and window-scoped Escape cancellation. Crop handles cannot intercept picker clicks.
- Rotation is also available in a screen-sized region outside each corner, using the same 15° snap state machine.
- 95 tests pass, including actual mouse sampling, all four corner rotation zones, repeated low-zoom redraws, draft previews, worker Apply and cached-result reload.

## Corrected crop semantics (supersedes the original automatic/default design)

The user clarified that the Load rectangle is a rough **search window**, not
an override of plate detection. Detection now runs inside that window (including
its rotation) and selects the largest background-colored external contour over
100 pixels. A dashed green outline distinguishes the detected plate from the
editable search window. Segmentation in Load is restricted to that window.

Preprocess always uses a fixed normalized rectangle relative to the detected
plate. Plant detection still supplies centers and leaf masking, but never moves
the analysis crop. With no explicit rectangle, the existing vertical ratios and
edge margins define a fixed rectangle; Reset crop restores those fixed margins.
A preprocessing cache version marker prevents reuse of old crop semantics.

Validation: eight groups / 32 full-resolution images across the current and
previous datasets, one rough search window reused for each group's time series.
All retained six plant centers and completed alignment and tracking. Maximum
center difference from the historical crop, transformed back to original oriented
image coordinates, was 1.414 pixels. Crop geometry intentionally changes, so root
measurements are not asserted equivalent. Existing tracking validation may still
report decreases between dates; this run verifies completion, not biological
accuracy. Results: `docs/performance-data/plate-search-fixed-crop-validation.json`.

## Load navigation and crop-frame transitions

Load browsing now reuses up to three prepared previews in a 256 MiB LRU and
prepares adjacent images on one background thread. Keys include file timestamp,
size, orientation, search ROI, and HSV bounds. Workers prepare NumPy arrays only;
all Qt updates remain on the GUI thread. Reset/close discards stale work. Load no
longer reloads heavy processed arrays, HSV is lazy, and QImage accepts BGR without
a channel-swap copy. On 24 MP JPEGs, prepared switches measured about 20 ms versus
256 ms before; a cold first image still took about 195 ms. Results are in
`docs/performance-data/image-switch-preview.json`.

Processed images now retain their plate-to-preview affine transform, including
registration offsets, in the model and disk-cache metadata. Entering/leaving crop
editing composes that transform with the viewport transform, preserving visible
pixel positions, including rotated crops. Zoom calculations use the transform's
scale magnitude. Cache version v2 ensures old results lacking registration/frame
metadata are regenerated. Tests cover multiple landmarks in both directions,
registration translations, numeric transform persistence, and existing view zoom
and pan interactions.

## Configurable Load detection and cropped preview

Load now exposes Detection (background color / off, box only) and Region
(largest matching region / all matching patches). Both settings participate in
configuration hashes, preview-cache keys, Apply/Discard, and per-group drafts.
Disabling detection bypasses color segmentation and uses the search box alone.
Finish editing or Apply shows only the final Load crop; re-entering Edit crop
restores the full source with the viewport affine mapping preserved. Barcode
rectangles are mapped into the displayed crop and omitted when entirely outside.

124 tests pass. On 16 full-resolution images from both datasets, largest-region
bounds exactly matched the prior implementation, all-patches bounds matched the
historical union, and disabled detection plus rotated finished previews matched
ROI extraction pixel-for-pixel. Results:
`docs/performance-data/crop-region-modes-validation.json`.

### Load preview and settings organization refinement

Finishing Load crop editing displays only the user-drawn search ROI, without
applying the detected plate bounds to the preview. The preview pixel cache depends
on the image and search geometry, so editing background colors or the region
method keeps the same visible area. Actual preprocessing still uses the detected
plate when detection is enabled; disabling detection uses only the drawn ROI.

The Plate search area section contains geometry controls only. Background color
contains a detection checkbox, the region selector, and the color/segmentation
controls. Disabling detection disables dependent controls while leaving the
checkbox available. Draft restoration uses the checkbox state.

Validation: all 124 tests pass, including drawn-box preview pixels and transforms,
unchanged preview canvas across color/region edits, and detection drafts across
groups. The revised layout was rendered and visually inspected offscreen.

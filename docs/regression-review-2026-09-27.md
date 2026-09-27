# Root continuity regression review (2026-09-27)

## Fixes and protected expectations

- RT40: plant 3 continues along the shared bundle on day29, while plant 4
  retains its longer exclusive tail; the day30 split keeps separate owners.
- RT45: plant 1 retains its complete observed distal main through local motion
  on days28–30. The new short day30 lateral remains nonmain.
- RT56: the day28 right exit belongs to plant 2, and the left to plant 3.
- RT69: the left exit belongs to plant 2 on days29–30; the right exit belongs
  to plant 1. Previously owned exits and the earlier day28 crossing are protected.

The implementation uses whole-route historical distance, measured crossing
footprints, and continued width observation for proven bundles. There are no
RT-number branches or thresholds fitted to individual screenshots. New recorded
fixtures replay all available days in chronological order. Synthetic tests cover
local movement, reconnections, image scale, contour subdivision, touching
laterals, missing width evidence, and short-edge lookahead.

The broader review also reproduced RT37's missing shared prefix: a skeleton
cycle delayed one arrival, and bulk fallback prematurely processed the contact.
Dependency-ordered cycle recovery restores the prefix. Recorded RT38/50 controls
protect grounded roots when physical and base dependencies interact. Across the
309-frame scheduler comparison, every plant tip depth is preserved. The only
owner transfer is a newly appearing RT67 branch, whose raw image supports the
corrected continuation; earlier ownership is unaffected.

## Broader checks and their limits

Fresh pipeline replays used cached preprocessing inputs for RT1, 3, 5, 7, 10,
11, 12, 15, 17, 18, 19, 21, 22, 25, 26, 33, 37, 40, 45, 51, 52, 53, 55, 56,
59, 61, 62, 69 and 70: 29 series and 115 images. The replay reruns thresholding,
skeletonization, ownership, main selection, rendering and measurements. It does
not overwrite user caches. All 115 replayed images pass these output invariants. The historical probe
checker reports 428 passing assertions, 25 unasserted expectations, and three
flagged comparisons (the two preexisting RT17 exits and the RT52 comparison
described below against an older frozen baseline). Output invariants include main length no greater
than owned length, agreement between samples and reported lengths, and main
geometry contained in owned geometry.

Separate stage comparisons used 309 saved graph frames across 78 series.
These comparisons detect unexpected output changes; unchanged output alone is
not anatomical proof. Selected historical probes and dedicated recorded tests
protect the reported cases, but some old reports lack precise reviewed
coordinates, so they cannot honestly be called fully anatomically verified.

The exact committed-code candidate was tested separately from unrelated work
in the shared working tree. Two additional RT17 day30 exit probes are already
reversed at the preceding commit; this change does not repair them. The older
RT17 shared-growth expectations remain covered by existing tests. RT52 plants5/6
ownership is unchanged. Plant6 loses nine main-highlight pixels on an side arm absent from the earlier main at the basal junction on day29; both the actual basal and
distal main contours, depth, and root ownership are preserved. No RT52-specific
correction was made after that report was withdrawn.

The exact candidate ran 382 unit tests; its only failures are two preexisting subcases in
`test_roi_geometry.test_bundled_default_crop_matches_plate_margins`: those expect
10% bottom crop while the bundled configuration has 5%. This change does not
alter cropping.

Run the new recorded and synthetic tests with:

```sh
python -m unittest discover -s tests -p 'test_main_route_motion.py'
python -m unittest discover -s tests -p 'test_bundle_growth.py'
python -m unittest discover -s tests -p 'test_new_crossing_exits.py'
python -m unittest discover -s tests -p 'test_junction_cycle_schedule.py'
```

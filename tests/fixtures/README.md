# Root-link regression data

`root_link_corners.npz` contains numeric corner/contour inputs captured immediately
before `RootLinker.link_corners` from the cached, preprocessed RT_26_2-18,
RT_26_2-19, RT_26_2-21 and RT_26_2-26 series, April 27–30, 2026. No source images
or pickle objects are included. Capture used `configs/in_vitro.yaml` and the
existing per-series masks. `metadata` is UTF-8 JSON; `contour_<frame>_<segment>`
contains segment pixels; `baseline_<frame>_<plant>` contains the assignments
before the connected-parent fix (zero-based frame/plant indices).

The regression replays each sequence with freshly computed previous-frame
samples. It verifies that every assigned segment traces to a plant origin,
that the identified disconnected pixels disappear, and that the remaining
plants retain exactly the same pixels. RT_26_2-21 is an unchanged control:
its reported plant 4 fragment has a link under the existing gap rules and
is not explained by the unassigned-parent bug.

`root_crossings.npz` contains the RT_26_2-25 and RT_26_2-19 corner sequences
(April 27–30), including the horizontal lateral branch from the report.
All crossing fixtures include physical junction IDs and direction-only terminal
exits, captured through the complete tracking pipeline after skeleton cleanup.
`root_shared_junctions.npz` contains RT_26_2-17 and RT_26_2-62 for those dates,
with physical-junction metadata for the merge/shared/split regression.
`root_crossing_controls.npz` contains RT_26_2-1, -7, -12, and -51: these protect
established main roots from being reassigned at a later lateral branch after
sharing a section. All three files use the same JSON metadata and numeric
contour representation as `root_link_corners.npz`, without baseline pixel arrays.
Expected plant identities are explicit, hand-traced endpoints in the tests.

The original disconnected-fragment fixture's comparisons now explicitly account
for corrected crossing assignments in RT_26_2-18 and -19; its debris rejection
and all other pixel-set comparisons remain enforced.

`root19_crossing_mask.npy` is the actual RT_26_2-19/day30 skeleton crop
at x=920:1050, y=415:535. Its end-to-end pipeline regression verifies that
discarded short bridges remain part of the crossing topology.
`root17_terminal_mask.npy` is the RT_26_2-17/day30 skeleton crop at
x=1060:1130, y=815:875. It verifies that a filtered terminal twig remains a
routing exit, including a horizontal-tip variant of the same crossing.

`root_temporal_contacts.npz` contains complete RT_26_2-1, -10, -12, -17 and -55
sequences through April 30, including junction and terminal-exit metadata.
These assert that touching roots cannot take over an established neighboring
trunk/lateral, that RT17's long shared root retains both owners, and that
RT55 retains a short historically shared portion inside a longer segment. Frame dates
use each series' actual final three/four-day sequence.

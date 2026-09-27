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

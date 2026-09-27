# RT_23_04_10 main-root regression

The previous main-route change could rank a short competing branch ahead of
a growing established main because of a small mean-distance advantage against
accumulated historical positions. Its sparse-lateral exception also admitted
unrelated terminal branches as ordinary main candidates. Plants 2 and 3 then
kept following those initial mistakes over subsequent days.

The fix keeps established-main support as the normal tip-selection criterion.
Whole-route geometry still selects ancestry at reconnections and recovers a
moved main when the support winner truncates its most recent observed extent.
Internal arrivals with admissible continuations do not compete as terminal
tips. Sparse lateral contact may preserve an established internal main bridge;
a contested terminal is recovery-only and must reach the historical depth and
improve whole-route fit. No image identifier or fitted numerical threshold is
used. Plant 1's earlier identity is preserved, as requested.

## Verification

- New numeric fixture replays nine chronological RT_23_04_10 frames. Checks
  plants 2/3 growth on March 6–9, proximal and distal main samples on March 9,
  and unchanged March 9 tips for plants 1/4/5/6.
- Synthetic controls reproduce stale-history competition at two image scales
  and a shifted terminal main touching sparse old lateral pixels.
- All 24 main-root tests pass, including RT45 movement, missing roots,
  reconnection ancestry, lateral overtaking and previous RT5/15/52/70 cases.
- Exact scoped candidate: 385 unit tests; only the two preexisting crop-margin
  subcases fail. The new tests fail against the preceding implementation.
- Replayed the 115 earlier reported images: 428 passing assertions, the same
  25 unasserted expectations and three flagged comparisons documented in the
  preceding review. No new failed historical probe.
- Compared 309 saved graph frames against the preceding commit: 26 plant/frame
  geometry changes, no changed main-tip coordinates. Independent review of the
  largest changes in RT70 and RT68 confirmed removal of unsupported basal
  lateral ancestry, not holes in an established distal main.

Both cached RT23 series (groups 1 and 10, 18 images) were replayed in the current
workspace and with the isolated commit candidate. In the current workspace,
plants 2/3 follow their growing mains through March 6–9 and plant 1 retains its
left branch. The isolated candidate also corrects both March 9 tips, but lacks
some plant-2 proximal main highlighting and selects a shorter tip on March 6.
Those routing inputs differ because unrelated in-progress routing changes were
excluded from this commit. This is an explicit limit of the isolated replay;
it is not counted as proof of a completely correct standalone pipeline.

Tests can be rerun with:

```sh
python -m unittest discover -s tests -p 'test_main*.py'
```

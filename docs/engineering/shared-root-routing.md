# Root crossings and shared paths

The tracker must preserve plant identity when roots cross, including crossings
that contain a visible shared section between two junctions. A shared segment
belongs to every plant entering it; its outgoing branches must not all inherit
the first plant ID. Disconnected fragments must still remain unassigned.

## Algorithm

1. Segment endpoint angles are outward tangents. Rotate a child's top tangent
   by 180 degrees before comparing it with a parent's bottom tangent, and use
   the shortest angular difference modulo 360. For exactly horizontal segments,
   orient the endpoints toward the nearest rooted endpoint and keep the endpoint
   and junction metadata consistent.
2. Label connected components of the skeleton pixels removed by splitting,
   including discarded short bridges (excluding background padding). Preserve
   discarded terminal twigs as direction-only exits in the matching problem;
   they remain excluded from drawing and measurement. Without these exits a
   root ending at a crossing could falsely merge into another root. Associate
   an endpoint with a junction only when its 3x3 neighborhood touches exactly
   one label. Spatially close but unconnected roots do not form a shared node.
3. Process segment dependencies from plant origins toward the tips. Non-junction
   gaps retain their existing links. At a physical junction, collect distinct
   incoming plant IDs and their tangents. A single plant may branch normally.
4. Match incoming directions to exits jointly with minimum-cost bipartite
   matching (SciPy's linear sum assignment). A 0.000001-degree order penalty
   breaks exact ties in image-left-to-image-right order. Historical ownership
   constrains the result: at 2-pixel tolerance require at least 5 exclusive
   matches, 20% segment coverage and 80% of exclusive votes. If inconclusive,
   try 4 then 8 pixels, requiring at least 10 matches and 50% coverage instead.
   Ambiguous pixels near multiple owners do not provide exclusive votes.
   Preserve the established owner even if another incoming direction scores
   better. A contact cannot add an owner to an established root unless an
   outgoing segment at its downstream junction independently supports that
   owner. Thus a root can end where it touches another root.
5. Preserve established shared ownership from actual previously shared pixels,
   allowing up to 8 pixels of alignment/thinning movement. Such evidence must
   be at least as close as any competing historical owner. At least 5 matching
   pixels are required. Retain whole-segment sharing only when the matches span
   at least 60% of its path, account for at least 60% of observed historical
   pixels, and no sustained exclusive historical portion conflicts with it.
   Otherwise preserve only the matched shared portions, separately for disjoint
   runs. Overlapping shared pairs combine all supported plant IDs. Sustained exclusive
   historical runs (at least 5 nearest-owner pixels within 2 pixels, with gaps
   of at most 2 intervening pixels) are also retained individually. Split a
   current contour at these ownership boundaries, keeping its new growth
   separately routed. This prevents both loss of old shared roots and expansion
   of sharing over an established unshared neighbor. Temporal preservation also
   runs in frames with no detected junctions.
6. While a segment has multiple IDs, retain their pre-merge directions and
   entry order. These guide the next split. A resolved, single-plant continuation
   uses its own bottom tangent at subsequent junctions. Store a parent endpoint
   separately for each plant so longest-path tracing follows the proper root.
   If a temporal boundary changes an endpoint owner, redirect its children to
   the last fragment retaining their own identity and rebuild links in parent
   order. Measurements, exported samples and drawing use the same fragments.
7. A cyclic/ambiguous directed junction falls back to already rooted base links;
   it never authorizes an unrooted fragment.

## Visualization and measurements

Shared contours are drawn last as adjacent color bands around their centerline,
ordered by the roots' entry positions. The bands follow local path normals.
Each band retains its own plant's main-root highlight (12-pixel total shared
width when a main path participates, otherwise 6 pixels). The same skeleton
pixels count in each participating plant's length and export geometry; image-wide
skeleton length and area remain unique-pixel measurements.

Singleton assignments remain integer IDs. Shared assignments are ordered tuples;
per-plant pixel samples contain the common segment in each participating plant.
Old tracking caches must be invalidated because assignments and lengths change.

## Verification

Synthetic tests cover straight and X-shaped continuations, angle wraparound,
merge/shared/split routing across frames, ordinary single-plant branching,
per-plant main-path measurement, and two-color rendering. Real fixtures cover
RT_26_2-25/19 crossings and RT_26_2-17/62 shared sections. The previous disconnected
fragment fixtures retain their rejection checks, with explicitly documented
corrections for previously misassigned crossings. A local before/after audit
also replays the cached dataset and checks rooted graph connections, retained
root geometry, and changed plant identities. No image-level automatic comparison
can establish biological ground truth for every ambiguous overlap.

The follow-up contact regressions cover RT1/10/12 day30 and RT17 shared-root
persistence. RT17's upper contact and left lateral remain with plant 2, as
established on earlier days; its long lower shared root keeps both plants 2
and 3. Earlier single-image expectations for that upper contact are superseded
by these temporal observations.

Validation: all 275 unittest tests passed. A 78-series/309-image replay retained
all previously assigned geometry, had no mismatched parent identities, and left
1,778 per-plant frame pixel sets unchanged. No run of five or more exactly
matching historical pixels changed plant ownership. Short isolated coincidences
are intentionally insufficient evidence for historical identity.
Run the committed tests with `python -m unittest discover -s tests`.

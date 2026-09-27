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
4. With multiple plants, compute the circular angle difference between each
   incoming tangent and each outgoing tangent. Previous-frame support is trusted
   only for pixels within 2 pixels of exactly one plant: at least 5 pixels, at
   least 20% of the outgoing segment, and at least 80% of its exclusive votes.
   Penalize a conflicting identity by 180 degrees times that coverage fraction.
   This prevents a later side branch from taking over an established main root;
   isolated overlaps and previously shared pixels cannot force an identity.
   Use minimum-cost bipartite
   matching (SciPy's linear sum assignment) to reserve separate exits where
   available. A 0.000001-degree order penalty breaks exact direction ties in
   image-left-to-image-right order. Extra exits use the best incoming identity;
   extra incoming identities share their best exit.
5. A new lateral does not end established sharing: retain the incoming shared
   IDs when at least 5 pixels and 60% of an outgoing segment match pixels that
   actually belonged to both plants previously (within 2 pixels). Mere proximity
   to two separate prior roots never counts as shared history.
6. While a segment has multiple IDs, retain their pre-merge directions and
   entry order. These guide the next split. A resolved, single-plant continuation
   uses its own bottom tangent at subsequent junctions. Store a parent endpoint
   separately for each plant so longest-path tracing follows the proper root.
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

Validation for this change: all 265 unittest tests passed. A complete tracking
rerun of 78 cached series (309 images) retained all previously assigned root
pixels, had no mismatched per-plant parent identities, and left 1,651 per-plant
frame pixel sets unchanged. Reported RT17/19/25/62 endpoints and four additional
series' established main roots are asserted separately against traced identities.
Run the committed tests with `python -m unittest discover -s tests`.

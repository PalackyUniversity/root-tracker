# Temporal root continuity

Tracking separates plant ownership, observed foreground, and main-root identity.
The fixes have no series IDs or coordinate-specific rules.

## Main-root identity

The first observation still starts from the deepest owned endpoint. In later
observations, historical main pixels vote for the nearest currently owned
segment, within its measured foreground radius (at least one diagonal raster
step). Other established roots, excluding the main history, estimate residual
translation by a converged median nearest-neighbour displacement. The main
itself cannot align a missing root onto a different lateral. Accept that motion
only when it improves support over the unshifted main geometry; missing lateral
anchors must not drag an intact main away.

Candidate root-to-tip paths maximize historical support; depth breaks ties for
new growth. At a physical reconnection, same-plant incoming segments are also
considered, so a lateral reconnecting to the main cannot replace its established
ancestry merely because the ownership solver picked that parent. Paths are
cycle guarded and cannot traverse another plant's exclusive segments.

History remembers previous main observations through missing or shortened
frames. Only current observed contours are rendered or measured; remembered
pixels do not manufacture foreground. Manual RSML replacement interrupts the
automatic main history. Masked runs keep internal main history separate from the
unmasked samples used by export.

## Thin fragments

Contour area is no longer sufficient to delete an established root. Otherwise
small current contours can survive with sustained support from a single previous
plant: at least ten contour samples and half the contour within four pixels,
covering at least five distinct historical samples. These reuse the existing
router's temporal evidence criteria. Missing foreground is never filled in.
Disconnected contours require majority temporal support to seed ownership;
minority contact is sufficient only to help disambiguate already rooted paths.

## Isolated debris

Registered tracking now carries unmasked physical-component identity, extent,
and foreground area into gap linking. Connected components retain their normal
links even when graph endpoints are far apart. Previously assigned pixels retain
historical ownership.

For a new disconnected object, compare the nearest separation from the proposed
parent component with the object's extent. Reject an unsupported gap larger than
the object only when the object is compact. An extent greater than twice the
diameter of an equal-area disk counts as elongated root evidence and bypasses
this gate. This dimensionless shape rule is invariant to image scale; it is a
conservative heuristic, not a claim that geometry can identify every impurity.
Missing component measurements and unregistered workflows retain prior behavior.
Component extent is measured before user exclusions, so an exclusion cannot make
an established long root look like a tiny unsupported island.

## Regression fixtures

- `main_root_continuity.npz`: observed assigned graphs for RT5 plant3, RT15
  plant1, RT52 plant2, days27–30. Tests cover the overtaking lateral, rejoining
  lateral, and restored main segments, including RT15's distal continuation.
- `historical_foreground.npz`: actual threshold crops and previous skeleton
  samples for RT12 plant6 day29 and RT7 plant2 day30.
- `gap_components.npz`: measured RT11 debris and real detached fine roots from
  RT71/RT39 that an unrestricted gap-distance gate incorrectly rejected.

Synthetic tests additionally cover subdivision, shared roots' individual
parents, scale changes, missing observations, partial historical contact, and
attempts to align a missing main onto another root. Cache invalidation forces
tracking to replay in time order with these rules.

## Contact provenance and merged main contours

A downstream exit proves a shared corridor only when that plant physically
arrives at the corridor's upstream junction. A temporal or gap link cannot use
a contact at the far end to acquire the entire upstream root. Independent
contact evidence and already established shared history remain valid.

For a tiny free terminal arm ending at an already rooted junction, an external
plant cannot enter over a gap longer than that arm's own observed extent. The
whole connected component's length is not evidence for this small arm. Existing
historical links, same-owner links, same-component connections and observed
longer incoming roots remain admissible. The rule scales with image geometry.

When segmentation merges an incoming lateral with a main contour, main identity
is applied to contour portions. Sustained historical lateral evidence excludes
the incoming prefix and its ancestry; established main evidence takes priority
within the measured root radius. Distal continuation remains possible even if
an earlier detection gap left some intermediate pixels without a main label.
An unsupported gap parent cannot grow the main backwards onto a newly observed
leaf edge. Direct support for its ancestry remains valid even when a global
alignment cannot explain local motion. Rendered highlights,
measurements and exported main samples all use the same selected portions.

Additional numeric regression fixtures record RT59 contact provenance across
four days, the RT61 day28 terminal junction and preceding ownership, and RT70
plant4's four-day assigned geometry, plus RT8 and RT26 main-continuity controls. Synthetic cases cover scale changes,
legitimate gap arrivals, partial main highlighting of shared roots, merged
lateral prefixes and incomplete historical main observations.

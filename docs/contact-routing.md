# Root contact evidence

Two incoming roots and one outgoing skeleton segment do not establish sharing:
one root can end against the other. The tracking pipeline now measures foreground
width before thinning and attaches a width profile to each ordered contour.
The router uses the same rules for every series; there are no RT-specific rules.

## Decision procedure

1. Match incoming identities to outgoing directions jointly. An unmatched identity
   may terminate instead of automatically being copied onto the remaining exit.
2. Look through degree-two junctions for independent continuations. Each supporting
   exit must uniquely prefer a different incoming direction and point forward.
   Arrivals at the far junction participate too: their exits cannot also prove
   that another root travelled through the corridor.
3. Otherwise compare foreground width with two explanations: the continuing root's
   own caliber, or the envelope of the arriving roots projected across the path.
   Individual calibers come from segment interiors. Width smoothing and observation
   windows are expressed in measured root diameters, not specimen-specific pixels.
   A one-pixel sampling uncertainty favors the simpler single-root explanation.
4. Carry individual calibers through a supported bundle. Calibrate its lane spacing
   to its observed silhouette so overlapping roots need not have additive widths.
   A lateral exit does not automatically remove an identity from a bundle that
   still has width evidence for both roots.
5. A sustained narrowing can mark a root ending only if the terminal portion also
   favors a single-root caliber. A temporary neck is insufficient. Distinct calibers
   identify the survivor; equal calibers require a centerline shift indicating
   which outer boundary continues. Inconclusive evidence does not pick a plant ID.
   Split the contour at a supported tip so rendering, measurements, and exported
   samples use the same ownership boundary.
6. Short crossings retain incoming headings. A measured bundle longer than the
   crossing footprint implied by the incoming calibers and angle keeps transverse
   order at separation. Exclusive temporal evidence takes priority over lane order.
7. Preserve previously assigned identities on surviving geometry. Positive evidence
   can add a shared owner, but cannot replace a historical owner with another plant.
   Shared-prefix proof and single-root suffix evidence are spatially bounded.

Width profiles reverse and split together with their contours. Older callers that
do not provide foreground widths retain the previous geometry-only behavior.
The tracking cache version changes so full-series recomputation starts with the new
contact decisions instead of freezing predictions from the old algorithm.

## Verification and limits

`tests/test_contact_evidence.py` covers touching versus crossing, mirror/scale
invariance, a downstream lateral, bounded shared prefixes, and complete temporal
replays from numeric contour/width fixtures. These include RT17 and RT62 controls,
and RT3, RT22, RT37, RT40, RT51, RT53, RT55, RT59 and RT69 reported endpoints.
RT3's pictured contact is interpreted as day 30; there is no corresponding contact
in the day 29 capture.

RT33 is included in replay and metadata-alignment checks, but its disputed branch
is not asserted as fixed. Its precise target remains to be confirmed; the likely
upper branch is already assigned to pink in the preceding frame.

A binary foreground silhouette cannot always establish whether one or two roots
occupy the same pixels. The lookahead currently stops at the first branching
junction; an incidental lateral can obscure a separating junction farther away.
The automated checks protect specified assignments and graph/history invariants;
they are not a claim that every unlabelled pixel has biologically correct ownership.

Validation on September 27, 2026:

- Replayed all 309 frozen frames across 78 series against the preceding router.
  No detected root pixels were lost, no parent pointed to another plant's segment,
  and no sustained historical ownership loss was found (runs of five pixels or
  more). 1,787 of 1,854 plant/frame sample sets were unchanged.
- Reran the complete image pipeline for 305 frames across 77 series. Per-plant
  lengths matched assigned sample counts, main-path lengths did not exceed total
  lengths, and RSML samples matched tracking samples. The current RT1 manual RSML
  replacement was left untouched; its original automatic inputs are covered by
  the frozen replay.
- Inspected corrected RT22, RT37, RT40 and RT53 crops and rechecked the earlier
  RT1, RT7, RT10, RT12, RT17, RT19, RT25, RT51 and RT62 endpoint controls.

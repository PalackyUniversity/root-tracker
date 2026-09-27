"""A small island cannot be rooted by a much longer unsupported gap."""
import unittest
import numpy as np
from root_tracker.tracking.gap_evidence import GapEvidence


class GapEvidenceTests(unittest.TestCase):
    def test_component_support_is_scale_independent(self):
        for scale in (1, 3):
            main = np.array([(100, y) for y in range(10, 201)]) * scale
            speck = np.array([(x, 80) for x in range(10, 21)]) * scale
            tip = np.array([(x, 220) for x in range(100, 151)]) * scale
            corners = [dict(point=tuple(p[0]), lower_point=tuple(p[-1]),
                            contour=p.reshape(-1, 1, 2), component_id=i+1,
                            component_area=len(p)*5*scale**2)
                       for i, p in enumerate((main, speck, tip))]
            evidence = GapEvidence(corners)
            self.assertFalse(evidence.allows(corners[1], tuple(main[0])))
            self.assertTrue(evidence.allows(corners[2], tuple(main[-1])))
            # A tiny branch in the SAME physical component remains valid even
            # if the graph's chosen parent is far away.
            corners[1]['component_id'] = 1
            self.assertTrue(GapEvidence(corners).allows(corners[1], tuple(main[0])))

    def test_legacy_callers_without_component_evidence_keep_existing_behavior(self):
        upper = dict(point=(10, 80), lower_point=(20, 80),
                     contour=np.array([[[10, 80]], [[20, 80]]]))
        self.assertTrue(GapEvidence([upper]).allows(upper, (100, 10)))

    def test_new_remote_speck_is_rejected_but_established_fragment_is_retained(self):
        from copy import deepcopy
        from unittest.mock import patch
        from root_tracker.config import Config
        from root_tracker.tracking.root_linker import RootLinker
        from test_root_crossings import detected_segment
        segment, _ = detected_segment((20, 80), (30, 90))
        segment['component_id'] = 1
        segment['component_area'] = 150
        origins = [{'point': (100, 10), 'angle': 90., 'plant_id': 0},
                   {'point': (500, 10), 'angle': 90., 'plant_id': 1}]
        history = {0: {(100, y) for y in range(10, 60)}, 1: set()}
        def assign(previous):
            return RootLinker(Config(n_clusters=2)).link_corners(
                [deepcopy(segment)], origins, [100, 500], [10, 10], previous)[1]
        with patch.object(GapEvidence, 'allows', return_value=True):
            self.assertIn((30, 90), assign(history))
        self.assertNotIn((30, 90), assign(history))
        history[0].update(map(tuple, segment['contour'][:, 0]))
        self.assertEqual(assign(history)[(30, 90)], 0)

    def test_long_thin_component_has_shape_evidence_despite_large_gap(self):
        for scale in (1, 4):
            points = np.array([(x, 100) for x in range(50)]) * scale
            upper = dict(point=tuple(points[0]), lower_point=tuple(points[-1]),
                         contour=points.reshape(-1, 1, 2), component_id=1,
                         component_area=150*scale**2, component_extent=50*scale)
            self.assertTrue(GapEvidence([upper]).allows(upper, (300*scale, 0)))

    def test_recorded_debris_and_detached_fine_roots(self):
        import json
        from pathlib import Path
        with np.load(Path(__file__).parent / 'fixtures/gap_components.npz') as data:
            for raw in json.loads(str(data['metadata'])):
                upper = dict(raw, point=tuple(raw['point']),
                             lower_point=tuple(raw['lower_point']), contour=data[raw['key']])
                with self.subTest(case=raw['key']):
                    self.assertEqual(GapEvidence([upper]).allows(upper, tuple(raw['parent'])),
                                     raw['expected'])

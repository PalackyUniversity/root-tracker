"""Recorded failures and general invariants for crossing/gap ownership."""
import unittest
import json
from pathlib import Path
import numpy as np

from root_tracker.config import Config
from root_tracker.tracking.root_linker import RootLinker
from root_tracker.tracking.temporal_fragments import contour_path
import test_contact_evidence as contact_fixtures


class FollowupRoutingTests(unittest.TestCase):
    def test_reported_failures_replayed_with_fresh_ownership_history(self):
        expected = {
            (17, 30): {(1041, 1024): 1, (934, 1002): 2, (930, 1100): 1},
            (34, 29): {(1699, 805): None},
            # By day 30 an observed fine root extends through this region.
            (34, 30): {(1713, 815): 3},
            (51, 30): {(1402, 933): None},
            (58, 30): {(395, 1653): None},
            (69, 30): {(425, 1046): 1, (506, 1000): 0, (338, 1075): 1},
        }
        history = {}
        fixture = Path(__file__).parent / 'fixtures/debris_followup.npz'
        with np.load(fixture, allow_pickle=False) as data:
            for i, case in enumerate(json.loads(data['metadata'].tobytes())):
                for j, upper in enumerate(case['upper']):
                    upper['point'] = tuple(upper['point'])
                    upper['lower_point'] = tuple(upper['lower_point'])
                    for field in ['contour', 'width_profile']:
                        if f'{field}_{i}_{j}' in data:
                            upper[field] = data[f'{field}_{i}_{j}']
                    for sink in upper.get('terminal_exits', []):
                        sink['point'] = tuple(sink['point'])
                        sink['lower_point'] = tuple(sink['lower_point'])
                for lower in case['lower']:
                    lower['point'] = tuple(lower['point'])
                group, day = case['group'], case['day']
                pairs, colored, samples = RootLinker(Config(n_clusters=len(case['x']))).link_corners(
                    case['upper'], case['lower'], case['x'], case['y'], history.get(group))
                history[group] = samples
                for point, owner in expected.get((group, day), {}).items():
                    with self.subTest(group=group, day=day, point=point):
                        self.assertEqual(colored.get(point), owner)
                        if owner is None:
                            self.assertTrue(all(point not in pixels for pixels in samples.values()))
                if (group, day) == (51, 30):
                    self.assertNotIn(((1402, 933), (1251, 915)), pairs)

    def test_reorienting_terminal_does_not_orphan_existing_downstream_gap(self):
        from root_tracker.tracking.terminal_orientation import orient_terminals
        upper = dict(point=(90, 80), lower_point=(98, 84), angle=210.,
                     contour=np.array([[[90, 80]], [[98, 84]]]))
        lowers = [dict(point=(98, 84), junction_id=1, angle=30.),
                  dict(point=(100, 82), junction_id=1, angle=90.)]
        pairs = [((90, 80), (130, 75)), ((110, 100), (98, 84))]
        colored = {(100, 82): 0, (98, 84): 0, (130, 75): 0}
        orient_terminals([upper], lowers, pairs, colored, set(), None)
        self.assertIn(((110, 100), (100, 82)), pairs)

    def test_terminal_anchor_cannot_be_its_own_descendant(self):
        from root_tracker.tracking.terminal_orientation import orient_terminals
        def segment(top, bottom):
            return dict(point=top, lower_point=bottom, angle=210.,
                        contour=np.array([[top], [bottom]]))
        uppers = [segment((90, 80), (98, 84)), segment((99, 85), (100, 86))]
        uppers[1]['junction_id'] = 1
        lowers = [dict(point=p, junction_id=1, angle=30.)
                  for p in [(98, 84), (100, 86), (104, 82)]]
        pairs = [((90, 80), (130, 75)), ((99, 85), (98, 84))]
        colored = {p: 0 for p in [(98, 84), (100, 86), (104, 82), (130, 75)]}
        orient_terminals(uppers, lowers, pairs, colored, set(), None)
        self.assertIn(((99, 85), (104, 82)), pairs)
        self.assertNotIn(((99, 85), (100, 86)), pairs)

    def test_disabled_registration_does_not_enable_gap_filter_with_history(self):
        points = np.array([(100, y) for y in range(60, 70)])
        upper = dict(point=(100, 60), lower_point=(100, 69), angle=270.,
                     contour=points[:, None], component_id=1,
                     component_area=100, component_extent=10.)
        config = Config(n_clusters=1)
        config.registration.enabled = False
        _, _, samples = RootLinker(config).link_corners(
            [upper], [{'point': (100, 10), 'angle': 90.}], [100], [10],
            {0: {(1, 1)}})
        self.assertEqual(samples[0], set(map(tuple, points)))

    def test_old_lateral_cannot_prove_arriving_root_shares_an_existing_trunk(self):
        segments, x = contact_fixtures.ContactEvidenceTests().scene(5., exits=True)
        previous = {0: set(map(tuple, contour_path(segments[0][0]))),
                    1: set().union(*(set(map(tuple, contour_path(u))) for u, _ in segments[1:]))}
        _, _, samples = RootLinker(Config(n_clusters=2)).link_corners(
            [u for u, _ in segments], [{'point': (a, 5), 'angle': 90.} for a in x]
            + [l for _, l in segments], x, [5, 5], previous)
        self.assertIn((70, 90), samples[1])
        self.assertNotIn((70, 90), samples[0])

    def test_short_horizontal_crossing_corridor_is_not_a_one_way_link(self):
        from test_root_crossings import detected_segment
        segments = [detected_segment((20, 10), (64, 55)),
                    detected_segment((120, 10), (76, 55)),
                    detected_segment((67, 58), (73, 58)),
                    detected_segment((64, 61), (20, 110)),
                    detected_segment((76, 61), (120, 110))]
        segments[0][1]['junction_id'] = 1
        segments[1][1]['junction_id'] = 2
        for corner in segments[2]:
            corner['junction_id'] = 1 if corner['point'][0] == 67 else 2
        segments[3][0]['junction_id'] = 1
        segments[4][0]['junction_id'] = 2
        for upper, _ in segments:
            upper['width_profile'] = np.full(len(contour_path(upper)), 5.)
        _, colored, samples = RootLinker(Config(n_clusters=2)).link_corners(
            [u for u, _ in segments],
            [{'point': (x, 5), 'angle': 90.} for x in [20, 120]] + [l for _, l in segments],
            [20, 120], [5, 5])
        self.assertEqual(colored[(20, 110)], 1)
        self.assertEqual(colored[(120, 110)], 0)
        self.assertTrue(set(map(tuple, contour_path(segments[2][0]))) <= samples[0] & samples[1])

    def test_foreground_flare_cannot_make_short_skeleton_an_elongated_root(self):
        from root_tracker.tracking.gap_evidence import GapEvidence
        points = np.array([(x, 80) for x in range(20, 32)])
        upper = dict(point=(20, 80), lower_point=(31, 80), contour=points[:, None],
                     component_id=1, component_area=146, component_extent=28.)
        self.assertFalse(GapEvidence([upper]).allows(upper, (100, 20)))

    def test_fuzzy_history_cannot_attach_compact_debris_to_a_remote_origin(self):
        points = np.array([(x, 200) for x in range(20, 33)])
        upper = dict(point=(20, 200), lower_point=(32, 200), angle=180.,
                     contour=points[:, None], component_id=1,
                     component_area=122, component_extent=19.)
        _, _, samples = RootLinker(Config(n_clusters=1)).link_corners(
            [upper], [{'point': (100, 10), 'angle': 90.}], [100], [10],
            {0: {(x, 201) for x in range(20, 33)}})
        self.assertFalse(samples[0])

    def test_upward_terminal_uses_its_physical_root_instead_of_a_remote_gap(self):
        import cv2
        from root_tracker.tracking.corner_detector import CornerDetector
        def segment(start, end):
            mask = np.zeros((600, 400), np.uint8)
            cv2.line(mask, start, end, 255, 1)
            contour = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0][0]
            upper, lower = CornerDetector(Config()).analyze_contour_corners(contour, mask)
            upper.update(contour=contour, lower_point=lower['point'])
            return upper, lower
        segments = [segment((20, 10), (20, 380)), segment((200, 10), (200, 497)),
                    segment((120, 450), (196, 500)), segment((200, 503), (200, 580))]
        segments[1][1]['junction_id'] = 1
        segments[2][1]['junction_id'] = 1
        segments[3][0]['junction_id'] = 1
        _, _, samples = RootLinker(Config(n_clusters=2)).link_corners(
            [u for u, _ in segments], [{'point': (x, 5), 'angle': 90.} for x in [0, 300]]
            + [l for _, l in segments], [0, 300], [5, 5])
        self.assertIn((120, 450), samples[1])
        self.assertNotIn((120, 450), samples[0])

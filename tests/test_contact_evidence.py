"""Contact decisions require independent continuation or measured bundle evidence."""
import unittest
import json
from pathlib import Path
import numpy as np
from root_tracker.config import Config
from root_tracker.tracking.root_linker import RootLinker
from test_root_crossings import detected_segment
from root_tracker.tracking.temporal_fragments import contour_path


class ContactEvidenceTests(unittest.TestCase):
    def scene(self, shared_width, exits=False, scale=1, mirror=False):
        segments = [detected_segment((20, 10), (67, 57)),
                    detected_segment((70, 10), (73, 57)),
                    detected_segment((70, 63), (70, 105))]
        for _, lower in segments[:2]:
            lower['junction_id'] = 1
        segments[2][0]['junction_id'] = 1
        for index, (upper, _) in enumerate(segments):
            # The new contact model receives measured foreground widths.
            upper['width_profile'] = np.full(len(np.unique(upper['contour'][:, 0], axis=0)),
                                             shared_width if index == 2 else 5., float)
        if exits:
            segments[2][1]['junction_id'] = 2
            for start, end in [((73, 111), (90, 128)), ((67, 111), (65, 128))]:
                upper, lower = detected_segment(start, end)
                upper.update(junction_id=2, width_profile=np.full(len(np.unique(upper['contour'][:, 0], axis=0)), 5.))
                segments.append((upper, lower))
        for upper, lower in segments:
            for corner in (upper, lower):
                x, y = corner['point']
                corner['point'] = ((140 - x if mirror else x) * scale, y * scale)
                if mirror:
                    corner['angle'] = (180 - corner['angle']) % 360
            upper['lower_point'] = lower['point']
            upper['contour'] = upper['contour'].copy()
            if mirror:
                upper['contour'][:, 0, 0] = 140 - upper['contour'][:, 0, 0]
            upper['contour'] *= scale
            upper['width_profile'] *= scale
        origins = [(140 - x if mirror else x) * scale for x in (20, 70)]
        return segments, origins

    def test_thin_single_exit_allows_touching_root_to_end(self):
        for scale, mirror in [(1, False), (3, False), (1, True)]:
            segments, x = self.scene(5., scale=scale, mirror=mirror)
            _, colored, _ = RootLinker(Config(n_clusters=2)).link_corners(
                [u for u, _ in segments], [{'point': (a, 5 * scale), 'angle': 90.} for a in x]
                + [l for _, l in segments], x, [5 * scale] * 2)
            with self.subTest(scale=scale, mirror=mirror):
                self.assertEqual(colored[segments[2][0]['lower_point']], 1)

    def test_measured_wide_bundle_preserves_both_arriving_roots(self):
        segments, x = self.scene(11.)
        _, colored, samples = RootLinker(Config(n_clusters=2)).link_corners(
            [u for u, _ in segments], [{'point': (a, 5), 'angle': 90.} for a in x]
            + [l for _, l in segments], x, [5, 5])
        self.assertEqual(set(colored[(70, 105)]), {0, 1})
        self.assertIn((70, 90), samples[0] & samples[1])

    def test_distinct_downstream_exits_establish_a_shared_crossing(self):
        segments, x = self.scene(5., exits=True)
        _, colored, _ = RootLinker(Config(n_clusters=2)).link_corners(
            [u for u, _ in segments], [{'point': (a, 5), 'angle': 90.} for a in x]
            + [l for _, l in segments], x, [5, 5])
        self.assertEqual(set(colored[(70, 105)]), {0, 1})
        self.assertEqual(colored[(90, 128)], 0)
        self.assertEqual(colored[(65, 128)], 1)

    def test_downstream_lateral_is_not_an_independent_crossing_exit(self):
        segments, x = self.scene(5., exits=True)
        # Both outgoing directions continue the vertical root; neither
        # continues the diagonal arrival. Merely counting two exits is wrong.
        for upper, _ in segments[3:]:
            upper['angle'] = 280.
        _, colored, _ = RootLinker(Config(n_clusters=2)).link_corners(
            [u for u, _ in segments], [{'point': (a, 5), 'angle': 90.} for a in x]
            + [l for _, l in segments], x, [5, 5])
        self.assertEqual(colored[(70, 105)], 1)

    def test_partial_bundle_upgrades_only_the_observed_prefix(self):
        for scale, mirror in [(1, False), (3, False), (1, True)]:
            segments, x = self.scene(11., scale=scale, mirror=mirror)
            segments[0][0]['width_profile'][:] = 3. * scale
            upper = segments[2][0]
            path = contour_path(upper)
            upper['width_profile'][len(path) // 2:] = 5. * scale
            previous = {0: set(), 1: set(map(tuple, path.tolist()))}
            _, _, samples = RootLinker(Config(n_clusters=2)).link_corners(
                [u for u, _ in segments], [{'point': (a, 5 * scale), 'angle': 90.} for a in x]
                + [l for _, l in segments], x, [5 * scale] * 2, previous)
            # Sharing can upgrade the observed prefix while the thinner
            # incoming root ends before the established owner's tail.
            self.assertTrue(previous[1] <= samples[1])
            self.assertIn(tuple(path[10]), samples[0] & samples[1])
            self.assertNotIn(tuple(path[-1]), samples[0])

    def test_rejected_third_arrival_keeps_existing_bundle_proof(self):
        segments, x = self.scene(11.)
        upper, lower = detected_segment((120, 10), (90, 57))
        lower['junction_id'] = 1
        upper['width_profile'] = np.full(len(contour_path(upper)), 5.)
        segments.append((upper, lower))
        x.append(120)
        previous = {0: set(), 1: set(map(tuple, contour_path(segments[2][0]))), 2: set()}
        _, colored, samples = RootLinker(Config(n_clusters=3)).link_corners(
            [u for u, _ in segments], [{'point': (a, 5), 'angle': 90.} for a in x]
            + [l for _, l in segments], x, [5] * 3, previous)
        self.assertEqual(set(colored[(70, 105)]), {0, 1})
        self.assertNotIn((70, 90), samples[2])

    def test_reported_contacts_replayed_from_first_timepoint(self):
        expected = {
            ('RT_26_2-17', 30): {(891, 1986): (1, 2)},
            ('RT_26_2-62', 30): {(1537, 1010): (2, 3), (1514, 1135): 3, (1584, 1140): 2},
            ('RT_26_2-22', 30): {(1004, 621): 2},
            ('RT_26_2-3', 30): {(1270, 733): 3},
            ('RT_26_2-37', 30): {(569, 1823): 0},
            ('RT_26_2-40', 27): {(1183, 1466): 3},
            ('RT_26_2-40', 30): {(1142, 2100): 2, (1174, 2223): 3},
            ('RT_26_2-51', 30): {(615, 719): 0},
            ('RT_26_2-53', 30): {(505, 1000): 1, (1278, 876): 3},
            ('RT_26_2-55', 29): {(600, 672): 1},
            ('RT_26_2-59', 30): {(1257, 989): 1},
            ('RT_26_2-69', 28): {(395, 604): (0, 1), (370, 688): 1},
            ('RT_26_2-69', 30): {(459, 926): 0},
        }
        previous_by_group = {}
        with np.load(Path(__file__).parent / 'fixtures' / 'root_contact_evidence.npz', allow_pickle=False) as data:
            cases = json.loads(data['metadata'].tobytes())
            for i, case in enumerate(cases):
                for j, upper in enumerate(case['upper']):
                    upper['point'] = tuple(upper['point'])
                    upper['lower_point'] = tuple(upper['lower_point'])
                    upper['contour'] = data[f'contour_{i}_{j}']
                    upper['width_profile'] = data[f'width_{i}_{j}']
                    for sink in upper.get('terminal_exits', []):
                        sink['point'] = tuple(sink['point'])
                        sink['lower_point'] = tuple(sink['lower_point'])
                for lower in case['lower']:
                    lower['point'] = tuple(lower['point'])
                group, day = case['group'], case['day']
                previous = previous_by_group.get(group)
                _, colored, samples = RootLinker(Config(n_clusters=len(case['x']))).link_corners(
                    case['upper'], case['lower'], case['x'], case['y'], previous)
                previous_by_group[group] = samples
                for point, owner in expected.get((group, day), {}).items():
                    with self.subTest(group=group, day=day, point=point):
                        self.assertEqual(colored.get(point), owner)
                if (group, day) == ('RT_26_2-37', 30):
                    self.assertTrue(samples[0] & samples[1])
                    self.assertNotIn((569, 1823), samples[1])
                for upper in case['upper']:
                    with self.subTest(group=group, day=day, width_alignment=upper['point']):
                        self.assertEqual(len(contour_path(upper)), len(upper['width_profile']))

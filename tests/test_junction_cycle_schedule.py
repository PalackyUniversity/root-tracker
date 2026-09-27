"""Upstream cycles must not bypass physical evidence at downstream contacts."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from root_tracker.config import Config
from root_tracker.tracking.root_linker import RootLinker
from root_tracker.tracking.temporal_fragments import contour_path
from test_root_crossings import detected_segment


class JunctionCycleScheduleTests(unittest.TestCase):
    def test_cycle_recovery_preserves_downstream_contact_for_any_input_order(self):
        segments = [detected_segment((20, 10), (50, 80)),
                    detected_segment((80, 10), (80, 30)),
                    detected_segment((80, 35), (82, 40)),
                    detected_segment((80, 45), (56, 80)),
                    detected_segment((53, 85), (53, 125))]
        for i, (upper, _) in enumerate(segments):
            upper['width_profile'] = np.full(len(contour_path(upper)), 11. if i == 4 else 5.)
        segments[0][1]['junction_id'] = 2
        segments[1][1]['junction_id'] = 1
        segments[2][0]['junction_id'] = 1
        segments[2][1]['junction_id'] = 1
        segments[3][0]['junction_id'] = 1
        segments[3][1]['junction_id'] = 2
        segments[4][0]['junction_id'] = 2
        for order in ([4, 0, 3, 2, 1], [1, 2, 3, 0, 4]):
            scene = copy.deepcopy(segments)
            _, colored, samples = RootLinker(Config(n_clusters=2)).link_corners(
                [scene[i][0] for i in order],
                [dict(point=(x, 5), angle=90.) for x in (20, 80)]
                + [lower for _, lower in scene], [20, 80], [5, 5])
            with self.subTest(order=order):
                self.assertEqual(set(colored[(53, 125)]), {0, 1})
                self.assertTrue((53, 100) in samples[0] & samples[1])

    def test_recorded_rt37_cycle_does_not_erase_its_downstream_shared_prefix(self):
        previous = None
        with np.load(Path(__file__).parent / 'fixtures' / 'rt37_cycle_contact.npz') as data:
            for i, case in enumerate(json.loads(str(data['metadata']))):
                for j, upper in enumerate(case['upper']):
                    upper['point'] = tuple(upper['point'])
                    upper['lower_point'] = tuple(upper['lower_point'])
                    upper['contour'] = data[f'contour_{i}_{j}']
                    upper['width_profile'] = data[f'width_profile_{i}_{j}']
                    for sink in upper.get('terminal_exits', []):
                        sink['point'] = tuple(sink['point'])
                        sink['lower_point'] = tuple(sink['lower_point'])
                for lower in case['lower']:
                    lower['point'] = tuple(lower['point'])
                _, colored, samples = RootLinker(Config(n_clusters=6)).link_corners(
                    case['upper'], case['lower'], case['x'], case['y'], previous)
                previous = samples
                if case['day'] == 30:
                    self.assertTrue((476, 1200) in samples[0] & samples[1])
                    self.assertEqual(colored[(569, 1823)], 0)
                    self.assertNotIn((569, 1823), samples[1])

    def test_unrooted_cycle_does_not_block_grounded_downstream_fragment(self):
        from root_tracker.tracking.junction_router import route_junctions
        loop, loop_end = detected_segment((30, 20), (32, 25))
        branch, branch_end = detected_segment((30, 30), (30, 60))
        loop['junction_id'] = 1
        loop_end['junction_id'] = 2
        branch['junction_id'] = 2
        origin = (30, 5)
        # The unresolved upstream piece depends on itself through its base
        # link. The downstream fragment independently has a grounded base.
        _, colored, samples = route_junctions(
            [branch, loop], [loop_end, branch_end], {origin: 0},
            [(loop['point'], loop['lower_point']), (branch['point'], origin)],
            1)
        self.assertEqual(colored[branch['lower_point']], 0)
        self.assertIn((30, 50), samples[0])
        self.assertNotIn(loop['lower_point'], colored)

    def test_recorded_cycles_retain_existing_grounded_roots(self):
        with np.load(Path(__file__).parent / 'fixtures' / 'junction_cycle_grounding.npz') as data:
            for i, case in enumerate(json.loads(str(data['metadata']))):
                for j, upper in enumerate(case['upper']):
                    upper['point'] = tuple(upper['point'])
                    upper['lower_point'] = tuple(upper['lower_point'])
                    upper['contour'] = data[f'contour_{i}_{j}']
                    upper['width_profile'] = data[f'width_profile_{i}_{j}']
                    for sink in upper.get('terminal_exits', []):
                        sink['point'] = tuple(sink['point'])
                        sink['lower_point'] = tuple(sink['lower_point'])
                for lower in case['lower']:
                    lower['point'] = tuple(lower['point'])
                previous = {pid: set(map(tuple, data[f'previous_{i}_{pid}'])) for pid in range(6)}
                _, colored, samples = RootLinker(Config(n_clusters=6)).link_corners(
                    case['upper'], case['lower'], case['x'], case['y'], previous)
                with self.subTest(group=case['group'], day=case['day']):
                    target = tuple(case['target'])
                    self.assertEqual(colored.get(target), case['owner'])
                    self.assertIn(target, samples[case['owner']])

"""Downstream contact evidence cannot invent an upstream root arrival."""
import unittest
import json
from pathlib import Path

import numpy as np

from root_tracker.config import Config
from root_tracker.tracking.junction_router import route_junctions
from root_tracker.tracking.root_linker import RootLinker
from test_root_crossings import detected_segment


class ContactProvenanceTests(unittest.TestCase):
    def test_reported_upstream_root_stays_exclusive_across_all_frames(self):
        path = Path(__file__).parent / 'fixtures' / 'root_contact_provenance.npz'
        expected = {27: (1318, 435), 28: (1313, 447),
                    29: (1304, 456), 30: (1300, 460)}
        previous = None
        with np.load(path, allow_pickle=False) as archive:
            def restore(value):
                if isinstance(value, dict):
                    if set(value) == {'array'}:
                        return archive[value['array']].copy()
                    return {key: tuple(item) if key in ('point', 'lower_point')
                            else restore(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [restore(item) for item in value]
                return value

            for frame in json.loads(archive['metadata'].tobytes()):
                data = restore(frame['inputs'])
                _, colored, previous = RootLinker(Config()).link_corners(
                    data['upper'], data['lower'], data['x'], data['y'], previous)
                trunk = next(upper for upper in data['upper']
                             if upper['point'] == expected[frame['day']])
                with self.subTest(day=frame['day']):
                    self.assertEqual(colored[trunk['lower_point']], 2)
                    self.assertEqual(set(trunk['parent_points']), {2})

    def scene(self):
        trunk, bottom = detected_segment((70, 10), (70, 100))
        child, _ = detected_segment((70, 105), (70, 130))
        bottom['junction_id'] = 1
        child['junction_id'] = 1
        previous = {0: {(70, 100)} | {(70, y) for y in range(105, 131)},
                    1: {(70, y) for y in range(10, 100)}}
        return trunk, bottom, child, previous

    def test_downstream_owner_cannot_upgrade_an_unproven_gap_arrival(self):
        trunk, bottom, child, previous = self.scene()
        # A single old pixel at the bottom can supply a stale base link to A.
        # The rest of this root belongs to B, including its upstream entrance.
        _, colored, samples = route_junctions(
            [trunk, child], [bottom], {(20, 5): 0, (70, 5): 1},
            [((70, 10), (20, 5))], 2, previous)
        self.assertEqual(colored[(70, 100)], 1)
        self.assertEqual(trunk['parent_points'], {1: (70, 5)})
        self.assertTrue(previous[1] <= samples[1])
        self.assertFalse(previous[1] & samples[0])

    def test_real_upstream_arrival_can_use_its_downstream_continuation(self):
        trunk, bottom, child, previous = self.scene()
        trunk['junction_id'] = 2
        arrival = {'point': (20, 5), 'angle': 90., 'junction_id': 2}
        _, colored, samples = route_junctions(
            [trunk, child], [arrival, bottom], {(20, 5): 0, (70, 5): 1},
            [], 2, previous)
        self.assertEqual(set(colored[(70, 100)]), {0, 1})
        self.assertIn((70, 50), samples[0] & samples[1])

    def test_established_shared_history_survives_without_upstream_junction(self):
        trunk, bottom, child, previous = self.scene()
        shared = {(70, y) for y in range(10, 101)}
        previous[0].update(shared)
        previous[1].update(shared)
        _, colored, samples = route_junctions(
            [trunk, child], [bottom], {(20, 5): 0, (70, 5): 1},
            [((70, 10), (20, 5))], 2, previous)
        self.assertEqual(set(colored[(70, 100)]), {0, 1})
        self.assertTrue(shared <= samples[0] & samples[1])

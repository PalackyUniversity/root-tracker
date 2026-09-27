"""Plant assignments must follow a chain back to a plant origin."""
import unittest

import numpy as np

from root_tracker.config import Config
from root_tracker.tracking.root_linker import RootLinker


def segment(x, top, bottom):
    return {'point': (x, top), 'lower_point': (x, bottom), 'angle': 270.,
            'contour': np.array([[[x, y]] for y in range(top, bottom + 1)], dtype=np.int32)}


def lower(upper):
    return {'point': upper['lower_point'], 'angle': 90.}


class RootLinkerTests(unittest.TestCase):
    def setUp(self):
        self.linker = RootLinker(Config(n_clusters=1))
        self.origin = {'point': (100, 10), 'angle': 90., 'plant_id': 0}

    def test_disconnected_fragments_cannot_assign_each_other_to_a_plant(self):
        debris = [segment(100, 200, 220), segment(100, 225, 250), segment(100, 255, 280)]
        pairs, colored, samples = self.linker.link_corners(
            debris, [self.origin, *map(lower, debris)], [100], [10])
        self.assertEqual(pairs, [])
        self.assertEqual(colored, {(100, 10): 0})
        self.assertEqual(samples, {0: set()})

    def test_unassigned_fragment_does_not_hide_a_valid_root_link(self):
        root = segment(100, 20, 200)
        debris = segment(130, 180, 225)
        continuation = segment(130, 230, 260)
        pairs, colored, samples = self.linker.link_corners(
            [continuation, debris, root], [self.origin, lower(root), lower(debris)], [100], [10])
        self.assertEqual(pairs, [((100, 20), (100, 10)), ((130, 230), (100, 200))])
        self.assertNotIn(debris['lower_point'], colored)
        self.assertEqual(colored[continuation['lower_point']], 0)
        self.assertNotIn((130, 180), samples[0])

    def test_connected_chain_and_previous_frame_identity_are_preserved(self):
        roots = [segment(100, 20, 100), segment(100, 105, 200)]
        args = (roots, [self.origin, *map(lower, roots)], [100], [10])
        first = self.linker.link_corners(*args)
        second = self.linker.link_corners(*args, previous_colored_samples=first[2])
        self.assertEqual(first, second)
        self.assertEqual(first[0], [((100, 20), (100, 10)), ((100, 105), (100, 100))])
        self.assertEqual(len(first[2][0]), 177)

    def test_disconnected_fragments_do_not_persist_into_later_frames(self):
        roots = [segment(100, 20, 80)]
        debris = [segment(100, 200, 220), segment(100, 225, 250)]
        previous = None
        for _ in range(3):
            _, colored, previous = self.linker.link_corners(
                roots + debris, [self.origin, *map(lower, roots + debris)], [100], [10], previous)
            self.assertEqual(colored, {(100, 10): 0, (100, 80): 0})
            self.assertEqual(len(previous[0]), 61)

    def test_saved_series_keep_rooted_assignments_without_changing_other_plants(self):
        import json
        from pathlib import Path

        previous = {}
        days = {}
        # Counts measured from the disconnected chains in the original graph.
        removed = {
            'RT_26_2-18': (0, [0, 80, 0, 30]),
            'RT_26_2-19': (2, [10, 11, 0, 0]),
            'RT_26_2-21': (3, [0, 0, 0, 0]),
            'RT_26_2-26': (0, [93, 85, 79, 33]),
        }
        # Independently traced crossing corrections. The original baseline
        # predates the fix for opposing endpoint-angle conventions.
        corrected = {
            3: (1, 0, [(493, 583)]),
            5: (2, 1, [(1035, 568), (993, 496)]),
            6: (2, 1, [(863, 576), (1096, 674), (994, 495)]),
            7: (2, 1, [(1190, 778), (1141, 729), (1120, 709), (863, 576),
                        (1095, 675), (1032, 559), (992, 493)]),
        }
        path = Path(__file__).parent / 'fixtures' / 'root_link_corners.npz'
        with np.load(path, allow_pickle=False) as data:
            for index, case in enumerate(json.loads(data['metadata'].tobytes())):
                group = case['group']
                day = days.get(group, 0)
                days[group] = day + 1
                with self.subTest(group=group, day=27 + day):
                    uppers, lowers = case['upper'], case['lower']
                    for j, upper in enumerate(uppers):
                        upper['point'] = tuple(upper['point'])
                        upper['lower_point'] = tuple(upper['lower_point'])
                        upper['contour'] = data[f'contour_{index}_{j}']
                    for corner in lowers:
                        corner['point'] = tuple(corner['point'])
                    pairs, colored, samples = RootLinker(Config()).link_corners(
                        uppers, lowers, case['x'], case['y'], previous.get(group))
                    previous[group] = samples
                    reached = set(zip(case['x'], case['y']))
                    by_top = {u['point']: u for u in uppers}
                    for top, parent in pairs:
                        self.assertIn(parent, reached)
                        reached.add(by_top[top]['lower_point'])
                    self.assertEqual(set(colored), reached)
                    changed_plant, counts = removed[group]
                    baselines = {plant: set(map(tuple, data[f'baseline_{index}_{plant}'].tolist()))
                                 for plant in range(6)}
                    if index in corrected:
                        source, target, ends = corrected[index]
                        for upper in uppers:
                            if upper['lower_point'] in ends:
                                pixels = set(map(tuple, upper['contour'][:, 0].tolist()))
                                baselines[source].difference_update(pixels)
                                baselines[target].update(pixels)
                    for plant in range(6):
                        baseline = baselines[plant]
                        self.assertFalse(samples[plant] - baseline)
                        self.assertEqual(len(baseline - samples[plant]),
                                         counts[day] if plant == changed_plant else 0)

    def test_old_tracking_cache_requires_recalculation(self):
        from root_tracker.models import ImageSeries
        from root_tracker.pipeline import RootTrackingPipeline

        config = Config.from_yaml('configs/in_vitro.yaml')
        series = ImageSeries('example')
        series.pipeline_state.preprocessed = True
        series.pipeline_state.tracked = True
        series.pipeline_state.preprocess_config_hash = config.preprocess_config_hash()
        # Hash persisted before direction-aware shared-root routing.
        series.pipeline_state.tracking_config_hash = '4638b313f4c78e1a46a1952b084f5f67'
        self.assertFalse(RootTrackingPipeline(config).is_tracking_current(series))
        series.pipeline_state.tracking_config_hash = config.tracking_config_hash()
        self.assertTrue(RootTrackingPipeline(config).is_tracking_current(series))

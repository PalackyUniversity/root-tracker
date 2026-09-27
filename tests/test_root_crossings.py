"""Direction-aware links preserve plant identity through root crossings."""
import unittest
import json
from pathlib import Path

import cv2
import numpy as np

from root_tracker.config import Config
from root_tracker.tracking.corner_detector import CornerDetector
from root_tracker.tracking.root_linker import RootLinker


def detected_segment(start, end):
    mask = np.zeros((140, 150), np.uint8)
    cv2.line(mask, start, end, 255, 1)
    contour = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0][0]
    upper, lower = CornerDetector(Config()).analyze_contour_corners(contour, mask)
    upper.update(contour=contour, lower_point=lower['point'])
    return upper, lower


def fixture_frames(filename):
    with np.load(Path(__file__).parent / 'fixtures' / filename, allow_pickle=False) as data:
        for i, case in enumerate(json.loads(data['metadata'].tobytes())):
            for j, upper in enumerate(case['upper']):
                upper['point'] = tuple(upper['point'])
                upper['lower_point'] = tuple(upper['lower_point'])
                upper['contour'] = data[f'contour_{i}_{j}']
                for terminal in upper.get('terminal_exits', []):
                    terminal['point'] = tuple(terminal['point'])
                    terminal['lower_point'] = tuple(terminal['lower_point'])
            for lower in case['lower']:
                lower['point'] = tuple(lower['point'])
            yield case


class RootCrossingTests(unittest.TestCase):
    def test_crossing_roots_follow_their_straight_continuations(self):
        segments = [detected_segment((20, 10), (67, 57)),
                    detected_segment((120, 10), (73, 57)),
                    detected_segment((73, 63), (120, 110)),
                    detected_segment((67, 63), (20, 110))]
        origins = [{'point': (x, 5), 'angle': 90., 'plant_id': i}
                   for i, x in enumerate((20, 120))]
        linker = RootLinker(Config(n_clusters=2))
        previous = None
        for day in range(3):
            with self.subTest(day=day):
                pairs, colored, previous = linker.link_corners(
                    [u for u, _ in segments], origins + [l for _, l in segments],
                    [20, 120], [5, 5], previous)
                self.assertEqual(colored[(120, 110)], 0)
                self.assertEqual(colored[(20, 110)], 1)
                self.assertIn(((73, 63), (67, 57)), pairs)
                self.assertIn(((67, 63), (73, 57)), pairs)

    def test_straight_vertical_gap_has_no_direction_penalty(self):
        child, _ = detected_segment((70, 60), (70, 110))
        _, parent = detected_segment((70, 10), (70, 55))
        cost = RootLinker(Config()).compute_link_cost(child, parent, 100, [20, 120], [5, 5])
        self.assertAlmostEqual(cost, 5.)

    def test_direction_comparison_wraps_at_zero_degrees(self):
        # Upper endpoints point out of the segment; 179 degrees therefore
        # describes a continuation heading at 359 degrees, close to 1 degree.
        child = {'point': (80, 60), 'angle': 179.}
        parent = {'point': (70, 60), 'angle': 1.}
        cost = RootLinker(Config()).compute_link_cost(child, parent, 100, [20, 120], [5, 5])
        self.assertAlmostEqual(cost, 10 + 2 * 100 / 180 / 4 + 2 * 100 / 180 / 2)

    def test_reported_crossings_and_horizontal_branch_keep_their_parent(self):
        # Hand-traced endpoints in the reported images, with zero-based IDs.
        expected = {
            ('RT_26_2-25', 29): {(1585, 766): 3, (1458, 657): 2},
            ('RT_26_2-25', 30): {(1656, 787): 3, (1109, 883): 1, (1485, 683): 2},
            ('RT_26_2-19', 30): {(1190, 778): 1, (863, 576): 1},
        }
        previous = {}
        for case in fixture_frames('root_crossings.npz'):
            group, day = case['group'], case['day']
            _, colored, samples = RootLinker(Config()).link_corners(
                case['upper'], case['lower'], case['x'], case['y'], previous.get(group))
            previous[group] = samples
            for point, plant in expected.get((group, day), {}).items():
                with self.subTest(group=group, day=day, point=point):
                    self.assertEqual(colored[point], plant)

    def test_merged_roots_share_pixels_then_recover_both_identities(self):
        segments = [detected_segment((20, 10), (67, 57)),
                    detected_segment((120, 10), (73, 57)),
                    detected_segment((70, 63), (70, 85)),
                    detected_segment((73, 91), (110, 128)),
                    detected_segment((67, 91), (30, 128))]
        for _, lower in segments[:2]:
            lower['junction_id'] = 1
        segments[2][0]['junction_id'] = 1
        segments[2][1]['junction_id'] = 2
        for upper, _ in segments[3:]:
            upper['junction_id'] = 2
        origins = [{'point': (x, 5), 'angle': 90., 'plant_id': i}
                   for i, x in enumerate((20, 120))]
        uppers = [u for u, _ in segments]
        previous = None
        for day in range(3):
            with self.subTest(day=day):
                pairs, colored, previous = RootLinker(Config(n_clusters=2)).link_corners(
                    uppers, origins + [l for _, l in segments], [20, 120], [5, 5], previous)
                self.assertEqual(colored[(70, 85)], (0, 1))
                self.assertEqual(colored[(110, 128)], 0)
                self.assertEqual(colored[(30, 128)], 1)
                self.assertIn((70, 75), previous[0])
                self.assertIn((70, 75), previous[1])
                self.assertEqual(uppers[2]['parent_points'], {0: (67, 57), 1: (73, 57)})

    def test_single_plant_branch_is_not_treated_as_two_roots(self):
        trunk = detected_segment((70, 10), (70, 57))
        branches = [detected_segment((73, 63), (120, 110)),
                    detected_segment((67, 63), (20, 110))]
        trunk[1]['junction_id'] = 1
        for upper, _ in branches:
            upper['junction_id'] = 1
        segments = [trunk, *branches]
        _, colored, samples = RootLinker(Config(n_clusters=1)).link_corners(
            [u for u, _ in segments], [{'point': (70, 5), 'angle': 90.}] + [l for _, l in segments],
            [70], [5])
        self.assertEqual(colored[(120, 110)], 0)
        self.assertEqual(colored[(20, 110)], 0)
        self.assertTrue(all(isinstance(owner, int) for owner in colored.values()))

    def test_filtered_terminal_exit_does_not_force_false_sharing(self):
        segments = [detected_segment((20, 10), (67, 57)),
                    detected_segment((120, 10), (73, 57)),
                    detected_segment((67, 63), (20, 110))]
        for _, lower in segments[:2]:
            lower['junction_id'] = 1
        segments[2][0].update(junction_id=1, terminal_exits=[
            {'point': (73, 63), 'lower_point': (76, 66),
             'angle': 225., 'junction_id': 1}])
        _, colored, samples = RootLinker(Config(n_clusters=2)).link_corners(
            [u for u, _ in segments],
            [{'point': (x, 5), 'angle': 90.} for x in (20, 120)]
            + [l for _, l in segments], [20, 120], [5, 5])
        self.assertEqual(colored[(20, 110)], 1)
        self.assertNotIn((76, 66), colored)
        self.assertNotIn((76, 66), samples[0])

    def test_pipeline_measures_and_draws_both_roots_on_shared_section(self):
        from datetime import datetime
        from tempfile import TemporaryDirectory
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.pipeline import RootTrackingPipeline

        process = np.zeros((180, 210), np.uint8)
        for start, end in [((30, 15), (100, 60)), ((150, 15), (100, 60)),
                           ((100, 60), (100, 100)), ((100, 100), (155, 160)),
                           ((100, 100), (45, 160))]:
            cv2.line(process, start, end, 255, 3)
        image = ImageData(datetime(2026, 4, 30), 'merge.png', 'merge')
        image.image = cv2.cvtColor(process, cv2.COLOR_GRAY2BGR)
        image.process = process
        image.positions_x, image.positions_y = [30, 150], [15, 15]
        with TemporaryDirectory() as folder:
            config = Config(n_clusters=2)
            config.data.output = folder
            RootTrackingPipeline(config).track_and_analyze_series(
                ImageSeries('merge', [image]), save_images=False)
        self.assertIn((100, 80), image.colored_samples[0])
        self.assertIn((100, 80), image.colored_samples[1])
        for plant in (0, 1):
            self.assertIn((100, 80), set(map(tuple, image.rsml_samples[plant].tolist())))
        self.assertIn((130, 133), image.colored_samples[0])
        self.assertIn((69, 133), image.colored_samples[1])
        self.assertEqual(image.plant_length, image.longest)
        self.assertGreater(image.plant_length[0], image.plant_length[1])
        # Shared main root: left-arriving plant is blue, right is green.
        self.assertTupleEqual(tuple(image.image_annotated[80, 97]), (255, 170, 170))
        self.assertTupleEqual(tuple(image.image_annotated[80, 103]), (170, 255, 170))

    def test_reported_shared_sections_split_back_to_the_correct_plants(self):
        expected = {
            'RT_26_2-17': {(930, 1100): (1, 2), (950, 1165): 1, (860, 1129): 2},
            'RT_26_2-62': {(1537, 1010): (2, 3), (1514, 1135): 3, (1584, 1140): 2},
        }
        previous = {}
        for case in fixture_frames('root_shared_junctions.npz'):
            group = case['group']
            pairs, colored, samples = RootLinker(Config()).link_corners(
                case['upper'], case['lower'], case['x'], case['y'], previous.get(group))
            previous[group] = samples
            # Every per-plant parent must carry that same plant identity.
            from root_tracker.tracking.junction_router import plant_ids
            for upper in case['upper']:
                for plant, parent in upper.get('parent_points', {}).items():
                    self.assertIn(plant, plant_ids(colored[parent]))
            if case['day'] == 30:
                for point, plant in expected[group].items():
                    with self.subTest(group=group, point=point):
                        self.assertEqual(colored[point], plant)

    def test_reliable_temporal_support_disambiguates_shared_split(self):
        segments = [detected_segment((20, 10), (67, 57)),
                    detected_segment((120, 10), (73, 57)),
                    detected_segment((70, 63), (70, 85)),
                    detected_segment((73, 91), (110, 128)),
                    detected_segment((67, 91), (30, 128))]
        for i, (_, lower) in enumerate(segments[:2]):
            lower.update(junction_id=1, angle=[60., 110.][i])
        segments[2][0]['junction_id'] = 1
        segments[2][1]['junction_id'] = 2
        for i, (upper, _) in enumerate(segments[3:]):
            upper.update(junction_id=2, angle=[270., 340.][i])
        origins = [{'point': (x, 5), 'angle': 90.} for x in (20, 120)]
        previous = {1: set(map(tuple, segments[3][0]['contour'][:, 0].tolist())),
                    0: set(map(tuple, segments[4][0]['contour'][:, 0].tolist()))}
        shifted = {pid: {(x + 1, y + 1) for x, y in points}
                   for pid, points in previous.items()}
        scenarios = [
            ('strong', previous, 1, 0),
            ('registration jitter', shifted, 1, 0),
            ('single pixel', {1: {next(iter(previous[1]))}}, 0, 1),
            ('nearby separate histories',
             {0: {(x - 1, y) for x, y in previous[1]},
              1: {(x + 2, y) for x, y in previous[1]}}, 0, 1),
            ('established shared history', {0: previous[1], 1: previous[1]}, (0, 1), 1),
            ('weaker competing branch', {1: previous[1] | set(sorted(previous[0])[:12])}, 1, 0),
        ]
        for name, evidence, right_plant, left_plant in scenarios:
            with self.subTest(evidence=name):
                _, colored, _ = RootLinker(Config(n_clusters=2)).link_corners(
                    [u for u, _ in segments], origins + [l for _, l in segments],
                    [20, 120], [5, 5], evidence)
                self.assertEqual(colored[(110, 128)], right_plant)
                self.assertEqual(colored[(30, 128)], left_plant)

    def test_shared_bands_keep_individual_main_root_highlighting(self):
        from root_tracker.tracking.shared_rendering import draw_shared_segments
        upper, _ = detected_segment((70, 20), (70, 110))
        upper['contour_index'] = 0
        image = np.zeros((140, 150, 3), np.uint8)
        draw_shared_segments(image, [upper], {(70, 110): (0, 1)},
                             {0: {0}}, RootLinker(Config()))
        self.assertTupleEqual(tuple(image[60, 67]), (255, 170, 170))
        self.assertTupleEqual(tuple(image[60, 73]), (0, 255, 0))

    def test_neighboring_unconnected_junctions_cannot_exchange_plants(self):
        from datetime import datetime
        from tempfile import TemporaryDirectory
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.pipeline import RootTrackingPipeline

        process = np.zeros((120, 120), np.uint8)
        for start, end in [((30, 10), (50, 60)), ((50, 60), (50, 110)),
                           ((50, 60), (15, 95)), ((73, 10), (53, 60)),
                           ((53, 60), (53, 110)), ((53, 60), (90, 95))]:
            cv2.line(process, start, end, 255, 1)
        image = ImageData(datetime(2026, 4, 30), 'separate.png', 'separate')
        image.image = cv2.cvtColor(process, cv2.COLOR_GRAY2BGR)
        image.process = process
        image.positions_x, image.positions_y = [30, 73], [10, 10]
        with TemporaryDirectory() as folder:
            config = Config(n_clusters=2)
            config.threshold.min_contour_area = 0
            config.data.output = folder
            RootTrackingPipeline(config).track_and_analyze_series(
                ImageSeries('separate', [image]), save_images=False)
        self.assertIn((20, 90), image.colored_samples[0])
        self.assertIn((85, 90), image.colored_samples[1])
        self.assertFalse(image.colored_samples[0] & image.colored_samples[1])

    def test_other_series_keep_their_established_main_roots_at_laterals(self):
        expected = {'RT_26_2-1': ((225, 2296), 1),
                    'RT_26_2-7': ((2577, 1823), 5),
                    'RT_26_2-12': ((2066, 2153), 4),
                    'RT_26_2-51': ((1240, 2313), 3)}
        previous = {}
        for case in fixture_frames('root_crossing_controls.npz'):
            group = case['group']
            _, colored, previous[group] = RootLinker(Config()).link_corners(
                case['upper'], case['lower'], case['x'], case['y'], previous.get(group))
            if case['day'] == 30:
                point, plant = expected[group]
                with self.subTest(group=group):
                    self.assertEqual(colored[point], plant)

    def test_new_lateral_does_not_split_an_established_shared_trunk(self):
        origins = [{'point': (x, 5), 'angle': 90.} for x in (20, 120)]
        previous = None
        for branched in (False, True):
            segments = [detected_segment((20, 10), (67, 57)),
                        detected_segment((120, 10), (73, 57)),
                        detected_segment((70, 63), (70, 85 if branched else 130))]
            for _, lower in segments[:2]:
                lower['junction_id'] = 1
            segments[2][0]['junction_id'] = 1
            if branched:
                segments[2][1]['junction_id'] = 2
                segments.extend([detected_segment((70, 91), (70, 130)),
                                 detected_segment((76, 91), (113, 128))])
                for upper, _ in segments[3:]:
                    upper['junction_id'] = 2
            _, colored, previous = RootLinker(Config(n_clusters=2)).link_corners(
                [u for u, _ in segments], origins + [l for _, l in segments],
                [20, 120], [5, 5], previous)
            with self.subTest(branched=branched):
                self.assertEqual(colored[(70, 130)], (0, 1))
                for plant in (0, 1):
                    self.assertIn((70, 120), previous[plant])
                if branched:
                    self.assertIsInstance(colored[(113, 128)], int)

    def test_discarded_short_connections_keep_the_complete_crossing_topology(self):
        from datetime import datetime
        from tempfile import TemporaryDirectory
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.pipeline import RootTrackingPipeline

        # Actual RT19/day30 skeleton crop. Tiny links inside the crossing are
        # removed by the segment-length filter, but still connect its junctions.
        process = np.load(Path(__file__).parent / 'fixtures' / 'root19_crossing_mask.npy')
        image = ImageData(datetime(2026, 4, 30), 'crossing.png', 'crossing')
        image.image = cv2.cvtColor(process, cv2.COLOR_GRAY2BGR)
        image.process = process
        image.positions_x, image.positions_y = [24, 129], [0, 7]
        with TemporaryDirectory() as folder:
            config = Config(n_clusters=2)
            config.threshold.min_contour_area = 0
            config.data.output = folder
            RootTrackingPipeline(config).track_and_analyze_series(
                ImageSeries('crossing', [image]), save_images=False)
        self.assertIn((99, 119), image.colored_samples[0])
        self.assertNotIn((99, 119), image.colored_samples[1])

    def test_pipeline_keeps_short_terminal_directions_after_filtering(self):
        from datetime import datetime
        from tempfile import TemporaryDirectory
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.pipeline import RootTrackingPipeline

        source = np.load(Path(__file__).parent / 'fixtures' / 'root17_terminal_mask.npy')
        for horizontal in (False, True):
            process = source.copy()
            if horizontal:
                # Replace the tiny southeast twig with a rightward horizontal
                # terminal, whose arbitrary upper/lower ordering is reversed.
                process[32:35, 33:37] = 0
                process[31, 30:37] = 255
            image = ImageData(datetime(2026, 4, 30), 'terminal.png', 'terminal')
            image.image = cv2.cvtColor(process, cv2.COLOR_GRAY2BGR)
            image.process = process
            image.positions_x, image.positions_y = [0, 44], [4, 0]
            with TemporaryDirectory() as folder:
                config = Config(n_clusters=2)
                config.threshold.min_contour_area = 0
                config.data.output = folder
                RootTrackingPipeline(config).track_and_analyze_series(
                    ImageSeries('terminal', [image]), save_images=False)
            with self.subTest(horizontal=horizontal):
                self.assertIn((20, 59), image.colored_samples[1])
                self.assertNotIn((20, 59), image.colored_samples[0])

"""Main-root identity survives lateral growth and segment subdivision."""
import unittest
import numpy as np
from root_tracker.tracking.main_root import select_main_path


def segment(start, end, index, parent):
    n = max(abs(end[0]-start[0]), abs(end[1]-start[1])) + 1
    points = np.rint(np.linspace(start, end, n)).astype(np.int32)
    return dict(point=start, lower_point=end, contour_index=index,
                contour=points.reshape(-1, 1, 2), parent_points={0: parent},
                width_profile=np.full(n, 4.))


class MainRootContinuityTests(unittest.TestCase):
    def test_lateral_overtaking_main_tip_does_not_replace_it(self):
        upper = [segment((20, 0), (20, 40), 0, (20, 0)),
                 segment((20, 41), (20, 80), 1, (20, 40)),
                 segment((21, 41), (60, 100), 2, (20, 40))]
        colored = {u['lower_point']: 0 for u in upper}
        history = {(20, y) for y in range(81)}
        tip, indices = select_main_path(upper, colored, [], 0, history)
        self.assertEqual(tip, (20, 80))
        self.assertEqual(set(indices), {0, 1})
        # With no previous identity, preserve deepest-tip initialization.
        self.assertEqual(select_main_path(upper, colored, [], 0, set())[0], (60, 100))

    def test_subdivision_and_skeleton_shift_preserve_continuation(self):
        upper = [segment((21, 0), (21, 40), 0, (21, 0)),
                 segment((21, 41), (21, 60), 1, (21, 40)),
                 segment((21, 61), (21, 90), 2, (21, 60)),
                 segment((22, 41), (60, 110), 3, (21, 40))]
        colored = {u['lower_point']: 0 for u in upper}
        tip, indices = select_main_path(upper, colored, [], 0,
                                        {(20, y) for y in range(81)})
        self.assertEqual(tip, (21, 90))
        self.assertEqual(set(indices), {0, 1, 2})

    def test_parent_walk_cannot_enter_another_plants_root(self):
        upper = [segment((20, 0), (20, 40), 0, (20, 0)),
                 segment((20, 41), (20, 80), 1, (20, 40))]
        tip, indices = select_main_path(upper, {(20, 40): 1, (20, 80): 0}, [], 0, set())
        self.assertEqual(indices, [1])

    def test_shared_segment_uses_each_plants_parent(self):
        upper = [segment((20, 0), (20, 40), 0, (20, 0)),
                 segment((60, 0), (60, 40), 1, (60, 0)),
                 segment((40, 41), (40, 80), 2, (20, 40))]
        upper[2]['parent_points'][1] = (60, 40)
        upper[1]['parent_points'] = {1: (60, 0)}
        colored = {(20, 40): 0, (60, 40): 1, (40, 80): (0, 1)}
        self.assertEqual(select_main_path(upper, colored, [], 0, set())[1], [2, 0])
        self.assertEqual(select_main_path(upper, colored, [], 1, set())[1], [2, 1])

    def test_rejoining_lateral_cannot_replace_main_ancestry(self):
        upper = [segment((20, 0), (20, 30), 0, (20, 0)),
                 segment((20, 31), (20, 70), 1, (20, 30)),
                 segment((21, 31), (50, 70), 2, (20, 30)),
                 segment((20, 71), (20, 100), 3, (50, 70))]
        upper[3]['junction_id'] = 2
        lowers = [{'point': (20, 70), 'junction_id': 2},
                  {'point': (50, 70), 'junction_id': 2}]
        colored = {u['lower_point']: 0 for u in upper}
        tip, indices = select_main_path(upper, colored, [], 0,
                                        {(20, y) for y in range(91)}, lowers)
        self.assertEqual(tip, (20, 100))
        self.assertEqual(indices, [3, 1, 0])

    def test_recorded_main_routes_survive_overtaking_and_reconnection(self):
        import json
        from pathlib import Path
        histories = {}
        previous_roots = {}
        restored = {(5, 30): (1094, 969), (15, 30): (336, 562),
                    (52, 28): (776, 1339)}
        rejected = {(5, 30): (1112, 978), (15, 30): (231, 658)}
        with np.load(Path(__file__).parent / 'fixtures/main_root_continuity.npz') as data:
            for frame in json.loads(str(data['metadata'])):
                upper = []
                for raw in frame['upper']:
                    u = dict(point=tuple(raw['point']), lower_point=tuple(raw['lower_point']),
                             contour_index=raw['contour_index'], junction_id=raw['junction_id'],
                             contour=data[raw['key']], width_profile=data[raw['key']+'w'])
                    if raw['parent'] is not None:
                        u['parent_points'] = {0: tuple(raw['parent'])}
                    upper.append(u)
                lower = [dict(point=tuple(l['point']), junction_id=l['junction_id'])
                         for l in frame['lower']]
                pairs = [(tuple(a), tuple(b)) for a, b in frame['pairs']]
                group = frame['group']
                _, indices = select_main_path(upper, {u['lower_point']: 0 for u in upper},
                                               pairs, 0, histories.get(group, set()), lower,
                                               previous_roots.get(group))
                pixels = {tuple(p) for u in upper if u['contour_index'] in indices
                          for p in u['contour'][:, 0]}
                histories.setdefault(group, set()).update(pixels)
                previous_roots[group] = {tuple(p) for u in upper for p in u["contour"][:, 0]}
                key = (group, frame['day'])
                with self.subTest(group=group, day=frame['day']):
                    if group == 15 and frame["day"] >= 28:
                        self.assertGreater(max(y for x, y in pixels), {28: 1800, 29: 2100, 30: 2200}[frame["day"]])
                    if key in restored:
                        self.assertIn(restored[key], pixels)
                    if key in rejected:
                        self.assertNotIn(rejected[key], pixels)

    def test_missing_main_cannot_be_aligned_onto_a_different_lateral(self):
        upper = [segment((0, 0), (0, 20), 0, (0, -1)),
                 segment((30, 0), (30, 120), 1, (0, -1))]
        history = {(0, y) for y in range(101)}
        tip, indices = select_main_path(upper, {u['lower_point']: 0 for u in upper},
                                        [], 0, history, previous_roots=history)
        self.assertEqual(tip, (0, 20))
        self.assertEqual(indices, [0])

    def test_pipeline_keeps_identity_across_empty_observation(self):
        import cv2
        from datetime import datetime
        from tempfile import TemporaryDirectory
        from root_tracker.config import Config
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.pipeline import RootTrackingPipeline
        images = []
        for day in range(3):
            mask = np.zeros((140, 100), np.uint8)
            if day != 1:
                cv2.line(mask, (20, 12), (20, 80), 255, 3)
            if day == 2:
                cv2.line(mask, (20, 40), (60, 110), 255, 3)
            image = ImageData(datetime(2026, 4, day+1), f'{day}.png', 'dropout')
            image.image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            image.process = mask
            image.positions_x, image.positions_y = [20], [10]
            images.append(image)
        with TemporaryDirectory() as folder:
            config = Config(n_clusters=1)
            config.threshold.min_contour_area = 0
            config.data.output = folder
            RootTrackingPipeline(config).track_and_analyze_series(
                ImageSeries('dropout', images), save_images=False)
        self.assertGreater(len(images[0].main_root_samples), 0)
        self.assertEqual(len(images[1].main_root_samples), 0)
        final = set(map(tuple, images[2].main_root_samples[:, 1:]))
        self.assertIn((20, 70), final)
        self.assertLess(max(y for x, y in final), 90)

    def test_missing_lateral_cannot_displace_an_intact_main(self):
        upper = [segment((0, 0), (0, 80), 0, (0, -1)),
                 segment((60, 0), (60, 120), 1, (0, -1))]
        history = {(0, y) for y in range(81)}
        previous_roots = history | {(25, y) for y in range(101)}
        tip, indices = select_main_path(upper, {u['lower_point']: 0 for u in upper},
                                        [], 0, history, previous_roots=previous_roots)
        self.assertEqual(tip, (0, 80))
        self.assertEqual(indices, [0])

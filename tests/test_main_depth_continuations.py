"""Main depth follows growth across gaps without promoting known laterals."""
import unittest
import numpy as np
from root_tracker.tracking.main_root import select_main_geometry
from test_main_root_continuity import segment


class MainDepthContinuationTests(unittest.TestCase):
    def test_gap_parent_cannot_skip_the_intermediate_main_fragment(self):
        upper = [segment((20, 0), (20, 10), 0, (20, -1)),
                 segment((20, 11), (20, 60), 1, (20, 10)),
                 segment((20, 85), (20, 100), 2, (20, 10))]
        tip, geometry = select_main_geometry(upper,
            {u['lower_point']: 0 for u in upper}, [], 0, set())
        self.assertEqual(tip, (20, 100))
        self.assertEqual(set(geometry), {0, 1, 2})

    def test_origin_linked_distal_fragment_continues_main_across_gap(self):
        for scale in (1, 3):
            upper = [segment((20, 10), (20, 50), 0, (20, 0)),
                     segment((20, 65), (20, 120), 1, (20, 0))]
            for node in upper:
                for key in ('point', 'lower_point'):
                    node[key] = tuple(scale * v for v in node[key])
                node['parent_points'][0] = (20 * scale, 0)
                node['contour'] *= scale
                node['width_profile'] *= scale
            history = {(20 * scale, y) for y in range(10 * scale, 51 * scale)}
            with self.subTest(scale=scale):
                tip, geometry = select_main_geometry(upper,
                    {u['lower_point']: 0 for u in upper}, [], 0, history,
                    previous_roots=history)
                self.assertEqual(tip, (20 * scale, 120 * scale))
                self.assertEqual(set(geometry), {0, 1})
                # Main inference must not rewrite plant ownership ancestry.
                self.assertEqual(upper[1]['parent_points'][0], (20 * scale, 0))

    def test_disconnected_known_lateral_cannot_extend_main(self):
        upper = [segment((20, 10), (20, 50), 0, (20, 0)),
                 segment((20, 65), (20, 120), 1, (20, 0))]
        history = {(20, y) for y in range(10, 51)}
        roots = history | {(20, y) for y in range(65, 101)}
        tip, geometry = select_main_geometry(upper,
            {u['lower_point']: 0 for u in upper}, [], 0, history, previous_roots=roots)
        self.assertEqual(tip, (20, 50))
        self.assertEqual(set(geometry), {0})

    def test_equally_near_fragments_do_not_invent_a_main_continuation(self):
        upper = [segment((20, 10), (20, 50), 0, (20, 0)),
                 segment((18, 65), (18, 120), 1, (20, 0)),
                 segment((22, 65), (22, 120), 2, (20, 0))]
        history = {(20, y) for y in range(10, 51)}
        for nodes in (upper, list(reversed(upper))):
            tip, geometry = select_main_geometry(nodes,
                {u['lower_point']: 0 for u in nodes}, [], 0, history, previous_roots=history)
            self.assertEqual(tip, (20, 50))
            self.assertEqual(set(geometry), {0})

    def test_side_by_side_root_is_not_a_distal_continuation(self):
        upper = [segment((20, 10), (20, 50), 0, (20, 0)),
                 segment((40, 10), (40, 120), 1, (20, 0))]
        history = {(20, y) for y in range(10, 51)}
        tip, geometry = select_main_geometry(upper,
            {u['lower_point']: 0 for u in upper}, [], 0, history, previous_roots=history)
        self.assertEqual(tip, (20, 50))
        self.assertEqual(set(geometry), {0})

    def test_first_compact_detection_does_not_lock_later_main_identity(self):
        import cv2
        from datetime import datetime
        from tempfile import TemporaryDirectory
        from root_tracker.config import Config
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.pipeline import RootTrackingPipeline
        images = []
        for day in range(2):
            mask = np.zeros((160, 160), np.uint8)
            cv2.ellipse(mask, (95, 45), (10, 16), 0, 0, 360, 255, -1)
            if day:
                cv2.line(mask, (60, 25), (60, 130), 255, 3)
            im = ImageData(datetime(2026, 4, day + 1), f'{day}.png', 'provisional')
            im.image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            im.process = mask
            im.positions_x, im.positions_y = [95], [25]
            images.append(im)
        with TemporaryDirectory() as folder:
            config = Config(n_clusters=1)
            config.threshold.min_contour_area = 0
            config.data.output = folder
            RootTrackingPipeline(config).track_and_analyze_series(
                ImageSeries('provisional', images), save_images=False)
        self.assertGreater(len(images[0].main_root_samples), 0)
        selected = set(map(tuple, images[-1].main_root_samples[:, 1:]))
        self.assertIn((60, 120), selected)
        self.assertNotIn((95, 45), selected)

"""Time-series evidence rejects debris without erasing established roots."""
import unittest

import cv2
import numpy as np

from root_tracker.tracking import temporal_debris


class TemporalDebrisTests(unittest.TestCase):
    def scene(self):
        frames = []
        for length in (45, 55, 65, 75):
            mask = np.zeros((120, 120), np.uint8)
            cv2.line(mask, (30, 10), (30, length), 255, 3)
            cv2.line(mask, (30, 30), (45, 40), 255, 3)  # old lateral
            cv2.line(mask, (65, 35), (70, 45), 255, 3)  # debris
            frames.append(mask)
        return frames

    def test_static_island_removed_and_entire_growing_root_kept(self):
        frames = self.scene()
        original = [m.copy() for m in frames]
        result = temporal_debris.filter_static_islands(frames, [(30, 10)])
        for mask, actual in zip(original, result):
            expected = mask.copy()
            expected[:, 60:] = 0
            np.testing.assert_array_equal(actual, expected)
        for before, after in zip(original, frames):
            np.testing.assert_array_equal(before, after)

    def test_small_registration_jitter_does_not_rescue_debris(self):
        frames = self.scene()
        frames[1][:, 60:] = np.roll(frames[1][:, 60:], 1, axis=0)
        frames[2][:, 60:] = np.roll(frames[2][:, 60:], -1, axis=1)
        result = temporal_debris.filter_static_islands(frames, [(30, 10)])
        self.assertTrue(all(not np.any(m[:, 60:]) for m in result))

    def test_stationary_root_attached_to_origin_is_kept(self):
        frames = self.scene()
        frames = [frames[0].copy() for _ in frames]
        result = temporal_debris.filter_static_islands(frames, [(30, 10)])
        self.assertTrue(all(np.all(m[10:45, 30] == 255) for m in result))

    def test_detached_fragment_with_extension_is_kept(self):
        frames = self.scene()
        for i, mask in enumerate(frames):
            cv2.line(mask, (90, 50), (90, 60 + 5*i), 255, 3)
        result = temporal_debris.filter_static_islands(frames, [(30, 10)])
        self.assertTrue(all(np.all(m[50:60, 90] == 255) for m in result))

    def test_new_late_fragment_is_kept(self):
        frames = self.scene()
        frames[0][:, 60:] = 0
        result = temporal_debris.filter_static_islands(frames, [(30, 10)])
        self.assertTrue(np.any(result[-1][:, 60:]))

    def test_insufficient_or_incompatible_frames_are_unchanged(self):
        mask = self.scene()[0]
        for frames in ([mask], [mask, None], [mask, np.zeros((30, 30), np.uint8)]):
            result = temporal_debris.filter_static_islands(frames, [(30, 10)])
            np.testing.assert_array_equal(result[0], mask)

    def test_stationary_fragment_joined_by_later_growth_is_kept(self):
        frames = self.scene()
        cv2.line(frames[-1], (30, 35), (65, 35), 255, 3)
        result = temporal_debris.filter_static_islands(frames, [(30, 10)])
        for before, after in zip(frames, result):
            np.testing.assert_array_equal(before, after)

    def track_scene(self, masks=None, registration=True, user_mask=None):
        from datetime import datetime
        from root_tracker.config import Config
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.pipeline import RootTrackingPipeline

        config = Config(n_clusters=1)
        config.registration.enabled = registration
        config.threshold.min_contour_area = 1
        config.threshold.min_contour_length = 1
        series = ImageSeries('synthetic')
        series.user_mask = user_mask
        for i, mask in enumerate(self.scene() if masks is None else masks):
            image = ImageData(datetime(2026, 1, i + 1), 'unused.png', 'synthetic')
            image.process = mask
            image.image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            image.positions_x, image.positions_y = [30], [10]
            series.images.append(image)
        RootTrackingPipeline(config).track_and_analyze_series(series, save_images=False)
        return series

    def test_pipeline_does_not_measure_or_remember_rejected_island(self):
        series = self.track_scene()
        for image in series:
            self.assertTrue(image.colored_samples[0])
            self.assertTrue(all(x < 60 for x, y in image.colored_samples[0]))
            self.assertTrue(all(x < 60 for x, y in image.rsml_samples[0]))

    def test_unregistered_series_keeps_existing_detections(self):
        series = self.track_scene(registration=False)
        self.assertTrue(any(x > 60 for x, y in series.images[0].colored_samples[0]))

    def test_user_exclusion_cannot_turn_old_root_into_debris(self):
        frames = [self.scene()[0]] * 3
        user_mask = np.zeros_like(frames[0])
        user_mask[20:25, :] = 255
        series = self.track_scene(frames, user_mask=user_mask)
        for image in series:
            self.assertTrue(any(y > 25 for x, y in image.colored_samples[0]))
            self.assertFalse(any(20 <= y < 25 for x, y in image.colored_samples[0]))
            self.assertFalse(any(x > 60 for x, y in image.rsml_unmasked_samples[0]))

    def test_fading_static_object_is_not_mistaken_for_growth(self):
        frames = self.scene()
        for i, mask in enumerate(frames):
            cv2.line(mask, (90, 30 + 3*i), (90, 60 - 3*i), 255, 5)
        result = temporal_debris.filter_static_islands(frames, [(30, 10)])
        self.assertTrue(all(not np.any(m[:, 80:]) for m in result))

    def test_background_translation_is_estimated_from_persistent_objects(self):
        frames = []
        for i in range(4):
            mask = np.zeros((180, 180), np.uint8)
            cv2.line(mask, (20, 10), (20, 40 + 10*i), 255, 3)
            for x, y in [(60, 30), (90, 60), (120, 90), (150, 120)]:
                cv2.circle(mask, (x + 2*i, y), 4, 255, -1)
            frames.append(mask)
        result = temporal_debris.filter_static_islands(frames, [(20, 10)])
        self.assertTrue(all(not np.any(m[:, 50:]) for m in result))
        self.assertTrue(all(np.any(m[:, :30]) for m in result))

    def test_slow_accumulated_extension_is_retained(self):
        frames = self.scene()
        for i, mask in enumerate(frames):
            cv2.line(mask, (90, 50), (90, 60+i), 255, 1)
        result = temporal_debris.filter_static_islands(frames, [(30, 10)])
        self.assertTrue(all(np.all(m[50:61, 90] == 255) for m in result))

    def test_common_root_extension_cannot_vote_as_camera_translation(self):
        frames = []
        for i in range(6):
            mask = np.zeros((100, 100), np.uint8)
            for x in (20, 50, 80):
                cv2.line(mask, (x, 30), (x, 40+i), 255, 3)
            frames.append(mask)
        result = temporal_debris.filter_static_islands(frames, [])
        for before, after in zip(frames, result):
            np.testing.assert_array_equal(before, after)

    def test_stationary_root_below_removed_green_stem_is_kept(self):
        mask = np.zeros((100, 100), np.uint8)
        cv2.line(mask, (30, 20), (30, 60), 255, 3)
        cv2.circle(mask, (65, 50), 5, 255, -1)
        result = temporal_debris.filter_static_islands([mask]*3, [(30, 10)])
        for actual in result:
            np.testing.assert_array_equal(actual[:, :45], mask[:, :45])
            self.assertFalse(np.any(actual[:, 45:]))

    def test_nearby_debris_cannot_steal_the_stationary_root_seed(self):
        mask = np.zeros((100, 100), np.uint8)
        cv2.line(mask, (30, 30), (30, 70), 255, 3)
        cv2.circle(mask, (39, 15), 2, 255, -1)
        result = temporal_debris.filter_static_islands([mask]*3, [(30, 10)])
        for actual in result:
            np.testing.assert_array_equal(actual[28:73, 28:33], mask[28:73, 28:33])

    def test_rt75_fading_impurity_is_removed_without_case_specific_settings(self):
        from pathlib import Path
        with np.load(Path(__file__).parent / 'fixtures/static_debris_rt75.npz') as fixture:
            masks = list(fixture['masks'])
        self.assertTrue(all(np.any(mask) for mask in masks))
        result = temporal_debris.filter_static_islands(masks, [])
        self.assertTrue(all(not np.any(mask) for mask in result))

"""Coarse registration preserves exact offsets and refuses uncertain evidence."""
import unittest

import cv2
import numpy as np

from root_tracker.config import Config
from root_tracker.registration.template_matcher import ImageRegistrator


class CoarseRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.registrator = ImageRegistrator(Config())
        self.rng = np.random.default_rng(821)

    def test_registration_updates_plate_coordinate_transform(self):
        from datetime import datetime
        from unittest.mock import patch
        from root_tracker.models import ImageData
        images = [ImageData(datetime(2026, 1, day), 'unused.jpg', 'test') for day in (1, 2)]
        for image in images:
            image.image = np.zeros((40, 50, 3), np.uint8)
            image.process = image.canny = np.zeros((40, 50), np.uint8)
            image.plate_transform = [1., 0., -20., 0., 1., -30.]
        target = images[1]
        with patch.object(self.registrator, 'align_to_template', return_value=(target.image, target.canny, target.process, 7, -4)):
            self.registrator.register_series(images)
        self.assertEqual(target.plate_transform, [1., 0., -13., 0., 1., -34.])
        self.assertEqual(images[0].plate_transform, [1., 0., -20., 0., 1., -30.])

    def edge_image(self, shape):
        result = np.zeros(shape, np.uint8)
        for _ in range(70):
            x, y = self.rng.integers([0, 0], [shape[1], shape[0]])
            dx, dy = self.rng.integers(-60, 61, size=2)
            cv2.line(result, (int(x), int(y)), (int(x+dx), int(y+dy)), 255, 1)
        return result

    def test_configured_border_matches_full_search(self):
        template = self.edge_image((203, 219))
        for shape in ((203, 219), (181, 241)):
            edges = cv2.warpAffine(template, np.float32([[1, 0, 17], [0, 1, -9]]),
                                   (shape[1], shape[0]))
            image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            for ratio in (0, .1, .25, .5):
                with self.subTest(shape=shape, ratio=ratio):
                    self.registrator.config.registration.margin_ratio = ratio
                    mx = max(int(shape[1] * ratio), (219-shape[1]+1)//2, 0)
                    my = max(int(shape[0] * ratio), (203-shape[0]+1)//2, 0)
                    padded = cv2.copyMakeBorder(edges, my, my, mx, mx, cv2.BORDER_CONSTANT)
                    left, top = cv2.minMaxLoc(cv2.matchTemplate(padded, template, cv2.TM_CCOEFF))[3]
                    actual = self.registrator.align_to_template(template, image, edges, edges)
                    self.assertEqual(actual[3:], (mx-left, my-top))
                    for output, source in zip(actual[:3], (image, edges, edges)):
                        reference = cv2.copyMakeBorder(source, my, my, mx, mx, cv2.BORDER_CONSTANT)
                        np.testing.assert_array_equal(output, reference[top:top+203, left:left+219])

    def test_large_translations_with_odd_and_different_shapes(self):
        template = self.edge_image((733, 819))
        for x, y in [(101, 97), (311, 223), (17, 151)]:
            target = np.zeros((1101, 1303), np.uint8)
            target[y:y+733, x:x+819] = template
            self.assertEqual(self.registrator._coarse_match_location(target, template), (x, y))
            self.assertEqual(self.registrator._match_location(target, template), (x, y))

    def test_blank_repeated_weak_and_boundary_evidence_falls_back(self):
        template = self.edge_image((401, 419))
        blank = np.zeros((1101, 1303), np.uint8)
        repeated = blank.copy()
        repeated[101:502, 101:520] = template
        repeated[601:1002, 801:1220] = template
        boundary = blank.copy()
        boundary[:401, :419] = template
        weak = self.edge_image(blank.shape)
        for target, reference in [(blank, template), (blank, np.zeros_like(template)),
                                  (repeated, template), (boundary, template), (weak, template)]:
            self.assertIsNone(self.registrator._coarse_match_location(target, reference))
            expected = cv2.minMaxLoc(cv2.matchTemplate(target, reference, cv2.TM_CCOEFF))[3]
            self.assertEqual(self.registrator._match_location(target, reference), expected)

    def test_sampling_phase_does_not_prefer_cell_aligned_decoy(self):
        rng = np.random.default_rng(42)
        template = np.zeros((320, 320), np.uint8)
        for y in range(0, 320, 4):
            for x in range(0, 320, 4):
                if rng.random() < .15:
                    template[y+rng.integers(4), x+rng.integers(4)] = 255
        decoy = template.copy()
        for y in range(0, 320, 4):
            for x in range(0, 320, 4):
                if rng.random() < .4:
                    decoy[y:y+4, x:x+4] = np.roll(decoy[y:y+4, x:x+4], 1, axis=1)
        target = np.zeros((1100, 1100), np.uint8)
        target[101:421, 101:421] = template
        target[600:920, 600:920] = decoy
        # Area reduction alone scores the aligned, damaged copy above the
        # exact match, whose edges straddle coarse cells after translation.
        self.assertEqual(self.registrator._match_location(target, template), (101, 101))

    def test_aligned_pixels_equal_padding_for_positive_and_negative_offsets(self):
        template = self.edge_image((703, 809))
        image = self.rng.integers(0, 256, (851, 999, 3), dtype=np.uint8)
        process = self.rng.integers(0, 256, (851, 999), dtype=np.uint8)
        for dx, dy in [(101, 73), (-59, -47)]:
            edges = cv2.warpAffine(template, np.float32([[1, 0, dx], [0, 1, dy]]), (999, 851))
            aligned = self.registrator.align_to_template(template, image, edges, process)
            mx, my = 999 // 4, 851 // 4
            padded = cv2.copyMakeBorder(edges, my, my, mx, mx, cv2.BORDER_CONSTANT)
            left, top = cv2.minMaxLoc(cv2.matchTemplate(padded, template, cv2.TM_CCOEFF))[3]
            self.assertEqual(aligned[3:], (mx-left, my-top))
            # Outputs must not retain much larger padded image allocations.
            self.assertIsNone(aligned[0].base)
            self.assertIsNone(aligned[2].base)
            for actual, source in zip(aligned[:3], (image, edges, process)):
                expected = cv2.copyMakeBorder(source, my, my, mx, mx, cv2.BORDER_CONSTANT)
                np.testing.assert_array_equal(actual, expected[top:top+703, left:left+809])


if __name__ == '__main__':
    unittest.main()

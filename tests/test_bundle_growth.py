"""A lateral must not freeze the growing shared portion of an observed bundle."""
import json
from pathlib import Path
import unittest
import numpy as np
from root_tracker.config import Config
from root_tracker.tracking.root_linker import RootLinker
from root_tracker.tracking.contact_evidence import ContactEvidence
from root_tracker.tracking.temporal_fragments import contour_path
from test_root_crossings import detected_segment


class BundleGrowthTests(unittest.TestCase):
    def test_partial_shared_tip_grows_before_the_roots_separate(self):
        previous = None
        with np.load(Path(__file__).parent / 'fixtures' / 'rt40_bundle_growth.npz') as data:
            cases = json.loads(str(data['metadata']))
            for i, case in enumerate(cases):
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
                if case['day'] == 27:
                    self.assertEqual(colored[(1183, 1466)], 3)
                    self.assertNotIn((1183, 1466), samples[2])
                if case['day'] == 29:
                    # This observed thick corridor is beyond yesterday's tip.
                    self.assertTrue((1184, 1829) in samples[2] & samples[3])
                    self.assertIn((1148, 2009), samples[3])
                if case['day'] == 30:
                    self.assertEqual(colored[(1142, 2100)], 2)
                    self.assertEqual(colored[(1174, 2223)], 3)


class BundleMeasurementTests(unittest.TestCase):
    def test_topological_bundle_retains_calibers_across_short_pieces(self):
        for scale in (1, 3):
            segments = [detected_segment((20, 0), (20, 40)),
                        detected_segment((26, 0), (26, 40)),
                        detected_segment((23, 44), (23, 50)),
                        detected_segment((23, 54), (23, 100))]
            uppers = [u for u, _ in segments]
            lowers = {l['point']: l for _, l in segments}
            for index, (upper, lower) in enumerate(segments):
                upper['width_profile'] = np.full(len(contour_path(upper)),
                                                (5. if index < 3 else 11.) * scale)
                for corner in (upper, lower):
                    corner['point'] = tuple(value * scale for value in corner['point'])
                upper['lower_point'] = lower['point']
                upper['contour'] *= scale
            lowers = {l['point']: l for _, l in segments}
            lowers[uppers[2]['lower_point']]['junction_id'] = 1
            contacts = ContactEvidence(uppers, lowers, {}, {('junction', 1): [3]},
                                       {u['lower_point']: i for i, u in enumerate(uppers)})
            routes = {pid: (uppers[pid]['lower_point'], 90.) for pid in (0, 1)}
            # Topology has already proved two identities in this short piece.
            contacts.remember(2, routes)
            bottom = uppers[2]['lower_point']
            with self.subTest(scale=scale):
                self.assertIn(bottom, contacts.bundles)
                self.assertNotEqual(contacts.sections[bottom][0][1],
                                    contacts.sections[bottom][1][1])
                extent = contacts.width_support(3, {pid: [(bottom, 90.)] for pid in routes}, 0)
                self.assertEqual(extent, (len(contour_path(uppers[3])), 0))

    def test_short_bundle_piece_observes_continuation_beyond_junction(self):
        segments = [detected_segment((20, 0), (20, 40)),
                    detected_segment((26, 0), (26, 40)),
                    detected_segment((23, 44), (23, 50)),
                    detected_segment((23, 54), (23, 100))]
        uppers = [u for u, _ in segments]
        lowers = {l['point']: l for _, l in segments}
        for index, upper in enumerate(uppers):
            upper['width_profile'] = np.full(len(contour_path(upper)), 5. if index < 2 else 11.)
        lowers[uppers[2]['lower_point']]['junction_id'] = 1
        contacts = ContactEvidence(uppers, lowers, {}, {('junction', 1): [3]},
                                   {u['lower_point']: i for i, u in enumerate(uppers)})
        routes = {pid: [(uppers[pid]['lower_point'], 90.)] for pid in (0, 1)}
        contacts.bundles.update(uppers[pid]['lower_point'] for pid in routes)
        self.assertEqual(contacts.width_support(2, routes, 0),
                         (len(contour_path(uppers[2])), 0))
        # A single-caliber continuation still provides no sharing evidence.
        uppers[2]['width_profile'][:] = 5.
        uppers[3]['width_profile'][:] = 5.
        self.assertIsNone(contacts.width_support(2, routes, 0))

    def test_contraction_in_lookahead_keeps_current_short_piece_shared(self):
        segments = [detected_segment((20, 0), (20, 40)),
                    detected_segment((26, 0), (26, 40)),
                    detected_segment((23, 44), (23, 50)),
                    detected_segment((23, 54), (23, 150))]
        uppers = [u for u, _ in segments]
        lowers = {l['point']: l for _, l in segments}
        for index, upper in enumerate(uppers):
            upper['width_profile'] = np.full(len(contour_path(upper)), 5. if index < 2 else 11.)
        uppers[3]['width_profile'][30:] = 5.
        lowers[uppers[2]['lower_point']]['junction_id'] = 1
        contacts = ContactEvidence(uppers, lowers, {}, {('junction', 1): [3]},
                                   {u['lower_point']: i for i, u in enumerate(uppers)})
        routes = {pid: [(uppers[pid]['lower_point'], 90.)] for pid in (0, 1)}
        contacts.bundles.update(uppers[pid]['lower_point'] for pid in routes)
        self.assertEqual(contacts.width_support(2, routes, 0),
                         (len(contour_path(uppers[2])), 0))

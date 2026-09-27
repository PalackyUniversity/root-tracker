"""New growth can cross an established root and leave by its own exit."""
import json
from pathlib import Path
import unittest

import numpy as np

from root_tracker.config import Config
from root_tracker.tracking.contact_evidence import ContactEvidence
from root_tracker.tracking.root_linker import RootLinker
from root_tracker.tracking.temporal_fragments import contour_path
from test_root_crossings import detected_segment


class NewCrossingExitTests(unittest.TestCase):
    def test_new_exits_keep_their_arriving_identity_through_all_frames(self):
        previous = {}
        expected = {
            ('56', 28): {(1047, 1476): 1, (1051, 1703): 2},
            ('56', 29): {(1070, 1695): 1, (1050, 1984): 2},
            ('56', 30): {(1090, 1914): 1, (986, 1461): 2},
            ('69', 29): {(311, 899): 1, (410, 880): 0},
            ('69', 30): {(301, 925): 1, (250, 1038): 1, (409, 878): 0},
        }
        with np.load(Path(__file__).parent / 'fixtures/new_crossing_exits.npz') as archive:
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
                case = restore(frame['inputs'])
                group = frame['group']
                _, colored, samples = RootLinker(Config()).link_corners(
                    case['upper'], case['lower'], case['x'], case['y'], previous.get(group))
                previous[group] = samples
                for point, owner in expected.get((group, frame['day']), {}).items():
                    with self.subTest(group=group, day=frame['day'], point=point):
                        self.assertEqual(colored.get(point), owner)
                        self.assertNotIn(point, samples[1 - owner] if group == '69' else samples[3 - owner])

    def contact(self, length=10, scale=1, split=False, exit_owner=None):
        segments = [detected_segment((20, 0), (39, 20)),
                    detected_segment((60, 0), (40, 20)),
                    detected_segment((40, 20), (40, 20 + length)),
                    detected_segment((40, 20 + length), (60, 40 + length)),
                    detected_segment((40, 20 + length), (20, 40 + length))]
        # Distinct physical entry endpoints belong to one junction.
        for upper, lower in segments:
            upper['width_profile'] = np.full(len(contour_path(upper)), 5. * scale)
            for corner in (upper, lower):
                corner['point'] = tuple(value * scale for value in corner['point'])
            upper['lower_point'] = lower['point']
            upper['contour'] *= scale
        uppers = [u for u, _ in segments]
        lowers = {l['point']: l for _, l in segments}
        end = uppers[2]['lower_point']
        lowers[end]['junction_id'] = 1
        tasks = {('junction', 1): [3, 4]}
        if split:
            first, second = detected_segment((40 * scale, 20 * scale),
                                              (40 * scale, (20 + length // 2) * scale))
            second['junction_id'] = 2
            first['width_profile'] = np.full(len(contour_path(first)), 5. * scale)
            old_path = contour_path(uppers[2]);uppers[2]['contour'] = old_path[len(old_path)//2:, None, :]
            uppers[2]['point'] = tuple(uppers[2]['contour'][0, 0])
            uppers.append(first);lowers[second['point']] = second
            tasks[('junction', 2)] = [2]
        history = {2: (0, 1.), 3: (0, 1.)}
        if split: history[5] = (0, 1.)
        if exit_owner is not None: history[4] = (exit_owner, 1.)
        contact = ContactEvidence(uppers, lowers, {}, tasks,
                                  {u['lower_point']: i for i, u in enumerate(uppers)},
                                  historical_owner=history.get)
        options = {0: [(uppers[0]['lower_point'], 45.)],
                   1: [(uppers[1]['lower_point'], 135.)]}
        return contact, options, 5 if split else 2

    def test_short_crossing_is_scale_and_segment_split_invariant(self):
        for scale in (1, 3):
            for split in (False, True):
                contact, options, index = self.contact(scale=scale, split=split)
                with self.subTest(scale=scale, split=split):
                    self.assertEqual(contact.independent_exits(index, options), {0, 1})

    def test_long_corridor_or_preexisting_other_owner_is_not_new_crossing(self):
        for kwargs in ({'length': 100}, {'exit_owner': 0}):
            contact, options, index = self.contact(**kwargs)
            self.assertEqual(contact.independent_exits(index, options), {0})

    def test_missing_widths_or_parallel_entries_do_not_prove_new_crossing(self):
        contact, options, index = self.contact()
        options[1] = [(options[1][0][0], 45.)]
        self.assertNotIn(1, contact.independent_exits(index, options))
        contact, options, index = self.contact()
        contact.uppers[0].pop('width_profile')
        self.assertEqual(contact.independent_exits(index, options), {0})

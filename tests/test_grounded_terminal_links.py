"""A small terminal arm cannot import a remote owner into a rooted junction."""
import json
from pathlib import Path
import unittest

import numpy as np

from root_tracker.config import Config
from root_tracker.tracking.root_linker import RootLinker


class GroundedTerminalLinkTests(unittest.TestCase):
    def test_recorded_rt61_terminal_does_not_import_plant_five(self):
        with np.load(Path(__file__).parent / 'fixtures/rt61_grounded_terminal.npz') as data:
            case = json.loads(data['metadata'].tobytes())
            for i, upper in enumerate(case['upper']):
                for key in ('point', 'lower_point'):
                    upper[key] = tuple(upper[key])
                for key in ('contour', 'width_profile'):
                    if f'{key}_{i}' in data:
                        upper[key] = data[f'{key}_{i}'].copy()
                for terminal in upper.get('terminal_exits', []):
                    for key in ('point', 'lower_point'):
                        terminal[key] = tuple(terminal[key])
            for lower in case['lower']:
                lower['point'] = tuple(lower['point'])
            previous = {pid: set(map(tuple, data[f'previous_{pid}'])) for pid in range(6)}
        pairs, colored, samples = RootLinker(Config(n_clusters=6)).link_corners(
            case['upper'], case['lower'], case['x'], case['y'], previous)
        self.assertNotIn(((1633, 1005), (1810, 996)), pairs)
        self.assertEqual(colored[(1636, 1073)], 3)
        self.assertNotIn((1640, 1030), samples[4])

    def test_unequal_height_junction_rejects_remote_entry_to_short_arm(self):
        for scale in (1, 3):
            with self.subTest(scale=scale):
                pairs, colored, _ = self.synthetic(scale=scale)
                self.assertNotIn(((90*scale, 80*scale), (130*scale, 75*scale)), pairs)
                self.assertEqual(colored[(100*scale, 130*scale)], 0)

    def test_observed_independent_arrival_can_cross_a_short_gap(self):
        pairs, _, _ = self.synthetic(long_arrival=True)
        self.assertIn(((120, 80), (130, 75)), pairs)

    def test_same_plant_terminal_keeps_ownership_through_its_physical_junction(self):
        pairs, colored, samples = self.synthetic(same_owner=True)
        # Only the trunks have history here, not the remote gap or tiny arm.
        # Same identity does not override its observed physical attachment.
        self.assertNotIn(((90, 80), (130, 75)), pairs)
        self.assertIn(((98, 84), (100, 82)), pairs)
        self.assertIn((90, 80), samples[0])
        self.assertNotIn((90, 80), samples[1])
        self.assertEqual(colored[(100, 130)], 0)

    def test_established_short_arrival_keeps_its_identity(self):
        pairs, colored, _ = self.synthetic(established=True)
        self.assertIn(((90, 80), (130, 75)), pairs)
        self.assertEqual(colored[(98, 84)], 1)

    @staticmethod
    def synthetic(scale=1, long_arrival=False, same_owner=False, established=False):
        def segment(start, end, component, angle, top_junction=None, bottom_junction=None):
            count = max(abs(end[0]-start[0]), abs(end[1]-start[1])) + 1
            path = np.rint(np.linspace(start, end, count)).astype(int)*scale
            upper = dict(point=tuple(path[0]), lower_point=tuple(path[-1]), angle=angle,
                         component_id=component, component_area=200*scale**2,
                         component_extent=150*scale,
                         contour=np.concatenate((path, path[-2:0:-1]))[:, None])
            lower = dict(point=tuple(path[-1]), angle=(angle+180)%360)
            if top_junction:
                upper['junction_id'] = top_junction
            if bottom_junction:
                lower['junction_id'] = bottom_junction
            return upper, lower
        segments = [segment((100, 15), (100, 82), 1, 270, bottom_junction=1),
                    segment((140, 15), (130, 75), 2, 280),
                    segment((120, 80) if long_arrival else (90, 80), (98, 84), 1,
                            350 if long_arrival else 210, bottom_junction=1),
                    segment((100, 86), (100, 130), 1, 270, top_junction=1)]
        origins = [{'point': (x*scale, -20*scale), 'angle': 90.} for x in (100, 220)]
        history = {0: {(100*scale, y*scale) for y in range(15, 83)},
                   1: set(map(tuple, segments[1][0]['contour'][:, 0]))}
        if same_owner:
            history[0].update(history[1])
            history[1].clear()
        if established:
            history[1].update(map(tuple, segments[2][0]['contour'][:, 0]))
        return RootLinker(Config(n_clusters=2)).link_corners(
            [u for u, _ in segments], origins+[l for _, l in segments],
            [100*scale, 220*scale], [-20*scale, -20*scale], history)

    def test_horizontal_reversal_can_create_a_new_lower_junction(self):
        config=Config(n_clusters=1)
        config.registration.enabled=False
        path=np.array([(x,10) for x in range(20,9,-1)],dtype=np.int32)
        upper=dict(point=(20,10),lower_point=(10,10),junction_id=1,angle=180.,
                   contour=path[:,None])
        lower=[dict(point=(10,0),angle=90.),dict(point=(10,10),angle=0.)]
        pairs,_,_=RootLinker(config).link_corners([upper],lower,[10],[0])
        self.assertIsInstance(pairs,list)

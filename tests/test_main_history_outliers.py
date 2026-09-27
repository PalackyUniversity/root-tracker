"""Current main evidence must outweigh stale unmatched historical geometry."""
import json
from pathlib import Path
import unittest
import numpy as np
from root_tracker.tracking.main_root import select_main_geometry
from test_main_route_motion import segment


class MainHistoryOutlierTests(unittest.TestCase):
    def test_unmatched_old_samples_do_not_pull_main_onto_another_branch(self):
        for scale in (1, 3):
            trunk = segment((0, 0), (0, 40), 0, None)
            main = segment((0, 41), (0, 80), 1, (0, 40))
            branch = segment((1, 41), (12, 85), 2, (0, 40))
            upper = [trunk, main, branch]
            history = {(0, y) for y in range(81)} | {(60, y) for y in range(85, 105)}
            for u in upper:
                for key in ('point', 'lower_point'):
                    u[key] = tuple(scale * v for v in u[key])
                u['contour'] *= scale
                u['width_profile'] *= scale
                parent = u['parent_points'][0]
                u['parent_points'][0] = tuple(scale * v for v in parent) if parent else None
            history = {(scale*x, scale*y) for x, y in history}
            tip, geometry = select_main_geometry(upper, {u['lower_point']: 0 for u in upper},
                                                 [], 0, history, previous_roots={(0, scale*y) for y in range(81)})
            with self.subTest(scale=scale):
                self.assertEqual(tip, (0, 80*scale))
                self.assertEqual(set(geometry), {0, 1})

    def test_moved_terminal_main_survives_sparse_old_lateral_contact(self):
        trunk = segment((20, 0), (20, 40), 0, None)
        main = segment((28, 41), (28, 120), 1, (20, 40))
        lateral = segment((27, 41), (20, 60), 2, (20, 40))
        anchor = segment((60, 0), (60, 130), 3, None)
        upper = [trunk, main, lateral, anchor]
        history = {(20, y) for y in range(101)}
        previous = history | {(28, y) for y in range(70, 81)} | {(60, y) for y in range(131)}
        tip, geometry = select_main_geometry(upper, {u['lower_point']: 0 for u in upper},
                                             [], 0, history, previous_roots=previous)
        self.assertEqual(tip, (28, 120))
        self.assertEqual(set(geometry), {0, 1})

    def test_rt23_04_10_keeps_plants_two_and_three_on_growing_main(self):
        histories = {}; previous = None
        with np.load(Path(__file__).parent/'fixtures/rt23_main_identity.npz') as data:
            for frame in json.loads(str(data['metadata'])):
                upper = []; colored = {}; roots = {p: set() for p in range(6)}
                for raw in frame['upper']:
                    u = {k: raw[k] for k in ('contour_index', 'junction_id', 'component_id') if k in raw}
                    u.update(point=tuple(raw['point']), lower_point=tuple(raw['lower_point']),
                             contour=data[raw['key']], width_profile=data[raw['key']+'w'],
                             parent_points={int(k): tuple(v) if v else None for k,v in raw.get('parent_points', {}).items()})
                    upper.append(u); colored[u['lower_point']] = raw['owner']
                    owners = raw['owner'] if isinstance(raw['owner'], list) else [raw['owner']]
                    for owner in owners:
                        if owner in roots: roots[owner].update(map(tuple, u['contour'][:,0]))
                lower = [dict(l, point=tuple(l['point'])) for l in frame['lower']]
                pairs = [(tuple(a), tuple(b)) for a,b in frame['pairs']]
                for plant in range(6):
                    tip, geometry = select_main_geometry(upper, colored, pairs, plant,
                        histories.get(plant, set()), lower, None if previous is None else previous[plant])
                    selected = {tuple(p) for pieces in geometry.values() for c in pieces for p in c[:,0]}
                    histories.setdefault(plant, set()).update(selected)
                    growth_tips = {6: {1: (519, 752), 2: (1059, 1000)},
                                   7: {1: (555, 943), 2: (1035, 1243)},
                                   8: {1: (580, 1163), 2: (1026, 1507)}}
                    if frame['day'] in growth_tips and plant in (1, 2):
                        with self.subTest(day=frame['day'], plant=plant+1):
                            self.assertEqual(tip, growth_tips[frame['day']][plant])
                    if frame['day'] == 9:
                        with self.subTest(plant=plant+1):
                            expected = {0: (1, 675), 1: (560, 1403), 2: (1048, 1789),
                                        3: (1371, 1868), 4: (1937, 2426), 5: (2465, 2312)}[plant]
                            self.assertEqual(tip, expected)
                            self.assertIn(expected, selected)
                            corridor = {
                                1: {(495, 450), (502, 600), (526, 800), (583, 1100), (569, 1300)},
                                2: {(989, 450), (1004, 600), (1057, 800), (1072, 1100), (1032, 1300)},
                            }.get(plant, set())
                            self.assertFalse(corridor - selected)
                previous = roots

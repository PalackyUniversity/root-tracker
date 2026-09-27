"""Main identity applies to geometry, not to an entire newly merged segment."""
import unittest
import numpy as np
from root_tracker.tracking.main_root import select_main_geometry


def line(a, b):
    n = max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1
    return np.rint(np.linspace(a,b,n)).astype(np.int32)


def node(points, index=0, parent=(20,0)):
    return dict(point=tuple(points[0]), lower_point=tuple(points[-1]),
                contour=points.reshape(-1,1,2), contour_index=index,
                width_profile=np.full(len(points),4.), parent_points={0:parent})


def pixels(geometry):
    return {tuple(p) for contours in geometry.values() for c in contours for p in c[:,0]}


class MainRoleFragmentsTests(unittest.TestCase):
    def test_merged_lateral_prefix_does_not_become_main(self):
        lateral=line((0,60),(20,20));main=line((20,20),(20,100))
        upper=node(np.concatenate((lateral[:-1],main)))
        old_main=set(map(tuple,line((20,20),(20,80))))
        roots=old_main|set(map(tuple,lateral))
        tip, geometry=select_main_geometry([upper],{upper['lower_point']:0},[],0,
                                           old_main,previous_roots=roots)
        selected=pixels(geometry)
        self.assertNotIn((0,60),selected)
        self.assertNotIn((10,40),selected)
        self.assertIn((20,70),selected)
        self.assertIn((20,95),selected)  # Genuine new tip growth remains main.
        self.assertEqual(tip,(20,100))

    def test_known_lateral_cannot_bridge_main_to_new_unobserved_growth(self):
        trunk=node(line((20,0),(20,40)),0,(20,-1))
        lateral=node(line((21,41),(60,80)),1,(20,40))
        new=node(line((60,81),(60,120)),2,(60,80))
        upper=[trunk,lateral,new]
        history=set(map(tuple,trunk['contour'][:,0]))
        roots=history|set(map(tuple,lateral['contour'][:,0]))
        tip,geometry=select_main_geometry(upper,{u['lower_point']:0 for u in upper},
                                         [],0,history,previous_roots=roots)
        self.assertEqual(tip,(20,40))
        self.assertNotIn((60,110),pixels(geometry))

    def test_clipped_curves_do_not_fill_background(self):
        import cv2
        lateral=line((0,60),(20,20))
        main=np.concatenate((line((20,20),(40,50))[:-1],line((40,50),(20,80))))
        upper=node(np.concatenate((lateral[:-1],main)))
        history=set(map(tuple,main))
        _, geometry=select_main_geometry([upper],{upper['lower_point']:0},[],0,
                                         history,previous_roots=history|set(map(tuple,lateral)))
        mask=np.zeros((100,100),np.uint8)
        cv2.drawContours(mask,[c for cs in geometry.values() for c in cs],-1,1,cv2.FILLED)
        self.assertTrue(set(zip(*np.nonzero(mask)[::-1]))<=set(map(tuple,upper['contour'][:,0])))

    def test_shared_rendering_keeps_partial_main_role_local(self):
        from root_tracker.tracking.shared_rendering import draw_shared_segments
        from root_tracker.tracking.root_linker import RootLinker
        from root_tracker.config import Config
        upper=node(line((70,20),(70,110)))
        image=np.zeros((140,150,3),np.uint8)
        draw_shared_segments(image,[upper],{(70,110):(0,1)},{0:{0}},
                             RootLinker(Config()),
                             main_samples={0:{(70,y) for y in range(20,61)},1:set()})
        self.assertEqual(tuple(image[40,67]),(255,170,170))
        self.assertEqual(tuple(image[90,68]),(255,0,0))
        self.assertEqual(tuple(image[90,72]),(0,255,0))
        self.assertEqual(tuple(image[90,65]),(0,0,0))

    def test_new_unobserved_basal_gap_parent_does_not_extend_main_backwards(self):
        leaf=node(line((0,0),(20,20)),0,(20,-1))
        main=node(line((20,40),(20,100)),1,(20,20))
        history=set(map(tuple,line((20,40),(20,80))))
        tip,geometry=select_main_geometry([leaf,main],{(20,20):0,(20,100):0},[],0,
                                         history,previous_roots=history)
        self.assertNotIn((10,10),pixels(geometry))
        self.assertIn((20,95),pixels(geometry))

    def test_recorded_rt70_keeps_lateral_and_leaf_out_of_main(self):
        import json
        from pathlib import Path
        histories={};previous_roots={}
        with np.load(Path(__file__).parent/'fixtures/rt70_main_roles.npz') as data:
            for frame in json.loads(str(data['metadata'])):
                group=frame['group']
                history=histories.setdefault(group,set())
                previous=previous_roots.get(group)
                upper=[]
                for raw in frame['upper']:
                    u={k:raw[k] for k in ('contour_index','junction_id','component_id')}
                    u.update(point=tuple(raw['point']),lower_point=tuple(raw['lower_point']),
                             contour=data[raw['key']],width_profile=data[raw['key']+'w'],
                             parent_points={0:tuple(raw['parent']) if raw['parent'] is not None else None})
                    upper.append(u)
                lower=[dict(point=tuple(l['point']),junction_id=l['junction_id']) for l in frame['lower']]
                pairs=[(tuple(a),tuple(b)) for a,b in frame['pairs']]
                _,geometry=select_main_geometry(upper,{u['lower_point']:0 for u in upper},
                                                pairs,0,history,lower,previous)
                selected=pixels(geometry)
                previous={tuple(p) for u in upper for p in u['contour'][:,0]}
                self.assertTrue(selected<=previous)
                history.update(selected)
                previous_roots[group]=previous
                if group != 70:
                    self.assertIn(tuple(frame['probe']),selected)
                    continue
                if frame['day']==29:
                    self.assertNotIn((1513,311),selected)
                    self.assertIn((1471,393),selected)
                if frame['day']==30:
                    self.assertNotIn((1372,484),selected)
                    self.assertIn((1491,514),selected)

    def test_missing_old_main_role_inside_continuation_does_not_stop_growth(self):
        main=node(line((20,20),(20,120)))
        history=set(map(tuple,line((20,20),(20,60))))
        # A prior disconnected observation existed beyond the labelled main.
        roots=history|set(map(tuple,line((20,61),(20,90))))
        tip,geometry=select_main_geometry([main],{main['lower_point']:0},[],0,
                                         history,previous_roots=roots)
        self.assertEqual(tip,(20,120))
        self.assertIn((20,110),pixels(geometry))

    def test_shifted_basal_history_cannot_drop_a_long_existing_main_ancestor(self):
        basal=node(line((20,10),(20,80)),0,(20,0))
        distal=node(line((20,90),(20,130)),1,(20,80))
        history={(21,y) for y in range(10,81)}|{(20,y) for y in range(90,111)}
        _,geometry=select_main_geometry([basal,distal],{(20,80):0,(20,130):0},[],0,
                                        history,previous_roots=history)
        self.assertIn((20,40),pixels(geometry))

    def test_rejoining_lateral_does_not_gain_main_role_from_basal_ancestor(self):
        trunk=node(line((20,0),(20,20)),0,(20,-1))
        lateral=line((21,21),(40,60));distal=line((40,60),(40,100))
        merged=node(np.concatenate((lateral[:-1],distal)),1,(20,20))
        history=set(map(tuple,trunk['contour'][:,0]))|set(map(tuple,distal))
        _,geometry=select_main_geometry([trunk,merged],{(20,20):0,(40,100):0},[],0,
                                        history,previous_roots=history|set(map(tuple,lateral)))
        selected=pixels(geometry)
        self.assertNotIn((28,36),selected)
        self.assertIn((20,10),selected)
        self.assertIn((40,90),selected)

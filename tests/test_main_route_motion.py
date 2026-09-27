"""Main identity follows a whole established route under local deformation."""
import unittest
import numpy as np
from root_tracker.tracking.main_root import select_main_geometry


def segment(start,end,index,parent):
    n=max(abs(end[0]-start[0]),abs(end[1]-start[1]))+1
    p=np.rint(np.linspace(start,end,n)).astype(np.int32)
    return dict(point=start,lower_point=end,contour_index=index,
                contour=p[:,None],width_profile=np.full(n,4.),parent_points={0:parent})


class MainRouteMotionTests(unittest.TestCase):
    def test_lateral_near_old_centerline_does_not_beat_shifted_whole_main(self):
        trunk=segment((20,0),(20,40),0,(20,-1))
        main=segment((28,41),(28,120),1,(20,40))
        lateral=segment((27,41),(20,60),2,(20,40))
        history={(20,y) for y in range(101)}
        tip,geometry=select_main_geometry([trunk,main,lateral],
            {u['lower_point']:0 for u in (trunk,main,lateral)},[],0,history,
            previous_roots=history)
        self.assertEqual(tip,(28,120))
        self.assertIn(1,geometry)
        self.assertNotIn(2,geometry)

    def test_recorded_rt45_main_stays_continuous_through_local_motion(self):
        import json
        from pathlib import Path
        history=set();previous=None
        with np.load(Path(__file__).parent/'fixtures/main_route_motion_rt45.npz') as data:
            for frame in json.loads(str(data['metadata'])):
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
                selected={tuple(p) for pieces in geometry.values() for c in pieces for p in c[:,0]}
                observed={tuple(p) for u in upper for p in u['contour'][:,0]}
                # Check the complete distal main, excluding the short lateral
                # that appears to its left on day30. A tip alone misses gaps.
                lateral={tuple(p) for u in upper if frame['day']==30
                         and u['point']==(72,1254) for p in u['contour'][:,0]}
                distal={p for p in observed-lateral if p[1]>1000}
                bridge_start={28:(178,480),29:(179,479),30:(178,479)}.get(frame['day'])
                bridge={tuple(p) for u in upper if u['point']==bridge_start
                        for p in u['contour'][:,0]}
                with self.subTest(day=frame['day']):
                    self.assertTrue(distal)
                    self.assertFalse(distal-selected)
                    self.assertFalse(bridge-selected)
                    self.assertFalse(lateral & selected)
                    self.assertTrue(selected<=observed)
                history.update(selected);previous=observed

    def test_rejoining_lateral_uses_same_route_metric_for_ancestry_and_tip(self):
        def bent(points,index,parent):
            parts=[segment(a,b,index,parent)['contour'][:,0]
                   for a,b in zip(points[:-1],points[1:])]
            path=np.concatenate([p[:-1] for p in parts]+[parts[-1][-1:]])
            u=segment(points[0],points[-1],index,parent)
            u.update(contour=path[:,None],width_profile=np.full(len(path),4.))
            return u
        trunk=segment((20,0),(20,40),0,(20,-1))
        main=bent([(28,41),(28,115),(20,120)],1,(20,40))
        lateral=bent([(20,41),(20,60),(60,80),(23,120)],2,(20,40))
        child=segment((21,122),(21,160),3,(20,120));child['junction_id']=1
        upper=[trunk,main,lateral,child]
        lower=[dict(point=(20,120),junction_id=1),dict(point=(23,120),junction_id=1)]
        history={(20,y) for y in range(101)}
        tip,geometry=select_main_geometry(upper,{u['lower_point']:0 for u in upper},
                                         [],0,history,lower,history)
        self.assertEqual(tip,(21,160))
        self.assertEqual(set(geometry),{0,1,3})

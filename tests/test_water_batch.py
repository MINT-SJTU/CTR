import numpy as np
import pytest
from real_robot_data_retime.timeline.water_batch import batch_schedule
from real_robot_data_retime.timeline.water import ORDERS
from real_robot_data_retime.compositing import layers
from real_robot_data_retime.compositing.interpolation import FlowFrames

@pytest.mark.parametrize('start',ORDERS)
@pytest.mark.parametrize('ret',ORDERS)
def test_initial_motion_and_terminal_supervision(start,ret):
 s=batch_schedule(663,270,555,start,ret,8);c=s['clocks']
 np.testing.assert_array_equal(c[:8],np.repeat(np.arange(8)[:,None],2,axis=1))
 assert (s['active'][:8]==1).all() and (s['active'][-45:]==1).all()
 assert ((np.diff(c,axis=0)>=0)&(np.diff(c,axis=0)<=1)).all()
 np.testing.assert_array_equal(c[s['shared'],0],c[s['shared'],1])

def test_exposure_and_background_converge_at_pairing(tmp_path,monkeypatch):
 frames=np.stack([np.full((80,80,3),v,np.uint8) for v in [160,180,200,220]])
 robots=np.zeros((4,2,80,80),bool)
 seg=dict(robots=np.packbits(robots,axis=-1),objects=np.packbits(np.zeros((0,4,80,80),bool),axis=-1))
 output=[];monkeypatch.setattr(layers,'write_video',lambda path,frames,fps:output.extend(frames))
 layers.composite(frames,dict(task='water',fps=30,episodes=[]),seg,[0,.0625,2.9375,3],[0,0,3,3],tmp_path/'o.mp4',tmp_path)
 assert np.abs(output[1].astype(float)-output[0]).mean()<2
 reference,_=FlowFrames(frames).sample(2.9375,[np.ones((80,80),bool)]*2)
 # Compare with the expected source exposure, allowing one uint8 rounding unit.
 assert np.abs(output[-2].astype(float)-reference).max()<=1

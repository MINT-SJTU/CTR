import numpy as np
from real_robot_data_retime.compositing import layers
from real_robot_data_retime.compositing.interpolation import FlowFrames

def test_disappeared_water_arm_does_not_leave_initial_border_anchor(tmp_path,monkeypatch):
 frames=np.full((4,80,80,3),180,np.uint8)
 robots=np.zeros((4,2,80,80),bool)
 robots[0,0,45:65,:6]=True;robots[1,0,45:65,20:26]=True
 for k in range(4):frames[k,robots[k].any(axis=0)]=20
 seg=dict(robots=np.packbits(robots,axis=-1),objects=np.packbits(np.zeros((0,4,80,80),bool),axis=-1))
 output=[]
 monkeypatch.setattr(layers,'write_video',lambda path,frames,fps:output.extend(frames))
 layers.composite(frames,dict(task='water',fps=30,episodes=[]),seg,[1],[0],tmp_path/'out.mp4',tmp_path)
 assert output[0][50:60,:4].mean()>150
 assert output[0][50:60,21:25].mean()<50

def test_paired_fractional_water_clock_preserves_full_frame(tmp_path,monkeypatch):
 frames=np.full((4,80,80,3),180,np.uint8)
 for k in range(4):frames[k,10:20,15+k:25+k]=40
 masks=np.zeros((4,2,80,80),bool)
 seg=dict(robots=np.packbits(masks,axis=-1),objects=np.packbits(np.zeros((0,4,80,80),bool),axis=-1))
 output=[];monkeypatch.setattr(layers,'write_video',lambda path,frames,fps:output.extend(frames))
 layers.composite(frames,dict(task='water',fps=30,episodes=[]),seg,[.5],[.5],tmp_path/'out.mp4',tmp_path)
 expected,_=FlowFrames(frames).sample(.5,[np.ones((80,80),bool)]*2)
 np.testing.assert_array_equal(output[0],expected)

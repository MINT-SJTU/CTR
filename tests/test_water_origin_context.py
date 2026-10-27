import numpy as np
import pytest
from real_robot_data_retime.compositing import layers

def scene():
 f=np.full((5,80,80,3),180,np.uint8);r=np.zeros((5,2,80,80),bool);o=np.zeros((1,5,80,80),bool)
 for k,y in enumerate([40,40,10,40,10]):o[0,k,y:y+10,12:22]=True;f[k,o[0,k]]=[40,90,40]
 r[4,0,30:60,2:32]=True;f[4,r[4,0]]=20
 event=dict(object_id=0,robot_id='left',pickup_frame=1,release_frame=5,grasp_frame=1,approach_start=0)
 return f,r,o,dict(task='water',fps=30,episodes=[event])

def test_origin_does_not_accumulate_bottle_from_other_source_times(tmp_path,monkeypatch):
 f,r,o,t=scene();out=[];monkeypatch.setattr(layers,'write_video',lambda path,frames,fps:out.extend(frames))
 layers.composite(f,t,dict(robots=np.packbits(r,axis=-1),objects=np.packbits(o,axis=-1)),[2],[0],tmp_path/'out.mp4',tmp_path)
 assert out[0][43:47,15:19].mean()>150,'Vacated original bottle location must not contain the bottle from frame3'

def test_context_crossed_by_other_arm_is_rejected(tmp_path,monkeypatch):
 f,r,o,t=scene();r[1,1,42:45,15:18]=True
 with pytest.raises(ValueError,match='crossed by the other arm'):
  layers.composite(f,t,dict(robots=np.packbits(r,axis=-1),objects=np.packbits(o,axis=-1)),[2],[0],tmp_path/'out.mp4',tmp_path)

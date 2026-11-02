"""Recover a cube identity from observed deposition inside an opened drawer."""
import cv2
import numpy as np
from .video import components
from .neural_tracks import mask_measurements
from ..tasks.drawer_constraints import discover_drawer_interior


def deposition_observations(frames, interior, proposal, robots, first, stop, fps, *, after):
    hue = float(proposal['color'][0])
    def chromatic(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        delta = np.abs(hsv[:,:,0].astype(float)-hue)
        return (np.minimum(delta,180-delta)<12)&(hsv[:,:,1]>35)&(hsv[:,:,2]>40)
    already_present = chromatic(frames[first])
    observations=[]
    for t in range(max(first+1,after),stop):
        for region,stat,center in components(chromatic(frames[t])&interior,max(10,round(proposal['area']*.15))):
            if stat[4]>proposal['area']*4 or already_present[region].mean()>.3:
                continue
            if robots[t].any(axis=0)[region].mean()>.2:
                continue
            if np.linalg.norm(center-proposal['origin'])<frames.shape[2]*.04:
                continue
            observations.append(dict(frame=t,center=center,mask=region,area=int(stat[4])))
    runs=[];current=[]
    for ob in sorted(observations,key=lambda x:x['frame']):
        if current and (ob['frame']!=current[-1]['frame']+1 or np.linalg.norm(ob['center']-current[-1]['center'])>max(2,frames.shape[2]*.008)):
            runs.append(current);current=[]
        current.append(ob)
    if current:runs.append(current)
    supported=[x for x in runs if len(x)>=max(3,round(fps*.2))]
    if not supported:raise ValueError('no persistent, robot-detached new object observed in the open drawer')
    sequence=max(supported,key=len)
    seed=max(sequence,key=lambda x:x['area'])
    return seed,dict(visible_deposition_frames=[x['frame'] for x in sequence],reference_frame=first,seed_frame=seed['frame'],method='new chromatic object inside observed drawer; static colors and robot overlap excluded')


def recover_deposited_cube(frames, proposals, tracks, grippers, event, motion, sam, fps):
    object_id=event['object_id'];proposal=proposals[object_id]
    robots=np.unpackbits(grippers['robot_masks'],axis=-1,count=frames.shape[2]).astype(bool)
    first,stop=motion['open_frame'],motion['close_start']
    interior=discover_drawer_interior(frames[first],sam=sam)
    seed,audit=deposition_observations(frames,interior,proposal,robots,first,stop,fps,after=event['pickup_frame'])
    yy,xx=np.where(seed['mask']);x0,y0=int(xx.min()),int(yy.min());bw,bh=int(xx.max()-x0+1),int(yy.max()-y0+1)
    distance=cv2.distanceTransform(seed['mask'].astype('uint8'),cv2.DIST_L2,5);py,px=np.unravel_index(distance.argmax(),distance.shape)
    prompt=dict(bbox=[x0,y0,bw,bh],mask=seed['mask'],positive_points=[[float(px),float(py)]])
    track=tracks[object_id];pickup=event['pickup_frame']
    centers=track['centers'].copy();areas=track['areas'].copy();packed=track['packed_masks'].copy()
    centers[pickup:]=np.nan;areas[pickup:]=0;packed[pickup:]=0
    for reverse,end in [(True,pickup-1),(False,stop)]:
        for t,masks in sam.propagate(frames,[prompt],seed_frame=seed['frame'],reverse=reverse,stop_frame=end):
            center,area=mask_measurements(masks[0])
            if area>proposal['area']*4 or area<10:continue
            if t>=seed['frame'] and not interior[np.rint(center[1]).astype(int),np.rint(center[0]).astype(int)]:continue
            centers[t]=center;areas[t]=area;packed[t]=np.packbits(masks[0],axis=-1)
    track.update(centers=centers,areas=areas,packed_masks=packed,confidence=np.minimum(areas/max(1,proposal['area']),1.0))
    return audit

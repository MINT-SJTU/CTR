"""Conservative batch phase clocks. Initial motion and all source actions retained."""
import numpy as np
from scipy.ndimage import median_filter
from .water import runs,block,smooth_clock,validate_schedule

def batch_phase(action,state):
 a=np.asarray(action);s=np.asarray(state);n=len(a);cycles=[]
 for k in [6,13]:cycles.append([(int(x),int(y)) for x,y in runs(median_filter(a[:,k],size=3)>20) if y-x>=6])
 if min(map(len,cycles))<3:raise ValueError('Missing bottle/tray gripper events')
 paired=[(x,y) for x in cycles[0][2:] for y in cycles[1][2:] if abs(x[0]-y[0])<=20 and min(x[1]-x[0],y[1]-y[0])>=14]
 if not paired:raise ValueError('No paired tray release opening')
 x,y=max(paired,key=lambda p:max(p[0][1],p[1][1]));base=max(x[1],y[1])+12;end=None
 for k in range(base,min(base+40,n-12)):
  if (s[k:k+3,[6,13]]<4).all():end=k+2;break
 if end is None:raise ValueError('No feedback-confirmed empty-gripper return boundary')
 start=min(cycles[0][1][0],cycles[1][1][0])-12
 first_open=min(cycles[0][0][0],cycles[1][0][0]);prefix=None;joints=[0,1,2,3,4,5,7,8,9,10,11,12]
 for k in range(max(1,first_open-6)):
  if np.ptp(s[k:k+6,joints],axis=0).max()<=.5 and np.ptp(a[k:k+6,joints],axis=0).max()<=.5:prefix=k;break
 initial_wait_supported=prefix is not None
 if prefix is None:prefix=0
 if not prefix+20<start<end<n-10:raise ValueError('Insufficient independent ranges')
 return dict(shared_start=start,shared_end=end,prefix=prefix,initial_wait_supported=initial_wait_supported,start_orders=['parallel','left_first','right_first'] if initial_wait_supported else ['parallel'],cycles=cycles,paired_release=[x,y],constraint_verified=True,image_verified=False,method='Conservative synchronization from12frames before bottle release through paired tray release plus12frames and actual empty closure; initial movement replayed jointly',bottle_events=[dict(object_id=i,robot_id=['left','right'][i],approach_start=c[0][0],pickup_frame=max(c[0][0],c[0][1]-25),grasp_frame=c[0][1],release_frame=start,retract_end=start-1) for i,c in enumerate(cycles)])

def pre_clock(prefix,stop,order):
 clocks,idle,wait=block(smooth_clock(prefix,stop),order,'pickup')
 head=np.arange(prefix,dtype=float);return np.vstack([np.c_[head,head],clocks]),np.vstack([np.zeros((prefix,2),bool),idle]),np.vstack([np.zeros((prefix,2),bool),wait])

def batch_schedule(n,S,E,start_order,return_order,prefix=0):
 pre,pi,pw=pre_clock(prefix,S-1,start_order);mid=smooth_clock(S-1,E+1);middle=np.c_[mid,mid];post,ri,rw=block(smooth_clock(E+1,n-1),return_order,'return')
 clock=np.vstack([pre,middle[1:],post[1:]]);idle=np.vstack([pi,np.zeros((len(mid)-1,2),bool),ri[1:]]);wait=np.vstack([pw,np.zeros((len(mid)-1,2),bool),rw[1:]]);stage=np.r_[np.zeros(len(pre),int),np.ones(len(mid)-1,int),np.full(len(post)-1,2,int)];length=len(clock)
 clock=np.vstack([clock,np.tile([n-1,n-1],(44,1))]);idle=np.vstack([idle,np.zeros((44,2),bool)]);wait=np.vstack([wait,np.zeros((44,2),bool)]);stage=np.r_[stage,np.full(44,3)];synthetic=np.arange(len(clock))>=length;moving=np.zeros_like(idle);moving[1:]|=np.diff(clock,axis=0)>0;moving[:-1]|=np.diff(clock,axis=0)>0;idle[moving]=False;active=(~idle).astype('float32');shared=((clock>=S)&(clock<=E)).any(axis=1);validate_schedule(clock,active,shared,wait,synthetic,n,S,E)
 return dict(clocks=clock,active=active,required_wait=wait,shared=shared,stage=stage,synthetic=synthetic)

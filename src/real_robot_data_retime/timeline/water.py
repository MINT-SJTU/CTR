"""Water tray CTR clocks and supervision policy. No calibration or joint projection."""
import numpy as np
from scipy.ndimage import median_filter

ORDERS=('parallel','left_first','right_first')
def runs(mask):return np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))).reshape(-1,2)
def phase_candidates(action,state,fps=30):
 action=np.asarray(action);state=np.asarray(state);n=len(action);assert action.shape==state.shape==(n,14)
 result=[]
 for side,k in enumerate([6,13]):
  opened=median_filter(action[:,k],size=3)>20
  seq=[(int(a),int(b)) for a,b in runs(opened) if b-a>=6]
  if len(seq)!=3:raise ValueError(f'Expected three opening cycles (bottle, tray approach, tray release); side {side}: {seq}')
  first,second,third=seq
  if not first[1]<second[0]<second[1]<third[0]<third[1]<n-5:raise ValueError('Invalid water event order')
  result.append(dict(open_runs=seq,pickup_command_close=first[1],bottle_open_command=second[0],tray_close_command=second[1],tray_release_open=third[0],empty_reclose=third[1]))
 start=min(x['tray_close_command'] for x in result)-6
 end=max(x['empty_reclose'] for x in result)+3
 if not 10<start<end<n-10:raise ValueError('Insufficient independent motion around synchronized carry')
 for side in range(2):
  v=state[:10,side*7:side*7+6]
  if np.ptp(v,axis=0).max()>0.5:raise ValueError('Initial state is already moving; cannot use it as an artificial onset hold')
 return dict(shared_start=start,shared_end=end,sides=result,method='gripper-cycle candidates; requires image confirmation of contact, settled tray and both arms disengaged',image_verified=False)

def smooth_clock(start,stop,ramp=8):
 d=stop-start
 if d<1:raise ValueError('Empty moving phase')
 r=min(ramp,d);rise=(np.arange(r)+0.5)/r
 steps=np.r_[rise,np.ones(d-r),rise[::-1]]
 clock=start+np.r_[0,np.cumsum(steps)];clock[-1]=stop
 assert np.all(np.diff(clock)>=0) and np.max(np.diff(clock))<=1+1e-8
 return clock

def block(clock,order,phase):
 if order not in ORDERS:raise ValueError(order)
 length=len(clock);delays=[0,0]
 if order!='parallel':delays[1 if order=='left_first' else 0]=length-1
 N=length+max(delays);clocks=np.stack([clock[np.clip(np.arange(N)-d,0,length-1)] for d in delays],axis=1)
 idle=np.zeros((N,2),bool);required=np.zeros((N,2),bool)
 for side,d in enumerate(delays):
  idle[:,side]=np.arange(N)<d
  if phase=='pickup':required[:,side]=np.arange(N)>=d+length
 # Both sides of the first advancing sample remain supervised. Never hide onset.
 advancing=np.zeros_like(idle);advancing[1:]|=np.diff(clocks,axis=0)>1e-9;advancing[:-1]|=np.diff(clocks,axis=0)>1e-9
 idle&=~advancing
 return clocks,idle,required

def make_schedule(n,shared_start,shared_end,start_order,return_order,fps=30,terminal_seconds=1.5):
 S,E=shared_start,shared_end
 if not 1<S<E<n-2:raise ValueError('Invalid synchronized range')
 pre,pi,pw=block(smooth_clock(0,S-1),start_order,'pickup')
 mid=smooth_clock(S-1,E+1);middle=np.stack([mid,mid],axis=1)
 post,ri,rw=block(smooth_clock(E+1,n-1),return_order,'return')
 clocks=np.concatenate([pre,middle[1:],post[1:]])
 idle=np.concatenate([pi,np.zeros((len(mid)-1,2),bool),ri[1:]])
 waits=np.concatenate([pw,np.zeros((len(mid)-1,2),bool),rw[1:]])
 stage=np.r_[np.zeros(len(pre),int),np.ones(len(mid)-1,int),np.full(len(post)-1,2,int)]
 original_length=len(clocks);tail=max(0,int(np.ceil(terminal_seconds*fps))-1)
 clocks=np.vstack([clocks,np.tile([n-1,n-1],(tail,1))]);idle=np.vstack([idle,np.zeros((tail,2),bool)]);waits=np.vstack([waits,np.zeros((tail,2),bool)]);stage=np.r_[stage,np.full(tail,3)]
 synthetic=np.arange(len(clocks))>=original_length
 active=(~idle).astype('float32');shared=((clocks>=S)&(clocks<=E)).any(axis=1)
 validate_schedule(clocks,active,shared,waits,synthetic,n,S,E)
 return dict(clocks=clocks,active=active,required_wait=waits,shared=shared,stage=stage,synthetic=synthetic)

def validate_schedule(clocks,active,shared,waits,synthetic,n,S,E):
 if not np.isfinite(clocks).all() or (np.diff(clocks,axis=0)<-1e-8).any():raise ValueError('Invalid source clocks')
 if clocks.min()<0 or clocks.max()>n-1 or np.max(np.diff(clocks,axis=0))>1+1e-8:raise ValueError('Source frames skipped or out of bounds')
 if not np.array_equal(clocks[shared,0],clocks[shared,1]):raise ValueError('Shared tray clocks diverge')
 if not np.all(active[shared]==1) or not np.all(active[waits]==1):raise ValueError('Required coordination masked')
 moving=np.zeros_like(active,dtype=bool);moving[1:]|=np.diff(clocks,axis=0)>1e-9;moving[:-1]|=np.diff(clocks,axis=0)>1e-9
 if not np.all(active[moving]==1):raise ValueError('Motion onset incorrectly masked')
 completed=clocks==n-1
 if not np.all(active[completed]==1) or not np.all(active[synthetic]==1):raise ValueError('Task completion must remain supervised')
 if not np.all(clocks[-1]==n-1):raise ValueError('Incomplete return')

"""Four-point sort-letters grid adapted to water's two independent phases."""
import numpy as np
from ..retime import ArmSegment, EpisodeSegments, build_uniform_schedule_plan
from .water import smooth_clock, validate_schedule

def grid_indices(source, count=75):
    if count < 1 or not 0 <= source < count:
        raise ValueError('invalid source/count')
    return [source + count * k for k in range(4)]

def uniform_block(clock, grid_index, grid_size):
    """Reuse the repository's exact duration-weighted onset rounding."""
    clock=np.asarray(clock,float);n=len(clock)
    if n < 2 or np.any(np.diff(clock)<=0):
        raise ValueError('moving source clock must strictly advance')
    segments=EpisodeSegments(0,ArmSegment(0,n),ArmSegment(0,n),0,0,0,0)
    plan=build_uniform_schedule_plan(segments,grid_index,grid_size)
    clocks=np.c_[clock[plan.left_source_indices],clock[plan.right_source_indices]]
    starts=np.array([plan.left_start_frame,plan.right_start_frame])
    idle=np.arange(len(clocks))[:,None]<starts
    moving=np.zeros_like(idle)
    moving[1:]|=np.diff(clocks,axis=0)>1e-9
    moving[:-1]|=np.diff(clocks,axis=0)>1e-9
    idle[moving]=False
    # Completed arms keep supervision. Activity labels from letters are not
    # copied: that would incorrectly mask completed water arms.
    required=np.arange(len(clocks))[:,None]>=starts+n
    return clocks,idle,required,plan.timing_receipt(30)

def make_uniform_schedule(n,S,E,source,variant,prefix=0,count=75):
    if variant not in range(4) or not 0<=prefix<S-1<E<n-2:
        raise ValueError('invalid phase or variant')
    grids=grid_indices(source,count)
    # A cyclic permutation keeps each marginal grid complete, with four full
    # tasks per source instead of the sixteen-point Cartesian product.
    return_variant=(variant+1+source%3)%4
    pre,pi,pw,pt=uniform_block(smooth_clock(prefix,S-1),grids[variant],4*count)
    head=np.arange(prefix,dtype=float)
    pre=np.vstack([np.c_[head,head],pre]);pi=np.vstack([np.zeros((prefix,2),bool),pi]);pw=np.vstack([np.zeros((prefix,2),bool),pw])
    mid=smooth_clock(S-1,E+1)[1:];middle=np.c_[mid,mid]
    post,ri,rw,rt=uniform_block(smooth_clock(E+1,n-1),grids[return_variant],4*count)
    clocks=np.vstack([pre,middle,post[1:]])
    idle=np.vstack([pi,np.zeros_like(middle,dtype=bool),ri[1:]])
    waits=np.vstack([pw,np.zeros_like(middle,dtype=bool),np.zeros_like(rw[1:])])
    stage=np.r_[np.zeros(len(pre),int),np.ones(len(mid),int),np.full(len(post)-1,2,int)]
    length=len(clocks)
    clocks=np.vstack([clocks,np.tile([n-1,n-1],(44,1))]);idle=np.vstack([idle,np.zeros((44,2),bool)]);waits=np.vstack([waits,np.zeros((44,2),bool)]);stage=np.r_[stage,np.full(44,3,int)]
    moving=np.zeros_like(idle);moving[1:]|=np.diff(clocks,axis=0)>1e-9;moving[:-1]|=np.diff(clocks,axis=0)>1e-9
    idle[moving]=False
    active=(~idle).astype('float32');shared=((clocks>=S)&(clocks<=E)).any(axis=1);synthetic=np.arange(len(clocks))>=length
    validate_schedule(clocks,active,shared,waits,synthetic,n,S,E)
    return dict(clocks=clocks,active=active,required_wait=waits,shared=shared,stage=stage,synthetic=synthetic,uniform=dict(source=source,variant=variant,grid_size=4*count,start=pt,return_timing=rt,return_variant=return_variant,prefix=prefix,policy='sort_letters_full_four_sample_grid'))

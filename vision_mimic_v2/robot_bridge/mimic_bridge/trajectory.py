"""Acceleration-limited references with a conservative stopping envelope.

This limits commands; it does not certify actual motor speed or collision safety.
"""
import numpy as np

class Trajectory:
    def __init__(self,cfg,initial):
        self.lo=cfg['demo_lower'].copy();self.hi=cfg['demo_upper'].copy()
        self.vmax=cfg['velocity_max'].copy();self.amax=cfg['acceleration_max'].copy()
        self.active=self.vmax>0
        self.q=np.asarray(initial,dtype=float).copy();self.v=np.zeros(27)
        if np.any(self.q<self.lo) or np.any(self.q>self.hi):raise ValueError('Feedback outside demo envelope; calibration needed')

    def _stopping_speed(self,distance,dt):
        # v*dt + v^2/(2*a) <= distance. Conservative for semi-implicit integration.
        a=self.amax[self.active]
        return np.maximum(0,np.sqrt((a*dt)**2+2*a*np.maximum(distance,0))-a*dt)

    def step(self,target,dt,brake=False):
        if not np.isfinite(dt) or dt<=0 or dt>.04:raise ValueError('Invalid control interval')
        target=np.clip(np.asarray(target,dtype=float),self.lo,self.hi)
        idx=self.active;q=self.q[idx];v=self.v[idx];a=self.amax[idx]
        error=target[idx]-q
        requested=np.zeros_like(v) if brake else np.sign(error)*np.minimum(self.vmax[idx],self._stopping_speed(np.abs(error),dt))
        lower=np.maximum.reduce([-self.vmax[idx],v-a*dt,-self._stopping_speed(q-self.lo[idx],dt)])
        upper=np.minimum.reduce([self.vmax[idx],v+a*dt,self._stopping_speed(self.hi[idx]-q,dt)])
        if np.any(lower>upper+1e-9):raise ValueError('Stopping envelope infeasible')
        new_v=np.minimum(upper,np.maximum(lower,requested))
        self.q[idx]=q+new_v*dt;self.v[idx]=new_v
        if np.any(self.q<self.lo-1e-9) or np.any(self.q>self.hi+1e-9):raise ValueError('Reference crossed envelope')
        return self.q.copy(),self.v.copy()

"""URDF-based FK and IK in the waist frame, x forward/y left/z up."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from .config import ROOT

def unit(v):
    n=np.linalg.norm(v)
    return v/n if n>1e-10 else np.zeros(3)

def rot(axis,angle):
    x,y,z=unit(axis)
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*k@k

def rpy(values):
    r,p,y=values
    return rot(np.array([0,0,1]),y)@rot(np.array([0,1,0]),p)@rot(np.array([1,0,0]),r)

class ArmModel:
    def __init__(self,path=ROOT/'models/robot.urdf'):
        root=ET.parse(path).getroot();joints={j.attrib['name']:j for j in root.findall('joint')}
        self.chains={}
        for side in ('left','right'):
            chain=[]
            for part in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_yaw'):
                j=joints[side.capitalize()+'_'+part+'_joint'];o=j.find('origin')
                chain.append((np.fromstring(o.get('xyz'),sep=' '),rpy(np.fromstring(o.get('rpy','0 0 0'),sep=' ')),np.fromstring(j.find('axis').get('xyz'),sep=' ')))
            self.chains[side]=chain

    def fk(self,side,angles,tool=(0,0,0)):
        angles=np.asarray(angles,dtype=float)
        if angles.shape!=(6,) or not np.isfinite(angles).all():raise ValueError('Expected 6 finite arm angles')
        p=np.zeros(3);r=np.eye(3);anchors=[]
        for angle,(offset,origin_rot,axis) in zip(angles,self.chains[side]):
            p=p+r@offset;r=r@origin_rot;anchors.append(p.copy());r=r@rot(axis,angle)
        return {'shoulder':anchors[0],'elbow':anchors[3],'wrist':anchors[5],'tip':p+r@np.asarray(tool),'rotation':r}

    def solve_position(self,side,target,seed,lo,hi,tolerance=.002):
        target=np.asarray(target,dtype=float)
        if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('Invalid IK target')
        q=np.asarray(seed,dtype=float).copy();lo=np.asarray(lo);hi=np.asarray(hi)
        if q.shape!=(6,) or lo.shape!=(6,) or hi.shape!=(6,) or not all(np.isfinite(x).all() for x in (q,lo,hi)) or np.any(lo>hi):raise ValueError('Invalid IK seed or bounds')
        q=np.clip(q,lo,hi)
        movable=np.flatnonzero(hi>lo)
        iterations=0
        for iterations in range(100):
            pos=self.fk(side,q)['wrist'];error=target-pos
            if np.linalg.norm(error)<tolerance or not len(movable):break
            jac=np.column_stack([(self.fk(side,q+np.eye(6)[i]*1e-5)['wrist']-pos)/1e-5 for i in movable])
            delta=jac.T@np.linalg.solve(jac@jac.T+1e-4*np.eye(3),error)
            best=q;cost=np.linalg.norm(error)
            for scale in (1.,.5,.25):
                candidate=q.copy();candidate[movable]+=np.clip(delta,-.12,.12)*scale;candidate=np.clip(candidate,lo,hi)
                score=np.linalg.norm(target-self.fk(side,candidate)['wrist'])
                if score<cost:best,cost=candidate,score
            if np.linalg.norm(best-q)<1e-8:break
            q=best
        residual=float(np.linalg.norm(target-self.fk(side,q)['wrist']))
        return {'angles':q,'residual_m':residual,'reachable':residual<tolerance,'iterations':iterations+1}

    def arm_points(self,q):
        return {side:self.fk(side,np.asarray(q)[base:base+6]) for side,base in [('left',6),('right',18)]}

def segment_distance(a,b,c,d):
    """Closest distance between finite line segments, including degenerate ones."""
    a,b,c,d=map(lambda p:np.asarray(p,dtype=float),(a,b,c,d))
    u=b-a;v=d-c;w=a-c
    aa=u@u;bb=u@v;cc=v@v;dd=u@w;ee=v@w
    if aa<1e-12 and cc<1e-12:return float(np.linalg.norm(a-c))
    if aa<1e-12:s=0.;t=np.clip(ee/cc,0,1)
    elif cc<1e-12:t=0.;s=np.clip(-dd/aa,0,1)
    else:
        denom=aa*cc-bb*bb
        s=np.clip((bb*ee-cc*dd)/denom,0,1) if denom>1e-12 else 0.
        t=(bb*s+ee)/cc
        if t<0:t=0.;s=np.clip(-dd/aa,0,1)
        elif t>1:t=1.;s=np.clip((bb-dd)/aa,0,1)
    return float(np.linalg.norm(w+s*u-t*v))

class GeometryGuard:
    def __init__(self,cfg):self.model=ArmModel();self.geometry=cfg['geometry']
    def check(self,q):
        points=self.model.arm_points(q);segments={}
        for side,p in points.items():segments[side]=[(p['shoulder'],p['elbow']),(p['elbow'],p['wrist'])]
        arm=self.geometry['arm_radius_m'];margin=self.geometry['clearance_m']
        torso=(np.array([0.,0.,-.15]),np.array([0.,0.,.3]))
        for side,links in segments.items():
            for a,b in links:
                if segment_distance(a,b,*torso)<arm+self.geometry['torso_radius_m']+margin:
                    return False,side+'_arm_torso_proximity'
        for a,b in segments['left']:
            for c,d in segments['right']:
                if segment_distance(a,b,c,d)<2*arm+margin:return False,'arms_proximity'
        return True,'ok'

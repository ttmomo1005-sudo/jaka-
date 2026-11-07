"""Retarget in a consistent torso frame using the actual URDF arm axes."""
from pathlib import Path
import math
import time
from contextvars import ContextVar
import xml.etree.ElementTree as ET
import numpy as np

INIT_POS = np.array([0]*6 + [0,-1.57,0,1.57,0,.3] + [0]*6 + [0,-1.57,0,1.57,0,.3] + [0]*3, dtype=float)
ARM_MAPPING_VERSION = 3
PAIRED_DEPTH_ENABLED = ContextVar('paired_depth_enabled',default=True)
_root = ET.parse(Path(__file__).resolve().parents[1]/'web_joint'/'robot.urdf').getroot()
_joints = {j.attrib['name']: j for j in _root.findall('joint')}
# Official physical limits, shared with the simulation copy.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'web_joint'))
from balance_check import DOWN_LIMIT as DOWN, UP_LIMIT as UP

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)

def rotation(axis, angle):
    x,y,z = unit(axis)
    k = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3) + math.sin(angle)*k + (1-math.cos(angle))*(k@k)

_chains = {}
for side in ('Left','Right'):
    names = [f'{side}_{part}_joint' for part in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_yaw')]
    _chains[side] = [(np.fromstring(_joints[n].find('origin').get('xyz'),sep=' '), np.fromstring(_joints[n].find('axis').get('xyz'),sep=' ')) for n in names]

def arm_directions(q, side):
    """Normalized shoulder→elbow and elbow→wrist in the waist frame."""
    r = np.eye(3)
    p = np.zeros(3)
    points = []
    for i,(offset,axis) in enumerate(_chains[side]):
        p = p + r@offset
        if i in (0,3,5): points.append(p.copy())
        r = r@rotation(axis, q[i] if i < len(q) else 0.)
    return np.concatenate([unit(points[1]-points[0]),unit(points[2]-points[1])])

def solve_arm(target, side, seed=None):
    base = 6 if side == 'Left' else 18
    lo,hi = DOWN[base:base+4],UP[base:base+4]
    target = np.asarray(target,dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError('Expected two finite arm directions')
    previous = np.clip(INIT_POS[base:base+4] if seed is None else np.asarray(seed,dtype=float),lo,hi)
    bend = math.acos(float(np.clip(np.dot(unit(target[:3]),unit(target[3:])), -1.,1.)))
    # Straight arms do not determine twist. Damping alone is not a posture prior:
    # it can retain a large yaw acquired while reducing elbow flexion.
    yaw_weight = .12 * max(0.,1.-bend/.7)**2
    weights = np.array([.0003,.0003,.0008,.0003]) if seed is not None else np.zeros(4)
    def cost(values):
        error = target-arm_directions(values,side)
        return float(error@error + np.sum(weights*(values-previous)**2) + yaw_weight*values[2]**2)
    natural = previous.copy()
    natural[2] = 0.
    natural[3] = np.clip(bend,lo[3],hi[3])
    candidates = [previous,natural]
    results = []
    for start in candidates:
      q = start.copy()
      for _ in range(24):
        pred = arm_directions(q, side)
        error = target-pred
        eps = 1e-4
        jac = np.column_stack([(arm_directions(q+np.eye(4)[i]*eps,side)-pred)/eps for i in range(4)])
        prior = weights*(q-previous)
        prior[2] += yaw_weight*q[2]
        diagonal = weights.copy()
        diagonal[2] += yaw_weight
        delta = np.linalg.solve(jac.T@jac + np.diag(diagonal) + .015*np.eye(4), jac.T@error-prior)
        best = q
        score = cost(q)
        for scale in (1.,.5,.25):
            candidate = np.clip(q+np.clip(delta,-.45,.45)*scale,lo,hi)
            new_cost = cost(candidate)
            if new_cost < score:
                best,score = candidate,new_cost
        if np.linalg.norm(best-q) < 1e-5: break
        q = best
      results.append(q)
    q = min(results,key=cost)
    return q, float(np.linalg.norm(target-arm_directions(q,side)))

# Bounded per-person history; stale/lost limbs and depth-source changes reset it.
# This changes IK branch selection, not the downstream trajectory speed limits.
_arm_history = {}

def reset_arm_history():
    _arm_history.clear()

def point_map(person, world=True):
    points = {}
    for k in person.get('world_keypoints' if world else 'keypoints',[]):
        if float(k.get('visibility',0) or 0) < .45: continue
        p = np.array([k['x'],k['y'],k.get('z',0) if world else 0.],dtype=float)
        if np.isfinite(p).all(): points[k['name']] = p
    return points

def torso_frame(points):
    required = ('left_shoulder','right_shoulder','left_hip','right_hip')
    if not all(k in points for k in required): return None
    ls,rs,lh,rh = [points[k] for k in required]
    left = unit(ls-rs)
    up = unit((ls+rs-lh-rh)*.5)
    forward = unit(np.cross(left,up))
    left = unit(np.cross(up,forward))
    if min(np.linalg.norm(forward), np.linalg.norm(left)) < .9: return None
    return np.stack([forward,left,up])

def paired_depth_targets(raw, points, frame):
    """Frontal projected symmetry can flag unreliable independent depth estimates.

    Preserve each arm's image direction and use a shared depth estimate only
    when both projected limbs agree. This is a prior, not measured 3D symmetry.
    Single-arm, side-on, missing and asymmetric poses keep independent targets.
    """
    info = {'applied':False,'reason':'independent_3d'}
    pixels = point_map(raw,False)
    image_frame = torso_frame(pixels)
    names = [side+'_'+part for side in ('left','right') for part in ('shoulder','elbow','wrist')]
    if image_frame is None or not all(n in pixels and n in points for n in names):
        info['reason']='missing_pair'
        return {},info
    if abs(frame[0,2]) < .85:
        info['reason']='side_view'
        return {},info
    projected,world = {},{}
    lengths={}
    for side in ('left','right'):
        for src,basis,destination in ((pixels,image_frame,projected),(points,frame,world)):
            sh,el,wr = [src[side+'_'+part] for part in ('shoulder','elbow','wrist')]
            if min(np.linalg.norm(el-sh),np.linalg.norm(wr-el))<1e-6:
                info['reason']='degenerate_pair'
                return {},info
            destination[side]=np.array([unit(basis@(el-sh)),unit(basis@(wr-el))])
            if src is pixels:
                lengths[side]=np.array([np.linalg.norm(el-sh),np.linalg.norm(wr-el)])
    reflected = projected['right']*np.array([1,-1,1])
    image_error = float(np.linalg.norm(projected['left']-reflected))
    world_error = float(np.linalg.norm(world['left']-world['right']*np.array([1,-1,1])))
    info.update(image_symmetry_error=image_error,world_symmetry_error=world_error)
    # Do not couple visibly different motions or already-consistent estimates.
    length_error=float(np.max(np.abs(np.log(lengths['left']/lengths['right']))))
    info['projected_length_log_ratio']=length_error
    if image_error>=.24 or world_error<=.25 or length_error>.25:
        info['reason']='asymmetric_image' if image_error>=.24 else ('foreshortened_pair' if length_error>.25 else 'consistent_depth')
        return {},info
    strength = float(np.clip((.24-image_error)/.06,0,1)*np.clip((world_error-.25)/.25,0,1))
    depth = (world['left'][:,0]+world['right'][:,0])*.5
    adjusted={}
    for side in ('left','right'):
        corrected=[]
        for i in range(2):
            in_plane=unit(np.array([0.,*projected[side][i,1:]]))
            candidate=np.array([depth[i],0.,0.])+in_plane*math.sqrt(max(0.,1.-depth[i]**2))
            corrected.append(unit((1-strength)*world[side][i]+strength*candidate))
        adjusted[side.capitalize()]=np.concatenate(corrected)
    info.update(applied=True,reason='frontal_depth_disagreement',strength=strength)
    return adjusted,info

def attach_robot_q(jaka_person, raw_person=None, *, stand_legs=True, sample_time=None):
    raw = raw_person or {}
    geom = np.asarray(jaka_person.get('q',[0]*27),dtype=float)
    q = INIT_POS.copy()
    valid = np.zeros(27,dtype=bool)
    points = point_map(raw)
    frame = torso_frame(points)
    source = 'world_3d'
    # Entire limb and torso fall back together; never combine meters and pixels.
    if frame is None:
        points = point_map(raw,False)
        frame = torso_frame(points)
        source = 'image_2d'
    errors = {}
    diagnostics = {}
    arm_targets = {}
    paired_enabled=PAIRED_DEPTH_ENABLED.get()
    paired,depth_info = paired_depth_targets(raw,points,frame) if paired_enabled and frame is not None and source=='world_3d' else ({},{'applied':False,'reason':'disabled' if not paired_enabled else 'no_world_frame'})
    now = time.monotonic() if sample_time is None else float(sample_time)
    if not math.isfinite(now):raise ValueError('Invalid sample time')
    for key in list(_arm_history):
        if now-_arm_history[key]['time'] > 1. or now<_arm_history[key]['time'] or (frame is None and key[0]==str(raw.get('id',1))):
            del _arm_history[key]
    if frame is not None:
        for side,base in [('Left',6),('Right',18)]:
            key = (str(raw.get('id',1)),side)
            names = [side.lower()+'_'+part for part in ('shoulder','elbow','wrist')]
            arm_points = points
            arm_frame = frame
            arm_source = source
            if not all(n in arm_points for n in names):
                arm_points = point_map(raw,False)
                arm_frame = torso_frame(arm_points)
                arm_source = 'image_2d'
            if arm_frame is None or not all(n in arm_points for n in names):
                _arm_history.pop(key,None)
                continue
            sh,el,wr = [arm_points[n] for n in names]
            if min(np.linalg.norm(el-sh),np.linalg.norm(wr-el)) < 1e-6:
                _arm_history.pop(key,None)
                continue
            target = np.concatenate([unit(arm_frame@(el-sh)),unit(arm_frame@(wr-el))])
            if arm_source=='world_3d' and side in paired:
                target=paired[side]
            arm_targets[side]=target.tolist()
            history = _arm_history.get(key)
            seed = history['q'] if history and history['source']==arm_source and history['paired_enabled']==paired_enabled else None
            if seed is not None and np.allclose(target,history['target'],atol=1e-10,rtol=0):
                solution,error = seed.copy(),history['error']
            else:
                solution,error = solve_arm(target,side,seed)
            errors[side] = error
            accepted = error < .20
            diagnostics[side] = {'source':arm_source,'accepted':accepted,
                                 'reason':'ok' if accepted else 'direction_unreachable',
                                 'continuous_seed':seed is not None}
            if accepted:
                q[base:base+4] = solution
                valid[base:base+4] = True
                if len(_arm_history)>=16 and key not in _arm_history:
                    _arm_history.pop(min(_arm_history,key=lambda k:_arm_history[k]['time']))
                _arm_history[key] = dict(q=solution.copy(),target=target.copy(),error=error,source=arm_source,time=now,paired_enabled=paired_enabled)
            else:
                _arm_history.pop(key,None)
        # Legs, waist and head remain explicitly approximate geometric estimates.
        # Full-body mode uses them; upper mode leaves the legs/waist to loco.
        for side,base in [('left',0),('right',12)]:
            if all(side+'_'+n in points for n in ('hip','knee','ankle')):
                q[base:base+6] = geom[base:base+6]
                valid[base:base+6] = True
        if source == 'world_3d':
            q[24:27] = geom[24:27]
            valid[24] = True
            valid[25:27] = 'nose' in points
    out = dict(jaka_person)
    out.update(geom_q=geom.tolist(), robot_q=np.clip(q,DOWN,UP).tolist(), robot_valid=valid.tolist(),
               q_space='geom',robot_q_space='mujoco_init',mapping_source=source,arm_fit_error=errors,
               arm_mapping=diagnostics,arm_mapping_version=ARM_MAPPING_VERSION,
               depth_consistency=depth_info,arm_targets=arm_targets)
    return out

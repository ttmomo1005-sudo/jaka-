from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
V2=ROOT.parent
DEFAULT_PROFILE=ROOT/'config/arms8_mock.json'
ARM_IDS=(6,7,8,9,18,19,20,21)

def vector(values,name):
    raw=np.asarray(values)
    if not np.issubdtype(raw.dtype,np.number) or np.iscomplexobj(raw):raise ValueError(name+' must contain real numeric angles')
    if isinstance(values,list) and any(isinstance(x,bool) for x in values):raise ValueError(name+' cannot contain booleans')
    arr=np.asarray(values,dtype=float)
    if arr.shape!=(27,) or not np.isfinite(arr).all():raise ValueError(name+' must contain 27 finite values')
    return arr

def load_profile(path=DEFAULT_PROFILE):
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg.get('robot_write_enabled') is not False or cfg.get('hardware_verified') is not False:
        raise ValueError('This offline package cannot enable hardware writes')
    if cfg['enabled_indices']!=list(ARM_IDS):raise ValueError('Only the approved 8 shoulder/elbow joints are permitted')
    if cfg['unit']!='rad' or cfg['q_space']!='mujoco_init':raise ValueError('Unsupported coordinate convention')
    names=cfg['joint_names']
    if len(names)!=27 or len(set(names))!=27:raise ValueError('Expected 27 unique joint names')
    for name in ('initial_q','physical_lower','physical_upper','demo_lower','demo_upper','velocity_max','acceleration_max'):
        cfg[name]=vector(cfg[name],name)
    lo,hi=cfg['demo_lower'],cfg['demo_upper']
    if np.any(lo>hi) or np.any(lo<cfg['physical_lower']) or np.any(hi>cfg['physical_upper']):raise ValueError('Demo bounds must be inside physical candidates')
    if np.any(cfg['initial_q']<lo) or np.any(cfg['initial_q']>hi):raise ValueError('Initial pose outside demo bounds')
    locked=[i for i in range(27) if i not in ARM_IDS]
    if not np.array_equal(lo[locked],hi[locked]):raise ValueError('Other joints must be held')
    if not np.array_equal(lo[locked],cfg['initial_q'][locked]):raise ValueError('Locked references must equal initial pose')
    for name in ('velocity_max','acceleration_max'):
        if np.any(cfg[name][list(ARM_IDS)]<=0) or np.any(cfg[name][locked]!=0):raise ValueError('Invalid motion limits')
    for name in ('period_s','frame_timeout_s','feedback_timeout_s','max_frame_age_ms','max_tilt_rad','max_angular_velocity_rad_s','tracking_error_rad','tracking_error_duration_s','max_step_dt_s','locked_feedback_error_rad','arming_velocity_rad_s'):
        if type(cfg[name]) not in (int,float) or not np.isfinite(cfg[name]) or cfg[name]<=0:raise ValueError('Invalid '+name)
    if cfg['period_s']>cfg['max_step_dt_s']:raise ValueError('Period exceeds step bound')
    if not np.isfinite(cfg['actual_arm_velocity_fault_rad_s']) or cfg['actual_arm_velocity_fault_rad_s']<=0:raise ValueError('Invalid feedback velocity threshold')
    for name in ('arm_radius_m','torso_radius_m','clearance_m'):
        value=cfg['geometry'][name]
        if not np.isfinite(value) or value<=0:raise ValueError('Invalid geometry '+name)
    cfg['profile_id']=hashlib.sha256(Path(path).read_bytes()).hexdigest()[:20]
    return cfg

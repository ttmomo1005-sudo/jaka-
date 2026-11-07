"""Explicit model/SDK arm conversion; unknown calibration never defaults to zero."""
import math
import numpy as np
from .config import ARM_IDS, vector


def audit_calibration(cfg, calibration):
    missing=[]
    for name in ('sdk_abi_verified','control_ownership_confirmed','stop_behavior_confirmed'):
        if calibration.get(name) is not True:missing.append(name)
    for name in ('controller_version','sdk_version','evidence'):
        if not calibration.get(name):missing.append(name)
    axes=calibration.get('axes',[])
    if not isinstance(axes,list) or len(axes)!=27:return missing+['27_axis_records']
    for i,axis in enumerate(axes):
        if not isinstance(axis,dict) or axis.get('sdk_name')!=cfg['joint_names'][i]:
            missing.append(f'axis_{i}_name');continue
        if i not in ARM_IDS:continue
        if type(axis.get('direction')) is not int or axis['direction'] not in (-1,1):missing.append(f'axis_{i}_direction')
        offset=axis.get('zero_offset_rad')
        if type(offset) not in (int,float) or not math.isfinite(offset or 0) or offset is None:missing.append(f'axis_{i}_offset')
        lo,hi=axis.get('approved_lower_model_rad'),axis.get('approved_upper_model_rad')
        if type(lo) not in (int,float) or type(hi) not in (int,float) or not all(map(math.isfinite,(lo,hi))) or lo>=hi:
            missing.append(f'axis_{i}_limits')
        elif lo<cfg['physical_lower'][i] or hi>cfg['physical_upper'][i]:missing.append(f'axis_{i}_limits_outside_candidates')
        for name in ('approved_velocity_rad_s','approved_acceleration_rad_s2','approved_torque_nm'):
            value=axis.get(name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:missing.append(f'axis_{i}_{name}')
        if not axis.get('evidence'):missing.append(f'axis_{i}_evidence')
    return missing


def model_arms_to_sdk(cfg, calibration, q):
    missing=audit_calibration(cfg,calibration)
    if missing:raise ValueError('Unverified calibration: '+', '.join(missing))
    q=vector(q,'model q');positions=[];limited=[]
    for i in ARM_IDS:
        a=calibration['axes'][i]
        lo,hi=a['approved_lower_model_rad'],a['approved_upper_model_rad']
        value=float(np.clip(q[i],lo,hi))
        if abs(value-q[i])>1e-9:limited.append(i)
        positions.append(a['direction']*value+a['zero_offset_rad'])
    return dict(joint_names=[cfg['joint_names'][i] for i in ARM_IDS],positions=positions,
                limited_joints=limited,unit='rad',hardware_write_supported=False)


def sdk_arms_to_model(cfg, calibration, sdk_positions):
    missing=audit_calibration(cfg,calibration)
    if missing:raise ValueError('Unverified calibration: '+', '.join(missing))
    values=np.asarray(sdk_positions,dtype=float)
    if values.shape!=(8,) or not np.isfinite(values).all():raise ValueError('Expected 8 finite SDK arm values')
    return np.array([(v-calibration['axes'][i]['zero_offset_rad'])/calibration['axes'][i]['direction']
                     for i,v in zip(ARM_IDS,values)])

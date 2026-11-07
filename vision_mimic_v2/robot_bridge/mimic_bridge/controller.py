"""Robot-side supervisor for a mock backend only; no enable/motor API calls."""
import uuid
import numpy as np
from .config import ARM_IDS, vector
from .trajectory import Trajectory
from .kinematics import GeometryGuard

class Controller:
    def __init__(self,cfg):
        self.cfg=cfg;self.planner=Trajectory(cfg,cfg['initial_q'])
        self.geometry=GeometryGuard(cfg);self.target=cfg['initial_q'].copy()
        self.state='DISARMED';self.reason='startup_requires_manual_arm'
        self.epoch='';self.seq=0;self.last_frame=None;self.armed_at=None
        self.pending='DISARMED';self.error_since=None;self.limited=[]
        self.feedback=None
        self.heartbeat_seq=0;self.last_heartbeat=None

    def feedback_reason(self,feedback,now):
        if feedback is None:return 'feedback_missing'
        if feedback.get('estop'):return 'hardware_estop'
        if feedback.get('fault'):return 'drive_fault'
        if not feedback.get('operational',False):return 'controller_not_operational'
        try:
            if any(type(feedback.get(k)) is not bool for k in ('estop','fault','operational')):return 'feedback_invalid'
            actual=vector(feedback['q'],'feedback q');vector(feedback['velocity'],'feedback velocity')
            age=now-float(feedback['received_at'])
            tilt=float(feedback['tilt_rad']);omega=float(feedback['angular_velocity_rad_s'])
            if not all(np.isfinite([age,tilt,omega])):return 'feedback_nonfinite'
        except (ValueError,TypeError,KeyError):return 'feedback_invalid'
        if age<0 or age>self.cfg['feedback_timeout_s']:return 'feedback_stale'
        if np.any(actual<self.cfg['physical_lower']) or np.any(actual>self.cfg['physical_upper']):return 'actual_position_outside_candidates'
        locked=[i for i in range(27) if i not in ARM_IDS]
        if np.max(np.abs(actual[locked]-self.cfg['initial_q'][locked]))>self.cfg['locked_feedback_error_rad']:return 'locked_joint_drift'
        if abs(tilt)>self.cfg['max_tilt_rad']:return 'tilt_exceeded'
        if abs(omega)>self.cfg['max_angular_velocity_rad_s']:return 'angular_velocity_exceeded'
        if np.max(np.abs(vector(feedback['velocity'],'feedback velocity')[list(ARM_IDS)]))>self.cfg['actual_arm_velocity_fault_rad_s']:return 'actual_arm_velocity_exceeded'
        return None

    def arm(self,feedback,now):
        if self.state=='FAULT':raise ValueError('Fault is latched; inspect and reset before arming')
        if self.state!='DISARMED':raise ValueError('Disarm and finish braking first')
        reason=self.feedback_reason(feedback,now)
        if reason:raise ValueError(reason)
        actual=vector(feedback['q'],'feedback q')
        if np.max(np.abs(vector(feedback['velocity'],'feedback velocity')))>self.cfg['arming_velocity_rad_s']:raise ValueError('Arm only from stationary feedback')
        if not self.geometry.check(actual)[0]:raise ValueError('Initial pose fails geometry guard')
        reference=self.cfg['initial_q'].copy()
        reference[list(ARM_IDS)]=actual[list(ARM_IDS)]
        # Inactive references remain fixed, but real feedback need not be exactly
        # equal to an ideal hold angle. Only active arms seed the command planner.
        self.planner=Trajectory(self.cfg,reference)
        self.target=reference.copy();self.feedback=feedback
        self.epoch=uuid.uuid4().hex;self.seq=0;self.last_frame=None;self.armed_at=now
        self.state='ARMED_WAITING';self.reason='waiting_for_fresh_frame'
        self.pending='DISARMED';self.error_since=None;self.limited=[]
        self.heartbeat_seq=0;self.last_heartbeat=now
        return self.epoch

    def brake(self,reason,pending='DISARMED'):
        if self.state in ('DISARMED','FAULT'):return
        if self.state=='BRAKING' and pending!='FAULT':return  # retain original stop/fault reason
        self.state='BRAKING';self.reason=reason
        if pending=='FAULT' or self.pending!='FAULT':self.pending=pending
        self.epoch=''  # Every stop invalidates all in-flight packets.

    def reset(self,feedback,now):
        if self.state!='FAULT':raise ValueError('Reset applies only to latched faults')
        reason=self.feedback_reason(feedback,now)
        if reason:raise ValueError(reason)
        if np.max(np.abs(self.planner.v))>1e-5:raise ValueError('Reference has not stopped')
        self.state='DISARMED';self.reason='fault_reset_requires_new_manual_arm';self.pending='DISARMED';self.error_since=None

    def heartbeat(self,message,now):
        if self.state not in ('ARMED_WAITING','TRACKING') or message.get('epoch')!=self.epoch:
            raise ValueError('Heartbeat requires current armed session')
        if message.get('profile_id')!=self.cfg['profile_id']:raise ValueError('Heartbeat profile mismatch')
        seq=message.get('seq')
        if type(seq) is not int or seq<=self.heartbeat_seq:raise ValueError('Duplicate/out-of-order heartbeat')
        self.heartbeat_seq=seq;self.last_heartbeat=now
        # A heartbeat proves connection liveness only. It never refreshes an image.

    def accept(self,message,now):
        if self.state not in ('ARMED_WAITING','TRACKING'):raise ValueError('Controller is not armed')
        if message.get('epoch')!=self.epoch:raise ValueError('Expired control session')
        if message.get('profile_id')!=self.cfg['profile_id']:raise ValueError('Joint/profile configuration mismatch')
        if message.get('unit')!='rad' or message.get('q_space')!='mujoco_init':raise ValueError('Coordinate convention mismatch')
        seq=message.get('seq')
        if type(seq) is not int or seq<=self.seq:raise ValueError('Duplicate or out-of-order frame')
        age=message.get('age_ms')
        if isinstance(age,bool) or not isinstance(age,(float,int)) or not np.isfinite(age) or age<0 or age>self.cfg['max_frame_age_ms']:raise ValueError('Frame is stale or has invalid age')
        q=vector(message.get('q'),'target q');valid=message.get('valid')
        if not isinstance(valid,list) or len(valid)!=27 or any(type(x) is not bool for x in valid):raise ValueError('Expected 27 boolean visibility flags')
        # Soft hold: missing landmarks or unsafe geometry never kill the session.
        # Keep the last safe reference, refresh timeouts, wait until motion is valid again.
        if not all(valid[i] for i in ARM_IDS):
            self.limited=[];self.seq=seq;self.last_frame=now;self.last_heartbeat=now
            self.state='TRACKING';self.reason='arm_landmarks_hold'
            return
        target=self.planner.q.copy()
        target[list(ARM_IDS)]=np.clip(q[list(ARM_IDS)],self.cfg['demo_lower'][list(ARM_IDS)],self.cfg['demo_upper'][list(ARM_IDS)])
        limited=[i for i in ARM_IDS if abs(q[i]-target[i])>1e-9]
        if not self.geometry.check(target)[0]:
            self.limited=limited;self.seq=seq;self.last_frame=now;self.last_heartbeat=now
            self.state='TRACKING';self.reason='target_geometry_hold'
            return
        for fraction in np.linspace(0,1,9):
            if not self.geometry.check(self.planner.q+fraction*(target-self.planner.q))[0]:
                self.limited=limited;self.seq=seq;self.last_frame=now;self.last_heartbeat=now
                self.state='TRACKING';self.reason='path_geometry_hold'
                return
        self.limited=limited
        self.target=target;self.seq=seq;self.last_frame=now
        self.last_heartbeat=now
        self.state='TRACKING';self.reason='target_limited' if self.limited else 'tracking'

    def tick(self,feedback,now,dt):
        self.feedback=feedback
        if self.state in ('DISARMED','FAULT'):return self.planner.q.copy(),self.planner.v.copy()
        reason=self.feedback_reason(feedback,now)
        if reason:self.brake(reason,'FAULT')
        if reason=='hardware_estop':
            # Hardware E-stop owns the stop; never simulate a graceful reset or enable.
            self.state='FAULT';self.pending='FAULT';self.planner.v[:]=0
            return self.planner.q.copy(),self.planner.v.copy()
        if self.state in ('ARMED_WAITING','TRACKING'):
            last=self.last_frame if self.last_frame is not None else self.armed_at
            if now-last>self.cfg['frame_timeout_s']:self.brake('target_timeout')
        if not reason and self.state=='TRACKING':
            error=np.max(np.abs(vector(feedback['q'],'feedback q')[list(ARM_IDS)]-self.planner.q[list(ARM_IDS)]))
            if error>self.cfg['tracking_error_rad']:
                if self.error_since is None:self.error_since=now
                if now-self.error_since>self.cfg['tracking_error_duration_s']:self.brake('tracking_error','FAULT')
            else:self.error_since=None
        if not np.isfinite(dt) or dt<=0 or dt>self.cfg['max_step_dt_s']:
            self.brake('control_timing_overrun','FAULT')
            dt=self.cfg['period_s']
        previous_q=self.planner.q.copy()
        q,v=self.planner.step(self.target,dt,brake=self.state=='BRAKING')
        if not self.geometry.check(q)[0]:
            self.brake('reference_geometry_violation','FAULT')
            # No unsafe reference is emitted. This is a mock freeze, not a physical stop law.
            self.state='FAULT';self.planner.q=previous_q;self.planner.v[:]=0
            return previous_q,self.planner.v.copy()
        if self.state=='BRAKING' and np.max(np.abs(v))<1e-5:
            self.state=self.pending;self.pending='DISARMED';self.target=q.copy()
            self.planner.v[:]=0;v=self.planner.v.copy()
        return q,v

    def status(self):
        return {'state':self.state,'reason':self.reason,'seq':self.seq,'profile_id':self.cfg['profile_id'],
                'q_reference':self.planner.q.tolist(),'velocity_reference':self.planner.v.tolist(),
                'limited_joints':self.limited,'hardware_writes':False,'enabled_indices':list(ARM_IDS),
                'q_actual':None if self.feedback is None else self.feedback['q'],
                'velocity_actual':None if self.feedback is None else self.feedback['velocity'],
                'heartbeat_seq':self.heartbeat_seq,
                'guard_scope':'Approximate geometry and reference limits; not a hardware safety certification'}

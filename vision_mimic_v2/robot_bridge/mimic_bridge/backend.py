import numpy as np
import time

class MockBackend:
    """Synthetic position servo. All effects are in memory."""
    hardware_writes=False
    def __init__(self,cfg,now=None):
        self.q=cfg['initial_q'].copy();self.velocity=np.zeros(27)
        self.estop=False;self.fault=False;self.operational=True
        self.tilt=0.;self.omega=0.;self.freeze=False;self.feedback_stale=False
        self.updated_at=time.monotonic() if now is None else now

    def advance(self,q_reference,dt,now):
        old=self.q.copy()
        if not (self.estop or self.fault or self.freeze):
            self.q+=(np.asarray(q_reference)-self.q)*(1-np.exp(-dt/.04))
        self.velocity=(self.q-old)/dt
        if not self.feedback_stale:self.updated_at=now

    def feedback(self):
        return {'q':self.q.tolist(),'velocity':self.velocity.tolist(),'received_at':self.updated_at,
                'estop':self.estop,'fault':self.fault,'operational':self.operational,
                'tilt_rad':self.tilt,'angular_velocity_rad_s':self.omega}

def make_backend(name,cfg):
    if name!='mock':raise RuntimeError('Hardware writes are not implemented. Use the read-only SDK probe after on-site review.')
    return MockBackend(cfg)

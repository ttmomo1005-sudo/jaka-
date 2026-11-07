"""Time-based adaptive smoothing with confidence hold and a speed limit."""
import numpy as np

class PoseFilter:
    def __init__(self, initial):
        self.q = np.array(initial,dtype=float)
        self.last = None

    def update(self, q, valid, now):
        q = np.asarray(q,dtype=float)
        valid = np.asarray(valid,dtype=bool)
        if q.shape != (27,) or valid.shape != (27,) or not np.isfinite(q).all():
            raise ValueError('Expected 27 finite joint values and confidence flags')
        dt = .033 if self.last is None else np.clip(now-self.last,.001,.2)
        speed = np.abs(q-self.q)/dt
        cutoff = 2.5 + .7*speed
        alpha = 1-np.exp(-2*np.pi*cutoff*dt)
        change = np.clip(alpha*(q-self.q),-7*dt,7*dt)
        self.q += np.where(valid,change,0.)
        self.last = now
        return self.q.tolist()

import copy
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mimic_bridge.config import load_profile, ARM_IDS
from mimic_bridge.trajectory import Trajectory
from mimic_bridge.kinematics import ArmModel, GeometryGuard, segment_distance
from mimic_bridge.controller import Controller
from mimic_bridge.backend import MockBackend, make_backend
from mimic_bridge.server import Runtime, make_server

class Tests(unittest.TestCase):
    def setUp(self):self.cfg=load_profile();self.backend=MockBackend(self.cfg,now=0)
    def feedback(self,now):
        self.backend.updated_at=now
        return self.backend.feedback()
    def packet(self,c,seq=1,q=None):
        return dict(epoch=c.epoch,profile_id=self.cfg['profile_id'],seq=seq,unit='rad',q_space='mujoco_init',
            age_ms=0,q=(self.cfg['initial_q'] if q is None else q).tolist(),valid=[True]*27)
    def armed(self):
        c=Controller(self.cfg);c.arm(self.feedback(0),0);return c
    def test_random_trajectory(self):
        p=Trajectory(self.cfg,self.cfg['initial_q']);rng=np.random.default_rng(994)
        locked=[i for i in range(27) if i not in ARM_IDS]
        for i in range(15000):
            dt=float(rng.uniform(.005,.04));oldq=p.q.copy();oldv=p.v.copy()
            q,v=p.step(rng.uniform(self.cfg['demo_lower']-.5,self.cfg['demo_upper']+.5),dt,brake=i%97==0)
            self.assertTrue(np.all(q>=p.lo-1e-9) and np.all(q<=p.hi+1e-9))
            self.assertTrue(np.all(np.abs(v)<=p.vmax+1e-9))
            self.assertTrue(np.all(np.abs(v-oldv)<=p.amax*dt+1e-9))
            np.testing.assert_allclose(q-oldq,v*dt,atol=1e-12)
            np.testing.assert_array_equal(q[locked],self.cfg['initial_q'][locked])
    def test_timeout_brakes_and_invalidates(self):
        c=self.armed();q=self.cfg['initial_q'].copy();q[list(ARM_IDS)]+=.15
        c.accept(self.packet(c,q=q),0);old=self.packet(c,seq=2)
        for k in range(1,65):
            now=k*.02;c.tick(self.feedback(now),now,.02);self.backend.advance(c.planner.q,.02,now)
        self.assertEqual(c.state,'DISARMED');self.assertEqual(c.reason,'target_timeout')
        self.assertTrue(np.all(c.planner.v==0));self.assertEqual(c.epoch,'')
        with self.assertRaises(ValueError):c.accept(old,1.3)
    def test_missing_landmarks(self):
        c=self.armed();previous=c.target.copy();epoch=c.epoch
        m=self.packet(c);m['valid'][9]=False
        c.accept(m,0)
        self.assertEqual(c.state,'TRACKING');self.assertEqual(c.reason,'arm_landmarks_hold')
        self.assertEqual(c.epoch,epoch);np.testing.assert_allclose(c.target,previous)
    def test_protocol_rejections(self):
        c=self.armed();c.accept(self.packet(c),0)
        for change in ({'seq':1},{'seq':True},{'age_ms':501},{'age_ms':float('nan')},
                       {'epoch':'old'},{'profile_id':'other'},{'unit':'deg'},{'q_space':'sdk'},
                       {'q':[0]*26},{'q':[float('nan')]*27},{'q':[True]*27},{'q':['0']*27},{'valid':[1]*27}):
            msg=self.packet(c,seq=2);msg.update(change)
            with self.assertRaises((ValueError,TypeError)):c.accept(msg,.01)
        self.assertEqual(c.seq,1)
    def test_lock_others_and_clip(self):
        c=self.armed();q=self.cfg['initial_q'].copy();q[0]=100;q[24]=100;q[6]=100
        c.accept(self.packet(c,q=q),0)
        self.assertEqual(c.target[0],self.cfg['initial_q'][0]);self.assertEqual(c.target[24],0)
        self.assertEqual(c.target[6],self.cfg['demo_upper'][6]);self.assertIn(6,c.limited)
    def test_faults_latch(self):
        for change,reason in [({'fault':True},'drive_fault'),({'received_at':-1},'feedback_stale'),
            ({'tilt_rad':1},'tilt_exceeded'),({'angular_velocity_rad_s':1},'angular_velocity_exceeded'),
            ({'operational':False},'controller_not_operational'),({'estop':True},'hardware_estop'),
            ({'velocity':[1]*27},'actual_arm_velocity_exceeded')]:
            c=self.armed();f=self.feedback(0);f.update(change)
            c.tick(f,.02,.02)
            self.assertEqual(c.state,'FAULT');self.assertEqual(c.reason,reason)
            with self.assertRaises(ValueError):c.arm(f,.02)
            c.reset(self.feedback(.04),.04)
            self.assertEqual(c.state,'DISARMED');self.assertEqual(c.epoch,'')
    def test_tracking_error(self):
        c=self.armed();q=self.cfg['initial_q'].copy();q[6]+=.2
        for k in range(1,100):
            now=k*.02
            if c.state in ('TRACKING','ARMED_WAITING'):c.accept(self.packet(c,seq=k,q=q),now)
            c.tick(self.feedback(now),now,.02)
        self.assertEqual(c.state,'FAULT');self.assertEqual(c.reason,'tracking_error')
    def test_timing_overrun(self):
        c=self.armed();c.tick(self.feedback(.1),.1,.1)
        self.assertEqual(c.state,'FAULT');self.assertEqual(c.reason,'control_timing_overrun')
    def test_no_hardware_backend(self):
        with self.assertRaises(RuntimeError):make_backend('real',self.cfg)
        bad=copy.deepcopy(self.cfg);bad['robot_write_enabled']=True
        path=ROOT/'runtime/test_rejected_profile.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        serial={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in bad.items()}
        path.write_text(json.dumps(serial),encoding='utf-8')
        try:
            with self.assertRaises(ValueError):load_profile(path)
        finally:path.unlink()
    def test_fk_legacy_and_ik(self):
        spec=importlib.util.spec_from_file_location('legacy_fk',ROOT/'reference/legacy_ik/fk_algorithm.py')
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        model=ArmModel();rng=np.random.default_rng(61)
        for side,base in [('left',6),('right',18)]:
            lo=self.cfg['demo_lower'][base:base+6];hi=self.cfg['demo_upper'][base:base+6]
            for k in range(30):
                q=rng.uniform(lo,hi);target=model.fk(side,q)['wrist']
                np.testing.assert_allclose(target,legacy.fk_algorithm(side,q)['wrist_waist'],atol=1e-7)
                result=model.solve_position(side,target,self.cfg['initial_q'][base:base+6],lo,hi)
                self.assertTrue(result['reachable'],result)
                self.assertTrue(np.all(result['angles']>=lo) and np.all(result['angles']<=hi))
                np.testing.assert_array_equal(result['angles'][4:],q[4:])
            self.assertFalse(model.solve_position(side,[10,10,10],q,lo,hi)['reachable'])
    def test_geometry(self):
        guard=GeometryGuard(self.cfg);self.assertTrue(guard.check(self.cfg['initial_q'])[0])
        # Inflated mock torso must reject, verifying the rejection path.
        inflated=copy.deepcopy(self.cfg);inflated['geometry']['torso_radius_m']=1
        self.assertFalse(GeometryGuard(inflated).check(self.cfg['initial_q'])[0])
        self.assertAlmostEqual(segment_distance([0,0,0],[1,0,0],[.5,-1,0],[.5,1,0]),0)
        self.assertAlmostEqual(segment_distance([0,0,0],[0,0,0],[1,0,0],[2,0,0]),1)
    def test_http_auth_and_state(self):
        rt=Runtime(self.cfg);server=make_server(rt,'test-secret',port=0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();rt.start()
        base='http://127.0.0.1:'+str(server.server_port)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def call(path,obj=None,auth=True):
            req=urllib.request.Request(base+path,data=None if obj is None else json.dumps(obj).encode(),
                headers={'Authorization':'Bearer test-secret' if auth else '','Content-Type':'application/json'})
            with opener.open(req,timeout=2) as response:return json.load(response)
        try:
            with self.assertRaises(urllib.error.HTTPError) as err:call('/v1/status',auth=False)
            self.assertEqual(err.exception.code,401)
            self.assertEqual(call('/v1/status')['state'],'DISARMED')
            arm=call('/v1/arm',{});msg=self.packet(rt.controller);msg['epoch']=arm['epoch']
            self.assertTrue(call('/v1/target',msg)['ok'])
            with self.assertRaises(urllib.error.HTTPError):call('/v1/target',msg)
            with self.assertRaises(urllib.error.HTTPError):call('/v1/target',{'q':[float('nan')]*27})
            call('/v1/stop',{});time.sleep(.1)
            self.assertEqual(call('/v1/status')['state'],'DISARMED')
            self.assertIsNone(call('/v1/status')['scheduler_error'])
        finally:server.shutdown();server.server_close();rt.close();thread.join(timeout=2)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'random_trajectory_steps':15000,'fk_comparisons':60,'ik_reachable_tests':60,
        'hardware_writes':False,'passed':result.wasSuccessful()}
    (ROOT/'reports/offline_tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)

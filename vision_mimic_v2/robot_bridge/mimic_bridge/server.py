"""Authenticated local mock receiver. Hardware backends are deliberately absent."""
import argparse
import hmac
import json
import secrets
import threading
import time
import uuid
import os
from collections import deque
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .config import ROOT, load_profile
from .controller import Controller
from .backend import make_backend

def strict_json(raw):
    def reject(value):raise ValueError('Nonfinite JSON number: '+value)
    obj=json.loads(raw,parse_constant=reject)
    if not isinstance(obj,dict):raise ValueError('Expected JSON object')
    return obj

class Runtime:
    def __init__(self,cfg,backend='mock',event_path=None):
        self.cfg=cfg;self.backend=make_backend(backend,cfg);self.controller=Controller(cfg)
        self.lock=threading.RLock();self.stop_event=threading.Event();self.thread=None
        self.scheduler_error=None;self.ticks=0
        self.instance_id=os.environ.get('MOCK_INSTANCE_ID') or uuid.uuid4().hex
        self.events=deque(maxlen=256);self.previous_transition=None;self.event_stream=None
        if event_path:
            path=Path(event_path).resolve()
            if not path.is_relative_to(ROOT.resolve()):raise ValueError('Event log must stay inside package')
            path.parent.mkdir(parents=True,exist_ok=True)
            self.event_stream=path.open('a',encoding='utf-8')
    def record_event(self,now):
        transition=(self.controller.state,self.controller.reason)
        if transition==self.previous_transition:return
        self.previous_transition=transition
        event=dict(monotonic_s=now,state=transition[0],reason=transition[1],seq=self.controller.seq)
        self.events.append(event)
        if self.event_stream:self.event_stream.write(json.dumps(event)+'\n');self.event_stream.flush()
    def start(self):
        self.thread=threading.Thread(target=self.run,name='mock-control',daemon=True);self.thread.start()
    def run(self):
        previous=time.monotonic();deadline=previous+self.cfg['period_s']
        while not self.stop_event.wait(max(0,deadline-time.monotonic())):
            now=time.monotonic();dt=now-previous;previous=now
            try:
                with self.lock:
                    feedback=self.backend.feedback()
                    q,_=self.controller.tick(feedback,now,dt)
                    # Even when the mock is frozen, its synthetic sensor stream continues.
                    self.backend.advance(self.backend.q if self.controller.state=='FAULT' else q,min(dt,.04),now)
                    self.ticks+=1
                    self.record_event(now)
            except Exception as error:
                with self.lock:
                    self.scheduler_error=str(error);self.controller.state='FAULT'
                    self.controller.reason='control_exception';self.controller.epoch='';self.controller.planner.v[:]=0
                    self.record_event(now)
            deadline+=self.cfg['period_s']
            if deadline<now:deadline=now+self.cfg['period_s']
    def status(self):
        with self.lock:
            state=self.controller.status();state.update(backend='mock',ticks=self.ticks,scheduler_error=self.scheduler_error)
            state['instance_id']=self.instance_id
            state.update(events=list(self.events),feedback_age_ms=max(0.,(time.monotonic()-self.backend.updated_at)*1000))
            return state
    def close(self):
        # On mock shutdown only; a production process exit needs controller-side watchdog.
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=2)
        if self.event_stream:self.event_stream.close()

def make_server(runtime,token,host='127.0.0.1',port=8870):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(1.)
        def log_message(self,*args):pass
        def reply(self,status,obj):
            raw=json.dumps(obj,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)));self.end_headers()
            self.wfile.write(raw)
        def authorized(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.reply(401,{'ok':False,'error':'Authentication required'});return False
            return True
        def do_GET(self):
            if not self.authorized():return
            if self.path!='/v1/status':self.reply(404,{'ok':False});return
            self.reply(200,runtime.status())
        def do_POST(self):
            if not self.authorized():return
            if self.headers.get('Transfer-Encoding'):self.reply(400,{'ok':False,'error':'Chunked bodies unsupported'});return
            try:
                length=int(self.headers.get('Content-Length','-1'))
                if length<0 or length>16384:raise ValueError('Body must be 0..16384 bytes')
                msg=strict_json(self.rfile.read(length) or b'{}')
                now=time.monotonic()
                with runtime.lock:
                    ctrl=runtime.controller
                    if self.path=='/v1/arm':
                        epoch=ctrl.arm(runtime.backend.feedback(),now)
                        result={'ok':True,'epoch':epoch,'profile_id':runtime.cfg['profile_id']}
                    elif self.path=='/v1/target':ctrl.accept(msg,now);result={'ok':True,'limited_joints':ctrl.limited}
                    elif self.path=='/v1/heartbeat':ctrl.heartbeat(msg,now);result={'ok':True,'target_age_renewed':False}
                    elif self.path=='/v1/stop':ctrl.brake('operator_stop');result={'ok':True}
                    elif self.path=='/v1/reset':ctrl.reset(runtime.backend.feedback(),now);result={'ok':True}
                    elif self.path=='/v1/shutdown':
                        ctrl.brake('mock_service_shutdown');result={'ok':True,'mock_only':True}
                        def shutdown_mock():
                            deadline=time.monotonic()+2
                            while time.monotonic()<deadline:
                                if runtime.status()['state'] in ('DISARMED','FAULT'):break
                                time.sleep(.02)
                            server.shutdown()
                        threading.Thread(target=shutdown_mock,daemon=True).start()
                    else:self.reply(404,{'ok':False});return
                    runtime.record_event(now)
                self.reply(200,result)
            except (ValueError,TypeError,KeyError,TimeoutError) as error:
                self.reply(400,{'ok':False,'error':str(error)})
    server=ThreadingHTTPServer((host,port),Handler);server.daemon_threads=True
    return server

def main():
    parser=argparse.ArgumentParser(description='Offline mock only; no physical robot commands')
    parser.add_argument('--port',type=int,default=8870);parser.add_argument('--profile',default=str(ROOT/'config/arms8_mock.json'))
    parser.add_argument('--backend',choices=['mock'],default='mock')
    args=parser.parse_args();cfg=load_profile(args.profile)
    token_path=ROOT/'runtime/local_token.txt';token_path.parent.mkdir(exist_ok=True)
    if token_path.exists():token=token_path.read_text().strip()
    else:
        token=secrets.token_urlsafe(32)
        descriptor=os.open(token_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:stream.write(token)
    if len(token)<32 or any(c.isspace() for c in token):raise ValueError('Invalid local mock authentication token')
    runtime=Runtime(cfg,args.backend,ROOT/'runtime/mock_events.jsonl');server=make_server(runtime,token,port=args.port);runtime.start()
    print('MOCK ONLY / DISARMED. Receiver: http://127.0.0.1:'+str(args.port),flush=True)
    print('Local authentication file: '+str(token_path),flush=True)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:server.server_close();runtime.close()

if __name__=='__main__':main()

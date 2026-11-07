"""Explicitly opted-in mock source client. No real robot addressing is supported."""
import argparse
import json
import time
import urllib.request
from .config import ROOT

class Client:
    def __init__(self,port=8870,token=None):
        self.base='http://127.0.0.1:'+str(port)
        self.token=token or (ROOT/'runtime/local_token.txt').read_text().strip()
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(self,path,payload=None):
        raw=None if payload is None else json.dumps(payload,allow_nan=False).encode()
        req=urllib.request.Request(self.base+path,data=raw,headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        with self.opener.open(req,timeout=.5) as response:return json.load(response)

def finish_mock(client,share=None,timeout=2.0):
    """Keep sharing bounded brake references until receiver stops, then latch off."""
    try:
        client.call('/v1/stop',{})
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            status=client.call('/v1/status')
            if share:share({'q':status['q_reference']})
            if status['state'] in ('DISARMED','FAULT'):return status
            time.sleep(.02)
        raise TimeoutError('Mock braking did not complete; inspect receiver')
    finally:
        if share:share({'release':True,'stop_vision':True})

def main():
    parser=argparse.ArgumentParser(description='Manual mock bridge; arm joints only')
    parser.add_argument('--arm-mock',action='store_true',required=True)
    parser.add_argument('--port',type=int,default=8870);parser.add_argument('--web-port',type=int,default=8788)
    parser.add_argument('--sim-follow',action='store_true',help='Make local web/MuJoCo targets follow the protected references')
    args=parser.parse_args();client=Client(args.port)
    armed=client.call('/v1/arm',{});seq=0;heartbeat_seq=0;last_source=None;source_session=None
    def share(payload):
        req=urllib.request.Request('http://127.0.0.1:'+str(args.web_port)+'/api/bridge/pose',
            data=json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json'})
        with client.opener.open(req,timeout=.5) as response:return json.load(response)
    print('Mock manually armed. Ctrl+C stops; any fault requires manual restart.',flush=True)
    try:
        while True:
            heartbeat_seq+=1
            client.call('/v1/heartbeat',dict(epoch=armed['epoch'],profile_id=armed['profile_id'],seq=heartbeat_seq))
            with client.opener.open('http://127.0.0.1:'+str(args.web_port)+'/api/bridge/frame',timeout=.5) as response:frame=json.load(response)
            if not frame.get('enabled'):break
            if source_session is None:source_session=frame.get('session')
            elif frame.get('session')!=source_session:break
            if frame.get('seq')!=last_source and frame.get('q') is not None:
                seq+=1
                client.call('/v1/target',dict(epoch=armed['epoch'],profile_id=armed['profile_id'],seq=seq,
                    unit='rad',q_space='mujoco_init',age_ms=frame['age_ms'],q=frame['q'],valid=frame['valid']))
                last_source=frame['seq']
            status=client.call('/v1/status')
            if status['state'] not in ('TRACKING','ARMED_WAITING'):break
            if args.sim_follow:
                share({'q':status['q_reference']})
            time.sleep(.02)
    except KeyboardInterrupt:pass
    finally:
        try:finish_mock(client,share if args.sim_follow else None)
        except Exception as error:print('Mock cleanup: '+str(error),flush=True)
        print('Mock stop requested. Re-arm is manual.',flush=True)

if __name__=='__main__':main()

"""Deterministic remapping/export; optional isolated headless MuJoCo, never hardware."""
from pathlib import Path
import argparse,json,os,socket,subprocess,sys,uuid
import numpy as np
from analyze import ROOT,load,report,array
sys.path.insert(0,str(ROOT/'web_joint'))
from pose_filter import PoseFilter
import robot_q as mapping

def export(events,mode='recorded',paired=True):
    mapping.reset_arm_history();filt=PoseFilter(mapping.INIT_POS)
    frames=[];remapped=[];held=mapping.INIT_POS.copy();first=None;session=None
    token=mapping.PAIRED_DEPTH_ENABLED.set(paired)
    try:
        for event in events:
            if event['kind']!='vision':continue
            d=event['data'];t=float(event['t'])
            if first is None:first=t
            if session!=d.get('session'):
                mapping.reset_arm_history();filt=PoseFilter(held);session=d.get('session')
            if mode=='remap':
                raw=d.get('raw_person') or {}
                result=mapping.attach_robot_q({'q':d.get('geom_q') or [0]*27},raw,sample_time=t)
                held=np.asarray(filt.update(result['robot_q'],result['robot_valid'],t))
                updated=dict(d,mapped_q=result['robot_q'],filtered_q=held.tolist(),valid=result['robot_valid'],
                             arm_targets=result['arm_targets'],arm_fit_error=result['arm_fit_error'],depth_consistency=result['depth_consistency'],
                             paired_depth=paired,arm_mapping_version=mapping.ARM_MAPPING_VERSION)
                remapped.append(dict(event,data=updated))
            else:
                value=array(d.get('filtered_q'))
                if value is not None:held=value.copy()
            held[24]=0
            frame=dict(time=t-first,q=held.tolist())
            if frames and frame['time']==frames[-1]['time']:frames[-1]=frame
            else:frames.append(frame)
    finally:mapping.PAIRED_DEPTH_ENABLED.reset(token)
    if not frames:raise ValueError('No vision frames')
    frames.append(dict(time=frames[-1]['time']+1,q=frames[-1]['q']))
    return frames,remapped

def physics(trajectory,duration,name):
    # Reserve a free loopback port without touching the live 8788 service.
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    env=os.environ.copy();env.pop('WEB_ONLY',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',SIM_HEADLESS='1',SIM_REALTIME='0',
               SIM_TEST_SECONDS=str(duration),WEB_JOINT_NO_BROWSER='1',WEB_PORT=str(port),SIM_POLICY='official_stand',
               SIM_REPORT_FILE=name+'.json',SIM_TEST_TRAJECTORY=str(trajectory),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
               TORCH_HOME=str(ROOT/'cache/torch'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'))
    python=Path(r'C:\Users\Public\JAKA_Dev\winvenv\Scripts\python.exe')
    if not python.is_file():raise ValueError('Configure a MuJoCo Python for this computer')
    log=ROOT/'logs'/f'{name}.log'
    with log.open('w',encoding='utf-8') as stream:
        child=subprocess.Popen([str(python),'-B','-u',str(ROOT/'web_joint/start_server.py')],cwd=ROOT,env=env,
            stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            code=child.wait(timeout=240)
        except subprocess.TimeoutExpired:
            # On Windows the venv launcher can have a real Python child; use an
            # isolated process group termination only for this owned test process.
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
            else:child.terminate()
            raise ValueError('Isolated physics test timed out; see '+str(log))
    if code:raise ValueError('Physics failed; see '+str(log))
    result=json.loads((ROOT/'reports'/f'{name}.json').read_text(encoding='utf-8'))
    if not result.get('q_finite') or result['metrics']['step_errors'] or result['metrics']['base_weld_active']:
        raise ValueError('Invalid/free-base physics test')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('recording');parser.add_argument('--mode',choices=['recorded','remap'],default='recorded')
    parser.add_argument('--independent-depth',action='store_true');parser.add_argument('--physics',action='store_true');args=parser.parse_args()
    events=load(args.recording);frames,remapped=export(events,args.mode,not args.independent_depth)
    name='replay_'+uuid.uuid4().hex[:8];trajectory=ROOT/'reports'/f'{name}_trajectory.json'
    trajectory.write_text(json.dumps(dict(frames=frames,mode=args.mode,hardware_writes=False),indent=2),encoding='utf-8')
    if remapped:
        remap_path=ROOT/'recordings'/f'{name}.jsonl';remap_path.parent.mkdir(exist_ok=True)
        metadata=dict(events[0],data=dict(events[0]['data'],source='remapped recording',physics_included=False))
        remap_path.write_text('\n'.join(json.dumps(e,ensure_ascii=False) for e in [metadata,*remapped])+'\n',encoding='utf-8')
        report(remap_path)
    result={'trajectory':str(trajectory),'duration_s':frames[-1]['time'],'mode':args.mode,'hardware_writes':False}
    if args.physics:result['physics']=physics(trajectory,frames[-1]['time'],name)
    print(json.dumps(result,ensure_ascii=False))

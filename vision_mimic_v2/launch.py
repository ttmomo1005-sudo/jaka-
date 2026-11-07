"""Launch isolated V2 services and clean up only our own child processes."""
from pathlib import Path
import json
import os
import socket
import subprocess
import time
import urllib.request
import sys

def open_page(url):
    """Use installed Edge without changing the user's browser settings."""
    candidates=[Path(os.environ.get('ProgramFiles(x86)',r'C:\Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe',
                Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'Microsoft/Edge/Application/msedge.exe']
    for browser in candidates:
        if browser.is_file():
            try:
                subprocess.Popen([str(browser),url],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                print('Web page opened in Microsoft Edge.')
                return True
            except OSError as error:
                print('Could not open Edge: '+str(error))
    print('Open this URL manually in a working browser: '+url)
    return False

def check_graphical_desktop(smoke=False):
    if smoke or os.name!='nt':return
    from web_joint import viewer_window
    context=viewer_window.status()
    if not context.get('interactive_desktop'):
        raise RuntimeError('GUI launch is on a hidden desktop ('+str(context.get('desktop'))+'). Double-click START_V2.cmd on your visible Windows desktop.')

def port_open(port):
    with socket.socket() as connection:
        connection.settimeout(.5)
        return connection.connect_ex(('127.0.0.1',port)) == 0

def service_status(port, endpoint):
    # Local readiness checks must not be redirected through a system proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}{endpoint}',timeout=2) as response:
            return json.load(response)
    except Exception:
        return None

def service_ready(port, endpoint):
    return (service_status(port, endpoint) or {}).get('version') == 'vision_mimic_v2'

def request_simulation_start():
    if os.name=='nt':
        # The foreground launcher may grant its existing viewer permission to focus.
        # This is normal application activation, not a change to Windows settings.
        import ctypes
        pid=(service_status(8788,'/api/status') or {}).get('process_id')
        if type(pid) is int and pid>0:ctypes.windll.user32.AllowSetForegroundWindow(pid)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request=urllib.request.Request('http://127.0.0.1:8788/api/simulation/start',data=b'{}',
        headers={'Content-Type':'application/json'})
    try:
        with opener.open(request,timeout=3) as response:reply=json.load(response)
    except Exception as error:
        raise RuntimeError('The web service is alive but MuJoCo is stopped. Could not reopen it: '+str(error)) from error
    if not reply.get('ok'):raise RuntimeError('MuJoCo reopen rejected: '+str(reply))

def ensure_simulation(smoke=False,children=()):
    deadline=time.monotonic()+45
    requested=False
    while time.monotonic()<deadline:
        if any(p.poll() is not None for p in children):raise RuntimeError('A service exited during startup. Check V2 logs.')
        state=service_status(8788,'/api/status')
        if state and state.get('version')=='vision_mimic_v2':
            native=state.get('window') or {}
            if not smoke and native.get('interactive_desktop') is False:
                raise RuntimeError('Existing MuJoCo is on a hidden desktop. Restart V2 from the visible Windows desktop; this is not a visible viewer.')
            status=state.get('mujoco')
            if status=='running':
                if not smoke and state.get('viewer')=='headless':
                    raise RuntimeError('An existing headless test is running, not a MuJoCo window. Close that test first.')
                if smoke:return state
                if not requested:
                    # The same start command also restores a hidden/minimized existing viewer.
                    request_simulation_start();requested=True
                native=state.get('window') or {}
                if native.get('presented') or native.get('supported') is False:return state
            if status in ('offline','error'):
                if requested:raise RuntimeError('MuJoCo failed to reopen: '+str(state.get('detail')))
                print('Web service exists, but MuJoCo is stopped. Reopening its window...')
                request_simulation_start();requested=True
        time.sleep(.2)
    raise RuntimeError('MuJoCo window startup timed out. Check logs/simulation.log')

def existing_services():
    existing = {}
    for name,port,endpoint in [('vision',8766,'/api/health'),('simulation',8788,'/api/status')]:
        existing[name] = port_open(port)
        if existing[name] and not service_ready(port,endpoint):
            # A model may still be warming up. Wait before declaring a conflict.
            deadline = time.monotonic()+15
            while time.monotonic()<deadline:
                if service_ready(port,endpoint): break
                if not port_open(port): existing[name]=False;break
                time.sleep(.3)
            else:
                raise RuntimeError(f'Port {port} is occupied by an unconfirmed service. No process was stopped.')
    return existing

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
VISION_PY = Path(os.environ.get('VISION_PYTHON', sys.executable))
SIM_PY = Path(os.environ.get('SIM_PYTHON', sys.executable))

def main():
    children = []
    logs = []
    try:
        smoke = '--smoke' in sys.argv
        check_graphical_desktop(smoke)
        for p in (VISION_PY,SIM_PY):
            if not p.is_file(): raise RuntimeError(f'Missing Python: {p}')
        existing = existing_services()
        if all(existing.values()):
            if not smoke:
                open_page('http://127.0.0.1:8788/mimic.html')
            ensure_simulation(smoke=smoke)
            print('V2 vision and the MuJoCo window are ready.')
            return 0
        env = os.environ.copy()
        for key in ('WEB_ONLY','SIM_HEADLESS','SIM_TEST_SECONDS','WEB_JOINT_NO_BROWSER'):
            env.pop(key,None)
        if smoke:
            env.update(SIM_HEADLESS='1',SIM_TEST_SECONDS='3',WEB_JOINT_NO_BROWSER='1',SIM_REPORT_FILE='launcher_smoke.json')
        else:
            env['WEB_JOINT_NO_BROWSER']='1'
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', VISION_PORT='8766', WEB_PORT='8788',
                   WEB_JOINT_START_PATH='/mimic.html', FACE_WEB_NO_BROWSER='1',
                   MPLCONFIGDIR=str(ROOT/'cache/matplotlib'), YOLO_CONFIG_DIR=str(ROOT/'cache/yolo'),
                   TORCH_HOME=str(ROOT/'cache/torch'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        (ROOT/'logs').mkdir(exist_ok=True)
        for name,py,script in [('vision',VISION_PY,ROOT/'vision/server.py'),('simulation',SIM_PY,ROOT/'web_joint/start_server.py')]:
            if existing[name]:
                print(f'Reusing existing V2 {name} service.')
                continue
            log = (ROOT/'logs'/f'{name}.log').open('a',encoding='utf-8')
            logs.append(log)
            if name == 'simulation':
                deadline = time.monotonic()+45
                while time.monotonic() < deadline:
                    if any(p.poll() is not None for p in children): raise RuntimeError('Vision failed. See logs/vision.log')
                    if service_ready(8766,'/api/health'): break
                    time.sleep(.3)
                else: raise RuntimeError('Vision startup timed out. See logs/vision.log')
            children.append(subprocess.Popen([str(py),'-B','-u',str(script)],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW))
        if not smoke:
            deadline=time.monotonic()+45
            while not service_ready(8788,'/api/status'):
                if time.monotonic()>deadline or any(p.poll() is not None for p in children):
                    raise RuntimeError('Web startup failed. Check logs/simulation.log')
                time.sleep(.2)
            open_page('http://127.0.0.1:8788/mimic.html')
        ensure_simulation(smoke=smoke,children=children)
        print('V2: http://127.0.0.1:8788/mimic.html')
        print('Logs: '+str(ROOT/'logs'))
        print('Keep this window open. Ctrl+C stops services launched by this window.')
        while all(p.poll() is None for p in children): time.sleep(.5)
        if smoke and children[-1].poll() == 0:
            print('Launcher smoke test passed.')
            return 0
        raise RuntimeError('A service exited. Check the V2 logs.')
    except KeyboardInterrupt:
        print('\nStopping V2 services...')
    except Exception as e:
        print('[ERROR]',e)
        return 1
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
        for f in logs: f.close()
    return 0

if __name__ == '__main__': sys.exit(main())

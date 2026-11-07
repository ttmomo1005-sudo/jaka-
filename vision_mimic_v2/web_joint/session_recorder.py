"""Bounded asynchronous numeric recorder; never records camera images or tokens."""
from pathlib import Path
import datetime,json,queue,threading,time,uuid

class SessionRecorder:
    def __init__(self,root):
        self.root=Path(root).resolve();self.lock=threading.Lock()
        self.active=False;self.worker=None;self.events=0;self.error=None;self.name=None
        self.queue=None;self.started=None
    def status(self):
        with self.lock:
            return dict(active=self.active,id=self.name,events=self.events,error=self.error,
                        elapsed_s=round(time.perf_counter()-self.started,2) if self.active else None)
    def start(self,metadata):
        with self.lock:
            if self.worker and self.worker.is_alive():raise ValueError('先停止当前记录')
            self.root.mkdir(parents=True,exist_ok=True)
            self.name=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.jsonl'
            self.queue=queue.Queue(maxsize=2048);self.events=0;self.error=None;self.started=time.perf_counter()
            # Queue the header before exposing active state to the physics thread.
            self.queue.put_nowait(dict(kind='metadata',t=0.0,monotonic=self.started,
                data=dict(**metadata, schema_version=1,clock='server perf_counter',
                          images_recorded=False,hardware_writes=False)))
            self.active=True
            self.worker=threading.Thread(target=self._write,args=(self.root/self.name,),daemon=True)
            self.worker.start()
        return self.status()
    def emit(self,kind,data):
        with self.lock:
            if not self.active:return
            if time.perf_counter()-self.started>600:
                self.active=False;self.error='记录达到10分钟上限';return
            event=dict(kind=kind,t=time.perf_counter()-self.started,monotonic=time.perf_counter(),data=data)
            try:self.queue.put_nowait(event)
            except queue.Full:self.active=False;self.error='写入队列已满，记录停止，不影响控制'
    def _write(self,path):
        total=0
        try:
            with path.open('x',encoding='utf-8') as stream:
                while True:
                    try:event=self.queue.get(timeout=.2)
                    except queue.Empty:
                        with self.lock:
                            if not self.active:break
                        continue
                    line=json.dumps(event,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
                    if total+len(line.encode('utf-8'))>64*1024*1024:
                        with self.lock:self.active=False;self.error='记录达到64MiB上限'
                        break
                    stream.write(line);total+=len(line.encode('utf-8'))
                    with self.lock:self.events+=1
                    if self.queue.empty():stream.flush()
        except Exception as error:
            with self.lock:self.active=False;self.error=str(error)
    def stop(self):
        self.emit('stop',{})
        with self.lock:self.active=False;worker=self.worker
        if worker:worker.join(timeout=5)
        if worker and worker.is_alive():raise ValueError('文件仍在写入，稍后再试')
        return self.status()
    def files(self):
        return [dict(id=p.name,size_bytes=p.stat().st_size) for p in sorted(self.root.glob('*.jsonl'),reverse=True)][:100]
    def file(self,name):
        if not isinstance(name,str) or Path(name).name!=name or '/' in name or '\\' in name or not name.endswith('.jsonl'):
            raise ValueError('Invalid recording id')
        path=self.root/name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.root):raise ValueError('Recording not found')
        return path

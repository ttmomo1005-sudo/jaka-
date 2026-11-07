"""Pose-only binary JPEG service. No face database or original project writes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import threading
import time
import cv2
import numpy as np
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pose_engine import get_pose_engine
from robot_q import ARM_MAPPING_VERSION, PAIRED_DEPTH_ENABLED

@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(lambda: get_pose_engine().process(np.zeros((480,640,3),dtype=np.uint8)))
    yield

app = FastAPI(title='JAKA Mimic V2', lifespan=lifespan)
busy = threading.Lock()

@app.get('/api/health')
def health():
    return {'ok': True, 'version': 'vision_mimic_v2', 'arm_mapping_version': ARM_MAPPING_VERSION,
            'process_id':os.getpid(), 'paired_depth_supported':True,'recording_landmarks_supported':True}

@app.post('/api/pose')
async def pose(request: Request):
    raw = await request.body()
    if not raw or len(raw) > 5_000_000: raise HTTPException(400,'Invalid frame size')
    if not busy.acquire(blocking=False): raise HTTPException(429,'Inference busy; retry with a fresh frame')
    try:
        t0 = time.perf_counter()
        image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise HTTPException(400,'JPEG decode failed')
        if max(image.shape[:2]) > 1280: raise HTTPException(400,'Maximum frame side: 1280')
        decoded = time.perf_counter()
        # Offload heavy inference so the event loop can reject overlapping requests.
        setting=request.headers.get('X-Paired-Depth','1')
        if setting not in ('0','1'):raise HTTPException(400,'X-Paired-Depth must be 0 or 1')
        context_token=PAIRED_DEPTH_ENABLED.set(setting=='1')
        try:
            out = await asyncio.to_thread(get_pose_engine().process,image)
        finally:
            PAIRED_DEPTH_ENABLED.reset(context_token)
        out.update(ok=True,timing_ms={'decode':round((decoded-t0)*1000,1),'total':round((time.perf_counter()-t0)*1000,1)})
        return out
    finally:
        busy.release()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=int(os.environ.get('VISION_PORT','8766')))

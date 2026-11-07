"""Batch photos using the same V2 mapping as the live service."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
from pose_engine import get_pose_engine

def main():
    root = Path(__file__).resolve().parents[1]
    items=[]
    for filename in sys.argv[1:]:
        image=cv2.imdecode(np.frombuffer(Path(filename).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise ValueError('Unreadable photo: '+filename)
        persons=get_pose_engine().process(image).get('persons') or []
        items.append({'source':filename,'persons':persons})
    (root/'reports/photos.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()

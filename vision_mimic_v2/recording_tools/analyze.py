"""Analyze and replay numeric recordings without driving any robot or live sim."""
from pathlib import Path
import argparse,json,math,sys,html
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vision'))
import robot_q as mapping

def load(path):
    path=Path(path)
    if path.stat().st_size>64*1024*1024:raise ValueError('Recording exceeds 64MiB')
    events=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        event=json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        if not isinstance(event,dict) or event.get('kind') not in ('metadata','vision','physics','stop'):raise ValueError('Unknown event')
        if not math.isfinite(float(event['t'])) or event['t']<0:raise ValueError('Invalid event time')
        events.append(event)
    if not events or events[0]['kind']!='metadata' or events[0]['data'].get('schema_version')!=1:raise ValueError('Unsupported recording schema')
    return sorted(events,key=lambda e:e['t'])

def array(value):
    a=np.asarray(value,dtype=float)
    return a if a.shape==(27,) and np.isfinite(a).all() else None

def degrees(a,b):
    return math.degrees(math.acos(float(np.clip(np.dot(mapping.unit(a),mapping.unit(b)),-1,1))))

def statistics(values):
    if not values:return None
    return dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=float(np.max(values)),count=len(values))

def analyze(events):
    mapped_angles=[];raw_angles=[];joint_gaps=[];image_angles=[];processing=[];missing=0;vision=0
    tracking=[];requested_tracking=[];speed=[];falls=[];upright=[];drift=[];sym_count=0;assisted=0;rows=[]
    for e in events:
        d=e['data'];kind=e['kind']
        if kind=='vision':
            vision+=1;q=array(d.get('mapped_q'));valid=d.get('valid',[])
            raw=d.get('raw_person') or {};pixels=mapping.point_map(raw,False);image_frame=mapping.torso_frame(pixels)
            points=mapping.point_map(raw);world_frame=mapping.torso_frame(points)
            if q is None or len(valid)!=27 or not all(valid[i] for i in (6,7,8,9,18,19,20,21)):
                missing+=1;continue
            paired=[]
            for side,base in [('Left',6),('Right',18)]:
                pred=mapping.arm_directions(q[base:base+4],side).reshape(2,3)
                target=np.asarray(d.get('arm_targets',{}).get(side,[]),dtype=float)
                if target.shape==(6,) and np.isfinite(target).all():
                    mapped_angles.extend(degrees(a,b) for a,b in zip(pred,target.reshape(2,3)))
                names=[side.lower()+'_'+part for part in ('shoulder','elbow','wrist')]
                if world_frame is not None and all(n in points for n in names):
                    sh,el,wr=[points[n] for n in names]
                    raw_angles.extend(degrees(a,b) for a,b in zip(pred,[world_frame@(el-sh),world_frame@(wr-el)]))
                if image_frame is not None and all(n in pixels for n in names):
                    sh,el,wr=[pixels[n] for n in names]
                    projection=[image_frame@(el-sh),image_frame@(wr-el)];paired.append(projection)
                    for a,b in zip(pred,projection):
                        if min(np.linalg.norm(a[1:]),np.linalg.norm(b[1:]))>.05:
                            image_angles.append(degrees(np.r_[0.,a[1:]],np.r_[0.,b[1:]]))
            # Only score bilateral asymmetry when image observations suggest symmetry.
            if len(paired)==2 and np.linalg.norm(np.array([mapping.unit(v) for v in paired[0]])-np.array([mapping.unit(v) for v in paired[1]])*[1,-1,1])<.24:
                sym_count+=1;joint_gaps.extend(np.rad2deg(np.abs(q[6:10]-q[18:22])).tolist())
            assisted+=bool(d.get('depth_consistency',{}).get('applied'))
            receipt=d.get('proxy_receipt_monotonic');done=d.get('mapping_complete_monotonic')
            if receipt is not None and done is not None and done>=receipt:processing.append((done-receipt)*1000)
            rows.append(dict(t=e['t'],kind='vision',q=q.tolist(),valid=valid,seq=d.get('source_seq')))
        elif kind=='physics':
            q=array(d.get('actual_q'));bounded=array(d.get('bounded_q'));requested=array(d.get('requested_q'));dq=array(d.get('actual_dq'))
            ids=[6,7,8,9,18,19,20,21]
            if q is not None and bounded is not None:tracking.extend((q[ids]-bounded[ids]).tolist())
            if q is not None and requested is not None:requested_tracking.extend((q[ids]-requested[ids]).tolist())
            if dq is not None:speed.append(float(np.max(np.abs(dq[ids]))))
            for key,destination in [('fall_count',falls),('upright',upright),('base_displacement_m',drift)]:
                if key in d:destination.append(d[key])
            if q is not None:rows.append(dict(t=e['t'],kind='physics',q=q.tolist(),seq=d.get('source_seq'),bounded_q=d.get('bounded_q')))
    def rms(values):return float(np.sqrt(np.mean(np.square(values)))) if values else None
    summary=dict(vision_frames=vision,invalid_arm_frames=missing,invalid_fraction=missing/vision if vision else None,
        paired_assist_frames=assisted,symmetric_image_frames=sym_count,
        mapped_direction_error_deg=statistics(mapped_angles),raw_estimated_3d_error_deg=statistics(raw_angles),
        projected_direction_error_deg=statistics(image_angles),symmetric_joint_gap_deg=statistics(joint_gaps),
        server_processing_ms=statistics(processing),actual_vs_bounded_rms_rad=rms(tracking),
        actual_vs_requested_rms_rad=rms(requested_tracking),actual_arm_speed_rad_s=statistics(speed),
        fall_counter_delta=max(falls)-min(falls) if falls else None,min_upright=min(upright) if upright else None,
        max_base_displacement_m=max(drift) if drift else None,
        physics_present=bool(tracking),hardware_test=False,
        caveats=['Raw 3D is an estimate, not human ground truth; assisted targets differ from raw depth.',
                 'Projection metric is torso image-plane agreement, not calibrated camera reprojection.',
                 'Server timing excludes capture/encode/browser rendering and is not end-to-end latency.',
                 'Physics tracking compares contemporaneous samples; no measured true motor limits.',
                 'Counter delta may miss a first reset; separate simulation runs must be analyzed separately.'])
    return summary,rows

def report(path):
    events=load(path);summary,rows=analyze(events)
    name=Path(path).stem
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):name='imported_recording'
    output=ROOT/'reports/recording_analysis'/name;output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    table=''.join('<tr><td>'+html.escape(k)+'</td><td>'+html.escape(json.dumps(v,ensure_ascii=False))+'</td></tr>' for k,v in summary.items() if k!='caveats')
    data=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>动作记录分析与回放</title><style>body{background:#101923;color:#e6edf3;font:16px system-ui;max-width:1100px;margin:25px auto;padding:15px}table{border-collapse:collapse;width:100%}td{border:1px solid #456;padding:8px;overflow-wrap:anywhere}canvas{background:#060d15;width:100%;height:300px}input{width:65%}button,select{padding:10px}</style><h1>数值动作回放</h1><p>本页不会驱动仿真或真机。记录不含摄像头图像。选择映射或物理反馈，逐帧检查左右肩肘。</p><select id="kind"><option value="vision">映射目标</option><option value="physics">仿真实际反馈</option></select><button id="play">播放</button><input id="slider" type="range" min="0" value="0"><span id="time"></span><canvas id="plot" width="1100" height="300"></canvas><pre id="angles"></pre><h2>分析结果</h2><table>'''+table+'''</table><h2>适用边界</h2><p>原始三维关键点不是人体真值。方向误差应区分原始估计与辅助后的目标；网页模型不是物理反馈。服务器处理时间不包括采集、编码与显示，不能称为端到端延迟。参考限速不等于实际关节硬限速。</p><script>
const all='''+data+''';let rows=[],playing=false,last=0;const slider=document.getElementById('slider'),kind=document.getElementById('kind'),ctx=document.getElementById('plot').getContext('2d');const ids=[6,7,8,9,18,19,20,21],names=['左肩pitch','左肩roll','左肩yaw','左肘','右肩pitch','右肩roll','右肩yaw','右肘'];
let cursor=0;function draw(){const row=rows[Number(slider.value)];ctx.clearRect(0,0,1100,300);if(!row){document.getElementById('time').textContent='没有该类数据';return}document.getElementById('time').textContent=row.t.toFixed(2)+' 秒';document.getElementById('angles').textContent=ids.map((id,i)=>names[i]+': '+(row.q[id]*180/Math.PI).toFixed(1)+'°').join('  ');const duration=rows[rows.length-1].t-rows[0].t||1;ids.forEach((id,j)=>{ctx.strokeStyle=['#66aaff','#55ddaa','#ffaa55','#ee7799'][j%4];ctx.setLineDash(j<4?[]:[6,4]);ctx.beginPath();rows.forEach((r,i)=>{const x=(r.t-rows[0].t)/duration*1100,y=150-r.q[id]*45;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()});ctx.setLineDash([]);ctx.strokeStyle='white';ctx.beginPath();const x=(row.t-rows[0].t)/duration*1100;ctx.moveTo(x,0);ctx.lineTo(x,300);ctx.stroke()}
function select(){playing=false;rows=all.filter(r=>r.kind===kind.value);slider.max=Math.max(0,rows.length-1);slider.value=0;draw()}kind.onchange=select;slider.oninput=()=>{playing=false;draw()};document.getElementById('play').onclick=()=>{playing=!playing;last=performance.now();cursor=rows[Number(slider.value)]?.t||0};function tick(now){if(playing&&rows.length){cursor+=(now-last)/1000;last=now;let i=Number(slider.value);while(i<rows.length-1&&rows[i+1].t<=cursor)i++;if(i===rows.length-1)playing=false;slider.value=i;draw()}requestAnimationFrame(tick)}select();requestAnimationFrame(tick);
</script>'''
    (output/'回放与分析.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(output=str(output),summary=summary),ensure_ascii=False))
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('recording');args=parser.parse_args();report(args.recording)

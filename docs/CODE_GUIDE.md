# 源码导读

## 视觉模仿主线

1. `vision_mimic_v2/vision/pose_engine.py`：MediaPipe / YOLO 姿态后端。
2. `vision_mimic_v2/vision/jaka_map.py`：骨架过滤和几何角。
3. `vision_mimic_v2/vision/robot_q.py`：人体坐标系、双臂求解、机器人空间映射。导入时需要外部 URDF。
4. `vision_mimic_v2/vision/server.py`：接收图像，返回姿态；含会话和并发处理。
5. `vision_mimic_v2/web_joint/mimic.html`：摄像头、骨架、模型与关节信息。
6. `vision_mimic_v2/web_joint/start_server.py`：网页服务、视觉代理、MuJoCo 生命周期和目标同步。
7. `vision_mimic_v2/web_joint/standing_policy.py`：官方 ONNX 观测和动作适配。
8. `vision_mimic_v2/robot_bridge/mimic_bridge/trajectory.py`：参考目标的速度与加速度限制。

`web_joint/index.html` 是独立关节编辑页面；`balance_check.py` 提供物理限位和粗略支撑检测；`session_recorder.py` 与 `recording_tools/` 提供数值记录、分析和回放源码，不附带实际记录。

## 离线双臂模块

`robot_bridge/mimic_bridge/` 包含配置校验、轨迹、几何检查、控制状态机、客户端和 mock 服务端。后端只实现内存中的合成伺服，不能据此控制真机。

`robot_bridge/config/arms8_mock.json` 为离线示例参数，`calibration_template.json` 为模板；两者都不是经过现场标定的硬件参数。完整 mock 流程还需要 `robot_bridge/models/robot.urdf`。`tests/test_offline.py` 保留历史离线回归，其中几何、IK 等测试需要未随仓库分发的资源。

## 早期安防模块

- `1.robot/1.code/perception/`：人体检测、数据源、目标方位、出现/丢失事件及无摄像头 smoke 检查。
- `1.robot/1.code/face_yolov8_arcface/`：命令行入口、识别引擎、录入、人脸库读写、匹配和追踪。保留库读写代码，不附带任何库数据。

早期人脸网页、厂商 SDK、其他机器人项目、助手记忆与重复交接包没有收录，以控制体积并突出当前主线。

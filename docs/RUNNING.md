# 运行与外部资源

本仓库是轻量源码版。源码可供阅读和局部验证，完整视觉与仿真演示需要自行取得外部资源。请先阅读各依赖与模型的许可。

## 可直接做的局部检查

在仓库根目录运行人体方位和目标出现/丢失滤波检查，不需要摄像头或模型：

```console
python 1.robot/1.code/perception/smoke_azimuth.py
```

安装 NumPy 后，可以运行离线轨迹随机检查与无硬件后端检查：

```console
cd vision_mimic_v2/robot_bridge
python -m unittest discover -s tests -p test_offline.py -k random_trajectory
python -m unittest discover -s tests -p test_offline.py -k no_hardware_backend
```

其余离线测试中的几何、IK 和接口检查需要额外资源，不能把上述局部检查理解成全套集成验收。

## 完整 V2 演示需要准备

视觉依赖列于 `vision_mimic_v2/vision/requirements.txt`。仿真还需要 MuJoCo、PyTorch、ONNX Runtime 及原厂适配依赖；本公开版没有提供经过全新机器验证的完整锁定环境。原工程使用 Windows 和 Python 3.13，两套解释器分别承担视觉与仿真。

需要按原工程布局补齐以下资源：

- `vision_mimic_v2/models/pose_landmarker_heavy.task`：MediaPipe 模型；YOLO 备用权重可选。模型需从各项目官方来源取得。
- `vision_mimic_v2/web_joint/robot.urdf`：网页与视觉映射所需的机器人描述。
- `vision_mimic_v2/deploy/`：与本平台匹配的 `Khan_mini.xml`、`Khan_mini_mujoco.py`、配置、踝运动学及 `meshes/` 等厂商资源和依赖。
- `vision_mimic_v2/policies/official/官方运动ONNX/policy_notarget/policy_base_v1.onnx`：原工程使用的官方站立权重。适配器校验模型摘要和输入输出尺寸，其他版本不能随意替换。
- `vision_mimic_v2/web_joint/vendor/`：与原工程匹配的 Three.js、addons 及 `URDFLoader.module.js`，需保留各自许可证。
- `vision_mimic_v2/robot_bridge/models/robot.urdf`：运行完整双臂 mock 几何检查所需的同平台描述。

取得匹配资源并配置依赖后，可在 Windows 设置解释器：

```bat
set "VISION_PYTHON=C:\path\to\vision-env\Scripts\python.exe"
set "SIM_PYTHON=C:\path\to\simulation-env\Scripts\python.exe"
vision_mimic_v2\START_V2.cmd
```

不设置时使用当前 Python。公开副本将原本固定的本机解释器路径改为环境变量；原工作目录中的控制代码没有因此改动。

预期电脑演示地址为 `http://127.0.0.1:8788/mimic.html`，视觉服务为 8766。摄像头需在本机浏览器授权。独立 MuJoCo 窗口显示带重力的实际仿真姿态，网页显示目标姿态。

## 早期安防源码

人体检测：进入 `1.robot/1.code/perception/`，安装其 `requirements.txt`，调整 `config.yaml`，再运行 `python run_detect.py --source webcam`。首次运行可能下载权重。

人脸识别：进入 `1.robot/1.code/face_yolov8_arcface/`，安装依赖，再用 `python face_app.py --check` 检查摄像头。配置中的缓存和库目录已改成相对路径；请从该目录运行。模型可能在首次使用时下载，仓库不附带已录入的人脸数据。

厂商模型、完整环境和现场部署资料可通过仓库 Issues 联系作者沟通。公开代码不提供真机启动入口。

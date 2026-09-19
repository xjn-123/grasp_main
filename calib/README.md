# 标定代码分析文档索引

这一套文档把 `carm_grasp-main/examples/common/src/` 下的四个标定相关脚本，按"零基础也能看懂"的标准逐行讲了一遍。

---

## 四篇文档

| 顺序 | 文档 | 源文件 | 干什么 | 需要硬件 |
|---|---|---|---|---|
| ① | [`calib_camera/calib_camera_analysis.md`](calib_camera/calib_camera_analysis.md) | `calib_camera.py`（231 行） | **相机内参标定**：求 `fx, fy, cx, cy` + 5 个畸变系数 | ❌ 只要照片 |
| ② | [`calib_handeyes/calib_handeyes_analysis.md`](calib_handeyes/calib_handeyes_analysis.md) | `calib_handeye.py`（363 行） | **手眼标定**：求 `T_end_cam`（相机 ↔ 机械臂末端） | ❌ 只要照片 + 位姿文件 |
| ③ | [`calib_gripper/calib_gripper_analysis.md`](calib_gripper/calib_gripper_analysis.md) | `calib_gripper.py`（479 行） | **夹爪标定**：求 `T_cam_gripper`（夹爪指尖 ↔ 相机） | ✅ 机械臂 + RGB-D 相机 |
| ④ | [`arm_node/arm_node_analysis.md`](arm_node/arm_node_analysis.md) | `arm_node.py`（207 行） | **ROS 状态节点**：广播位姿/关节/夹爪/TF，键盘手动摆位 | ✅ 机械臂 |

---

## 标定顺序（不能乱）

```
① calib_camera.py
      输入：多角度拍摄的 AprilTag 标定板照片
      输出：cam_params.json      （fx,fy,cx,cy + k1..k3）
        ↓ 必须用它的结果
② calib_handeye.py
      输入：照片 + arm_pose.json + cam_params.json
      输出：calib_handeye.json   （T_end_cam）
        ↓ 必须用它的结果
③ calib_gripper.py
      输入：RGB-D 实时图像 + cam_params.json + calib_handeye.json
      输出：gripper_params.json  （T_cam_gripper）
        ↓
④ arm_node.py
      输入：calib_handeye.json + gripper_params.json
      输出：ROS 话题 + TF（不产生标定文件）
```

**每一步都依赖上一步的输出**，跳步或顺序颠倒会得到错误结果且不报错。

---

## 坐标系链（整套系统的骨架）

```
   gripper（夹爪指尖中心）
        │  T_cam_gripper   ← ③ 夹爪标定
        ▼
   camera（相机光心）
        │  T_end_cam       ← ② 手眼标定
        ▼
   end（机械臂末端法兰）
        │  T_base_end      ← 机械臂控制器实时上报
        ▼
   base（机械臂基座 = 世界原点）
```

**下标记法**：`T_A_B` = 把 B 坐标系里的点换算到 A 坐标系里。

**矩阵乘法 = 坐标系接力**，中间下标相同才能消掉：

```python
T_base_cam = T_base_end @ T_end_cam      # base←end × end←cam = base←cam ✅
T_base_end = T_base_cam @ T_end_cam      # ❌ 下标接不上，错误
```

这是检查公式对不对的万能方法。

---

## 三个最容易踩的坑（跨文档通用）

### 1. 四元数顺序不一致

不同库用不同的四元数排列：

| 来源 | 顺序 |
|---|---|
| `arm_pose.json` / ROS / OpenCV | `[qx, qy, qz, qw]`（实部在**后**） |
| `transforms3d` / scipy | `[qw, qx, qy, qz]`（实部在**前**） |
| `calib_handeye.json` | `[qw, qx, qy, qz]`（文件里写了 `QuaternionFormat`） |

代码里到处都有重排操作（`q = [pose[6], pose[3], pose[4], pose[5]]`），就是在做这件事。**弄反了旋转结果完全错误，且不报错。**

### 2. numpy 与 OpenCV 的维度顺序相反

```python
img.shape[0]   # 高（行）
img.shape[1]   # 宽（列）

cv2 的 imageSize 参数要的是 (宽, 高)   ← 和 numpy 相反！
```

写反了 `cx` 和 `cy` 会互换，程序不报错但结果是错的。

### 3. 标定板尺寸填错

`--calib_board_info "[0.0245, 0.0075, 6, 6]"` 里的数字是**米**。用尺子量实际板子核对。

单位错了会让所有距离计算系统性偏移（比如差 10 倍），而且**所有自检指标看起来都正常**——这是最隐蔽的一类错误。

---

## 各步骤的"合格线"

| 步骤 | 看什么指标 | 合格标准 |
|---|---|---|
| ① 相机内参 | 日志里的 `RMS error` | < 0.5 px 优秀，< 1.0 px 可用 |
| ① 相机内参 | `cx ≈ 宽/2`、`cy ≈ 高/2`、`fx ≈ fy` | 偏差应在 5% 内 |
| ② 手眼 | 日志里的 `rotation(deg)` / `translation(mm)` | < 1° / < 5 mm |
| ③ 夹爪 | 日志里的 `inliers ratio` | > 0.8（平面拟合质量） |
| ③ 夹爪 | `T_cam_gripper` 的 Z 平移 | 5~20 cm（爪子伸出的长度） |
| ④ ROS 节点 | `ros2 topic hz /arm_pose` | ≈ 14 Hz（受 `spin_once(0.05)` 限制） |

---

## 采集数据的通用原则

三个标定里，**采集质量比代码正确性更决定成败**：

| 原则 | 说明 |
|---|---|
| **相机内参** | 15~30 张，覆盖画面四角和中心，有远有近，姿态多样 |
| **手眼（最关键）** | **姿态必须变化**！只平移不旋转会导致 AX=XB 退化，结果完全错误。每次建议 15°~45° 的不同轴旋转 |
| **夹爪** | 先按 `a` 把末端摆正朝下；换 2~3 个姿态各标一次，对比结果差异应 < 5 mm |

---

## 通用排查顺序

结果不对时，按这个顺序查（**九成不是代码的问题**）：

```
1. 打开被注释掉的可视化代码（各脚本里都有），看 tag 认全了没、框得准不准
2. 用尺子重新量标定板，核对 --calib_board_info
3. 看自检指标（RMS / 误差 / inliers ratio）
4. 检查中间结果的量级是否合理（平移是几厘米还是几米？）
5. 检查标定顺序，确认用的标定文件是这次刚生成的
6. 重新采集数据
```

---

## 源文件位置

所有被分析的脚本都在：

```
C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\examples\common\src\
├── calib_camera.py
├── calib_handeye.py      ← 注意是单数，不是 calib_handeyes
├── calib_gripper.py
└── arm_node.py
```

依赖的工程模块：

```
C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\core\
├── utils.py         → 颜色常量、KeyboardReader、read_cam_params、inv_tf、read_calib_handeye
├── arm_wrapper.py   → ArmWrapper（机械臂硬件封装）
├── arm_utils.py     → GripperBody、compute_axis_aligned_pose、CollisionDetector
├── arm_ros_utils.py → ArmNode、pose_to_transform_stamped
├── cam_ros_utils.py → CamNode
└── vision_utils.py  → depth_mean_filter、compute_locate_error
```

> ⚠️ `apriltag2` 不在工程目录内，是外部依赖（`pip` 安装或另行提供），四个脚本都靠它做 AprilTag 检测。

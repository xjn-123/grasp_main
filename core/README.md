# core 模块分析文档索引

这一套文档把 `carm_grasp-main/core/` 下的公共模块，按"零基础也能看懂"的标准逐行讲了一遍。

---

## 五篇文档

| 文档 | 源文件 | 干什么 | 关键词 |
|---|---|---|---|
| [`utils/utils_analysis.md`](utils/utils_analysis.md) | `utils.py`（371 行） | 通用工具：读标定文件、位姿求逆、键盘读取 | `inv_tf`、`read_calib_handeye`、`KeyboardReader` |
| [`vision_utils/vision_utils_analysis.md`](vision_utils/vision_utils_analysis.md) | `vision_utils.py`（876 行） | 视觉核心：点云、去畸变、AprilTag 2D/3D 定位 | `TagMatcher2D/3D`、平面拟合、射线求交 |
| [`arm_utils/arm_utils_analysis.md`](arm_utils/arm_utils_analysis.md) | `arm_utils.py`（529 行） | 夹爪几何 + 碰撞检测 + 姿态对齐 | `GripperBody`、`CollisionDetector`、`compute_axis_aligned_pose` |
| [`arm_ros_utils/arm_ros_utils_analysis.md`](arm_ros_utils/arm_ros_utils_analysis.md) | `arm_ros_utils.py`（353 行） | 机械臂的 ROS2 发布（给 rviz 看） | `ArmNode`、`TargetArmNode`、`Marker` |
| [`cam_ros_utils/cam_ros_utils_analysis.md`](cam_ros_utils/cam_ros_utils_analysis.md) | `cam_ros_utils.py`（278 行） | 相机的 ROS2 订阅 + 多图时间同步 | `CamNode`、`ApproximateTimeSynchronizer` |

**按要求本次未分析**：`arm_wrapper.py`（机械臂硬件封装，361 行）和空的 `__init__.py`。需要的话可以补一篇。

---

## 模块依赖关系

```
              ┌──────────────┐
              │   utils.py   │  ← 谁都依赖它（颜色常量、inv_tf、读标定文件）
              └──────┬───────┘
        ┌────────────┼────────────┬─────────────┐
        ▼            ▼            ▼             ▼
  vision_utils   arm_utils   arm_ros_utils  cam_ros_utils
        ▲            │            │             │
        └────────────┘            │             │
     （arm_utils import           │             │
      rgbd_to_point_cloud）       │             │
                                  ▼             ▼
                             （都被 examples/ 下的脚本直接使用）
```

`arm_wrapper.py` 只依赖 `utils.py`，被所有需要动机械臂的脚本直接用。

---

## 三条贯穿全部模块的约定

### 1. 位姿记法 `T_a_b`

```
T_a_b  =  "把 b 坐标系里的点，换算到 a 坐标系里"  =  "从 b 到 a"
```

**矩阵乘法 = 坐标系接力**，中间下标相同才消得掉：

```python
T_base_cam = T_base_end @ T_end_cam      # base←end × end←cam = base←cam ✅
T_base_cam = T_end_cam  @ T_base_end     # ❌ 下标接不上
```

这是检查任何公式对不对的万能方法。**注意**：`arm_utils.py` 里 `get_rects_3d` 的参数名叫 `T_target_cam`（从相机到目标），而 `GripperBody` 的属性叫 `T_cam_gripper`（从夹爪到相机）——方向相反，纯靠记法分辨。

### 2. 四元数顺序

| 来源 | 顺序 |
|---|---|
| ROS / `arm_pose.json` / 命令行 `--detect_pose` | `[qx, qy, qz, qw]`（实部在**后**） |
| `transforms3d` / `scipy` | `[qw, qx, qy, qz]`（实部在**前**） |
| `calib_handeye.json` | `[qw, qx, qy, qz]` |

`arm_wrapper.array_to_matrix` 里那行 `q = [pose[6], pose[3], pose[4], pose[5]]` 就是在做重排。**弄反了旋转完全错误且不报错。**

### 3. 图像格式约定

| 名字 | 格式 | 含义 |
|---|---|---|
| `rgb_img` / `bgr_img` | `CV_8UC3` | 3 通道 8 位彩色图（注意 RGB/BGR 之争，见下） |
| `depth_img`（输入给 matcher） | `CV_16UC1` | 原始整数深度，**未**乘 `depth_scale` |
| `depth_img`（点云函数内部） | `CV_32FC1` | 已乘 `depth_scale` 的**米**制深度 |

**RGB / BGR 冲突**：`cam_ros_utils` 用 `desired_encoding=img_msg.encoding` 原样解码，ROS 里彩色话题通常是 `rgb8` → 得到 RGB 数组；但 `vision_utils` 的参数名写的是 `bgr_img`，OpenCV 的 `imwrite` / 画彩色框也按 BGR。apriltag 检测走灰度所以不影响识别，**只影响调试图的颜色**。

---

## 各模块的"一句话职责"

```
相机拍到图  ──→  cam_ros_utils.CamNode  （收图 + 同步）
             ──→  vision_utils.TagMatcher2D/3D  （认 tag + 算位姿）
             ──→  arm_utils.check_arm_pose / CollisionDetector  （这姿势安全吗）
             ──→  arm_ros_utils.TargetArmNode  （发到 rviz 给人看）
             ──→  arm_wrapper.ArmWrapper  （真正下发运动指令）
```

---

## 跨模块最值得记住的四个坑

### 1. 同一个 AprilTag，两处建的坐标系差 45°

| 位置 | X 轴取法 |
|---|---|
| `vision_utils.compute_tag_pose`（第 376–380 行） | `corners[1] - corners[0]`（一条**边**） |
| `arm_utils.GripperBody.initialize`（第 76–77 行） | `corners[2] - corners[0]`（一条**对角线**） |

后果：**夹爪标定时的 tag 坐标系 ≠ 物体定位时的 tag 坐标系**。所以夹爪上的 AprilTag 必须**贴成菱形（转 45°）**，X 轴（夹爪张开方向）才对得准。这个要求在任何注释里都没写，是从代码推出来的。

### 2. `get_frames()` 的 "谁来 spin" 契约

`CamNode.get_frames(do_spin_once=False)` 默认不自己 spin。**如果调用它的线程没有别人在 spin 这个节点，它会一直 sleep 到超时（5 秒）然后返回 `None`**。

- `test_tmpl_grasp_2d/3d`：主线程在 spin → 子线程传默认 `False`，✅
- `create_tmpl_grasp_2d`：单线程 → 必须传 `do_spin_once=True`，✅

混用会表现为"永远取不到图"。

### 3. 去畸变映射表只按第一张图的尺寸算一次

`ImageUndistorter.undistort_img` 缓存 `initUndistortRectifyMap` 的结果，**之后换分辨率的图会静默用错尺寸**。同一进程里如果彩色图和深度对齐图分辨率不同，必踩。

### 4. Windows 上 `utils.py` 直接 import 失败

`utils.py` 顶部 `import termios, tty, select` ——这三个是 Unix 专有模块，Windows 没有（对应的是 `msvcrt`）。所以整个工程**只能在 Linux / macOS 上跑**。

---

## 逐模块"合格线"

| 模块 | 看什么 | 正常表现 |
|---|---|---|
| `vision_utils` | `compute_tag_corners3d` 日志里的 `inliers ratio` | > 0.8（平面拟合质量） |
| `vision_utils` | `plane equation` 的 D 值 | 量级 = 物体到相机的距离（米） |
| `arm_utils` | `check_collision cost time` | 几十 ms（>200 ms 说明逐像素循环太慢） |
| `arm_utils` | `Arm gripper height check` | 夹爪最低点 > -0.01 m |
| `cam_ros_utils` | `get_frames cost time` | 30~100 ms（接近 5000 ms 说明没人在 spin） |
| `utils` | `read_calib_handeye` 日志 | 打印 `T_end_cam` 而不是 `T_base_cam` |

---

## 源文件位置

```
C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\core\
├── __init__.py       （空文件，1 行）
├── utils.py          ✅ 已分析（371 行）
├── vision_utils.py   ✅ 已分析（876 行）
├── arm_utils.py      ✅ 已分析（529 行）
├── arm_ros_utils.py  ✅ 已分析（353 行）
├── cam_ros_utils.py  ✅ 已分析（278 行）
└── arm_wrapper.py    ⏸ 本次按要求跳过（361 行）
```

> ⚠️ `apriltag2` 是外部已安装依赖，仓库里没有源码，`vision_utils` 的两个 Matcher 都靠它做检测。

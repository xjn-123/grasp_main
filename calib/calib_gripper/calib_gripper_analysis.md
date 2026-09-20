# `calib_gripper.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、也没接触过机器人标定的同学。
> 源文件：`C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\examples\common\src\calib_gripper.py`（479 行）
>
> 前置依赖（建议先读）：
> - `calib_camera_analysis.md`：相机内参标定
> - `calib_handeyes_analysis.md`：手眼标定（`T_end_cam`）
>
> 本篇还会大量引用 `core/arm_utils.py` 里的 `GripperBody` 类——那个类才是夹爪标定的真正计算核心，脚本只是它的"驾驶员"。

---

## 目录

- [0. 一句话概括](#0-一句话概括)
- [1. 背景知识](#1-背景知识)
  - [1.1 为什么还要标定夹爪](#11-为什么还要标定夹爪)
  - [1.2 完整坐标系链](#12-完整坐标系链)
  - [1.3 TCP 与夹爪坐标系的定义](#13-tcp-与夹爪坐标系的定义)
  - [1.4 核心思路：贴个 AprilTag 在爪子上](#14-核心思路贴个-apriltag-在爪子上)
  - [1.5 关键几何：射线与平面求交](#15-关键几何射线与平面求交)
- [2. 整体流水线](#2-整体流水线)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 导入区](#31-导入区)
  - [3.2 `compute_corners3d()`：从 RGB-D 算角点 3D 坐标](#32-compute_corners3d从-rgb-d-算角点-3d-坐标)
  - [3.3 `GripperBody.initialize()`：由 4 个角点建立夹爪坐标系](#33-gripperbodyinitialize由-4-个角点建立夹爪坐标系)
  - [3.4 `GripperBody.get_rects_3d()`：夹爪的几何模型](#34-gripperbodyget_rects_3d夹爪的几何模型)
  - [3.5 主程序 `__main__`](#35-主程序-__main__)
- [4. 依赖的 ROS2 / 硬件模块科普](#4-依赖的-ros2--硬件模块科普)
- [5. 操作手册：怎么跑这个标定](#5-操作手册怎么跑这个标定)
- [6. 输入输出规范](#6-输入输出规范)
- [7. 坑与改进建议](#7-坑与改进建议)
- [8. 一句话总结](#8-一句话总结)

---

## 0. 一句话概括

> **相机看到了物体，但真正去抓东西的是夹爪。这两个不是同一个点——夹爪尖端在相机前面伸出去一截。这个脚本用一个贴在夹爪上的 AprilTag，把"夹爪尖端相对相机的位置和朝向"量出来。**

算出来的结果记作 **`T_cam_gripper`**（从夹爪坐标系到相机坐标系的变换）。

**输入 → 输出：**

| | 内容 |
|---|---|
| 输入 1 | `cam_params.json`：内参 + 畸变 + `depth_scale` |
| 输入 2 | `calib_handeye.json`：`T_end_cam` |
| 输入 3 | ROS2 实时图像话题（彩色 + 深度） |
| 输入 4 | 夹爪尺寸 `[width, thickness]` |
| 输入 5 | 贴在夹爪上的 AprilTag（**ID 必须是 0**） |
| 输出 | `gripper_params.json`：`{width, thickness, T_cam_gripper}` |

**和其他两个脚本的区别（重要）：**

| 脚本 | 运行方式 | 是否需要机械臂 | 是否需要 ROS |
|---|---|---|---|
| `calib_camera.py` | 一次性跑完 | ❌ | ❌ |
| `calib_handeye.py` | 一次性跑完 | ❌（只读位姿文件） | ❌ |
| **`calib_gripper.py`** | **交互式循环，键盘操作** | ✅ **必须** | ✅ **必须** |

---

## 1. 背景知识

### 1.1 为什么还要标定夹爪

前面已经标定了：

- 相机内参（`calib_camera`）：知道像素 ↔ 三维射线的关系
- 手眼（`calib_handeye`）：知道相机 ↔ 机械臂末端的关系

那还差什么？

**差"末端"到"真正夹东西的那个点"的距离。** 机械臂控制器报的位姿是**末端法兰盘**（安装面）的位置，而真正接触物体的是**夹爪指尖**，两者之间隔着爪子的长度。

如果忽略它，抓取时会发生什么？——**机械臂会把物体"怼"过去，深度差了一整个爪子的长度**（通常 5~15 厘米）。这必然撞坏东西或抓空。

### 1.2 完整坐标系链

把所有标定串起来，就是这条链：

```
   gripper（夹爪指尖中心）
        │  T_cam_gripper   ← ③ 本篇标定
        ▼
   camera（相机）
        │  T_end_cam       ← ② 手眼标定
        ▼
   end（机械臂末端法兰）
        │  T_base_end      ← 机械臂控制器实时上报
        ▼
   base（机械臂基座）
        │  T_world_base    ← 通常就是单位阵（基座即世界原点）
        ▼
   world
```

**用法举例**：相机看到物体在 `T_cam_obj`，要算末端该去哪：

```python
T_end_obj = T_end_cam @ T_cam_obj            # 物体相对末端
T_gripper_obj = inv(T_cam_gripper) @ T_cam_obj   # 物体相对夹爪指尖
```

有了 `T_cam_gripper`，才算真正打通"看见"到"抓到"。

### 1.3 TCP 与夹爪坐标系的定义

`GripperBody` 类的文档字符串（`core/arm_utils.py` 第 35-42 行）定义了四条预设：

```
1) 仅适用于眼在手的场景
2) 夹爪坐标系的原点位于两片爪尖的几何中心
3) 以夹爪张开方向为夹爪坐标系的 X 轴，从左爪指向右爪
4) 夹爪坐标系的 Z 轴垂直于夹爪平面，朝向与末端坐标系 Z 轴的夹角近似平行
```

画出来是这样（从相机侧看夹爪）：

```
              Z 轴（垂直于夹爪平面，朝向相机）
              ↑
              |
      左爪    |    右爪
     ┌────┐   |   ┌────┐
     │    │   ●   │    │      ● = 坐标系原点（两爪尖的几何中心）
     │    │  /    │    │
     └────┘ /     └────┘
            /
           ●────────────→  X 轴（张开方向，左爪 → 右爪）

   Y 轴 = X × Z 的方向（右手定则）
```

### 1.4 核心思路：贴个 AprilTag 在爪子上

**怎么量出 `T_cam_gripper`？** 用一个巧妙的办法：

1. 在夹爪上贴一张 **AprilTag 贴纸**（必须是 **ID = 0**，代码写死查找 id 0）
2. 用 RGB-D 相机拍它，测出这张 tag **四个角点的三维坐标**
3. 由这 4 个角点的位置关系，反推出一个坐标系——这个坐标系就是"夹爪坐标系"

**为什么 4 个角点能确定一个坐标系？**

一个刚体坐标系需要 6 个自由度（3 位置 + 3 姿态）。一个正方形 tag 的 4 个角点提供了：
- 4 个点的中心 → 位置（3 个自由度）
- 两条对角线的方向 → 两个方向向量 → 姿态（3 个自由度）

刚好够。

### 1.5 关键几何：射线与平面求交

这是 `compute_corners3d()` 最后一步的数学，也是全脚本最精妙的地方。

**问题**：相机只知道角点在照片上的像素位置 `(u, v)`，不知道它有多远（深度）。怎么求它的三维坐标？

**方案**：

**第一步：把像素变成一条射线。**

相机内参告诉我们，像素 `(u,v)` 对应一条从相机光心射出去的射线。射线上所有点的方向是固定的：

$$n_x = \frac{u - c_x}{f_x}, \quad n_y = \frac{v - c_y}{f_y}$$

射线上的任意一点可以写成（$z$ 是到相机的深度）：

$$P = z \cdot (n_x,\ n_y,\ 1)$$

即 $x = n_x z,\ y = n_y z,\ z = z$。**$z$ 是多少还不知道。**

**第二步：用平面方程把 $z$ 解出来。**

因为 tag 是贴在夹爪上的**平面**，它的 4 个角点都在这个平面上。我们用深度点云拟合出了这个平面的方程：

$$Ax + By + Cz + D = 0$$

把 $x = n_x z,\ y = n_y z$ 代进去：

$$A(n_x z) + B(n_y z) + C(z) + D = 0$$

$$z(A n_x + B n_y + C) = -D$$

$$\boxed{z = \frac{-D}{A n_x + B n_y + C}}$$

**就这一行公式**，对应代码第 177 行：

```python
z = -D / (A * nx + B * ny + C)
```

求出 $z$ 后，$x = n_x z,\ y = n_y z$，角点的三维坐标就完整了。

**为什么要用平面而不是直接用深度图？**

深度相机（尤其是消费级）在边缘、反光、黑色图案（AprilTag 正好是黑白的！）上噪声很大，单个像素的深度值不可靠。但**一整片点云拟合出来的平面**非常稳——这是用"面"的信息补偿"点"的噪声。

---

## 2. 整体流水线

```
 ┌────────────────────────────────────────────────────────────┐
 │  启动阶段                                                    │
 │  读 cam_params.json  → intrinsic, distortion, depth_scale   │
 │  读 calib_handeye.json → T_end_cam                          │
 │  连机械臂 ArmWrapper()                                       │
 │  起 ROS2 节点：ArmNode + CamNode + TF 广播器                 │
 │  造 GripperBody(width, thickness)                           │
 └───────────────────────────┬────────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────────┐
 │  主循环（每秒十几次）                                        │
 │  ① arm.get_pose() → T_base_end                              │
 │  ② T_base_cam = T_base_end @ T_end_cam                      │
 │  ③ 广播 TF：base → camera_link                              │
 │  ④ 发布末端位姿 / 夹爪 Marker                                │
 │  ⑤ 读键盘                                                    │
 └───────────────────────────┬────────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────────┐
 │  按键盘 't' → 执行一次夹爪标定                                │
 │                                                             │
 │  cam_node.get_frames(5) → 5 帧 RGB-D                        │
 │        ↓                                                    │
 │  深度图 5 帧均值滤波（降噪）                                  │
 │        ↓                                                    │
 │  compute_corners3d():                                       │
 │    a. cv2.undistort 去畸变                                   │
 │    b. detect → 找 ID=0 的 tag → 4 个 2D 角点                │
 │    c. 画可视化（人工确认角点顺序）                            │
 │    d. 角点向外扩 3 倍 → 掩码                                 │
 │    e. 掩码 + 深度 → 点云                                     │
 │    f. RANSAC 平面拟合 → 平面方程 (A,B,C,D)                   │
 │    g. 射线 ∩ 平面 → 4 个角点的 3D 坐标                       │
 │        ↓                                                    │
 │  gb.initialize(corners3d) → 算出 T_cam_gripper              │
 └───────────────────────────┬────────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────────┐
 │  按键盘 's' → mmengine.dump 存 gripper_params.json          │
 └────────────────────────────────────────────────────────────┘
```

---

## 3. 逐段代码精读

### 3.1 导入区（第 6–42 行）

```python
import argparse, json, logging, os, sys

import apriltag2         # AprilTag 检测
import cv2               # 图像处理
import mmengine          # 通用配置/文件读写（支持 json/yaml/pkl）
import numpy as np
import open3d            # 点云处理（平面拟合）
import rclpy             # ROS2 Python 接口

code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{code_dir}/../../../")
sys.path.append(root_dir)

from core.arm_ros_utils import ArmNode, pose_to_transform_stamped
from core.arm_utils import GripperBody, compute_axis_aligned_pose
from core.arm_wrapper import ArmWrapper
from core.cam_ros_utils import CamNode
from core.utils import (BLUE, GREEN, RED, RESET,
                        KeyboardReader, read_calib_handeye, read_rgbd_params)
from core.vision_utils import depth_mean_filter
from tf2_ros import TransformBroadcaster
from typing_extensions import Tuple
```

**新增的库：**

| 库 | 干什么 |
|---|---|
| `open3d` | 点云库。这里只用了一个功能：`segment_plane`（RANSAC 平面拟合） |
| `rclpy` | ROS2 的 Python 客户端。创建节点、发布消息 |
| `tf2_ros` | ROS 的 TF2 坐标变换库。这里用 `TransformBroadcaster` 广播坐标系关系 |
| `mmengine` | 微软/MMLab 的配置库。这里当"万能文件读写器"用，按后缀自动选择 json/yaml 格式 |

**新增的工程模块：**

| 模块 | 提供什么 |
|---|---|
| `core.arm_ros_utils` | `ArmNode`（ROS 节点类：发布位姿/关节/夹爪 Marker）、`pose_to_transform_stamped`（矩阵转 TF 消息） |
| `core.arm_utils` | `GripperBody`（夹爪几何模型，**核心**）、`compute_axis_aligned_pose`（姿态对齐） |
| `core.arm_wrapper` | `ArmWrapper`（机械臂硬件封装） |
| `core.cam_ros_utils` | `CamNode`（ROS 相机节点，同步接收 RGB-D） |
| `core.vision_utils` | `depth_mean_filter`（多帧深度均值滤波） |

### 3.2 `compute_corners3d()`：从 RGB-D 算角点 3D 坐标（第 50–188 行）

**这是全脚本技术含量最高的函数。** 签名：

```python
def compute_corners3d(
    gray_img: np.ndarray,     # 灰度图
    depth_img: np.ndarray,    # 深度图（CV_16UC1，单位是"深度计数值"）
    intrinsic: np.ndarray,    # [fx, fy, cx, cy]
    distortion: np.ndarray,   # [k1,k2,p1,p2,k3] 或 None
    depth_scale: float,       # 深度计数值 → 米 的换算系数
) -> Tuple[np.ndarray, np.ndarray]:
```

> ⚠️ **类型标注 bug**：标注说返回两个数组，但函数实际只返回**一个** `corners3d`（第 188 行），失败时返回 `None`（第 74、87 行）。应改为 `-> np.ndarray`。调用方写法 `corners3d = compute_corners3d(...)` 是对的，只是标注写错了。

#### 第 1 步：去畸变（第 58–64 行）

```python
K = np.array(
    [[intrinsic[0], 0, intrinsic[2]],
     [0, intrinsic[1], intrinsic[3]],
     [0, 0, 1]],
    dtype=np.float32,
)
D = np.array(distortion, dtype=np.float32) if distortion is not None else None
un_img = cv2.undistort(gray_img, K, D)
```

把 `[fx,fy,cx,cy]` 组装成 3×3 矩阵，然后 `cv2.undistort` 把扭曲的图像"掰直"。

**为什么这里要 undistort，而 `calib_handeye.py` 不用？**

见手眼标定文档 3.5 节。简单说：这里后面要用**深度图**做几何计算，深度图的每个像素和彩色图一一对应，如果彩色图有畸变，角点的像素位置和深度图对不上。先掰直最省事。

#### 第 2 步：检测 ID=0 的 AprilTag（第 69–88 行）

```python
detector = apriltag2.Detector(tag_family="tag36h11", black_border=2)
tags = detector.detect(un_img)
if len(tags) == 0:
    logging.error("No AprilTag detected.")
    return None

tag = None
for t in tags:
    if t.id == 0:
        tag = t
        break

if tag is None:
    logging.error("AprilTag ID 0 not detected.")
    return None
```

**语法点：** `for ... break` 是"找到就停"的标准写法。`break` 立刻跳出循环。

**业务含义：** 画面里可以有很多 tag，但**只认 ID=0 这一个**。

> 📌 **实操要求**：你贴在夹爪上的那张 AprilTag，**必须是 tag36h11 家族里编号为 0 的那一张**。如果贴错了编号，这里会一直报 "AprilTag ID 0 not detected."

> ⚠️ **性能小问题**：`Detector` 在**函数内部**创建，而这个函数每次按 `t` 都会调用一次。检测器初始化要生成模板（耗时），应该像 `calib_camera.py` 那样在循环外创建一次。交互式场景影响不大，但不规范。

#### 第 3 步：画可视化（第 93–125 行）

```python
vis_img = cv2.cvtColor(un_img, cv2.COLOR_GRAY2BGR)
pt0 = (int(corners[0][0]), int(corners[0][1]))
pt1 = (int(corners[1][0]), int(corners[1][1]))
pt2 = (int(corners[2][0]), int(corners[2][1]))
pt3 = (int(corners[3][0]), int(corners[3][1]))
cv2.line(vis_img, pt0, pt1, (0, 0, 255), 1)      # 红
cv2.line(vis_img, pt1, pt2, (255, 0, 0), 1)      # 蓝
cv2.line(vis_img, pt2, pt3, (255, 0, 0), 1)      # 蓝
cv2.line(vis_img, pt3, pt0, (0, 255, 0), 1)      # 绿
cv2.rectangle(vis_img, (int(pt0[0]-5), int(pt0[1]-5)),
                       (int(pt0[0]+5), int(pt0[1]+5)), (0, 255, 255), 1)   # 黄框
cv2.rectangle(vis_img, (int(pt2[0]-5), int(pt2[1]-5)),
                       (int(pt2[0]+5), int(pt2[1]+5)), (0, 255, 255), 1)   # 黄框
```

**这一段极其重要，但很多人会忽略它。**

四条边画了**三种不同的颜色**，两个角点（0 和 2）画了黄色小方块。这是**故意的**——让你能用肉眼确认角点的排列顺序：

```
     pt0 ●━━━━━━━━● pt1
         ┃        ┃          红边：pt0 → pt1
      绿 ┃        ┃ 蓝       蓝边：pt1 → pt2, pt2 → pt3
         ┃        ┃          绿边：pt3 → pt0
     pt3 ●━━━━━━━━● pt2
         ↑        ↑
       黄框      黄框
```

**为什么要确认？** 因为下一步建立坐标系时，用的就是"从 pt0 指向 pt2"和"从 pt3 指向 pt1"这两条对角线的方向。**如果顺序理解错了，整个夹爪坐标系就转错方向了。**

**OpenCV 颜色格式提醒**：`(B, G, R)` 不是 RGB！

- `(0, 0, 255)` = **红**（B=0, G=0, R=255）
- `(0, 255, 0)` = 绿
- `(255, 0, 0)` = 蓝
- `(0, 255, 255)` = 黄（绿+红）

这是 OpenCV 的经典坑（历史原因，早期摄像头数据是 BGR 顺序）。

```python
import platform
if platform.machine() == "x86_64":
    cv2.imshow("tag detection", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

**为什么要判断架构？** 因为实际部署在 **ARM 工控机**（`aarch64`）上，那台机器没接显示器，`cv2.imshow` 会直接崩溃。所以只在 x86_64（开发机）上显示。

- `cv2.waitKey(0)`：参数是 0 表示**无限等待**，直到你按任意键才继续。所以标定时会弹窗让你确认，看完按一下键继续。
- `cv2.destroyAllWindows()`：关掉窗口，防止堆积。

> ⚠️ **`import platform` 写在函数内部**（第 119 行）不是好习惯，应该放到文件顶部。功能上没问题（Python 会缓存已导入的模块），但可读性差。

#### 第 4 步：角点向外扩张 3 倍做掩码（第 127–144 行）

```python
center = tag.center
expanded_corners = []
for corner in corners:
    vec = corner - center
    expanded_corner = center + vec * 3.0
    expanded_corners.append(expanded_corner)
expanded_corners = np.array(expanded_corners)

mask = np.zeros_like(gray_img, dtype=np.uint8)
pts = expanded_corners.astype(np.int32)
cv2.fillConvexPoly(mask, pts, 255)

masked_depth = depth_img.copy()
masked_depth[mask == 0] = 0
```

**这一步在干什么？** 把 tag 的四个角点**以中心为原点向外放大 3 倍**，得到一个比 tag 大的四边形，用它做掩码，从深度图里圈出一片区域。

```
    ┌─────────────────┐
    │   ┌───────┐     │   ← 外层是"扩张 3 倍"的四边形（掩码区域）
    │   │  tag  │     │   ← 内层是 tag 本身
    │   └───────┘     │
    └─────────────────┘
```

**为什么要放大 3 倍？** 因为要拟合平面，需要**足够多的点**。tag 本身面积小，深度图在黑色图案上又经常测不到（红外被黑墨吸收），点数太少拟合不稳。放大到 3 倍能圈进周围的平面区域（夹爪本体表面），点多拟合才稳。

**语法点：布尔索引（boolean indexing）**

```python
masked_depth[mask == 0] = 0
```

这是 numpy 最强大的特性之一。`mask == 0` 产生一个同形状的 True/False 数组，然后用它当索引，**把所有 True 位置的元素一次性赋值为 0**。等价于：

```python
for v in range(H):
    for u in range(W):
        if mask[v, u] == 0:
            masked_depth[v, u] = 0
```

一行顶十行，而且快几十倍。

> ⚠️ **注意**：扩张后的角点**只用来做掩码**，最后算 `corners3d` 用的是**原始 corners**（第 184-186 行）。这点很容易看错。

#### 第 5 步：深度图转点云（第 146–160 行）

```python
pc = open3d.geometry.PointCloud.create_from_depth_image(
    open3d.geometry.Image(masked_depth),
    open3d.camera.PinholeCameraIntrinsic(
        gray_img.shape[1],    # 宽
        gray_img.shape[0],    # 高
        intrinsic[0],         # fx
        intrinsic[1],         # fy
        intrinsic[2],         # cx
        intrinsic[3],         # cy
    ),
    np.eye(4),
    depth_scale=1.0 / depth_scale,
    depth_trunc=0.5,
)
```

用内参把深度图"炸开"成三维点云：每个像素变成一个 3D 点。

**两个容易搞错的参数：**

**① `depth_scale=1.0 / depth_scale`**

`cam_params.json` 里的 `depth_scale` 是"**1 个深度计数值 = 多少米**"（比如 0.0001 表示计数值 10000 = 1 米）。

而 Open3D 的 `depth_scale` 参数是"**深度值要乘上多少才变成米**"——嗯，其实两个含义一样。等等，如果一样为什么要取倒数？

看 `core/vision_utils.py` 里的用法（第 395 行）：`depth_img.astype(np.float32) * self.depth_scale`，即**深度值 × depth_scale = 米**。所以工程里 `depth_scale` 确实是"米/计数值"。

而 Open3D 文档里 `depth_scale` 的含义是"深度值除以它得到米"（即 depth / depth_scale = 米，单位 mm 时 depth_scale=1000）。所以这里传 `1.0/depth_scale` 是把两种约定对接起来。

**结论**：这行是**单位约定转换**，不是 bug，但极易看错，改代码时要小心。

**② `depth_trunc=0.5`**

只保留 0.5 米以内的点，更远的丢弃。

> ⚠️ **这是个硬编码的隐患**：如果夹爪离相机超过 0.5 米（比如大臂展工况、或者相机装得远），点云会全空，平面拟合必然失败。建议做成命令行参数。

#### 第 6 步：RANSAC 平面拟合（第 162–168 行）

```python
plane, inliers = pc.segment_plane(
    distance_threshold=0.002,   # 2mm
    ransac_n=6,                 # 每次随机取 6 个点
    num_iterations=1000,
)
logging.info(f"plane equation: {plane}, inliers count: {len(inliers)}, "
             f"inliers ratio: {len(inliers) / len(pc.points)}")
```

**RANSAC 是什么？**（Random Sample Consensus，随机采样一致性）

用大白话讲：

1. 随机抓 3 个点（这里 `ransac_n=6` 是抓 6 个），算一个平面
2. 数一数有多少点落在这个平面附近（2mm 以内）→ 这些叫**内点（inliers）**
3. 重复 1000 次，选**内点最多**的那个平面

**为什么要用 RANSAC 而不是直接最小二乘？** 因为深度图里有**离群点（噪点、飞点）**，最小二乘会被这些点拽偏，RANSAC 能无视它们。

**返回的 `plane` 是 4 个数 `[A, B, C, D]`**，对应平面方程：

$$Ax + By + Cz + D = 0$$

**日志里的 `inliers ratio` 是重要的健康指标**：

| 内点比例 | 说明 |
|---|---|
| > 0.8 | 很好的平面 |
| 0.5 ~ 0.8 | 还行 |
| < 0.5 | ⚠️ 掩码区域里混进了别的物体 / 深度图太烂 → 标定不可靠 |

#### 第 7 步：射线与平面求交，算角点 3D 坐标（第 170–188 行）

```python
def compute_pt3d(corner, intrinsic, plane) -> np.ndarray:
    nx = (corner[0] - intrinsic[2]) / intrinsic[0]     # (u - cx) / fx
    ny = (corner[1] - intrinsic[3]) / intrinsic[1]     # (v - cy) / fy
    A, B, C, D = plane
    z = -D / (A * nx + B * ny + C)
    x = nx * z
    y = ny * z
    return np.array([x, y, z])

corners3d = np.array([compute_pt3d(corner, intrinsic, plane) for corner in corners])
```

这就是 [1.5 节](#15-关键几何射线与平面求交) 推导的那个公式，一行不少地实现了。

**语法点：函数内定义函数 + 列表推导式**

```python
[compute_pt3d(corner, intrinsic, plane) for corner in corners]
```

对 `corners` 的每个元素调用一次 `compute_pt3d`，把结果收集成列表。结果形状 `(4, 3)`。

**业务含义**：4 个角点的 3D 坐标算出来了，单位**米**，坐标系是**相机坐标系**（相机光心为原点，Z 轴指向前方）。

### 3.3 `GripperBody.initialize()`：由 4 个角点建立夹爪坐标系（core/arm_utils.py 第 61–99 行）

**这段不在脚本里，但它是夹爪标定的真正计算核心，必须理解。**

```python
def initialize(self, corners3d: np.ndarray):
    assert corners3d.shape == (4, 3), "corners3d must have shape (4, 3)"

    # 坐标系定义：
    # 从 corners3d[0] 指向 corners3d[2] 的方向为 X 轴
    # 从 corner[3] 指向 corner[1] 的方向为 Y 轴
    # Z 轴为平面的法向量，由右手定则确定
    # 原点为 corners3d[0] 和 corners3d[2] 的中点

    center_3d = corners3d.mean(axis=0)          # 更鲁棒的中心
    edge_x = corners3d[2] - corners3d[0]        # 对角线 1
    edge_y = corners3d[1] - corners3d[3]        # 对角线 2
    if np.linalg.norm(edge_x) < 1e-9 or np.linalg.norm(edge_y) < 1e-9:
        raise ValueError("角点退化,无法建立坐标系")

    axis_x = edge_x / np.linalg.norm(edge_x)                       # 归一化

    axis_y_raw = edge_y / np.linalg.norm(edge_y)
    axis_y = axis_y_raw - np.dot(axis_y_raw, axis_x) * axis_x      # 正交化
    axis_y /= np.linalg.norm(axis_y)

    axis_z = np.cross(axis_x, axis_y)                              # 叉乘
    axis_z /= np.linalg.norm(axis_z)
    assert axis_z[2] > 0, "gripper Z axis direction error"

    self.T_cam_gripper[:3, :3] = np.column_stack((axis_x, axis_y, axis_z))
    self.T_cam_gripper[:3, 3] = center_3d.reshape(3)
    logging.info(f'Gripper pose set. T_cam_gripper:\n{self.T_cam_gripper}')
```

**逐步拆解：**

**① 画个图看清楚用的是哪两条对角线**

```
     c[0] ●━━━━━━━━● c[1]
          ┃        ┃
          ┃        ┃          X 轴方向 = c[2] - c[0] （左上 → 右下）
          ┃        ┃          Y 轴方向 = c[1] - c[3] （左下 → 右上）
     c[3] ●━━━━━━━━● c[2]
```

**注意用的是对角线，不是边！** 对于正方形 tag，两条对角线恰好互相垂直（长度相等、夹角 90°），所以天然适合当正交基。

> 📌 **这意味着贴 tag 时有朝向要求**：配合 `GripperBody` 的定义"X 轴 = 夹爪张开方向"，你贴 tag 时要让 **tag 的对角线方向**沿着**夹爪张开方向**。也就是 tag 要**菱形朝上**（转 45°）贴，而不是边水平地贴。
>
> 这是文档里没写、但从代码能推出来的实操要求。**如果你把 tag 正着贴（边水平），X 轴会歪 45°，导致后续碰撞检测/抓取点的矩形全错。**

**② `corners3d.mean(axis=0)`：4 个角点的平均**

`axis=0` 表示"沿着第 0 维（行）求平均"，结果是把 4 个 (3,) 的点平均成 1 个 (3,) 的中心点。等价于 `(c0+c1+c2+c3)/4`。

> ⚠️ **注释与代码不一致**：上面注释写"原点为 corners3d[0] 和 corners3d[2] 的中点"，但代码用的是**4 点均值** `mean(axis=0)`。4 点均值其实**更鲁棒**（对单个角点的噪声不敏感），代码是对的，是注释没跟着更新。

**③ Gram-Schmidt 正交化**

```python
axis_y = axis_y_raw - np.dot(axis_y_raw, axis_x) * axis_x
```

理论和上面是：两条对角线应该垂直，但**实测数据有噪声**，不可能严格垂直。所以要强制正交化：

$$\hat{y} = y - (y \cdot \hat{x})\hat{x}$$

几何含义：**把 y 里"掺入 x 方向的那一份"减掉**，剩下的部分就严格垂直于 x 了。

```
       y_raw
        ↗
       /     ← 减去在 x 上的投影分量
      /        
     ●━━━━━━━━━→ x
     
     结果：axis_y 严格垂直于 axis_x
```

**④ 叉乘得 Z 轴**

```python
axis_z = np.cross(axis_x, axis_y)
```

叉乘 $x \times y$ 得到同时垂直于 x 和 y 的向量，方向由**右手定则**决定。

```
      z ↑
        |
        |      y
        |    ↗
        |  /
        |/──────→ x
```

**⑤ `assert axis_z[2] > 0`**

要求 Z 轴的**第 3 个分量（相机坐标系的 Z 方向）为正**，即夹爪平面的法向量要**朝向相机**（不能背对）。

如果 tag 贴反了（或者角点顺序反了），这里会直接 `AssertionError` 崩溃。这是**保护性检查**，很好。

> 💡 如果实际运行中遇到 `gripper Z axis direction error`，检查：tag 是不是贴反了？是不是从背面拍的？

**⑥ 组装成矩阵**

```python
self.T_cam_gripper[:3, :3] = np.column_stack((axis_x, axis_y, axis_z))
self.T_cam_gripper[:3, 3] = center_3d.reshape(3)
```

`np.column_stack((a, b, c))` 把三个 (3,) 的向量**按列**拼成 3×3 矩阵。结果：

```
T_cam_gripper = [ ax_x  ay_x  az_x  |  cx ]
                [ ax_y  ay_y  az_y  |  cy ]
                [ ax_z  ay_z  az_z  |  cz ]
                [  0     0     0    |  1  ]
                  └─ 旋转（3个轴向量）─┘   └平移┘
```

**这就是最终标定的 `T_cam_gripper`。**

### 3.4 `GripperBody.get_rects_3d()`：夹爪的几何模型（core/arm_utils.py 第 101–134 行）

标定完之后，夹爪要参与**碰撞检测**和**可视化**，所以需要一个几何模型。

```python
def get_rects_3d(self, dist: float, T_target_cam=np.eye(4)) -> np.ndarray:
    w = self.width         # 夹爪宽度
    t = self.thickness     # 夹爪厚度
    hd = dist / 2.0        # 半间距
    hw = w / 2.0           # 半宽

    left_rect = np.array([[-hd - t, hw, 0, 1],
                          [-hd,     hw, 0, 1],
                          [-hd,    -hw, 0, 1],
                          [-hd - t,-hw, 0, 1]], dtype=np.float32).T   # (4,4)

    right_rect = np.array([[hd,      hw, 0, 1],
                           [hd + t,  hw, 0, 1],
                           [hd + t, -hw, 0, 1],
                           [hd,     -hw, 0, 1]], dtype=np.float32).T  # (4,4)

    T_target_gripper = T_target_cam @ self.T_cam_gripper
    left_rect_3d = (T_target_gripper @ left_rect).T[:, :3]
    right_rect_3d = (T_target_gripper @ right_rect).T[:, :3]
    rects_3d = np.vstack((left_rect_3d, right_rect_3d))   # (8,3)
    return rects_3d
```

**在夹爪坐标系下，两片爪子是两个矩形：**

```
   X 轴（张开方向）→
   
   ┌────┐                    ┌────┐
   │ 左 │                    │ 右 │      hd = dist/2（半间距）
   │ 爪 │                    │ 爪 │      t  = thickness（厚度）
   └────┘                    └────┘      hw = width/2（半宽）
   ↑                              ↑
 -hd-t      -hd        0        hd      hd+t
   
        ←── 间距 dist ──→
              ● 原点
```

**语法点：齐次坐标 + 转置技巧**

```python
left_rect = np.array([[x1,y1,z1,1],        # 4 行
                      [x2,y2,z2,1],
                      [x3,y3,z3,1],
                      [x4,y4,z4,1]]).T     # 转置成 4 列
```

- 每个点写成 `[x, y, z, 1]` —— 末尾补 1 叫**齐次坐标**，这样才能用 4×4 矩阵一次性做"旋转+平移"
- `.T` 转置成 (4,4)，让每个点变成一列，这样 `T @ left_rect` 能一次变换 4 个点
- 变换后再 `.T` 转回来，取 `[:, :3]` 丢掉末尾的 1，得到 (4,3)

**这个"列向量 + 转置"技巧在图形学里到处都是，值得记住。**

业务含义：`dist` 是夹爪当前张开的距离（实时从机械臂读），所以这个矩形的大小是动态的。

### 3.5 主程序 `__main__`（第 199–479 行）

#### 参数定义（第 200–284 行）

```python
parser.add_argument("--cam_params_path", ...)
parser.add_argument("--calib_handeye_path", ...)
parser.add_argument("--gripper_path", ...)
parser.add_argument("--color_img_topic", default="/gemini305/color/image_raw", ...)
parser.add_argument("--depth_img_topic", default="/gemini305/depth/image_raw", ...)
parser.add_argument("--pc_frame_id", default="camera_link", ...)
parser.add_argument("--gripper_size", default="[0.0015,0.002]", ...)
```

| 参数 | 含义 |
|---|---|
| `--color_img_topic` | 彩色图的 ROS 话题名（这里默认是 **Gemini 305** 相机） |
| `--depth_img_topic` | 深度图的 ROS 话题名 |
| `--pc_frame_id` | 点云所在的坐标系名，默认 `camera_link`，用于 TF 广播 |
| `--gripper_size` | `[width, thickness]`，单位**米**。默认 `[0.0015, 0.002]` = 宽 1.5mm、厚 2mm |

> ⚠️ 默认 `[0.0015, 0.002]` 是 1.5 毫米宽、2 毫米厚的爪子——这非常细，像针一样。**大概率是作者测试用的占位值，实际夹爪要改**。用卡尺量一下你的爪子。

#### 参数校验（第 254–269 行）

```python
color_img_topic = args.color_img_topic
if color_img_topic is None:
    logging.error("Error: color_img_topic is not provided.")
    exit(0)
```

> ⚠️ **这段校验是无效的**：因为这些参数都有 `default`，`args.color_img_topic` 永远不可能是 `None`（除非用户显式传 `--color_img_topic=None`）。所以这三个 `if` 永远不触发。属于**防御性冗余代码**。

> ⚠️ 另外 `exit(0)` 用的是 Python 内置的 `exit()`（交互式解释器用的），正式代码应该用 `sys.exit(1)`。其他地方（第 301 行）用的是 `exit(1)`，不一致。

#### 加载已有标定（第 316–350 行）

```python
calib_data = {}
if os.path.exists(gripper_path):
    calib_data = mmengine.load(gripper_path)
    logging.info(f"Loaded existing gripper calib data from: {GREEN}{gripper_path}{RESET}")

if "width" in calib_data:
    saved_width = calib_data["width"]
    saved_thickness = calib_data["thickness"]
    if abs(saved_width - gripper_width) > 1e-6 or abs(saved_thickness - gripper_thickness) > 1e-6:
        logging.warning(f"Loaded gripper_size [{saved_width}, {saved_thickness}] is different from "
                        f"current setting [{gripper_width}, {gripper_thickness}], use loaded values.")
else:
    calib_data["width"] = gripper_width
    calib_data["thickness"] = gripper_thickness

if "T_cam_gripper" in calib_data:
    gb.T_cam_gripper = np.array(calib_data["T_cam_gripper"], dtype=np.float32)
    logging.info(f"Loaded T_cam_gripper: \n{GREEN}{gb.T_cam_gripper}{RESET}")
else:
    calib_data["T_cam_gripper"] = np.eye(4).tolist()
    logging.warning("No T_cam_gripper found in loaded calib_data, use identity matrix as default.")
```

**业务含义**：支持**增量标定**——上次标过就加载出来，这次在此基础上继续。

> ⚠️ **发现一个逻辑不一致**：警告文字说 "use loaded values"（使用已加载的值），但代码**并没有把 `gripper_width` 改成加载的值**！而 `gb` 对象在第 314 行已经用命令行参数构造好了。所以实际生效的是**命令行传入的值**，警告文字是误导的。
>
> 要么改成真的用加载值（`gripper_width = saved_width; gb.width = saved_width`），要么改警告文字为 "use command line values"。

#### 初始化 ROS2 与硬件（第 297–314 行）

```python
arm = ArmWrapper()
if not arm.is_connected():
    logging.error(f"{RED}failed to connect to arm, exiting {RESET}")
    exit(1)

rclpy.init(args=None)
arm_node = ArmNode(pub_gripper_msg=True)
cam_node = CamNode(img_topic_list=[color_img_topic, depth_img_topic])
tf_broadcaster = TransformBroadcaster(arm_node)

keyboard_reader = KeyboardReader()

gb = GripperBody(width=gripper_width, thickness=gripper_thickness)
```

- **`ArmWrapper()`**：连机械臂。默认 IP `10.42.0.101`；如果在 ARM 工控机上（aarch64）会自动改用 `127.0.0.1`（本地）。
- **`rclpy.init(args=None)`**：初始化 ROS2 运行时，必须最先调用
- **`ArmNode(pub_gripper_msg=True)`**：创建发布夹爪 Marker 的节点
- **`CamNode([color, depth])`**：订阅两个话题，用 `ApproximateTimeSynchronizer` 做**时间同步**（允许 50ms 误差），保证彩色图和深度图是同一时刻的
- **`TransformBroadcaster(arm_node)`**：TF 广播器
- **`KeyboardReader()`**：非阻塞键盘读取（见下）

#### 主循环（第 364–472 行）

```python
while rclpy.ok():
    rclpy.spin_once(arm_node, timeout_sec=0.005)

    T_base_end = arm.get_pose()
    gripper_dist = arm.get_gripper_dist()

    T_base_cam = T_base_end @ T_end_cam
    ts = pose_to_transform_stamped(arm_node.frame_id, pc_frame_id, T_base_cam)
    ts.header.stamp = arm_node.get_clock().now().to_msg()
    tf_broadcaster.sendTransform(ts)

    arm_node.publish_pose(T_base_end)
    arm_node.publish_grippers(gripper_body=gb, gripper_dist=gripper_dist,
                              T_base_end=T_base_end, T_end_cam=T_end_cam)

    key = keyboard_reader.read_key()
    if key is None:
        continue
    ...
```

**每帧做四件事：**

**① 广播 TF：`base → camera_link`**

```python
T_base_cam = T_base_end @ T_end_cam
```

把相机位姿算出来（末端位姿 × 手眼矩阵），广播到 ROS 的 TF 树里。

**这有什么用？** 相机发布的点云带 `frame_id=camera_link`。有了这个 TF，RViz 和其他程序就能**自动把点云从相机坐标系转换到基座坐标系**显示/计算。这是 ROS 的核心机制。

**② 发布末端位姿**（`/arm_pose` 话题）

**③ 发布夹爪 Marker**（`/grippers` 话题，绿色半透明长方体）

**④ 读键盘并处理**

#### 键盘命令（第 390–470 行）

帮助文字（第 353–362 行）：

```
  q: 退出程序
  a: 调整末端姿态，使末端坐标系的 Z 轴指向下方
  t: 从当前 RGB-D 图像计算夹爪在相机坐标系下的位姿
  <: 缩小夹爪
  >: 放大夹爪
  s: 保存标定结果到文件
```

| 键 | 代码位置 | 做什么 |
|---|---|---|
| `q` | 390-392 | 退出循环 |
| `a` | 396-410 | 姿态对齐：让末端 Z 轴朝下 |
| `t` | 429-457 | **执行夹爪标定**（核心） |
| `,` | 414-419 | 夹爪缩小 1mm |
| `.` | 420-425 | 夹爪放大 1mm |
| `s` | 460-469 | 保存 |

> ⚠️ **帮助文字和实际按键不符**：帮助里写的是 `<` 和 `>`，但代码判断的是 `,` 和 `.`。
>
> **原因**：`<` 是 `Shift + ,` 产生的，而 `KeyboardReader` 工作在 `cbreak` 模式下，读到的是原始字符，不会帮你识别 Shift 组合键。所以实际要按**逗号键和句号键**。
>
> 这是个小的易用性问题，改帮助文字即可。

**`a` 键：姿态对齐**

```python
target_T_base_end = compute_axis_aligned_pose(T_base_end, base_axis_idx=-3, obj_axis_idx=3)
```

`base_axis_idx=-3` 表示基座坐标系的 **-Z 方向**（向下），`obj_axis_idx=3` 表示物体（这里物体 = 末端本身）的 **+Z 方向**。合起来就是"**把末端的 +Z 轴转到指向下方**"。

详见 `arm_node_analysis.md` 里对 `compute_axis_aligned_pose` 的详解。

**为什么要先按 `a`？** 让夹爪摆正（朝下），这样贴在夹爪上的 AprilTag 就正对相机，检测精度最高、平面拟合最稳。

**`t` 键：执行标定**

```python
frames = cam_node.get_frames(do_spin_once=True, frames_num=5)
if frames is None:
    logging.warning("No RGB-D frame available yet.")
    continue

color_img = frames[0][0]                                # 第 1 帧的彩色图
depth_img_list = [frame[1] for frame in frames]         # 5 帧的深度图
depth_img = depth_mean_filter(depth_img_list)           # 均值滤波

gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

corners3d = compute_corners3d(gray_img, depth_img, intrinsic, distortion, depth_scale)
if corners3d is None:
    logging.error("failed to compute tag plane, skip this round")
    continue

gb.initialize(corners3d)
calib_data["T_cam_gripper"] = gb.T_cam_gripper.tolist()
```

**取 5 帧做深度均值滤波**是降噪的关键。`depth_mean_filter`（`core/vision_utils.py` 第 100 行）的逻辑：

```
对每个像素：
  统计 5 帧里深度 > 0 的次数 obs_count
  深度和 depth_sum
  如果 obs_count >= 0.5 × 5（至少一半帧有值）→ 取 depth_sum / obs_count
  否则 → 置 0（认为测不到）
```

**为什么能降噪？** 深度相机的噪声是随机跳变的，多帧平均会互相抵消；而"某些帧测不到"（黑色图案、反光）会被过滤掉。

**`s` 键：保存**

```python
mmengine.dump(calib_data, gripper_path, indent=4)
```

`mmengine.dump` 会根据文件后缀自动选格式（`.json` / `.yaml` / `.pkl`）。这里后缀是 `.json`，所以写 JSON。

#### 退出清理（第 474–477 行）

```python
arm_node.destroy_node()
cam_node.destroy_node()
rclpy.shutdown()
logging.info("shutdown")
```

标准 ROS2 退出流程。

---

## 4. 依赖的 ROS2 / 硬件模块科普

给完全没接触过 ROS 的同学补一点背景。

### 4.1 ROS2 的几个核心概念

| 概念 | 大白话 |
|---|---|
| **Node（节点）** | 一个独立的程序单元。`arm_node`、`cam_node` 都是节点 |
| **Topic（话题）** | 数据管道。发布者往里塞数据，订阅者从里取。`/arm_pose`、`/grippers` 都是话题 |
| **Publisher（发布者）** | 往话题塞数据的一端 |
| **Subscriber（订阅者）** | 从话题取数据的一端 |
| **TF** | 坐标变换树。告诉系统"camera_link 在 base_link 的什么位置"，系统就能自动换算 |
| **Marker** | 可视化用的几何体（长方体、球、线等），在 RViz 里能看到 |
| **`spin_once`** | 让节点"处理一次"待办事件（收到的数据、定时器）。主循环里必须周期性调用 |

### 4.2 `KeyboardReader`（core/utils.py 第 335-370 行）

```python
class KeyboardReader:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = self._tcgetattr(self.fd)
        tty.setcbreak(self.fd)          # 切到 cbreak 模式
        self._closed = False
        atexit.register(self._close)    # 程序退出时自动恢复

    def read_key(self):
        rlist, _, _ = select.select([sys.stdin], [], [], 0)   # 超时=0，非阻塞
        return sys.stdin.read(1) if rlist else None
```

**两个关键点：**

**① `tty.setcbreak(fd)`**：把终端从"行缓冲模式"（要按回车才提交）切成"字符模式"（按一个键立刻可读）。这是实现"按 q 立刻退出"的基础。

**② `select.select([sys.stdin], [], [], 0)`**：超时设为 **0**，表示"看一眼有没有输入，没有就立刻返回"。这就是**非阻塞**——主循环不会卡在等键盘上。

> ⚠️ **副作用**：`cbreak` 模式下 **Ctrl+C 不再触发 `KeyboardInterrupt`**，而是被读成一个字符 `'\x03'`。本脚本没处理它，所以**标定过程中按 Ctrl+C 是没反应的，只能按 `q` 退出**。建议加上：
> ```python
> if key == '\x03':   # Ctrl+C
>     break
> ```

### 4.3 `ArmWrapper`（core/arm_wrapper.py）

机械臂硬件封装。关键方法：

```python
arm.get_pose()           # → T_base_end (4x4)，从控制器读当前末端位姿
arm.get_gripper_dist()   # → 夹爪张开距离（米）
arm.get_joints()         # → 6 个关节角
arm.set_pose(T)          # 移动到目标位姿
arm.set_gripper_dist(d)  # 设置夹爪张开距离
```

内部调用 `carm` 库（自研机械臂的 Python API），通过**网络（默认 IP 10.42.0.101）**通信。

> 类的设计注释写得很清楚："后续适配其他机械臂时，只需要修改该类的实现即可，不需要修改其他代码"——这是标准的**适配器模式**，值得学习。

---

## 5. 操作手册：怎么跑这个标定

### 5.1 前置准备

```
✅ calib_camera.py  已跑完 → cam_params.json（必须含 depth_scale）
✅ calib_handeye.py 已跑完 → calib_handeye.json
✅ 机械臂通电、网连通（能 ping 通 10.42.0.101）
✅ 相机 ROS 驱动已启动（rostopic list 能看到 /gemini305/... 话题）
✅ AprilTag（ID=0）已贴在夹爪上，贴平整，对角线沿夹爪张开方向
```

### 5.2 运行

```bash
python calib_gripper.py \
    --cam_params_path     "D:/calib/collect_image/cam_params.json" \
    --calib_handeye_path  "D:/calib/collect_image_handeye/calib_handeye.json" \
    --gripper_path        "D:/calib/gripper/gripper_params.json" \
    --color_img_topic     "/gemini305/color/image_raw" \
    --depth_img_topic     "/gemini305/depth/image_raw" \
    --gripper_size        "[0.02,0.005]"
```

### 5.3 标定步骤

```
1. 程序启动，终端显示按键说明
2. 按 'a'  → 机械臂自动摆正，末端朝下
             （如果报 "angle > 45 deg, skip align"，说明当前姿态太歪，
               先手动拖动机械臂到大致朝下再按 a）
3. 按 '.'  → 把夹爪张开一点（让 tag 露出来、不被爪子挡住）
4. 按 't'  → 执行标定
             ⚠️ 会弹出可视化窗口，确认：
                · 红/绿/蓝边框是否正好框住 AprilTag
                · 两个黄色小方块是否在对角
                · 按任意键关闭窗口继续
             终端会打印：
                · plane equation / inliers ratio  ← 看内点比例
                · Gripper pose set. T_cam_gripper: ...
5. 换个姿态（手动拖或再按 a），重复 3-4 步 2~3 次，看结果是否稳定
6. 按 's'  → 保存
7. 按 'q'  → 退出
```

### 5.4 怎么判断标定成功

| 检查项 | 期望 |
|---|---|
| 弹出的可视化窗口 | 彩色边框**精确**贴在 tag 四边，没有歪 |
| `inliers ratio` | > 0.8（内点比例高 = 平面拟合好） |
| `plane equation` | 前三个数（法向量）的 Z 分量应该较大（平面大致正对相机） |
| `T_cam_gripper` 的平移量 | 相机到夹爪的距离，一般 **5~20 cm**。出现几米肯定是错的 |
| 多次标定结果的**一致性** | 换个姿态重标，平移量差异应 < 5 mm |

---

## 6. 输入输出规范

### 6.1 输出的 `gripper_params.json`

```json
{
    "width": 0.02,
    "thickness": 0.005,
    "T_cam_gripper": [
        [0.998, -0.012, 0.055, 0.031],
        [0.011,  0.999, 0.021, -0.008],
        [-0.055, -0.020, 0.998, 0.142],
        [0.0,    0.0,   0.0,   1.0]
    ]
}
```

| 字段 | 含义 |
|---|---|
| `width` | 夹爪宽度（米），沿 Y 轴方向 |
| `thickness` | 夹爪厚度（米），沿 X 轴方向 |
| `T_cam_gripper` | 4×4 变换矩阵：从**夹爪坐标系**到**相机坐标系** |

**读法**：
- 右上角 3×1 = `[0.031, -0.008, 0.142]` 是夹爪原点在相机坐标系里的位置
  - X = 3.1 cm（偏右）
  - Y = -0.8 cm（偏下）
  - **Z = 14.2 cm（在相机前方 14.2 厘米）** ← 这个数最重要，它就是"爪子伸出去多长"
- 左上 3×3 = 旋转，描述夹爪相对相机的朝向

### 6.2 输入 `cam_params.json` 需要额外字段

注意这里用的是 `read_rgbd_params()`（不是 `read_cam_params()`），它**要求有 `depth_scale` 字段**：

```json
{
    "camera_type": "Pinhole",
    "IntrinsicFormat": "fx,fy,cx,cy",
    "DistortionFormat": "k1,k2,p1,p2,k3",
    "resolution": [1280, 720],
    "intrinsic": [912.3, 911.9, 638.2, 361.5],
    "distortion": [-0.123, 0.056, 0.0008, -0.0003, 0.0012],
    "depth_scale": 0.0001
}
```

`depth_scale = 0.0001` 表示深度计数值 10000 对应 1 米（即深度单位是 0.1 毫米，常见）。

> ⚠️ 这个字段 `calib_camera.py` **不会自动写**（它只写内参和畸变），需要你手动从相机驱动那边拿到并补进去。

---

## 7. 坑与改进建议

| # | 位置 | 问题 | 影响 | 建议 |
|---|---|---|---|---|
| 1 | 第 79 行 | **tag ID 写死为 0** | 贴错编号永远检测不到 | 做成命令行参数 |
| 2 | 第 159 行 | **`depth_trunc=0.5` 硬编码** | 夹爪离相机 >0.5m 时点云全空，标定失败 | 做成参数，或按 `depth_trunc=1.0` |
| 3 | 第 70 行 | `Detector` 在**函数内**创建 | 每次按 t 都重建一次，慢 | 提到主程序创建一次，作为参数传入 |
| 4 | 第 119 行 | `import platform` 写在函数内部 | 不规范 | 移到文件顶部 |
| 5 | 第 56 行 | 返回类型标注 `Tuple[...]` 但实际返回单个数组 | 误导读者 | 改成 `-> np.ndarray` |
| 6 | 第 332-335 行 | 警告说 "use loaded values" 但**没实际使用加载值** | 行为与提示不符 | 真正赋值，或改提示文字 |
| 7 | 第 354-358 行 | 帮助写 `<` `>`，实际判断 `,` `.` | 用户按错键没反应 | 改帮助文字 |
| 8 | 第 254-269 行 | 参数空值校验**永远不触发**（有 default） | 冗余代码 | 删掉，或去掉 default 让校验有意义 |
| 9 | 第 256/301 行 | 混用 `exit(0)` / `exit(1)` | 不规范，`exit()` 是交互式用的 | 统一用 `sys.exit(1)` |
| 10 | 键盘处理 | **没有处理 Ctrl+C**（`\x03`） | cbreak 模式下 Ctrl+C 失效，只能按 q | 加 `if key == '\x03': break` |
| 11 | 第 132 行 | 角点扩张 3 倍可能圈进**背景物体** | 平面拟合被污染（内点比例会变低，是个信号） | 根据 `inliers ratio` 自动判断，过低就报警 |
| 12 | 全局 | 没有**多次标定的一致性检查** | 单次结果可能偶然错误 | 标定 2~3 次，对比平移量差异 |
| 13 | 第 456 行 | `gb.T_cam_gripper.tolist()` 直接覆盖 | 无法回溯历史结果 | 保存前备份旧文件 |
| 14 | 默认参数 | `--gripper_size` 默认 `[0.0015, 0.002]`（1.5mm 宽） | 明显是占位值，实际必改 | 改成更合理的默认或设为必填 |
| 15 | 默认路径 | 全是作者的 Linux 路径 | 直接运行必失败 | 换掉 |

### 7.1 一个推荐的加固片段

```python
# ① 检测器提到外面创建（避免重复初始化）
detector = apriltag2.Detector(tag_family="tag36h11", black_border=2)

# ② 平面拟合后加健康检查
inlier_ratio = len(inliers) / len(pc.points)
if inlier_ratio < 0.6:
    logging.error(f"{RED}Plane fitting unreliable, inlier ratio only {inlier_ratio:.2f}. "
                  f"Check if the mask covers other objects.{RESET}")
    return None

# ③ 多次标定一致性检查
history = calib_data.get("T_cam_gripper_history", [])
history.append(gb.T_cam_gripper.tolist())
calib_data["T_cam_gripper_history"] = history[-5:]

if len(history) >= 2:
    prev_t = np.array(history[-2])[:3, 3]
    curr_t = gb.T_cam_gripper[:3, 3]
    diff_mm = np.linalg.norm(curr_t - prev_t) * 1000
    logging.info(f"与上次标定结果差异: {GREEN}{diff_mm:.2f} mm{RESET}")
    if diff_mm > 5.0:
        logging.warning(f"{YELLOW}两次标定差异 {diff_mm:.2f} mm > 5 mm, 请检查!{RESET}")

# ④ 支持 Ctrl+C
if key in ("q", "\x03"):
    logging.info("exit by user command")
    break
```

---

## 8. 一句话总结

**用一条链串起来：**

```
在夹爪上贴 ID=0 的 AprilTag
  → 取 5 帧 RGB-D，深度图均值滤波降噪
  → 彩色图去畸变，检测出 tag 的 4 个亚像素角点
  → 角点向外扩 3 倍做掩码，圈出深度点云
  → RANSAC 拟合出平面方程 Ax+By+Cz+D=0
  → 每个角点发一条射线 (nx, ny, 1)，与平面求交：z = -D/(A·nx + B·ny + C)
  → 得到 4 个角点的 3D 坐标
  → 用两条对角线建正交基（Gram-Schmidt + 叉乘），中心当原点
  → 组装成 T_cam_gripper
  → 按 s 存成 gripper_params.json
```

**三个最需要记住的点：**

1. **核心公式就一行**：`z = -D / (A·nx + B·ny + C)` —— 射线与平面求交。用"面"的稳定弥补"点"的噪声，这是整个设计的精髓。
2. **tag 必须贴成菱形**：坐标系的 X 轴用的是 tag 的**对角线**方向（c[2]-c[0]），所以贴的时候要让对角线对齐夹爪张开方向。贴正了（边水平）X 轴会歪 45°。
3. **看 `inliers ratio`**：这是平面拟合质量的唯一指标，< 0.5 说明掩码圈进了别的东西，标定不可信。

**排查清单（标定失败时按顺序查）：**

```
① 报 "AprilTag ID 0 not detected"？
   → tag 编号贴错了 / 光照太暗 / tag 被爪子挡住了 → 按 '.' 张开夹爪
② 弹窗里的彩色边框歪了 / 没框住 tag？
   → 检测有问题，检查 black_border、tag 家族、是否糊了
③ inliers ratio < 0.5？
   → 掩码圈进了背景 → 检查夹爪周围有没有别的物体
④ 报 "gripper Z axis direction error"？
   → tag 贴反了 / 从背面拍的
⑤ T_cam_gripper 的 Z 平移是几米？
   → 深度尺度 depth_scale 填错了
⑥ 点云计算为空？
   → 夹爪离相机超过 depth_trunc=0.5 米了
```

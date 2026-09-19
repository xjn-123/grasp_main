# `arm_utils.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、没学过线性代数、没接触过 ROS / rviz 的同学。
> 目标：读完之后，你能逐行看懂 `arm_utils.py` 在干什么——它怎么用两个矩形近似夹爪、怎么把夹爪投影到深度图上检测碰撞、怎么检查机械臂位姿是否合理、以及怎么算一个"让某个轴对齐"的新位姿。每一段都会讲三件事：算的是什么（数学）、这段代码在整套系统里干什么（业务）、这个 Python 写法什么意思（语法）。
>
> 被分析的源文件：`C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\core\arm_utils.py`（共 529 行）

---

## 目录

- [0. 一句话概括](#0-一句话概括)
- [1. 背景知识](#1-背景知识)
  - [1.1 夹爪的"几何模型"是什么、为什么抓之前要检查碰撞](#11-夹爪的几何模型是什么为什么抓之前要检查碰撞)
  - [1.2 坐标系约定 T_a_b = "从 b 到 a"](#12-坐标系约定-t_a_b--从-b-到-a)
  - [1.3 旋转矩阵的第 N 列 = 第 N 个坐标轴方向](#13-旋转矩阵的第-n-列--第-n-个坐标轴方向)
  - [1.4 叉乘求法向量、点乘求夹角](#14-叉乘求法向量点乘求夹角)
  - [1.5 深度图碰撞检测的思路](#15-深度图碰撞检测的思路)
  - [1.6 姿态对齐 compute_axis_aligned_pose 在解决什么问题](#16-姿态对齐-compute_axis_aligned_pose-在解决什么问题)
- [2. 模块地图](#2-模块地图)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 文件头与导入（第 1–22 行）](#31-文件头与导入第-1-22-行)
  - [3.2 全局常量（第 24–30 行）](#32-全局常量第-24-30-行)
  - [3.3 GripperBody 类的定义与构造函数（第 35–59 行）](#33-gripperbody-类的定义与构造函数第-35-59-行)
  - [3.4 GripperBody.initialize：用对角线建坐标系（第 61–99 行）](#34-gripperbodyinitialize用对角线建坐标系第-61-99-行)
  - [3.5 GripperBody.get_rects_3d：算出 8 个角点（第 101–134 行）](#35-gripperbodyget_rects_3d算出-8-个角点第-101-134-行)
  - [3.6 CollisionDetector 构造函数（第 138–175 行）](#36-collisiondetector-构造函数第-138-175-行)
  - [3.7 _undistort_color_img：彩色图去畸变（第 177–215 行）](#37-_undistort_color_img彩色图去畸变第-177-215-行)
  - [3.8 _proj_rect_3d：把矩形投影到图像（第 217–277 行）](#38-_proj_rect_3d把矩形投影到图像第-217-277-行)
  - [3.9 _compute_depth_diff：算深度差（第 279–328 行）](#39-_compute_depth_diff算深度差第-279-328-行)
  - [3.10 CollisionDetector.check：总入口（第 330–418 行）](#310-collisiondetectorcheck总入口第-330-418-行)
  - [3.11 check_arm_pose：检查位姿是否合理（第 424–468 行）](#311-check_arm_pose检查位姿是否合理第-424-468-行)
  - [3.12 compute_axis_aligned_pose：对齐某个轴（第 471–528 行）](#312-compute_axis_aligned_pose对齐某个轴第-471-528-行)
- [4. Python 基础语法速查](#4-python-基础语法速查)
- [5. 输入输出规范](#5-输入输出规范)
- [6. 核心数学：一步一步算给你看](#6-核心数学一步一步算给你看)
  - [6.1 GripperBody.initialize 用两条对角线建坐标系的完整推导](#61-gripperbodyinitialize-用两条对角线建坐标系的完整推导)
  - [6.2 get_rects_3d 的 8 个角点坐标怎么排](#62-get_rects_3d-的-8-个角点坐标怎么排)
  - [6.3 碰撞检测核心：投影、深度差、反直觉的判定条件](#63-碰撞检测核心投影深度差反直觉的判定条件)
  - [6.4 阈值 bad_cnt > max(20, 0.4*valid_cnt) 的含义](#64-阈值-bad_cnt--max20-04valid_cnt-的含义)
  - [6.5 check_arm_pose 两个判据的几何含义](#65-check_arm_pose-两个判据的几何含义)
  - [6.6 compute_axis_aligned_pose 的推导](#66-compute_axis_aligned_pose-的推导)
  - [6.7 数值演练](#67-数值演练)
- [7. 这段代码里的坑与改进建议](#7-这段代码里的坑与改进建议)
- [8. 一句话总结](#8-一句话总结)

---

## 0. 一句话概括

> **`arm_utils.py` 给"眼在手"的抓取机器人提供了一套"夹爪几何 + 碰撞检测 + 位姿合法性检查 + 姿态对齐"的工具箱。** 它不连机械臂、不连 ROS，纯算数学；上层脚本（如 `test_tmpl_grasp_3d.py`、`calib_gripper.py`）调它来决定"这个姿态能不能抓、会不会撞、手该怎么转"。

**输入 → 输出（从整文件角度看）：**

| | 内容 |
|---|---|
| 输入 1 | 夹爪尺寸 `width`/`thickness`、标定出的 `T_cam_gripper` |
| 输入 2 | AprilTag 的 4 个 3D 角点（标定夹爪时用） |
| 输入 3 | 参考位姿与目标位姿的末端矩阵、参考视角的 RGB-D 图（碰撞检测用） |
| 输出 1 | `GripperBody`：能算出夹爪 8 个 3D 顶点的模型对象 |
| 输出 2 | `CollisionDetector.check()`：布尔值，目标姿态会不会撞 |
| 输出 3 | `check_arm_pose()`：布尔值，末端位姿是否合法 |
| 输出 4 | `compute_axis_aligned_pose()`：一个新的末端位姿矩阵（让某轴对齐） |

---

## 1. 背景知识

### 1.1 夹爪的"几何模型"是什么、为什么抓之前要检查碰撞

**夹爪（gripper）** 就是机械臂最前端的"手指"。本工程用的是一种两片式平行夹爪：两片爪子沿一条轴（`GripperBody` 里叫 X 轴，爪张开方向）左右张开/合拢，每片爪有宽度（width）和厚度（thickness）。

代码里没有用真实的三维 CAD 模型，而是用 **两个矩形薄片** 来近似两片爪子（见 `get_rects_3d`）。每个矩形代表一片爪的内侧面（朝向物体的那一面）。为什么用矩形就够？因为碰撞检测只关心"爪子会碰到环境里的什么东西"，而爪子能碰到东西的主要就是这两片内侧面，用矩形近似既便宜又够用。

**为什么抓之前要检查碰撞？** 机械臂在"目标位姿"上张开夹爪去抓物体时，如果目标位姿离桌面/其他物体太近，爪子会插进桌面里——这叫"碰撞"。在真机上盲目移动可能撞坏设备。所以代码先把夹爪在目标位姿下的矩形"投影"到一张参考视角的深度图上，比较"爪子表面应该离相机多远"和"真实场景离相机多远"，如果爪子比真实表面还靠后（更深），就判定会撞。这就是 `CollisionDetector` 的全部思想。

### 1.2 坐标系约定 T_a_b = "从 b 到 a"

机器人里到处都是坐标系（基座 base、末端 end、相机 cam、物体 obj、夹爪 gripper……）。怎么表示一个坐标系相对另一个的位姿？用变换矩阵，而且本工程**统一用下标记法：**

$$T_{a\_b} = \text{从 b 坐标系变换到 a 坐标系的矩阵}$$

读法："把 b 下的坐标，变成 a 下的坐标"。下标里最后一个是"起点/源"，下划线前面是"终点/目标"。

本模块里出现的：
- `T_cam_gripper`：从夹爪坐标系 → 相机坐标系（第 53 行注释）。
- `T_target_cam`：从相机坐标系 → 目标坐标系（第 108 行注释）。
- `T_end_cam`：在 `CollisionDetector` 里注释写的是"从相机 → 末端"（第 155 行），**但后面实际传进去的是反方向**——这是本文件最大的命名坑，见第 7 节第 7 条。

> 牢记一条链式规则：矩阵相乘时**相邻下标会消掉**。例如
> $$T_{base\_cam} = T_{base\_end} \cdot T_{end\_cam}$$
> 中间的 `end` 消掉，得到 `base ← cam`。想不起来就念一遍"base→end→cam"。

### 1.3 旋转矩阵的第 N 列 = 第 N 个坐标轴方向

一个 4×4 位姿矩阵长这样：

$$
T = \begin{bmatrix}
R_{3\times3} & t_{3\times1} \\
0\ 0\ 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

左上角的 3×3 叫**旋转块** $R$。它的：
- **第 0 列** = 子坐标系的 **X 轴**在父坐标系下的方向向量（即 $R$ 乘 $[1,0,0]^T$ 的结果）。
- **第 1 列** = 子坐标系的 **Y 轴**方向。
- **第 2 列** = 子坐标系的 **Z 轴**方向。

所以 `R[:, 2]` 就是"子坐标系的 Z 轴指向哪"（在父坐标系下）。这是 `check_arm_pose` 和 `compute_axis_aligned_pose` 反复用到的性质。代码里 `R_base_end[:, 2]`（第 445 行）取的就是"末端 Z 轴在基座系下的方向"。

### 1.4 叉乘求法向量、点乘求夹角

- **叉乘（cross）** `np.cross(a, b)`：结果是一个**同时垂直于 a 和 b** 的向量，长度 = $|a||b|\sin\theta$，方向由右手定则决定。代码用它求平面的法向量（第 239 行、第 90 行、第 520 行）。
- **点乘（dot）** `np.dot(a, b)`：结果 = $|a||b|\cos\theta$。如果两个向量都是**单位向量**（长度 1），点乘就直接等于它们夹角的余弦 $\cos\theta$。代码用它求夹角余弦（第 447 行、第 507 行）。

这两者在 `arm_utils.py` 里无处不在：建坐标系要用叉乘，检查/对齐角度要用点乘。

### 1.5 深度图碰撞检测的思路：把夹爪"投影"到参考视角的深度图上

**深度图**是一张和彩色图一样大的图，但每个像素存的不是颜色，而是"这个像素对应的真实点到相机的距离（深度 z）"。值越大表示越远。

碰撞检测的核心想法（第 330–418 行）：
1. 先有一个**参考视角**的彩色图 + 深度图（机械臂在某个安全位置拍的，代表"环境本来长这样"）。
2. 假设机械臂移动到**目标位姿**去抓，把此时夹爪的两个矩形（在目标坐标系下算好）先变换回**参考视角相机坐标系**下，得到 8 个 3D 点。
3. 把这两个矩形"拍扁"投影到参考图像上，得到爪子会覆盖哪些像素；并对每个覆盖像素，用平面方程反算"如果爪子表面在这，它离相机该有多深"——这就是**投影深度图**。
4. 拿**投影深度图**和**真实深度图**逐像素比：如果某个像素上爪子算出来的深度 `d_proj` 比真实表面的深度 `d_real` 还大（即爪子比真实表面更靠后/更深），说明爪子应该插进真实物体里了 → 这一像素"撞了"。
5. 统计撞了的像素数，超过阈值就判定碰撞。

第 6.3 节会一步一步算给你看，那个"d_proj 比 d_real 大 = 撞"的点非常反直觉，务必看。

### 1.6 姿态对齐 compute_axis_aligned_pose 在解决什么问题

有时我们想让机械臂末端的某个轴（比如 Z 轴）指向某个特定方向（比如竖直向下）。比如夹爪标定时要让夹爪正对相机、抓取时要让夹爪垂直朝下。

`compute_axis_aligned_pose`（第 471 行）做的就是：**给定"当前末端位姿"和"想让物体系第几根轴对齐到基体系第几根轴"，算出一个新的末端位姿，使那根轴指向目标方向，且只旋转不改位置。**

一句话："把末端（或物体）的某根轴，转到指向下方/某方向，末端原地转。"

---

## 2. 模块地图

| 名称 | 行号 | 输入 | 输出 | 被谁调用 | 一句话作用 |
|---|---|---|---|---|---|
| `TH_ANGLE_Z` | 26 | 无 | 浮点常量 | `check_arm_pose`（经 `test_tmpl_grasp_3d`） | 末端 Z 轴与 -Z 轴夹角阈值（弧度，45°） |
| `TH_GRIPPER_HEIGHT` | 29 | 无 | 浮点常量 | `check_arm_pose` | 夹爪最低点的高度阈值（米，-0.01） |
| `GripperBody.__init__` | 44 | width, thickness, T_cam_gripper | GripperBody 对象 | `calib_gripper.py`、`test_tmpl_grasp_3d.py` | 存夹爪尺寸与标定位姿 |
| `GripperBody.initialize` | 61 | corners3d (4×3) | 填充 T_cam_gripper | `calib_gripper.py`（第 454 行） | 用 AprilTag 四角点建夹爪坐标系 |
| `GripperBody.get_rects_3d` | 101 | dist, T_target_cam | (8,3) 角点 | `check`、`check_arm_pose`、`ArmNode.publish_grippers` | 算两片爪的 8 个 3D 顶点 |
| `CollisionDetector.__init__` | 144 | gripper_body, T_end_cam, 内参… | 检测器对象 | 上层（碰撞检测调用方） | 存相机/夹爪信息 |
| `CollisionDetector._undistort_color_img` | 177 | 彩色图 | 去畸变彩色图 | `check`（debug 时） | 去镜头畸变 |
| `CollisionDetector._proj_rect_3d` | 217 | rect_3d, 掩码图, 深度图 | 轮廓 + 填充投影图 | `check` | 投影一个矩形并算投影深度 |
| `CollisionDetector._compute_depth_diff` | 279 | 真实/投影深度图、掩码、轮廓 | (valid_cnt, bad_cnt) | `check` | 比较深度、数撞了的像素 |
| `CollisionDetector.check` | 330 | 两末端位姿 + 参考 RGB-D | 是否碰撞(bool) | 上层抓取流程 | 碰撞检测总入口 |
| `check_arm_pose` | 424 | T_base_end, T_end_cam, 夹爪, 阈值 | 是否合法(bool) | `test_tmpl_grasp_3d.py`（第 320/410/482 行） | 检查末端朝向与夹爪高度 |
| `compute_axis_aligned_pose` | 471 | T_base_end, 轴索引, T_end_obj | 新末端位姿或 None | `calib_gripper.py`、`test_tmpl_grasp_2d.py` | 算对齐轴的新位姿 |

---

## 3. 逐段代码精读

### 3.1 文件头与导入（第 1–22 行）

```python
"""
检测夹爪与环境的碰撞情况
"""

import os
import logging
import time

from typing_extensions import List, Tuple, Dict

import transforms3d
import numpy as np
import cv2
import open3d

# 导入本工程的模块
from .utils import (
    GREEN, YELLOW, RESET,
    inv_tf
)
from .vision_utils import rgbd_to_point_cloud, depth_to_point_cloud
```

- 第 1–3 行：模块说明字符串（文档字符串 docstring）。它说"检测夹爪与环境的碰撞情况"，但本模块其实还有"建夹爪模型""检查位姿""对齐姿态"三件事，说明这个 docstring 写得有点窄。
- `numpy as np`：数值计算库，矩阵/向量都靠它。**ndarray** 是它的数组类型，比如 `np.array([1,2,3])` 是一维数组，`np.eye(4)` 是 4×4 单位阵。
- `transforms3d`：三维变换库，用来在"旋转矩阵、四元数、轴角"之间互转。`mat2quat`（矩阵→四元数）、`axangle2mat`（轴角→矩阵）后面都会用到。
- `cv2`：OpenCV，处理图像（去畸变、画多边形）。
- `open3d`：点云库，只在 debug 存 `.pcd` 点云时用。
- `from .utils import ... inv_tf`：同一包下的 `utils.py` 里的 `inv_tf`（第 210 行），求变换矩阵的逆（见第 6 节会用）。`GREEN/YELLOW/RESET` 是彩色终端打印用的 ANSI 转义码。
- `from .vision_utils import rgbd_to_point_cloud, depth_to_point_cloud`：把深度图转成点云的函数，只在 debug 可视化用。

### 3.2 全局常量（第 24–30 行）

```python
TH_ANGLE_Z = 45.0 * np.pi / 180.0
"""两个坐标系之间的 Z 轴夹角阈值, 单位: 弧度"""

TH_GRIPPER_HEIGHT = -0.01
"""夹爪中心在基座坐标系的高度阈值, 单位: 米"""
```

- 第 26 行：`45.0 * np.pi / 180.0` 把 45 度转成弧度。因为 Python 的 `np.arccos` 等三角函数都用**弧度**。$\pi$ 弧度 = 180°，所以"度→弧度"要乘 $\pi/180$。结果约 0.7854。
  - ⚠️ 注意：这个常量单位是**弧度**。但后面 `compute_axis_aligned_pose` 里的 `th_angle` 参数单位是**度**（第 514 行用 `np.deg2rad` 转）。**同一模块两种单位**，极易传错，见第 7 节第 9 条。
- 第 29 行：`TH_GRIPPER_HEIGHT = -0.01`，单位米。意思是"夹爪最低点不能低于基座系下 -1 厘米"。负值说明允许夹爪略微低于基座原点平面（桌面大概就在基座系 z≈0 附近，但允许一点点余量）。

### 3.3 GripperBody 类的定义与构造函数（第 35–59 行）

```python
class GripperBody:
    """
    夹爪几何体,使用两个矩形来表示夹爪与环境的接触面.预设条件:
    1) 仅适用于眼在手的场景
    2) 夹爪坐标系的原点位于两片爪尖的几何中心
    3) 以夹爪张开方向为夹爪坐标系的 X 轴,从左爪指向右爪;
    4) 夹爪坐标系的 Z 轴垂直于夹爪平面,朝向与末端坐标系 Z 轴的夹角近似平行;
    """

    def __init__(self,
                 width: float,
                 thickness: float,
                 T_cam_gripper: np.ndarray = np.eye(4)):
        """
        构造函数
        Args:
            width (float): 夹爪宽度 (单位: m)
            thickness (float): 夹爪厚度 (单位: m)
            T_cam_gripper (np.ndarray): 从夹爪坐标系到相机坐标系的位姿变换, 形状为 (4, 4)
        """

        self.width = width
        self.thickness = thickness
        self.T_cam_gripper = T_cam_gripper  # 夹爪在相机坐标系下的位姿
    # end def __init__
```

**语法点：什么是类（class）？**
- `class GripperBody:` 定义一个"类"，可以理解成一张"夹爪的图纸"。按这张图纸 `GripperBody(width=..., thickness=...)` 造出来的东西叫**实例（instance）**，代码里叫 `gripper_body`。
- `def __init__(self, ...)` 是**构造函数**，每次 `GripperBody(...)` 时自动运行，用来给新造的实例"上初始属性"。
- `self.width = width`：`self` 指"这个实例自己"。`self.width` 是**实例属性**，每个实例各自存一份。之后 `gripper_body.width` 就能取出来。
- `T_cam_gripper: np.ndarray = np.eye(4)`：这是**默认参数**——调用时不传就用 4×4 单位阵。`np.eye(4)` 是单位阵 $I_4$（对角线为 1，其余为 0），代表"夹爪坐标系和相机坐标系重合"的初始假设。
  - ⚠️ 坑：`calib_gripper.py` 第 314 行造 `gb` 时**没传** `T_cam_gripper`，于是初值是单位阵；随后第 454 行 `gb.initialize(corners3d)` 会**覆盖** `self.T_cam_gripper`（第 94–95 行）。所以默认单位阵只在"还没标定"时短暂存在。但也意味着：如果你忘了调 `initialize`，`T_cam_gripper` 就还是单位阵——后面算出来全错却不会报错。

**文档字符串里的 4 条预设**（很重要）：
1. 只支持"眼在手"（相机装在末端上）。
2. 夹爪坐标系原点在两片爪尖几何中心。
3. X 轴 = 爪张开方向（左→右）。
4. Z 轴垂直夹爪平面，且大致平行于末端 Z 轴。

这 4 条决定了后面所有公式怎么摆。

### 3.4 GripperBody.initialize：用对角线建坐标系（第 61–99 行）

```python
def initialize(self,
               corners3d: np.ndarray):
    """
    初始化: 计算从夹爪坐标系到相机坐标系的位姿变换
    Args:
        corners3d (np.ndarray): 夹爪平面的 AprilTag 标签的四个角点的3D坐标,形状为 (4, 3)
    """

    assert corners3d.shape == (4, 3), "corners3d must have shape (4, 3)"

    # 坐标系定义, 从 corners3d[0] 指向 corners3d[2] 的方向为 X 轴, 从 corner[3] 指向 corner[1] 的方向为 Y 轴
    # Z 轴为平面的法向量,由右手定则确定
    # 原点为 corners3d[0] 和 corners3d[2] 的中点

    center_3d = corners3d.mean(axis=0)  # 更鲁棒的中心
    edge_x = corners3d[2] - corners3d[0]
    edge_y = corners3d[1] - corners3d[3]
    if np.linalg.norm(edge_x) < 1e-9 or np.linalg.norm(edge_y) < 1e-9:
        raise ValueError("角点退化,无法建立坐标系")
    # end if

    axis_x = edge_x / np.linalg.norm(edge_x)

    # 先正交化 Y
    axis_y_raw = edge_y / np.linalg.norm(edge_y)
    axis_y = axis_y_raw - np.dot(axis_y_raw, axis_x) * axis_x
    axis_y /= np.linalg.norm(axis_y)

    # 右手系 Z
    axis_z = np.cross(axis_x, axis_y)
    axis_z /= np.linalg.norm(axis_z)
    assert axis_z[2] > 0, "gripper Z axis direction error"

    self.T_cam_gripper[:3, :3] = np.column_stack((axis_x, axis_y, axis_z))
    self.T_cam_gripper[:3, 3] = center_3d.reshape(3)

    logging.info(f'Gripper pose set. T_cam_gripper:\n{self.T_cam_gripper}')
```

**这段在干什么（业务）：** 夹爪上贴了一张 AprilTag（黑白方块）。相机看到这张 tag，能解出它的 4 个角点在相机系下的 3D 坐标 `corners3d`（4×3）。但光有角点还不够——我们需要知道"夹爪坐标系"相对相机在哪、朝哪。这段就是**用这 4 个角点反推出夹爪坐标系**（也就是填 `T_cam_gripper` 的旋转块和平移块）。

**关键点（重点问题 1）：它用的是对角线，不是边！**
- 第 76 行：`edge_x = corners3d[2] - corners3d[0]` —— 用**第 0 号角到 第 2 号角**的对角线当 X 轴。
- 第 77 行：`edge_y = corners3d[1] - corners3d[3]` —— 用**第 3 号角到 第 1 号角**的对角线当 Y 轴。

而同工程的 `vision_utils.compute_tag_pose`（第 376–380 行）对**同一张 AprilTag** 用的是**边**：
- `axis_x = corners3d[1] - corners3d[0]`（第 0→1 号边）
- `axis_y = corners3d[3] - corners3d[0]`（第 0→3 号边）

**两条对角线互相垂直，而相邻的两条边（0→1 和 0→3）夹角是 90°**。但"对角线方向"和"边方向"之间差 **45°**！也就是说：同一个物理 tag，在"夹爪标定"流程里建立的 tag 坐标系（对角线为轴），和"物体定位"流程里 `compute_tag_pose` 建立的 tag 坐标系（边为轴），**旋转差了 45°**。

这带来一个非常重要的实际后果：**贴 AprilTag 给夹爪标定时，必须把它转 45° 贴成"菱形"（对角线水平/竖直），这样 `initialize` 用的对角线才正好对齐夹爪张开方向（X 轴）。** 如果按常规正着贴（边水平），那 `initialize` 算出的 X 轴是 45° 斜的，夹爪坐标系就歪了——后面所有抓取都歪 45°。第 6.1 节会画 ASCII 图说明。

逐行语法/数学：
- 第 69 行 `assert corners3d.shape == (4, 3), "..."`：`assert` 是"断言"，条件为真就啥也不发生，为假就抛 `AssertionError`。这里检查输入形状。⚠️ 坑：见第 7 节第 2 条，`assert` 在 `python -O` 下会被**整个删掉**，且有用 `assert` 做业务校验的风险。
- 第 75 行 `corners3d.mean(axis=0)`：沿第 0 轴（行）求平均，得到 `(3,)`——4 个角点的平均位置，当作原点。`axis=0` 表示"把 4 行压成 1 行（逐列平均）"。
- 第 78 行 `np.linalg.norm(edge_x)`：向量的长度（模）。`< 1e-9` 判断是否为退化（长度几乎为 0）。
- 第 79 行 `raise ValueError("...")`：**主动抛异常**，强制中断并给出可读信息。这比 `assert` 好（不受 `-O` 影响）。
- 第 82 行 `edge_x / np.linalg.norm(edge_x)`：把向量**归一化**成单位向量（长度 1，方向不变）。
- 第 84–87 行（**正交化**，重点）：AprilTag 的两条对角线虽然基本垂直，但数值上可能不严格垂直。`axis_y = axis_y_raw - (axis_y_raw·axis_x) * axis_x` 是把 `axis_y_raw` 里"平行于 X 轴的分量"减掉，剩下的就是垂直于 X 轴的分量（这就是**格拉姆-施密特正交化**的第一步）。最后再归一化。这样保证 X、Y 严格垂直。
- 第 90–91 行 `axis_z = np.cross(axis_x, axis_y)`：叉乘得到同时垂直 X、Y 的法向量（Z 轴），右手系。
- 第 92 行 `assert axis_z[2] > 0`：要求 Z 轴在相机系下朝"前/上"（z 分量为正）。⚠️ 坑：见第 7 节第 2 条。
- 第 94 行 `np.column_stack((axis_x, axis_y, axis_z))`：把三个列向量并成 3×3 旋转矩阵，第 0 列是 X、第 1 列是 Y、第 2 列是 Z（对应第 1.3 节的性质）。
- 第 95 行 `center_3d.reshape(3)`：把 `(3,)` 形状确认一下，填进平移块。

### 3.5 GripperBody.get_rects_3d：算出 8 个角点（第 101–134 行）

```python
def get_rects_3d(self,
                 dist: float,
                 T_target_cam: np.ndarray = np.eye(4)) -> np.ndarray:
    """
    用两个空间矩形表示夹爪的位置,计算夹爪的八个角点的3D坐标( 目标坐标系下 )
    Args:
        dist (float): 两个夹爪矩形之间的距离 (单位: m)
        T_target_cam (np.ndarray): 从相机坐标系到目标坐标系的变换矩阵, 形状为 (4, 4)
    Returns:
        (np.ndarray): 两个夹爪夹爪共八个角点的3D坐标, 形状为 (8, 3), 每个夹爪四个角点按[左下,右下,右上,左上]顺序排列
    """

    w = self.width
    t = self.thickness
    hd = dist / 2.0
    hw = w / 2.0

    # 计算夹爪四个角点在夹爪坐标系下的坐标
    left_rect = np.array([[-hd - t, hw, 0, 1],
                          [-hd, hw, 0, 1],
                          [-hd, -hw, 0, 1],
                          [-hd - t, -hw, 0, 1]], dtype=np.float32).T  # (4,4)

    right_rect = np.array([[hd, hw, 0, 1],
                           [hd + t, hw, 0, 1],
                           [hd + t, -hw, 0, 1],
                           [hd, -hw, 0, 1]], dtype=np.float32).T  # (4,4)

    T_target_gripper = T_target_cam @ self.T_cam_gripper      # 从夹爪坐标系到目标坐标系的变换矩阵
    left_rect_3d = (T_target_gripper @ left_rect).T[:, :3]    # (4,3)
    right_rect_3d = (T_target_gripper @ right_rect).T[:, :3]  # (4,3)
    rects_3d = np.vstack((left_rect_3d, right_rect_3d))       # (8,3)
    return rects_3d
```

**这段在干什么（业务）：** 给定"两片爪之间的距离 `dist`"和"夹爪坐标系→目标坐标系的变换 `T_target_cam`"，算出两片爪的内侧面矩形的 8 个 3D 顶点（在目标坐标系下）。这是碰撞检测、位姿检查、`publish_grippers` 可视化的共同前置。

逐行：
- 第 115 行 `hd = dist / 2.0`：half distance，两片爪中心间距的一半。左爪在 $-hd$ 附近，右爪在 $+hd$ 附近。
- 第 116 行 `hw = w / 2.0`：half width，爪宽度一半。
- 第 119–122 行 `left_rect`：左爪 4 个角点在**夹爪坐标系**下的坐标。每个点是 `(x, y, z, 1)` 的**齐次坐标**（加个 1 才能用 4×4 矩阵做平移）。`.T` 转置，让它从 `(4,4)`（每行一个点）变成 `(4,4)`（每列一个点）——这样后面 `矩阵 @ 点阵` 时每列是一个点。
  - 四个点的 x 是 `-hd-t, -hd, -hd, -hd-t`（从左外侧到左内侧），y 是 `hw, hw, -hw, -hw`（上到下），所以顺序是 `[左下, 右下, 右上, 左上]`（代码注释第 110 行）。注意 y 先正后负，配合 x 顺序，得到的是逆时针还是顺时针看 6.2 节图。
- 第 124–127 行 `right_rect`：右爪，x 在 `+hd` 附近。
- 第 129 行 `T_target_gripper = T_target_cam @ self.T_cam_gripper`：链式相乘。按第 1.2 节记法，`T_target_cam` 是"相机→目标"，`T_cam_gripper` 是"夹爪→相机"，中间 `cam` 消掉，得到"夹爪→目标"。⚠️ 注意这里 `T_target_cam` 这个名字在注释里写"从相机到目标"，但传入的常常是 `cam_T_ref_target` 或 `T_base_cam`，命名混乱，见第 7 节第 7 条。
- 第 130 行 `(T_target_gripper @ left_rect).T[:, :3]`：`4×4 @ 4×4` 得到每个点的变换后齐次坐标，`.T` 转回"每行一个点"，`[:, :3]` 取 x,y,z 丢掉末尾的 1。
- 第 132 行 `np.vstack(...)`：竖直拼接，左 4 点在上、右 4 点在下，得到 `(8,3)`。

### 3.6 CollisionDetector 构造函数（第 138–175 行）

```python
class CollisionDetector:
    """
    碰撞检测器( 仅适用于眼在手的场景 )
    通过将夹爪在目标位置下的空间矩形投影到参考位置下的深度图像上,计算投影深度图与真实深度图的差异来判断是否发生碰撞
    """

    def __init__(self,
                 gripper_body: GripperBody,
                 T_end_cam: np.ndarray,
                 intrinsic: List[float],
                 depth_scale: float,
                 distortion: List[float] = None,
                 debug_dir: str = None):
        """
        ...
        Args:
            gripper_body (GripperBody): 夹爪几何体
            T_end_cam (np.ndarray): 从相机坐标系到机械臂末端的变换矩阵, 形状为 (4, 4)
            intrinsic (List[float]): 相机内参 [fx, fy, cx, cy]
            depth_scale (float): 深度图像的缩放比例
            distortion (List[float], optional): 相机畸变参数,格式与 OpenCV 一致. 默认值为 None
            debug_dir (str, optional): 用于保存调试信息的目录. 默认值为 None
        """
        assert len(intrinsic) == 4, "intrinsic must have 4 elements: [fx, fy, cx, cy]"

        self.gripper_body = gripper_body
        self.T_end_cam = T_end_cam

        self.intrinsic = intrinsic
        self.depth_scale = depth_scale
        self.distortion = distortion
        self.color_undistort_maps = []  # 去畸变映射表, [map1, map2], 由 cv2.initUndistortRectifyMap 计算得到

        self.debug_dir = debug_dir
        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)
        # end if
    # end def __init__
```

**这段在干什么（业务）：** 把"夹爪模型 + 相机参数"打包成一个检测器对象，后面 `check()` 反复用。

- 第 161 行 `assert len(intrinsic) == 4`：检查内参是 4 个数 `[fx, fy, cx, cy]`。
- `intrinsic` 是相机内参：`fx, fy` 是焦距（像素单位），`cx, cy` 是主点（图像中心像素）。投影公式（第 248 行）就靠它们。
- `depth_scale`：深度图里每个整数值代表多少**米**。比如深度图存 1000，乘 `depth_scale=0.001` 才是 1 米。
- `distortion`：镜头畸变系数，去畸变用。
- 第 169 行 `self.color_undistort_maps = []`：先留空，真正用到彩色图去畸变时才计算并缓存（第 192–206 行），避免每次都重算。
- 第 172–173 行 `os.makedirs(self.debug_dir, exist_ok=True)`：`exist_ok=True` 表示"目录已存在也不报错"。只有传了 `debug_dir` 才建。

### 3.7 _undistort_color_img：彩色图去畸变（第 177–215 行）

```python
def _undistort_color_img(self,
                         color_img: np.ndarray) -> np.ndarray:
    """
    对彩色图像去畸变
    ...
    """
    if self.distortion is None:
        return color_img
    # end if

    # 初始化畸变校正映射表
    if len(self.color_undistort_maps) == 0:
        K = np.array([[self.intrinsic[0], 0, self.intrinsic[2]],
                      [0, self.intrinsic[1], self.intrinsic[3]],
                      [0, 0, 1]], dtype=np.float32)
        D = np.array(self.distortion, dtype=np.float32)
        img_w, img_h = color_img.shape[1], color_img.shape[0]
        map1, map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=K,
            distCoeffs=D,
            R=None,
            newCameraMatrix=K,
            size=(img_w, img_h),
            m1type=cv2.CV_16SC2
        )
        self.color_undistort_maps = [map1, map2]

        logging.info("undistort rectify maps initialized.")
    # end if

    undistorted_img = cv2.remap(color_img, self.color_undistort_maps[0], self.color_undistort_maps[1],
                                interpolation=cv2.INTER_LINEAR)

    return undistorted_img
```

**这段在干什么（业务）：** 真实镜头拍出来是"弯"的（畸变），`cv2.initUndistortRectifyMap` 预先算好一张"重映射表"，`cv2.remap` 用它把弯曲的图拉直。只在 `check()` 的 debug 可视化（画投影轮廓）时用——碰撞检测本身用的是深度图，不直接去畸变彩色图。

逐行：
- 第 187–189 行：没畸变参数就原样返回。
- 第 192 行 `if len(self.color_undistort_maps) == 0:`：**懒初始化**——只在第一次调用时算映射表，算过就缓存，避免重复计算。
- 第 193–195 行 `K`：把 `[fx,fy,cx,cy]` 拼成 3×3 内参矩阵。
- 第 197 行 `color_img.shape[1]` 是宽、`shape[0]` 是高（numpy 数组 shape 是 `(高, 宽, 通道)`）。
- 第 198–205 行 `cv2.initUndistortRectifyMap`：核心去畸变函数，返回 `map1, map2` 两张查找表。
- 第 211–212 行 `cv2.remap`：按表重采样，得到去畸变图。

### 3.8 _proj_rect_3d：把矩形投影到图像（第 217–277 行）

```python
def _proj_rect_3d(self,
                  intrinsic: List[float],
                  rect_3d: np.ndarray,
                  proj_mask_img: np.ndarray,
                  proj_depth_img: np.ndarray,
                  mask_value: int = 255) -> np.ndarray:
    """
    将3D矩形投影到图像平面上,并计算投影深度图、投影掩码图、投影轮廓
    ...
    """

    # 计算平面方程 Ax + By + Cz + D = 0
    pt0, pt1, pt2 = rect_3d[0], rect_3d[1], rect_3d[2]
    v1 = pt1 - pt0
    v2 = pt2 - pt0
    normal = np.cross(v1, v2)
    D = -np.dot(normal, pt0)
    A, B, C = normal[0], normal[1], normal[2]

    # 计算投影
    fx, fy, cx, cy = intrinsic
    contour = []
    for i in range(4):
        pt3d = rect_3d[i]
        u = int(pt3d[0] * fx / pt3d[2] + cx)
        v = int(pt3d[1] * fy / pt3d[2] + cy)
        contour.append([u, v])
    # end for
    contour = np.array(contour, dtype=np.int32)

    # 填充多边形区域
    cv2.fillPoly(proj_mask_img, [contour], color=mask_value)

    # 计算投影深度
    vs, us = np.where(proj_mask_img == mask_value)
    for u, v in zip(us, vs):
        # 计算归一化图像坐标
        nx, ny = (u - cx) / fx, (v - cy) / fy

        # 计算平面方程与视线的交点深度值
        z = -D / (A * nx + B * ny + C)
        if z <= 0.001:
            continue
        # end if

        d = np.uint16(z / self.depth_scale)

        if d > proj_depth_img[v, u]:
            proj_depth_img[v, u] = d
        # end if
    # end for

    return contour
```

**这段在干什么（业务）：** 把一个 3D 矩形（4 个顶点）拍到图像上：
1. 算出这个矩形所在平面的方程。
2. 把 4 个顶点投影成图像上的 4 个像素 `(u,v)`，连成轮廓。
3. 用 `cv2.fillPoly` 把轮廓内部全部涂成 `mask_value`（左爪 200，右爪 255），得到"这个爪覆盖哪些像素"。
4. 对覆盖到的每个像素，用平面方程反算"若爪子表面在此像素，离相机多深"，填进 `proj_depth_img`。

逐行（数学见第 6.3 节）：
- 第 236–241 行：用 3 点求平面。`v1, v2` 是矩形两条边，`normal = cross(v1, v2)` 是法向量。平面方程 `Ax+By+Cz+D=0`，其中 `D = -n·pt0`。
- 第 244–251 行：投影公式。`pt3d[2]` 是该点的深度 z。`u = x*fx/z + cx` 是针孔相机模型（把相机系下的 (x,y,z) 投到像素 (u,v)）。⚠️ 坑：第 248–249 行用 `int()` **截断取整**（向 0 舍入），不是四舍五入，会带来最多半像素误差；且没有 clamp，角点可能落在图像外（第 7 节第 4 条）。
- 第 255 行 `cv2.fillPoly`：填充多边形，参数 `[contour]` 要包一层列表（OpenCV 要求"多个多边形"的列表）。
- 第 258 行 `vs, us = np.where(proj_mask_img == mask_value)`：`np.where` 对布尔/值数组返回满足条件的**下标**。`==` 比较得到布尔掩码，于是返回 `(行下标数组, 列下标数组)`。这里 `proj_mask_img` 是 `(高,宽)`，`np.where` 返回 `(vs, us)`——`vs` 是 v（行=y），`us` 是 u（列=x）。注意返回顺序是 `(行, 列)` 即 `(v, u)`。
- 第 259 行 `for u, v in zip(us, vs):`：把 u、v 两个数组"拉链"式配对遍历。⚠️ 坑：这是逐像素 Python 循环，是性能热点（第 7 节第 3 条）。
- 第 261 行 `(u-cx)/fx`：像素坐标反算"归一化图像坐标" nx（去掉了内参，相当于单位焦距下的 x）。
- 第 264 行 `z = -D / (A*nx + B*ny + C)`：由平面方程和视线方向 `(nx, ny, 1)` 求交点深度。推导见 6.3。
- 第 269 行 `d = np.uint16(z / self.depth_scale)`：把米转成深度图的"原始整数值"。`np.uint16` 是无符号 16 位整数（0~65535）。
- 第 271–272 行 `if d > proj_depth_img[v, u]:`：⚠️ 坑：同一个像素若被左右两片爪都覆盖（实际不太可能）或多次循环，取**较大**的（更远的）。这里取"较大"的取舍含义见第 7 节第 3 条。

### 3.9 _compute_depth_diff：算深度差（第 279–328 行）

```python
def _compute_depth_diff(self,
                        real_depth_img: np.ndarray,
                        proj_depth_img: np.ndarray,
                        proj_mask_img: np.ndarray,
                        proj_contour: np.ndarray,
                        mask_value: int,
                        max_depth_diff: float,) -> Tuple[float, float]:
    """
    计算投影深度图与真实深度图的深度差异
    ...
    Returns:
        (Tuple[float, float]): 有效像素点数量, 深度差异过大( 投影深度大于实际深度 )的像素点数量
    """

    th_diff = max_depth_diff / self.depth_scale
    valid_cnt = 0
    bad_cnt = 0
    bbox = cv2.boundingRect(proj_contour)  # x, y, w, h
    v0 = max(bbox[1] - 1, 0)
    v1 = min(bbox[1] + bbox[3], real_depth_img.shape[0] - 1)
    u0 = max(bbox[0] - 1, 0)
    u1 = min(bbox[0] + bbox[2], real_depth_img.shape[1] - 1)
    for v in range(v0, v1):
        for u in range(u0, u1):
            if proj_mask_img[v, u] != mask_value:
                continue
            # end if

            d_proj = proj_depth_img[v, u]
            d_real = real_depth_img[v, u]
            if d_real == 0 or d_proj == 0:
                continue
            # end if

            valid_cnt += 1

            if d_proj > d_real + th_diff:
                bad_cnt += 1
            # end if
        # end for
    # end for

    return valid_cnt, bad_cnt
```

**这段在干什么（业务）：** 在一个矩形覆盖的像素范围内，逐像素比较"投影深度（爪子表面深度）"和"真实深度（环境表面深度）"。统计：(1) 有效像素数 `valid_cnt`；(2) 爪子比真实表面深超过 `max_depth_diff` 的像素数 `bad_cnt`。

逐行：
- 第 299 行 `th_diff = max_depth_diff / self.depth_scale`：把"米"阈值转成深度图整数阈值。
- 第 302 行 `cv2.boundingRect(proj_contour)`：求轮廓的外接矩形 `(x, y, w, h)`。`x=u0, y=v0, w, h`。
- 第 303–306 行：把遍历范围限制在包围盒（且 clamp 到图像内），避免遍历全图。⚠️ 这里用 `max(..., 0)` 和 `min(..., shape-1)` 做了边界保护，但第 3.8 节的角点投影**没**做，不一致。
- 第 307–308 行：双层 `for v in range(...): for u in range(...)` 遍历。⚠️ 坑：又是逐像素 Python 循环（第 7 节第 3 条）。
- 第 309 行 `if proj_mask_img[v, u] != mask_value: continue`：只处理本矩形覆盖的像素。
- 第 315–317 行：深度为 0 表示"无数据/太远"，跳过。
- 第 321 行 `if d_proj > d_real + th_diff:`：**核心判定**。爪子投影深度比真实深度大（更深）超过阈值 → 这一像素算"撞"。`th_diff` 是允许的误差（容差）。反直觉之处见第 6.3 节。

返回 `(valid_cnt, bad_cnt)`——注意是 `float` 标注但实际是整数，小瑕疵。

### 3.10 CollisionDetector.check：总入口（第 330–418 行）

```python
def check(self,
          gripper_dist: float,
          ref_T_base_end: np.ndarray,
          target_T_base_end: np.ndarray,
          ref_bgr_img: np.ndarray,
          ref_depth_img: np.ndarray,
          max_depth_diff: float = 0.01,
          debug_level: int = 0) -> bool:
    """
    检测夹爪在目标位置是否与环境碰撞
    ...
    Returns:
        (bool): 是否发生碰撞
    """

    st = time.time()

    # 从 target 位置到 ref 位置,相机位姿的变换
    cam_T_ref_target = inv_tf(ref_T_base_end @ self.T_end_cam) @ (target_T_base_end @ self.T_end_cam)

    # 计算 target 位置下的夹爪在 ref 位置下的相机坐标系的空间矩形的顶点 8*3
    gripper_rects_3d = self.gripper_body.get_rects_3d(gripper_dist, cam_T_ref_target)

    # 计算投影
    proj_mask_img = np.zeros_like(ref_depth_img, dtype=np.uint8)
    proj_depth_img = np.zeros_like(ref_depth_img, dtype=np.uint16)

    left_contour = self._proj_rect_3d(self.intrinsic, gripper_rects_3d[0:4],
                                      proj_mask_img, proj_depth_img, mask_value=200)  # 左夹爪
    right_contour = self._proj_rect_3d(self.intrinsic, gripper_rects_3d[4:8],
                                       proj_mask_img, proj_depth_img, mask_value=255)  # 右夹爪

    left_valid_cnt, left_bad_cnt = self._compute_depth_diff(ref_depth_img, proj_depth_img, proj_mask_img, left_contour,
                                                            mask_value=200, max_depth_diff=max_depth_diff)
    right_valid_cnt, right_bad_cnt = self._compute_depth_diff(ref_depth_img, proj_depth_img, proj_mask_img, right_contour,
                                                              mask_value=255, max_depth_diff=max_depth_diff)

    logging.info(f'check_collision cost time( ms ): {(time.time() - st) * 1000:.1f}')
    logging.info(f'check_collision (bad_cnt/valid_cnt), left: {left_bad_cnt}/{left_valid_cnt}, right: {right_bad_cnt}/{right_valid_cnt}')

    if debug_level >= 1:
        ... # 去畸变、画轮廓、存 pcd，略

    if left_bad_cnt > max(20, 0.4 * left_valid_cnt) or right_bad_cnt > max(20, 0.4 * right_valid_cnt):
        is_obstacled = True
    else:
        is_obstacled = False
    # end if

    return is_obstacled
```

**这段在干什么（业务）：** 碰撞检测的总入口。上层（抓取流程）在准备把一个目标位姿发给机械臂之前，调 `check(...)` 问"在这个位姿张爪去抓，会不会撞？"。返回 `True` 表示会撞，上层就放弃这个位姿。

逐行：
- 第 352 行 `st = time.time()`：记录开始时间，最后第 374 行打印耗时。⚠️ 作者自己打了耗时日志，说明他知道慢（第 7 节第 3 条）。
- 第 355 行（**最绕的一行**，见第 6.3 节）：
  `cam_T_ref_target = inv_tf(ref_T_base_end @ self.T_end_cam) @ (target_T_base_end @ self.T_end_cam)`
  - `ref_T_base_end @ T_end_cam` = 参考视角下相机在基座系的位姿 `T_base_cam_ref`。
  - `inv_tf(...)` 取逆 → `T_cam_ref`（从基座→参考相机）。
  - `target_T_base_end @ T_end_cam` = 目标视角下相机位姿 `T_base_cam_target`。
  - 两者相乘：`T_cam_ref @ T_base_cam_target` = 从 target 相机 → ref 相机 的变换。⚠️ 坑：这个量**语义上是"从 target 相机到 ref 相机"**，但马上被当成 `get_rects_3d` 的 `T_target_cam` 参数用（第 358 行），而那个参数名写着"从相机到目标"。命名完全反着，但数学上恰好对（因为 `get_rects_3d` 里 `T_target_cam @ T_cam_gripper` 需要"相机→目标"）。详见第 7 节第 7 条。
- 第 358 行 `get_rects_3d(gripper_dist, cam_T_ref_target)`：把夹爪顶点从"夹爪系"经 `cam_T_ref_target`（相机→ref相机）变换到"ref 相机系"——即"假设机械臂在目标位姿时，爪子在参考相机看来在哪"。
- 第 361–362 行：初始化两张全零图（掩码 uint8、深度 uint16），和参考深度图同形状。
- 第 364–367 行：分别投影左、右爪，用不同 mask 值区分（200/255）。
- 第 369–372 行：分别数左右爪的坏像素。
- 第 411 行（**判定阈值**，见第 6.4 节）：`bad_cnt > max(20, 0.4*valid_cnt)`。即"坏像素数超过 20 个、或者超过有效像素的 40%"→ 判撞。⚠️ 坑：那个 `20` 是绝对像素数，跟分辨率强相关（第 7 节第 5 条）。
- 第 417 行：返回是否碰撞。

### 3.11 check_arm_pose：检查位姿是否合理（第 424–468 行）

```python
def check_arm_pose(T_base_end: np.ndarray,
                   T_end_cam: np.ndarray,
                   gripper_body: GripperBody,
                   gripper_dist: float,
                   th_angle_z: float,
                   th_gripper_height: float) -> bool:
    """
    检查机械臂位姿是否合理:
        - 末端 Z 轴与基座坐标系的 -Z 轴的夹角小于 th_angle_z
        - 夹爪在基座坐标系下的高度高于 th_gripper_height
    ...
    """

    # 1. 与 -Z 轴的夹角检查
    R_base_end = T_base_end[:3, :3]
    z_dir = R_base_end[:, 2]  # 机械臂末端 Z 轴方向
    z_axis = np.array([0, 0, -1], dtype=np.float32)  # 基座坐标系 Z 轴负方向
    cosine = np.dot(z_dir, z_axis) / (np.linalg.norm(z_dir) * np.linalg.norm(z_axis))
    angle = np.arccos(cosine)
    logging.info(f'Arm end-effector Z axis angle check, angle: {angle * 180.0 / np.pi:.2f} deg')

    if angle > th_angle_z:  # 夹角大于45度
        logging.error(f'Arm end-effector Z axis angle check failed, should be less than {th_angle_z* 180.0 / np.pi:.2f} deg')
        return False
    # end if

    # 2. 夹爪位置检查, z 坐标高于一定高度
    T_base_cam = T_base_end @ T_end_cam
    gripper_rect3_3d = gripper_body.get_rects_3d(gripper_dist, T_base_cam)  # 计算夹爪在基座坐标系下的8个顶点坐标 8*3
    gripper_height = np.min(gripper_rect3_3d[:, 2])  # 夹爪最低点的高度
    logging.info(f'Arm gripper height check, height: {gripper_height:.3f} m')

    if gripper_height < th_gripper_height:  # 夹爪高度低于阈值
        logging.error(f'Arm gripper height check failed, should be higher than {th_gripper_height} m')
        return False
    # end if

    return True
```

**这段在干什么（业务）：** 抓取流程在每一步算出目标末端位姿后，调它做两道安全检查（见 `test_tmpl_grasp_3d.py` 第 320、410、482 行）：
1. **朝向检查**：末端 Z 轴不能太歪（不能偏离"竖直向下"超过 `th_angle_z`，默认 45°）。否则夹爪斜着戳，抓不稳。
2. **高度检查**：夹爪最低点不能低于阈值（默认 -0.01m），否则会插进桌面。

逐行：
- 第 444 行 `R_base_end = T_base_end[:3, :3]`：取旋转块。
- 第 445 行 `z_dir = R_base_end[:, 2]`：末端 Z 轴在基座系下的方向（第 1.3 节性质）。
- 第 446 行 `z_axis = [0,0,-1]`：基座系 -Z 方向（竖直向下）。
- 第 447 行 `cosine = dot(z_dir, z_axis) / (|z_dir|*|z_axis|)`：夹角余弦。⚠️ 坑：这里 `z_dir` 来自旋转矩阵列，本来是单位向量，除以 `norm` 是冗余的（防御性），但**没有 clip**（第 7 节第 11 条）。
- 第 448 行 `np.arccos(cosine)`：由余弦反求角度（弧度）。
- 第 451 行 `if angle > th_angle_z: return False`：超阈值 → 不合格。
- 第 457 行 `T_base_cam = T_base_end @ T_end_cam`：末端位姿乘手眼矩阵 → 相机在基座系下的位姿（作为 `get_rects_3d` 的"目标系"，这里目标系就是基座系）。
- 第 458 行 `get_rects_3d(gripper_dist, T_base_cam)`：算夹爪 8 顶点（基座系下）。
- 第 459 行 `np.min(gripper_rect3_3d[:, 2])`：取所有顶点 z 的**最小**值 = 夹爪最低点高度。
- 第 462 行 `if gripper_height < th_gripper_height: return False`：太低 → 不合格。

### 3.12 compute_axis_aligned_pose：对齐某个轴（第 471–528 行）

```python
def compute_axis_aligned_pose(T_base_end: np.ndarray,
                              base_axis_idx: int,
                              obj_axis_idx: int,
                              T_end_obj: np.ndarray = np.eye(4),
                              th_angle: float = 45.0) -> np.ndarray:
    """
    计算一个新的末端位姿,使得物体坐标系的第 obj_axis_idx 个轴与机械臂基座坐标系的第 base_axis_idx 个轴对齐
    ...
    """

    if abs(base_axis_idx) not in [1, 2, 3]:
        logging.error(f'Invalid base_axis_idx: {base_axis_idx}')
        return None
    # end if

    if abs(obj_axis_idx) not in [1, 2, 3]:
        logging.error(f'Invalid obj_axis_idx: {obj_axis_idx}')
        return None
    # end if

    T_base_obj = T_base_end @ T_end_obj

    # 期望的对齐方向, 由 base_axis_idx 决定
    target_dir = np.eye(3)[:, abs(base_axis_idx) - 1] * np.sign(base_axis_idx)

    # 当前的方向, 由 obj_axis_idx 决定
    current_dir = T_base_obj[:3, abs(obj_axis_idx) - 1] * np.sign(obj_axis_idx)

    # 计算夹角
    cosine = np.dot(target_dir, current_dir)
    cosine = np.clip(cosine, -1.0, 1.0)  # 数值稳定性
    angle = np.arccos(cosine)
    logging.info(f'compute_axis_aligned_pose, target_dir: {GREEN}{target_dir}{RESET}, '
                 f'current_dir: {GREEN}{current_dir}{RESET}, '
                 f'angle: {GREEN}{angle * 180.0 / np.pi:.2f}{RESET} deg')

    if angle > np.deg2rad(th_angle):  # 夹角过大
        logging.warning(f'angle > {th_angle} deg, skip align')
        return None
    # end if

    # 计算调整后的姿态
    axis = np.cross(current_dir, target_dir)
    delta_R = transforms3d.axangles.axangle2mat(axis, angle)

    target_T_base_obj = T_base_obj.copy()
    target_T_base_obj[:3, :3] = delta_R @ T_base_obj[:3, :3]
    target_T_base_end = target_T_base_obj @ inv_tf(T_end_obj)

    return target_T_base_end
```

**这段在干什么（业务）：** 见第 1.6 节。典型调用在 `calib_gripper.py` 第 397 行（`base_axis_idx=-3, obj_axis_idx=3`：让末端 Z 轴指向下方）和 `test_tmpl_grasp_2d.py` 第 457 行（同样参数）。返回新的末端位姿，机械臂照着走就能把某轴转正。

逐行（推导见第 6.6 节）：
- 第 488–496 行：检查轴索引合法（±1/±2/±3）。
- 第 498 行 `T_base_obj = T_base_end @ T_end_obj`：把物体位姿换到基座系下。
- 第 501 行 `target_dir = np.eye(3)[:, abs(base_axis_idx)-1] * np.sign(base_axis_idx)`：`np.eye(3)` 是第 1.3 节的"轴矩阵"，取第 `abs(idx)-1` 列得到对应轴方向，再乘 `sign` 决定正负。如 `base_axis_idx=-3` → 取第 2 列 `[0,0,1]` 乘 `-1` → `[0,0,-1]`（向下）。
- 第 504 行 `current_dir = T_base_obj[:3, abs(obj_axis_idx)-1] * np.sign(...)`：物体在基座系下那根轴的当前方向（旋转矩阵第 N 列，第 1.3 节）。
- 第 507–509 行：`np.clip(cosine, -1, 1)` 把数值误差导致的 `1.0000001` 夹回合法范围，避免 `arccos` 返回 NaN。✅ 这是好的写法（对比 `check_arm_pose` 没 clip，第 7 节第 11 条）。
- 第 514 行 `if angle > np.deg2rad(th_angle):`：`th_angle` 单位是**度**，这里转弧度比。⚠️ 坑：和模块顶部 `TH_ANGLE_Z`（弧度）单位不一致（第 7 节第 9 条）。
- 第 520–521 行：`axis = cross(current, target)` 旋转轴；`axangle2mat(axis, angle)` 把"绕 axis 转 angle"变成旋转矩阵（罗德里格斯公式）。⚠️ 坑：当 `angle≈0` 时 `cross` 接近零向量，`axangle2mat` 内部归一化会除零 → NaN 旋转矩阵（第 7 节第 8 条）。
- 第 523–525 行：只改旋转块（左乘 `delta_R`），**平移块是 copy 原值** → 末端位置不变，绕末端法兰原点转（不是绕爪尖转，第 7 节第 10 条、第 6.6 节）。
- 第 525 行 `target_T_base_end = target_T_base_obj @ inv_tf(T_end_obj)`：把"物体该到的位姿"换回"末端该到的位姿"。

---

## 4. Python 基础语法速查

每条给一个最小可运行示例。

**1. 类的定义与 `self`**
```python
class Dog:
    def __init__(self, name):
        self.name = name      # 实例属性
    def bark(self):
        print(self.name, "wang")
d = Dog("A")                  # 造实例
d.bark()                      # 调方法，self 自动传 d
```

**2. 实例属性**：`self.x = x` 后可用 `obj.x` 读取（第 56–58 行）。

**3. 默认参数 `np.eye(4)`**：`def f(x=np.eye(4))` 不传 x 就用单位阵（第 47 行）。

**4. `assert` 与自定义报错**：`assert 条件, "报错信息"`（第 69 行）。`raise ValueError("...")` 主动抛错（第 79 行）。区别：`assert` 在 `python -O` 下被删除。

**5. 私有方法下划线**：`_proj_rect_3d` 以单下划线开头，是"内部方法"的约定（不是强制私有）。

**6. `np.column_stack` / `np.vstack`**：
```python
np.column_stack(([1,0,0],[0,1,0],[0,0,1]))  # 把列向量并成 3x3 矩阵
np.vstack((a, b))                           # 上下拼两个数组
```

**7. `np.cross` / `np.dot` / `np.linalg.norm`**：
```python
np.cross([1,0,0],[0,1,0])   # -> [0,0,1]  叉乘（垂直向量）
np.dot([1,0,0],[0,1,0])     # -> 0        点乘
np.linalg.norm([3,4])       # -> 5.0      长度
```

**8. 布尔掩码索引**：
```python
a = np.array([1,2,3,4])
a[a > 2]                     # -> array([3, 4])  只取满足条件的元素
```

**9. `np.where` 返回 `(rows, cols)`**：
```python
img = np.array([[0,1],[1,0]])
np.where(img == 1)           # -> (array([0,1]), array([1,0]))  即 (行下标, 列下标)
```
第 258 行 `vs, us = np.where(...)` 中 `vs` 是行(v)、`us` 是列(u)。

**10. `zip` 双循环**：
```python
for u, v in zip([10,20], [30,40]):   # 配对：(10,30) (20,40)
    print(u, v)
```

**11. `cv2.fillPoly` / `boundingRect` / `polylines`**：
```python
cv2.fillPoly(mask, [contour], color=255)   # 把轮廓内部填成 255（contour 要包一层列表）
cv2.boundingRect(contour)                  # -> (x, y, w, h) 外接矩形
cv2.polylines(img, [contour], isClosed=True, color=(255,0,0), thickness=1)
```

**12. `time.time()` 计时**：
```python
st = time.time()
... 干活 ...
print((time.time() - st) * 1000, "ms")    # 毫秒
```

**13. 整除与 `np.rint`**：`//` 是整数除法，`np.rint` 四舍五入成整数。`int()` 是**截断**（向 0 舍入），不是四舍五入（第 248 行）。

**14. 嵌套函数**：`grippers_to_msg` 里 `create_marker` 定义在函数内部（arm_ros_utils，第 75 行），能直接用外层变量。

**15. `enumerate`**：
```python
for i, x in enumerate(["a","b"]):   # i=0,x="a" 然后 i=1,x="b"
    print(i, x)
```

---

## 5. 输入输出规范

**1. `gripper.json`（夹爪标定文件）字段**（由 `calib_gripper.py` 写出，见其第 456–466 行）：
```json
{
  "width": 0.0015,
  "thickness": 0.002,
  "T_cam_gripper": [[...4x4 浮点...]]
}
```
- `width`：夹爪宽度（米）。
- `thickness`：夹爪厚度（米）。
- `T_cam_gripper`：4×4 矩阵（列表的列表），夹爪→相机。

**2. `get_rects_3d` 返回的 8×3 点序**（第 110 行注释）：
- 前 4 行 = 左爪，顺序 `[左下, 右下, 右上, 左上]`（按 y 从高到低、x 从外到内）。
- 后 4 行 = 右爪，同样顺序。
- `rects_3d[:, 2]` 是各点 z 坐标；`np.min(rects_3d[:, 2])` 即夹爪最低点高度。

**3. `check()` 的返回语义**：
- 返回 `True` = **会碰撞**（obstacled），不应执行该位姿。
- 返回 `False` = 安全。
- 日志会打 `bad_cnt/valid_cnt`（左右爪各一组），可据此调阈值。

**4. `compute_axis_aligned_pose` 返回语义**：
- 返回 4×4 矩阵 = 对齐后的新末端位姿。
- 返回 `None` = 轴索引非法，或当前夹角已超过 `th_angle`（拒绝调整）。调用方必须判 `None`（见 `calib_gripper.py` 第 400 行）。

---

## 6. 核心数学：一步一步算给你看

### 6.1 GripperBody.initialize 用两条对角线建坐标系的完整推导

假设 AprilTag 四角点（相机系下）近似为一个正方形，按 OpenCV/AprilTag 惯例编号 0→1→2→3 逆时针：

```
         角1
        /   \
     角0     角2
        \   /
         角3
```
（实际 AprilTag 角点顺序是 0(左上)→1(右上)→2(右下)→3(左下) 顺时针，但下面的向量推导只看相对位置。）

- `edge_x = corners[2] - corners[0]`：第 0 号到第 2 号是**一条对角线**（横穿正方形）。
- `edge_y = corners[1] - corners[3]`：第 3 号到第 1 号是**另一条对角线**（垂直方向横穿）。

两条对角线互相**垂直**，所以拿它们当 X、Y 轴，相当于把坐标系"转了 45°"相对于用边当轴。

**正交化（第 84–87 行）为什么必要**：即使对角线理论上垂直，数值噪声会让它们不完全垂直。`axis_y = axis_y_raw - (axis_y_raw·axis_x)*axis_x` 是"把 Y 在 X 上的投影减掉"，剩下纯垂直分量。举例：
- 设 `axis_x = [1, 0, 0]`，`axis_y_raw = [0.1, 1, 0]`（略带 x 分量）。
- `axis_y_raw·axis_x = 0.1`。
- `axis_y = [0.1,1,0] - 0.1*[1,0,0] = [0, 1, 0]`。垂直了。✅

**Z = X × Y（第 90 行）**：叉乘结果垂直 X、Y，构成右手系。`axis_z /= norm` 归一化。

**为什么 `assert axis_z[2] > 0`（第 92 行）**：期望夹爪 Z 轴在相机系下朝前/朝上（z 分量正）。如果贴反了（tag 镜像/绕错），Z 会朝下（z 负），断言失败提示"方向错了"。

**ASCII 图：AprilTag 必须贴成菱形（转 45°）**

用边当轴（`compute_tag_pose`，物体定位）：
```
 边(0→1) = X轴
 +-----------+
 | 1       2 |
 |           |
 | 0       3 |
 +-----------+
 边(0→3) = Y轴
```
此时 X 轴水平、Y 轴竖直（正方向）。

用对角线当轴（`initialize`，夹爪标定）：
```
 角1
  |\  <- 对角线(3→1) 当 Y轴
  | \
 角0--角2   对角线(0→2) 当 X轴(爪张开方向)
  | /
 角3
```
为了让 `initialize` 的 X 轴（0→2 对角线）**正好等于夹爪张开方向**，必须让这条对角线水平——也就是把 tag **转 45° 贴成菱形**。否则（正着贴）"0→2 对角线"是斜的 45°，算出的夹爪 X 轴就歪 45°，后续 `T_cam_gripper` 整体错 45° → 抓取全歪。

> 这就是"重点问题 1"的实际后果：**标定夹爪的 tag 要菱形贴；定位物体的 tag 要正着贴**。两套约定差 45°，混用必错。

### 6.2 get_rects_3d 的 8 个角点坐标怎么排

俯视图（夹爪坐标系下，X 右、Y 上、Z 垂直纸面朝外）。`hd = dist/2`，`hw = width/2`，`t = thickness`：

```
        Y(上)
        ^
        |     右爪            左爪
        |   [hd,hw]──[hd+t,hw]
        |     |          |
        |   [hd,-hw]─[hd+t,-hw]        (右爪四点, x>0)
        |
        |   [-hd-t,hw]──[-hd,hw]
        |      |           |
        |   [-hd-t,-hw]─[-hd,-hw]      (左爪四点, x<0)
        +----------------------------> X(右, 爪张开方向)
```
- 左爪四点（第 119–122 行）x 在 `-hd-t` 与 `-hd` 之间（爪在 X 负侧，厚度朝更负），y 在 `hw` 到 `-hw`。顺序 `[-hd-t,hw]→[-hd,hw]→[-hd,-hw]→[-hd-t,-hw]` = 左下→右下→右上→左上（注释第 110 行说"左下,右下,右上,左上"，注意它把 y 正当成"上"）。
- 右爪（第 124–127 行）镜像到 x>0 侧。
- `z = 0`：所有点都在夹爪系 z=0 平面——**零厚度薄片**（⚠️ 坑第 6 条：可视化靠 `gripper_length=0.05` 拉成体，但碰撞检测用薄片，两处不一致）。

`T_target_gripper = T_target_cam @ T_cam_gripper` 把"夹爪系坐标"变到"目标系坐标"。每个齐次点 `[x,y,z,1]^T` 左乘该矩阵即得目标系坐标。

### 6.3 碰撞检测核心：投影、深度差、反直觉的判定条件

**（1）cam_T_ref_target 的推导（第 355 行）**

记 `C = self.T_end_cam`（相机→末端，即 `T_end_cam`）。
- 参考视角相机在基座系位姿：`T_base_cam_ref = ref_T_base_end @ C`。
- 目标视角相机在基座系位姿：`T_base_cam_target = target_T_base_end @ C`。
- 要的是"从 target 相机看，ref 相机在哪"：`T_cam_target_cam_ref`（即代码里的 `cam_T_ref_target`）。
  $$
  T_{cam\_target\_cam\_ref} = (T_{base\_cam\_ref})^{-1} \cdot T_{base\_cam\_target}
  = \text{inv\_tf}(ref\_T_{base\_end} @ C) @ (target\_T_{base\_end} @ C)
  $$
  下标链：`base→cam_ref` 的逆 = `cam_ref→base`，再 `@` `base→cam_target` = `cam_ref→cam_target`。✅ 与代码一致。

**注意**：这个量的语义是"从 target 相机到 ref 相机"，但传给 `get_rects_3d` 当 `T_target_cam`（参数注释写"相机→目标"）。名字反了，但数学对——因为 `get_rects_3d` 需要"相机→目标(这里目标是 ref 相机)"来把爪点变到 ref 相机系。见第 7 节第 7 条。

**（2）矩形投影到图像（第 248 行）**

针孔相机模型：相机系下点 `(x,y,z)`（z 是深度，前方为正），投到像素：
$$u = x\cdot f_x / z + c_x,\quad v = y\cdot f_y / z + c_y$$
代码用 `int()` 截断（⚠️ 半像素误差）。

**（3）逐像素求深度 `z = -D/(A·nx + B·ny + C)`（第 264 行）**

平面方程：`A x + B y + C z + D = 0`，法向量 `n=(A,B,C)`。
像素 `(u,v)` 对应的视线方向（相机系）为 `(nx, ny, 1)`，其中 `nx=(u-cx)/fx, ny=(v-cy)/fy`（已去掉内参，相当于 z=1 平面上的点）。
视线上任一点可写为 `k·(nx, ny, 1)`（k 为缩放）。代入平面方程：
$$A(k\,nx) + B(k\,ny) + C(k\cdot1) + D = 0$$
$$k\,(A\,nx + B\,ny + C) = -D$$
$$k = \frac{-D}{A\,nx + B\,ny + C}$$
而这个 k 正是该视线与平面交点的 **z（深度）**。所以 `z = -D/(A*nx + B*ny + C)`。✅

**（4）反直觉的判定：`d_proj > d_real + th` = 撞（第 321 行）**

深度图里**值越大 = 离相机越远**。
- `d_proj`：如果爪子表面在这个像素，它"应该"离相机多远。
- `d_real`：真实环境表面离相机多远。

如果 `d_proj > d_real`：爪子表面比真实表面**更远（更靠后）**。可爪子是要"伸到"这个位置的——它本该停在真实表面**前面**（更近）。现在算出来它该在真实表面**后面**，说明爪子得穿过真实表面才能到那 → **插进去了 = 撞**。

直觉反的地方：很多人以为"爪子比东西近=撞"，其实这里"爪子更深(值更大)=撞"。记住：**深度值大 = 远 = 后；爪子算出来比真实还后，就是穿模。**

### 6.4 阈值 bad_cnt > max(20, 0.4*valid_cnt) 的含义与调参（第 411 行）

- `valid_cnt`：矩形覆盖的、有真实深度数据的像素总数。
- `bad_cnt`：其中爪子比真实表面深超过 `th_diff` 的像素数。
- 判定：`bad_cnt > max(20, 0.4*valid_cnt)`。即满足"坏像素 > 20 **且** 坏像素占比 > 40%"才算撞。两个同时满足（取 max 的"或"语义：只要超过两者中较大者）。

**为什么用 max**：`0.4*valid_cnt` 是相对比例（防止大面积轻微穿透被忽略），`20` 是绝对下限（防止少量噪声像素误判）。⚠️ 坑：那个 `20` 是**绝对像素数**，和图像分辨率强绑定——换更高分辨率相机，`valid_cnt` 暴涨，但 20 不变，导致同样物理穿模可能判不撞（第 7 节第 5 条）。建议改成相对或按面积比例。

### 6.5 check_arm_pose 两个判据的几何含义

1. **夹角判据（第 443–454 行）**：`z_dir`（末端 Z 轴）与 `[0,0,-1]`（向下）的夹角 `angle`。`angle > th_angle_z`（默认 45°）→ 末端太歪，夹爪斜戳，不稳。几何意义：要求夹爪"大致朝下"。
2. **高度判据（第 456–465 行）**：夹爪 8 顶点最小 z（最低点）`< th_gripper_height`（默认 -0.01m）→ 夹爪低于桌面，会插桌。几何意义：要求夹爪"离桌面足够高"。

### 6.6 compute_axis_aligned_pose 的推导

目标：让物体系第 `obj_axis_idx` 轴对齐到基体系第 `base_axis_idx` 轴。

- `target_dir`：基座系下目标方向（第 1.3 节轴矩阵取列×符号）。
- `current_dir`：物体当前在基座系下那根轴的方向（旋转矩阵列×符号）。
- 夹角 `angle = arccos(clip(dot, -1, 1))`。
- 旋转轴 `axis = cross(current, target)`：同时垂直两者，绕它转能把 current 转到 target。
- `delta_R = axangle2mat(axis, angle)`。
- **左乘 vs 右乘（第 524 行）**：`delta_R @ T_base_obj[:3,:3]` 是**左乘** = 在**父坐标系（基座）**下施加旋转（我们要在基座系下把物体轴转正，所以左乘）。右乘则会在物体自身系下转。
- `target_T_base_end = target_T_base_obj @ inv_tf(T_end_obj)`：把"物体该到的位姿"换回"末端该到的位姿"（中间 `obj` 消掉）。
- ⚠️ **只改旋转块**：平移 `target_T_base_obj[:3,3]` 是 `copy` 原 `T_base_obj` 的平移。所以末端原点位置不变 → 绕**末端法兰中心**转，不是绕爪尖（第 7 节第 10 条）。

### 6.7 数值演练

**（a）手算一次 get_rects_3d（简化）**

设 `width=0.04, thickness=0.01, dist=0.06`，则 `hd=0.03, hw=0.02, t=0.01`。
忽略旋转（`T_target_cam @ T_cam_gripper = I`，即目标系=夹爪系）：

左爪四点（夹爪系）：
- `[-hd-t, hw, 0] = [-0.04, 0.02, 0]`
- `[-hd, hw, 0] = [-0.03, 0.02, 0]`
- `[-hd, -hw, 0] = [-0.03, -0.02, 0]`
- `[-hd-t, -hw, 0] = [-0.04, -0.02, 0]`

右爪四点：
- `[hd, hw, 0] = [0.03, 0.02, 0]`
- `[hd+t, hw, 0] = [0.04, 0.02, 0]`
- `[hd+t, -hw, 0] = [0.04, -0.02, 0]`
- `[hd, -hw, 0] = [0.03, -0.02, 0]`

即左爪占 x∈[-0.04,-0.03]、右爪占 x∈[0.03,0.04]、y∈[-0.02,0.02]，两片爪间距 0.06，符合预期。✅

**（b）手算一次投影（简化）**

设相机内参 `fx=fy=500, cx=cy=320`，夹爪系一点 `P=(0.03, 0, 0.5)`（右爪中心，深度 z=0.5m）。
投影：
- `u = 0.03*500/0.5 + 320 = 30 + 320 = 350`
- `v = 0*500/0.5 + 320 = 320`

该像素在图像 (350, 320)。若 `d_real` 在此像素=0.5m 对应深度值，`d_proj` 也算≈0.5，则 `d_proj > d_real+th` 不成立 → 不撞。若该处真实桌面只有 0.45m（更近），则 `d_real < d_proj`，`d_proj > d_real+th` 成立 → 撞（爪子要穿过桌面）。✅

---

## 7. 这段代码里的坑与改进建议

| # | 位置 | 现象 | 根因 | 改法 |
|---|---|---|---|---|
| 1 | 第 76–77 行 vs `vision_utils` 第 376–380 行 | 夹爪标定与物体定位的 tag 坐标系差 45° | 一个用对角线、一个用边建轴 | 全文统一一种约定；标定贴 tag 时按约定转 45° 菱形贴，并在文档/注释写明 |
| 2 | 第 92 行 `assert axis_z[2] > 0` | `python -O` 下断言被删，方向错误查不出；失败信息无数值 | 用 assert 做业务校验 | 改成抛异常并带数值（见下） |
| 3 | 第 258–259、307–308 行 | 逐像素 Python 双重循环，碰撞检测很慢（日志第 374 行自证） | 没向量化 | 用 numpy 一次性算（见下） |
| 4 | 第 248–249 行 | 半像素误差；角点可能落图像外越界 | `int()` 截断、无 clamp | 用 `np.rint` 四舍五入并 clamp 到 `[0, H-1]/[0,W-1]` |
| 5 | 第 411 行 `max(20, 0.4*valid_cnt)` | 换分辨率要重调阈值 | `20` 是绝对像素数，跟分辨率强相关 | 改用相对比例或按像素面积比例 |
| 6 | 第 119–127 行 | 碰撞用零厚度薄片，可视化用 0.05 长方体，两处不一致 | `get_rects_3d` 全在 z=0 | 明确薄片是近似；或让碰撞也用厚度方向拉伸 |
| 7 | 第 155、358 行 | `T_end_cam` 注释"相机→末端"，实际与 `cam_T_ref_target` 方向相反 | 命名混乱但用法对 | 统一命名（如存为 `T_cam_end`），注释写清真实方向 |
| 8 | 第 520–521 行 | `angle≈0` 时 `cross` 近零 → `axangle2mat` 归一化除零 → NaN 矩阵 | 没保护零角 | 加 `if angle < 1e-6: return T_base_end.copy()`（见下） |
| 9 | 第 514 行 vs 第 26 行 | `th_angle` 是度、`TH_ANGLE_Z` 是弧度，同模块两单位 | 约定不统一 | 统一弧度，或函数内显式标注单位并转 |
| 10 | 第 523–525 行 | 只改旋转块，绕末端法兰转，不是绕爪尖转 | 平移块 copy 原值 | 若需绕爪尖，先把原点平移到爪尖再转 |
| 11 | 第 447–448 行 | `cosine` 未 clip，数值误差可能 >1 → `arccos` 出 NaN | 与第 508 行对比，这里漏了 clip | 加 `cosine = np.clip(cosine, -1.0, 1.0)` |

**改法 1（第 92 行，替换 assert）：**
```python
if axis_z[2] <= 0:
    raise ValueError(
        f"gripper Z axis must point +z in camera frame, got axis_z={axis_z}. "
        f"Check AprilTag orientation (should be rotated 45deg / diamond)."
    )
```

**改法 2（第 8 条，compute_axis_aligned_pose 加保护，放在第 519 行前）：**
```python
if angle < 1e-6:
    # 已经对齐，无需旋转，直接返回副本避免除零
    return T_base_end.copy()
```

**改法 3（第 3 条，向量化投影深度，替换第 258–274 行）：**
```python
ys, xs = np.where(proj_mask_img == mask_value)
nx = (xs - cx) / fx
ny = (ys - cy) / fy
denom = A * nx + B * ny + C
z = np.zeros_like(denom)
valid = denom > 1e-9
z[valid] = -D / denom[valid]
z[(z <= 0.001) | (~valid)] = 0.0
d = (z / self.depth_scale).astype(np.uint16)
# 只更新比已有更深的像素
update = (d > proj_depth_img[ys, xs])
proj_depth_img[ys[update], xs[update]] = d[update]
```
这样把 Python 双层循环换成 numpy 向量化，速度可提升一个数量级。

**改法 4（第 4 条，投影取整与 clamp，替换第 248–249 行）：**
```python
u = int(np.clip(np.rint(pt3d[0] * fx / pt3d[2] + cx), 0, proj_mask_img.shape[1] - 1))
v = int(np.clip(np.rint(pt3d[1] * fy / pt3d[2] + cy), 0, proj_mask_img.shape[0] - 1))
```

**改法 5（第 11 条，check_arm_pose clip）：**
```python
cosine = np.dot(z_dir, z_axis) / (np.linalg.norm(z_dir) * np.linalg.norm(z_axis))
cosine = np.clip(cosine, -1.0, 1.0)   # 加这行
angle = np.arccos(cosine)
```

---

## 8. 一句话总结

`arm_utils.py` 是抓取系统的"纯数学工具箱"：用 `GripperBody` 把夹爪近似成两个矩形并算出 8 个顶点；用 `CollisionDetector.check` 把目标位姿下的爪子投影到参考深度图、比较深度来判断会不会撞；用 `check_arm_pose` 检查末端朝向和高度是否合法；用 `compute_axis_aligned_pose` 算出一个"把某轴转正"的新末端位姿。读它要记住三件事：**(1) 变换矩阵下标 `T_a_b` = 从 b 到 a，相乘时相邻下标相消；(2) 旋转矩阵第 N 列就是子坐标系第 N 个轴的方向；(3) 深度图"值越大越远"，所以"爪子算出来比真实更深=撞"这个判定是反直觉的。** 最要小心的两个真问题：夹爪标定用对角线建轴、物体定位用边建轴，差 45°（贴 tag 要菱形）；以及 `compute_axis_aligned_pose` 在夹角≈0 时会除零出 NaN 矩阵。

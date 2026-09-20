# `vision_utils.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、没学过线性代数、没用过 OpenCV / Open3D 的同学。
> 目标：读完之后，你能逐行看懂这个文件在干什么，知道每一个数字是怎么算出来的，知道它在整个抓取系统里被谁调用、为什么需要它，并且顺手把 Python 基础语法也补上。
>
> 被分析的源文件：
> `C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\core\vision_utils.py`（共 876 行）

---

## 目录

- [0. 一句话概括](#0-一句话概括)
- [1. 背景知识：这个模块在整套系统里的位置](#1-背景知识这个模块在整套系统里的位置)
  - [1.1 从"相机拍到的一张图"到"物体在相机前的位姿"](#11-从相机拍到的一张图到物体在相机前的位姿)
  - [1.2 针孔相机模型复习：像素 → 归一化坐标 → 三维射线](#12-针孔相机模型复习像素--归一化坐标--三维射线)
  - [1.3 深度图是什么、depth_scale 是什么、为什么深度图常有洞](#13-深度图是什么depth_scale-是什么为什么深度图常有洞)
  - [1.4 AprilTag 是什么、为什么用它](#14-apriltag-是什么为什么用它)
  - [1.5 点云是什么、Open3D 是什么](#15-点云是什么open3d-是什么)
  - [1.6 平面方程与"射线和平面求交"](#16-平面方程与射线和平面求交)
- [2. 模块地图：函数与类一览表](#2-模块地图函数与类一览表)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 导入区（第 5–21 行）](#31-导入区第-521-行)
  - [3.2 `rgbd_to_point_cloud`（第 26–61 行）](#32-rgbd_to_point_cloud第-2661-行)
  - [3.3 `depth_to_point_cloud`（第 64–97 行）](#33-depth_to_point_cloud第-6497-行)
  - [3.4 `depth_mean_filter`（第 100–142 行）](#34-depth_mean_filter第-100142-行)
  - [3.5 `compute_locate_error`（第 145–182 行）](#35-compute_locate_error第-145182-行)
  - [3.6 `compute_projective_transformation`（第 185–248 行）](#36-compute_projective_transformation第-185248-行)
  - [3.7 `compute_tag_pose_2d`（第 251–273 行）](#37-compute_tag_pose_2d第-251273-行)
  - [3.8 `compute_tag_mask`（第 276–301 行）](#38-compute_tag_mask第-276301-行)
  - [3.9 `compute_tag_corners3d`（第 304–362 行）](#39-compute_tag_corners3d第-304362-行)
  - [3.10 `compute_tag_pose`（第 365–395 行）](#310-compute_tag_pose第-365395-行)
  - [3.11 `ImageUndistorter`（第 401–485 行）](#311-imageundistorter第-401485-行)
  - [3.12 `TagMatcher2D`（第 488–641 行）](#312-tagmatcher2d第-488641-行)
  - [3.13 `TagMatcher3D`（第 644–876 行）](#313-tagmatcher3d第-644876-行)
- [4. Python 基础语法速查](#4-python-基础语法速查)
- [5. 输入输出规范](#5-输入输出规范)
- [6. 核心数学：一步一步算给你看](#6-核心数学一步一步算给你看)
  - [6.1 反投影：从 (u,v,depth) 到 (x,y,z)](#61-反投影从-uvdepth-到-xyz)
  - [6.2 射线与平面求交 z = -D/(A·nx+B·ny+C) 的推导](#62-射线与平面求交-z--da-nxb-nyc-的推导)
  - [6.3 用三个角点定平面法向量：叉乘](#63-用三个角点定平面法向量叉乘)
  - [6.4 用角点建坐标系：边向量与正交化](#64-用角点建坐标系边向量与正交化)
  - [6.5 旋转矩阵的迹求夹角](#65-旋转矩阵的迹求夹角)
  - [6.6 射影变换 H = K_dst · R · K_src⁻¹](#66-射影变换-h--k_dst--r--k_src)
  - [6.7 depth_mean_filter 的逐像素统计（手算示例）](#67-depth_mean_filter-的逐像素统计手算示例)
  - [6.8 对角外延 ex_ratio 的几何含义](#68-对角外延-ex_ratio-的几何含义)
- [7. 这段代码里的坑与改进建议](#7-这段代码里的坑与改进建议)
- [8. 一句话总结](#8-一句话总结)

---

## 0. 一句话概括

> **这个文件把"相机看到的一张彩色图 + 一张深度图"变成"贴在某物体上的 AprilTag 在相机坐标系下的三维位姿（位置 + 朝向）"，同时提供去畸变、点云生成、对称误差评估、掩码生成、射影变换等一堆机器人抓取前处理工具。**

它在整套抓取系统里的角色是"眼睛到大脑的翻译官"：上游是相机（`CamNode` 给出的彩色图和深度图），下游是抓取的视觉伺服/位姿计算（`test_tmpl_grasp_2d.py` / `test_tmpl_grasp_3d.py`）。没有它，机械臂就不知道物体在哪、该怎么摆。

**输入 → 输出（以 3D 版为例）：**

| | 内容 |
|---|---|
| 输入 1 | `bgr_img`：彩色图（OpenCV 的 BGR 格式，CV_8UC3） |
| 输入 2 | `depth_img`：深度图（每个像素一个距离值，单位由 depth_scale 决定） |
| 输入 3 | `intrinsic`：相机内参 `[fx, fy, cx, cy]` |
| 输入 4 | `depth_scale`：深度缩放系数 |
| 输出 | 一个 `Result` 列表，每个元素含 `T_cam_tag`（4×4 位姿矩阵，描述 tag 相对相机的位姿） |

---

## 1. 背景知识：这个模块在整套系统里的位置

### 1.1 从"相机拍到的一张图"到"物体在相机前的位姿"

机器人抓取要解决的核心问题是：**"物体到底在相机坐标系的哪个位置、摆成什么角度？"**

相机拍出来的是**二维图像**（一张像素网格），而机器人要移动的是**三维空间坐标**。两者之间的转换需要好几步，本模块就负责其中"识别标签 → 估计标签位姿"这一段：

```
相机原始帧 (彩色 + 深度)
      │
      ▼
[去畸变]  ImageUndistorter.undistort_img
      │  把被镜头"扭曲"的像素掰直
      ▼
[识别]    apriltag2.Detector.detect
      │  找到画面里的 AprilTag，给出 2D 角点
      ▼
[反投影]  compute_tag_corners3d
      │  用深度图把 2D 角点"抬"成 3D 角点
      ▼
[拟合平面] pc.segment_plane
      │  标签贴在物体表面，表面是平面 → 拟合出一个平面
      ▼
[建坐标系] compute_tag_pose
      │  用角点建立 tag 的局部坐标系
      ▼
[输出]    T_cam_tag   (物体相对相机的 4×4 位姿矩阵)
      │
      ▼
下游: compute_ready_pose / compute_delta_end_pose → 机械臂该去哪
```

2D 版（`TagMatcher2D`）更简单：它不求深度，只求标签中心在"归一化平面"上的坐标 `(nx, ny)` 和一条朝向角 `theta`，用于"物体平放在桌上、只需 x/y/旋转"的场景（见 `test_tmpl_grasp_2d.py`）。

### 1.2 针孔相机模型复习：像素 → 归一化坐标 → 三维射线

先把术语讲清楚：

- **像素坐标 (u, v)**：图像是一张网格，`u` 是列（横向，从左到右）、`v` 是行（纵向，从上到下）。单位是"第几个像素"。
- **相机内参 `[fx, fy, cx, cy]`**：相机的"身份证"。
  - `fx, fy`：焦距，单位是"像素"。意思是一米远的东西，在图像上被放大成多少像素。
  - `cx, cy`：主点（principal point），就是相机光轴（镜头正中心）打在图像上的像素位置，通常接近图像中心。

**针孔相机模型**（把镜头想成一个小针孔，光直线穿过它投到后面的感光面上）：

```
         物体点 (X, Y, Z) 在相机坐标系里
                  │
                  │ 光线沿直线穿过针孔(原点)
                  ▼
             ┌───────── 成像平面(在针孔前方,距离=焦距 f) ─────────┐
             │   像素 (u, v)                                        │
             └──────────────────────────────────────────────────────┘

相似三角形告诉我们：
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
```

为什么是 `X/Z`？因为物体越远（Z 越大），它在图像上就被压得越小——除以深度 Z 就是"透视缩小"。`fx` 把"米"放大成"像素"，`cx/cy` 把原点从图像左上角挪到光轴位置。

**反过来（反投影）**：已知像素 `(u, v)` 和深度 `Z`（从深度图得到），求相机坐标系下的 3D 点 `(X, Y, Z)`：

```
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
```

本文件里反复出现 `(u - cx)/fx` 这种写法，它就是"把像素坐标变成**归一化坐标**（去掉焦距和主点影响），再乘深度得到三维坐标"。**归一化坐标** `(nx, ny)` 可以理解为"假设焦距是 1 米时的像素坐标"，它只描述方向、不描述远近。

### 1.3 深度图是什么、depth_scale 是什么、为什么深度图常有洞

**彩色图**每个像素存的是颜色（R、G、B 三个 0~255 的整数）。**深度图**每个像素存的是"这个像素对应的点到相机的距离（深度）"，单位通常是毫米或米。

深度图有两种常见存储方式，本文件两种都涉及：

- **CV_16UC1**：16 位无符号整数（`uint16`），每个像素 0~65535。常见的是"原始深度"，比如 RealSense 出的深度是"毫米整数"，`500` 表示 0.5 米。
- **CV_32FC1**：32 位单精度浮点（`float32`），每个像素存真实的小数米数，比如 `0.5` 表示 0.5 米。

**depth_scale 是什么**：深度相机硬件测出的往往不是"米"，而是一个需要换算的整数。比如硬件读出 `500`，实际是 0.5 米，那么 `depth_scale = 0.001`（米/计数），真实深度 = 读数 × depth_scale。本模块 `read_rgbd_params` 从 JSON 读出它。`compute_tag_corners3d` 第 327 行 `flt_depth_img = depth_img.astype(np.float32) * depth_scale` 就是这个换算。

**为什么深度图常有"洞"（像素值为 0 或无效）**：
1. 物体边缘、强光、透明/反光表面，深度相机算不出距离。
2. 距离太近或太远，超出相机量程。
3. 两个表面重叠的遮挡区。

所以本文件里大量代码在做"掩码过滤"：把深度为 0 / 无效的像素剔除。比如 `compute_tag_corners3d` 第 328 行 `flt_depth_img[mask_img == 0] = 0.0`——只保留 tag 区域里的深度。

### 1.4 AprilTag 是什么、为什么用它

**AprilTag** 是一种印在纸上的二维方形条码（黑白方块拼成的图案，类似二维码但更简单）。它的好处：

1. **检测快、稳**：`apriltag2` 库能在毫秒级从图像里找出所有 tag，并给出四个角点的精确像素坐标 `corners`（顺序固定，见下文）。
2. **自带编号**：每个 tag 有个唯一 `id`，可以区分多个物体。
3. **能算位姿**：因为 tag 是已知尺寸的平面方块，知道它在图上的四个角点，就能反推出它相对相机的三维位姿（这就是 PnP / 平面拟合在做的事）。

本工程把 AprilTag 贴在**物体表面**和**夹爪**上，于是相机拍一张图就知道"物体在哪、夹爪在哪"。

`apriltag2` 是外部已安装的三方包，本文件只用它三个接口：
- `apriltag2.Detector(tag_family=..., black_border=...)`：创建检测器。
  - `tag_family`：tag 的图案家族，默认 `"tag36h11"`。
  - `black_border`：tag 外围黑边框宽度（格数）。
- `detector.detect(img, -1)`：在图像 `img` 上检测，返回 `Tag2D` 列表。
- `Tag2D` 对象有三个属性：
  - `.id`：标签编号（整数）。
  - `.center`：标签中心点的像素坐标 `(u, v)`。
  - `.corners`：标签四个角点的像素坐标，形状 `(4, 2)`，顺序是 `corners[0]`=左上、`corners[1]`=右上、`corners[2]`=右下、`corners[3]`=左下（这是 apriltag2 的约定，后文所有"第几个角点"都按这个顺序理解）。

### 1.5 点云是什么、Open3D 是什么

**点云（Point Cloud）**：把三维空间里一堆点（每个点有 x,y,z，还可以带颜色 rgb）集合在一起表示物体表面。相机 + 深度图可以"炸"出一张点云：每个有效深度像素都变成一个 3D 点。

**Open3D**：一个开源的 3D 数据处理库。本文件用它的几个能力：
- `open3d.geometry.Image`：把 numpy 数组包装成 Open3D 的"图像"对象。
- `open3d.geometry.RGBDImage.create_from_color_and_depth`：把彩色图 + 深度图合成 RGBD 图像。
- `open3d.geometry.PointCloud.create_from_rgbd_image` / `create_from_depth_image`：从图像生成点云（内部就是按 1.2 的反投影公式逐像素算 3D 点）。
- `pc.segment_plane(...)`：用 **RANSAC** 算法从点云里拟合一个平面（见 1.6）。
- `open3d.camera.PinholeCameraIntrinsic`：用内参构造一个相机模型对象，供 `create_from_*` 使用。

**RANSAC**（RANdom SAmple Consensus，随机采样一致）是一种"从有噪声、有离群点的数据里找模型"的算法。拟合平面时：随机抽几个点算一个平面，看有多少点落在这个平面附近（"内点" inliers），反复多次取内点最多的那个平面。本文件 `segment_plane(distance_threshold=0.002, ransac_n=6, num_iterations=1000)` 中：
- `distance_threshold=0.002`：离平面 2 毫米以内算"内点"。
- `ransac_n=6`：每次随机抽 6 个点来算平面（平面上 3 点定一面，用 6 个更稳）。
- `num_iterations=1000`：最多随机试 1000 次。

返回一个平面方程系数 `[A, B, C, D]`（见 1.6）和"内点索引数组 `inliers`"。

### 1.6 平面方程与"射线和平面求交"

这是本模块**最核心的数学**，出现了两次：第 349 行（`compute_tag_corners3d` 内部 `compute_pt3d`）和第 264 行附近（`arm_utils._proj_rect_3d` 里也用了一模一样的公式 `z = -D/(A*nx+B*ny+C)`）。务必吃透。

**平面方程**写成：

```
A·x + B·y + C·z + D = 0
```

- `(A, B, C)` 是平面的**法向量**（垂直于平面的方向向量），长度通常是 1（单位法向量）。
- `D` 是常数项，决定平面离原点多远。

例如水平桌面：`A=0, B=0, C=1, D=-0.5`，方程是 `z = 0.5`，表示桌面在相机下方 0.5 米处。

**射线和平面求交**：我们已知一个像素 `(u, v)`，由 1.2 知道它对应的三维射线方向是 `(nx, ny, 1)` 的倍数（归一化坐标再补一个 z 分量）。也就是说，射线上任意一点可以写成：

```
P = (nx·z,  ny·z,  z)       其中 z > 0 是待求的深度
```

把 P 代入平面方程：

```
A·(nx·z) + B·(ny·z) + C·(z) + D = 0
z·(A·nx + B·ny + C) + D = 0
z = -D / (A·nx + B·ny + C)
```

求出 `z` 后，再带回：

```
x = nx·z
y = ny·z
```

就得到了这个角点的三维坐标 `(x, y, z)`。

> 注意分母 `A·nx + B·ny + C`。它就是"射线方向"和"平面法向量"的点积（撇开 z 分量不看）。如果射线**平行于**平面，这个点积≈0，除出来 `z` 会非常大甚至无穷大——这时候说明这个角点方向上看不到平面。本模块第 349 行没有对分母做保护，是个隐患（见第 7 节）。

**为什么需要"射线与平面求交"**：深度图可能只覆盖 tag 区域的一部分（有洞），但 tag 的四个角点像素坐标我们能从 `apriltag2` 精确拿到。于是我们用 tag 区域内部那片有深度的点**拟合出整个平面**，再让四条"角点方向的射线"与平面相交，就能补出角点的完整 3D 坐标——即使角点本身正好落在深度图的洞上也没关系。这就是 `compute_tag_corners3d` 的精髓。

---

## 2. 模块地图：函数与类一览表

下面这张表是整个文件的"地图"。后面第 3 节会逐个展开。

| 名称 | 行号 | 输入 | 输出 | 被谁调用 | 一句话作用 |
|---|---|---|---|---|---|
| `rgbd_to_point_cloud` | 26–61 | rgb_img, depth_img, intrinsic, depth_range | `PointCloud` | `arm_utils.CollisionDetector.check`（debug 用） | 彩色+深度 → 点云 |
| `depth_to_point_cloud` | 64–97 | depth_img, intrinsic, depth_range | `PointCloud` | `arm_utils.CollisionDetector.check`（debug 用） | 只有深度 → 点云 |
| `depth_mean_filter` | 100–142 | depth_img_list, obs_ratio | 单张滤波后深度图 | `test_tmpl_grasp_3d.py` 导入（未在本次给出片段中调用） | 多帧深度取有效均值，去噪/补洞 |
| `compute_locate_error` | 145–182 | expected/actual 两个 4×4 矩阵, sym_tfs | (pos_err_mm, rot_err_deg) | `test_tmpl_grasp_3d.py` 的 `do_grasp` | 算定位误差（考虑物体对称性） |
| `compute_projective_transformation` | 185–248 | src_K, src_img_size, R_dst_src | (dst_K, dst_size, H) | 系统其它处（把末端相机图对齐到末端坐标系） | 算把一张图投影到另一视角的变换 |
| `compute_tag_pose_2d` | 251–273 | tag, intrinsic | (nx, ny, theta) | `TagMatcher2D.match` | 由 2D 角点算平面位姿 |
| `compute_tag_mask` | 276–301 | tag, mask_img, ex_ratio | 无（原地改 mask_img） | `compute_tag_corners3d` | 在掩码图上画出 tag 区域（可外延） |
| `compute_tag_corners3d` | 304–362 | tag, depth_img, intrinsic, depth_scale, ex_ratio | (4,3) 角点 3D | `TagMatcher3D.match`、`TagMatcher3D.track` | 2D 角点 + 深度 → 3D 角点（核心） |
| `compute_tag_pose` | 365–395 | corners3d (4,3) | T_cam_tag (4×4) | `TagMatcher3D.match`、`TagMatcher3D.track` | 由 3D 角点建 tag 坐标系 |
| `ImageUndistorter` | 401–485 | intrinsic, distortion | 类实例 | `TagMatcher2D`/`TagMatcher3D` 内部 | 去畸变（图 / 点） |
| `TagMatcher2D` | 488–641 | Config / bgr_img | Result 列表 | `test_tmpl_grasp_2d.py` | 2D 标签识别 + 平面位姿 |
| `TagMatcher3D` | 644–876 | Config / bgr_img + depth_img | Result 列表 / T_cam_tag | `test_tmpl_grasp_3d.py` | 3D 标签识别 + 位姿（含 track） |

---

## 3. 逐段代码精读

### 3.1 导入区（第 5–21 行）

```python
import os
import logging
import dataclasses
from typing import List, Dict, Tuple

import math

import numpy as np
import cv2
import open3d

import apriltag2

# 导入本工程的模块
from .utils import (
    GREEN, YELLOW, BLUE, RED, RESET
)
```

逐个解释：

- `os`：操作系统接口，这里主要用 `os.makedirs` 创建调试目录（保存中间结果用）。
- `logging`：日志。本文件大量用 `logging.info(...)`、`logging.error(...)` 打印运行信息，比 `print` 更规范（能带时间、文件名、行号）。
- `dataclasses`：Python 的"数据类"装饰器 `@dataclasses.dataclass`。它专门用来定义"主要装数据的类"，能自动生成 `__init__`、`__repr__` 等。后面 `Config`、`Result` 都用它。
- `from typing import List, Dict, Tuple`：类型注解用的"容器类型"。`List[float]` 表示"元素都是 float 的列表"，`Tuple[float, float, float]` 表示"三个 float 组成的元组"。**注意：这只在函数签名里做"文档说明"，Python 运行时不强制检查类型**——写错了类型不会报错，只是给人看的。
- `math`：数学库，这里用 `math.ceil`（向上取整）、`math.sqrt`（开平方）。
- `numpy as np`：数值计算基石。所有矩阵、数组运算都靠它。
- `cv2`：OpenCV，计算机视觉库，做图像处理（去畸变、画多边形、裁剪等）。
- `open3d`：3D 点云库（见 1.5）。
- `apriltag2`：AprilTag 检测三方包（见 1.4）。
- `from .utils import (GREEN, YELLOW, BLUE, RED, RESET)`：从**同一个包**（`core`）的 `utils.py` 导入几个颜色控制字符串。这些字符串是终端 ANSI 转义码（`\033[92m` 之类），`print` 出来能让文字变绿/变红。`GREEN` 等定义在 `core/utils.py` 第 33–46 行。

### 3.2 `rgbd_to_point_cloud`（第 26–61 行）

```python
def rgbd_to_point_cloud(rgb_img: np.ndarray,
                        depth_img: np.ndarray,
                        intrinsic: List[float],
                        depth_range: Tuple[float, float] = None) -> open3d.geometry.PointCloud:
    """
    将 RGB-D 图像转换为点云
    ...
    """
    assert rgb_img.dtype == np.uint8 and rgb_img.ndim == 3 and rgb_img.shape[2] == 3, "rgb_img must be CV_8UC3"
    assert depth_img.dtype == np.float32 and depth_img.ndim == 2, "depth_img must be CV_32FC1"
    assert rgb_img.shape[:2] == depth_img.shape, "rgb_img and depth_img must have the same size"
    assert intrinsic is not None and len(intrinsic) == 4, "intrinsic must be a list of 4 elements [fx, fy, cx, cy]"

    if depth_range is not None:
        min_depth, max_depth = depth_range
        filter_depth_img = depth_img.copy()
        filter_depth_img[(filter_depth_img < min_depth) | (filter_depth_img > max_depth)] = 0.0
    else:
        filter_depth_img = depth_img
    # end if

    o3d_depth = open3d.geometry.Image(filter_depth_img)
    o3d_color = open3d.geometry.Image(rgb_img)
    o3d_rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, depth_scale=1.0, convert_rgb_to_intensity=False)
    rgbd_pc = open3d.geometry.PointCloud.create_from_rgbd_image(
        o3d_rgbd,
        open3d.camera.PinholeCameraIntrinsic(rgb_img.shape[1], rgb_img.shape[0], intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]))
    return rgbd_pc
```

**逐行讲解：**

1. **四个 `assert`（断言）**：在真正干活前先"体检"。`assert 条件, 报错信息` 的意思是"如果条件不成立，立刻抛异常并打出报错信息"。这是防御性编程——提前把乱传进来的数据挡在门外。
   - 第 41 行要求 `rgb_img` 是 `uint8`（0~255 整数）、三维、第三维是 3（RGB 三通道）——这正是 **CV_8UC3** 的含义：C=Channels，8U=8位无符号，3=3通道。
   - 第 42 行要求 `depth_img` 是 `float32` 且二维——**CV_32FC1**：32位浮点、1通道。注意：这里要求的是"已经换算成米"的浮点深度，不是 16 位整数！
   - 第 43 行要求彩色图和深度图一样大。
   - 第 44 行要求内参是 4 个数。
2. **`depth_range` 过滤（第 46–52 行）**：如果调用方给了 `(min_depth, max_depth)`，就把深度图里不在这个范围的点清零（表示"不要"）。`filter_depth_img[(filter_depth_img < min_depth) | (filter_depth_img > max_depth)] = 0.0` 这行用了**布尔掩码索引**：`(...)` 生成一个和图像一样大的 True/False 数组，True 的位置被赋值为 0.0。`|` 是"或"（位运算层面的逻辑或，对布尔数组逐元素做或）。
3. **构造 Open3D 对象（第 54–59 行）**：把 numpy 数组包成 `Image`，再合成 `RGBDImage`，最后 `create_from_rgbd_image` 生成点云。`depth_scale=1.0` 是因为深度已经是米（float32）了，不需要再缩放。`convert_rgb_to_intensity=False` 表示保留彩色，不要把它压成灰度。
4. **`PinholeCameraIntrinsic`**：用图像的宽（`shape[1]`）、高（`shape[0]`）和内参四元组构造相机模型。

**业务作用**：这个函数其实在 `vision_utils.py` 自身里没被调用，它在 `arm_utils.py` 的 `CollisionDetector.check`（debug_level>=2，第 394–397 行）里被调用来生成场景点云做可视化。也就是说它属于"调试/可视化"用途，不是抓取主链路。

### 3.3 `depth_to_point_cloud`（第 64–97 行）

```python
def depth_to_point_cloud(depth_img: np.ndarray,
                         intrinsic: List[float],
                         depth_range: Tuple[float, float] = None) -> open3d.geometry.PointCloud:
    assert depth_img.dtype == np.float32 and depth_img.ndim == 2, "depth_img must be CV_32FC1"
    assert intrinsic is not None and len(intrinsic) == 4, "intrinsic must be a list of 4 elements [fx, fy, cx, cy]"

    if depth_range is not None:
        assert depth_range[0] >= 0 and depth_range[1] > depth_range[0], "depth_range must be ..."
        min_depth, max_depth = depth_range
        filter_depth_img = depth_img.copy()
        filter_depth_img[(filter_depth_img < min_depth) | (filter_depth_img > max_depth)] = 0.0
    else:
        filter_depth_img = depth_img
    # end if

    pc = open3d.geometry.PointCloud.create_from_depth_image(
        open3d.geometry.Image(filter_depth_img),
        open3d.camera.PinholeCameraIntrinsic(depth_img.shape[1], depth_img.shape[0], intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]),
        depth_scale=1.0)
    return pc
```

和上一节几乎一样，区别是**只有深度、没有彩色**。它也是 `arm_utils.py` 的 `CollisionDetector.check` 可视化里用到的（`gripper_pc` 那段，第 398 行）。

> 重要：**这两个函数都要求深度是 `float32` 且单位是米**（因为 `depth_scale=1.0`）。如果传进来的是 `uint16` 原始深度，必须先在调用方乘 `depth_scale` 转成 float（见 `arm_utils.py` 第 395 行 `ref_depth_img.astype(np.float32) * self.depth_scale`）。

### 3.4 `depth_mean_filter`（第 100–142 行）

```python
def depth_mean_filter(depth_img_list: List[np.ndarray],
                      obs_ratio: float = 0.5) -> np.ndarray:
    """
    对深度图像列表进行均值滤波:
    1) 每个像素仅统计深度 >0 的观测次数
    2) 平均深度 = 该像素所有深度和 / 观测次数
    3) 观测次数 < obs_ratio * 列表长度时,该像素深度置 0
    ...
    """
    if len(depth_img_list) == 0:
        raise ValueError("depth_img_list is empty.")
    if obs_ratio < 0.1 or obs_ratio > 1:
        raise ValueError("obs_ratio must be in the range [0.1, 1].")

    # (N, H, W), 使用无符号整型避免求和溢出
    depth_imgs = np.stack(depth_img_list, axis=0).astype(np.uint32)
    n = depth_imgs.shape[0]

    # 每像素观测次数（ 深度非 0 ）
    obs_count = np.count_nonzero(depth_imgs, axis=0)  # (H, W), int

    # 每像素深度和（ 0 值自然不贡献 ）
    depth_sum = np.sum(depth_imgs, axis=0, dtype=np.uint32)  # (H, W)

    # 有效条件: 观测次数 >= obs_ratio * 帧数,且观测次数 >0
    th_ob_cnt = max(1, math.ceil(obs_ratio * n))  # 观测次数阈值,至少为1
    valid_mask = (obs_count >= th_ob_cnt)

    # 对有效像素做除法, depth_sum / obs_count, 无效像素结果置 0
    depth_mean = np.zeros_like(depth_sum, dtype=np.float32)
    np.divide(depth_sum, obs_count, out=depth_mean, where=valid_mask)

    # 转回 CV_16UC1
    return np.rint(depth_mean).astype(np.uint16)
```

**这个函数在做什么**：相机一帧一帧地出深度图，但单帧有很多洞和噪声。于是我们把多帧深度图叠起来，对每个像素：
- 数一数它有多少帧"有有效深度"（depth>0）→ `obs_count`。
- 把这些帧的深度值加起来 → `depth_sum`。
- 平均深度 = 总和 / 次数，但**只在"有效观测足够多"的像素上才算**，否则这个像素直接置 0（视为不可信）。

**逐行讲解：**

- 第 114–120 行：两个 `raise ValueError`——参数不合法时主动报错（和 `assert` 类似，但 `raise` 更明确地表达"这是业务错误"）。
- 第 123 行 `np.stack(depth_img_list, axis=0)`：把 N 张 `(H, W)` 的图在第 0 维堆叠，变成 `(N, H, W)` 的三维数组。`.astype(np.uint32)` 转成 32 位无符号整数——注释说"避免求和溢出"，因为原始 `uint16` 最大 65535，几百帧加起来可能爆掉 16 位。
- 第 127 行 `np.count_nonzero(..., axis=0)`：沿第 0 维（帧维度）统计非零个数，结果是 `(H, W)`。
- 第 130 行 `np.sum(depth_imgs, axis=0, dtype=np.uint32)`：沿帧维度求和，得 `(H, W)`。这里 `axis=0` 是"压缩掉第 0 维"，把 N 帧合并成一张图。
- 第 133 行 `math.ceil(obs_ratio * n)`：`obs_ratio=0.5`、`n=10` → `ceil(5.0)=5`，意思是"至少被 5 帧看到过才算有效"。`max(1, ...)` 保证至少 1。
- 第 134 行 `valid_mask`：布尔数组，只在"观测次数达标"处为 True。
- 第 137–138 行是重点：`np.divide(depth_sum, obs_count, out=depth_mean, where=valid_mask)`。
  - 普通写法是 `depth_mean = depth_sum / obs_count`，但那样会在"无效像素"（obs_count=0）处触发"除以 0"，得到 `inf` 或 `nan`（非数），污染整个数组。
  - `out=depth_mean`：结果写进预先分配好的 `depth_mean` 数组（初始全 0）。
  - `where=valid_mask`：**只在 `valid_mask` 为 True 的位置执行除法**；False 的位置保持原值（0）。这是"安全除法"，完美避开了除零。
- 第 141 行 `np.rint(depth_mean)`：四舍五入（round to nearest integer），`.astype(np.uint16)` 转回 16 位整数。

**类型隐患（务必看第 7 节第 10 点）**：docstring 第 108、111 行明确说输入/输出是 **CV_16UC1**（16 位整数深度）。但第 141 行最后又转回 `uint16`。如果调用方传进来的是"已经乘过 depth_scale 的 float 米制深度"（比如 0.5 米），那第 123 行 `.astype(np.uint16)` 会把 0.5 截成 0，整张图报废。所以**这个函数期望的是原始整数深度，不是米**。

### 3.5 `compute_locate_error`（第 145–182 行）

```python
def compute_locate_error(expected_T_cam_model: np.ndarray,
                         actual_T_cam_model: np.ndarray,
                         sym_tfs: np.ndarray = None) -> Tuple[float, float]:
    if sym_tfs is None:
        sym_tfs = np.array([np.eye(4)], dtype=np.float32)
    # end if

    N = sym_tfs.shape[0]  # 对称变换数量
    min_delta_deg = 1000.0
    for i in range(N):
        sym_tf = sym_tfs[i]
        T_cam_model = actual_T_cam_model @ sym_tf
        delta_rot = expected_T_cam_model[:3, :3] @ T_cam_model[:3, :3].T  # 旋转误差
        cosine = (np.trace(delta_rot) - 1) / 2
        cosine = np.clip(cosine, -1.0, 1.0)  # 数值稳定性保护
        delta_deg = abs(np.arccos(cosine))
        if delta_deg < min_delta_deg:
            min_delta_deg = delta_deg
            delta_pos = expected_T_cam_model[:3, 3] - T_cam_model[:3, 3]  # 位置误差
        # end if
    # end for

    pos_err = np.linalg.norm(delta_pos) * 1000.0  # 位置误差(毫米)
    rot_err = min_delta_deg * 180.0 / np.pi

    return pos_err, rot_err  # 返回位置误差(毫米)和角度误差(度)
```

**这个函数解决什么问题**：抓取系统估计出一个物体的位姿 `actual_T_cam_model`，而我们期望它到达 `expected_T_cam_model`。怎么衡量"差多少"？直接比位置差 + 旋转差。

但有些物体是**对称**的（比如一个圆形瓶子，绕中心转 90° 看起来一样）。这时"实际位姿"和"期望位姿"可能差了 90°，但对抓取来说完全正确。所以传入 `sym_tfs`（对称变换矩阵的数组），把实际位姿分别套上每一种对称变换，取"和期望位姿差最小"的那个作为误差。

**逐行讲解：**

- 第 159–160 行：如果没传 `sym_tfs`（None），就用一个单位阵 `[I]`（4×4 单位阵），表示"物体不对称，只有一种姿态"。
- 第 163 行 `N = sym_tfs.shape[0]`：对称变换的个数。
- 第 165–176 行循环：对每个对称变换 `sym_tf`：
  - 第 167 行 `T_cam_model = actual_T_cam_model @ sym_tf`：把实际位姿应用一次对称变换（相当于物体"转了一下"）。
  - 第 168 行 `delta_rot = expected[:3,:3] @ T_cam_model[:3,:3].T`：两个旋转矩阵的差。`.T` 是转置，旋转矩阵的逆 = 转置，所以 `R_expected @ R_actual.T` 就是"从 actual 转到 expected 的旋转"。
  - 第 169 行 `cosine = (np.trace(delta_rot) - 1) / 2`：由旋转矩阵求旋转角的余弦（**公式推导见 6.5 节**）。
  - 第 170 行 `np.clip(cosine, -1.0, 1.0)`：把余弦限制在 [-1,1]，防止浮点误差让它变成 1.0000001 导致 `arccos` 返回 nan。
  - 第 171 行 `np.arccos` 得弧度，`abs` 取绝对值（旋转角没有方向）。
  - 第 172–175 行：如果这次旋转差更小，就更新最小值，并**同时**记下位置差。
- 第 178 行 `np.linalg.norm(delta_pos)`：`delta_pos` 是位置差向量（3 个数），`norm` 求它的长度（欧几里得距离），`* 1000` 把米变毫米。
- 第 179 行把弧度 `min_delta_deg` 乘 `180/π` 变角度。

**两个真问题（详见第 7 节第 3 点）**：
1. 如果 `sym_tfs` 是形状 `(0,)` 的空数组，`N=0`，`for` 循环一次都不执行，第 178 行要用的 `delta_pos` 从未被赋值 → 抛 `UnboundLocalError`。
2. 位置差 `delta_pos` 只在"旋转差更新"时才记录，把"位置误差"和"旋转误差"耦合在一起了。严格说应该对每个对称变换都分别算位置差和旋转差，再综合取最小。

### 3.6 `compute_projective_transformation`（第 185–248 行）

```python
def compute_projective_transformation(src_K: np.ndarray,
                                      src_img_size: Tuple[int, int],
                                      R_dst_src: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    src_w, src_h = src_img_size
    corners = np.array([
        [0, 0, 1.0],                  # 左上
        [src_w - 1, 0, 1.0],          # 右上
        [src_w - 1, src_h - 1, 1.0],  # 右下
        [0, src_h - 1, 1.0],          # 左下
    ], dtype=np.float64).T  # (3,4)

    src_rays = np.linalg.inv(src_K) @ corners    # (3,4)  src 相机坐标系中的射线
    dst_rays = R_dst_src @ src_rays              # (3,4)  dst 相机坐标系中的射线

    z = dst_rays[2, :]
    if np.any(z <= 1e-6):
        logging.error("有 src 的图像角点在旋转后落在 dst 相机的后方(z<=0), 无法投影到 dst 图像平面")
        return None, None, None
    # end if

    fx, fy = src_K[0, 0], src_K[1, 1]
    u_hat = fx * (dst_rays[0, :] / z)
    v_hat = fy * (dst_rays[1, :] / z)

    u_min, u_max = float(np.min(u_hat)), float(np.max(u_hat))
    v_min, v_max = float(np.min(v_hat)), float(np.max(v_hat))

    dst_cx = -u_min
    dst_cy = -v_min

    dst_w = int(np.ceil(u_max - u_min)) + 1
    dst_h = int(np.ceil(v_max - v_min)) + 1

    dst_K = np.array([
        [fx, 0.0, dst_cx],
        [0.0, fy, dst_cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    H_dst_src = dst_K @ R_dst_src @ np.linalg.inv(src_K)

    return dst_K, (dst_w, dst_h), H_dst_src
```

**它在干什么**：假设有两台相机看同一个场景，src 相机和 dst 相机之间只差一个**旋转** `R_dst_src`（没有平移）。我们想把 src 拍的图"重新投影"成 dst 相机视角下的图。这个函数算出：dst 相机的内参 `dst_K`、dst 图该多大 `dst_size`、以及一个 3×3 的射影变换矩阵 `H_dst_src`（可以用 `cv2.warpPerspective` 直接把 src 图变过去）。

**逐行讲解：**

- 第 204–211 行：先取 src 图像四个角的像素坐标，补一个 `1.0` 变成**齐次坐标** `(u, v, 1)`。`.T` 转置后变成 `(3, 4)`——每一列是一个角点。
  - **齐次坐标（homogeneous coordinates）**：普通的 2D 点 `(u, v)` 加一维变成 `(u, v, 1)`，好处是能用矩阵乘法统一表示"平移、旋转、缩放、透视"等变换。透视变换（射影变换）必须用 3×3 矩阵作用在齐次坐标上。
- 第 213 行 `src_rays = inv(src_K) @ corners`：用内参的逆，把像素角点反投回**src 相机坐标系下的三维射线方向**（见 1.2，反投影去掉了 fx/cx 等影响，得到方向向量）。
- 第 214 行 `dst_rays = R_dst_src @ src_rays`：把这条射线旋转到 dst 相机坐标系。
- 第 216–220 行：取每条射线的 `z` 分量。如果任何一个 `z <= 0`，说明这个角点转到了相机背后，没法投影，直接返回 None。
- 第 222–225 行：在 dst 坐标系里，用"不含主点偏移"的公式 `u_hat = fx * (x/z)`、`v_hat = fy * (y/z)` 算出角点落在 dst 图像上的大致像素位置（注意这里没加 cx/cy，因为我们要自己选主点）。
- 第 227–236 行：找出所有角点的 `u_hat/v_hat` 的最小最大值，算出内容在 dst 图上的包围盒。`dst_cx = -u_min` 是把最左边缘挪到像素 0（主点偏移），`dst_w = ceil(u_max - u_min) + 1` 保证所有内容都装得下。
- 第 238–242 行：按上面算出的主点构造 dst 内参。
- 第 245 行 `H_dst_src = dst_K @ R_dst_src @ inv(src_K)`：**射影变换的核心公式**（推导见 6.6 节）。

**业务用途**：docstring 第 200–201 行说"一般用于将机械臂末端相机拍摄的图像变换到与机械臂末端坐标系方向对齐的视角"。也就是把"相机歪着拍的图"扳正成"从末端正前方看"的图，方便后续处理。

**真问题（第 7 节第 11 点）**：第 231–236 行没考虑 `u_max==u_min`（内容退化成一条线、宽高为 0）的极端情况；另外它**假设图像已经去畸变**——调用方必须先 `undistort`，否则会错。

### 3.7 `compute_tag_pose_2d`（第 251–273 行）

```python
def compute_tag_pose_2d(tag: apriltag2.Tag2D,
                        intrinsic: List[float]) -> Tuple[float, float, float]:
    center = tag.center  # 标签中心点的像素坐标 (u,v)
    u, v = center[0], center[1]
    nx = (u - intrinsic[2]) / intrinsic[0]  # 标签中心点在相机归一化坐标系中的 x 坐标
    ny = (v - intrinsic[3]) / intrinsic[1]  # 标签中心点在相机归一化坐标系中的 y 坐标

    dir = tag.corners[1] - tag.corners[0]   # 计算物体朝向, 向量从第一个角点指向第二个角点
    theta = np.arctan2(dir[1], dir[0])

    return nx, ny, theta
```

**它在干什么**：只做 2D（不碰深度图）。输出 `(nx, ny, theta)`：
- `(nx, ny)`：tag 中心在相机**归一化平面**上的坐标（见 1.2，相当于"假设焦距=1 米时中心在哪"）。
- `theta`：tag 的朝向角。取 `corners[1] - corners[0]`（右上角减左上角，即 tag 的上边）的方向，用 `arctan2` 求它与水平轴的夹角。

**逐行讲解**：
- 第 266–267 行就是 1.2 的反投影公式去掉 Z：`nx = (u - cx)/fx`、`ny = (v - cy)/fy`。这里直接把归一化坐标当成了平面坐标（因为 2D 场景里物体平放在桌上，归一化坐标的方向就代表桌面上相对相机的方向）。
- 第 269 行 `dir = corners[1] - corners[0]`：从角点 0（左上）指向角点 1（右上）的向量，代表 tag 的"上边"方向。
- 第 270 行 `np.arctan2(dir[1], dir[0])`：`arctan2(y, x)` 是"带象限的四则 arctan"，能返回 `(-π, π]` 之间的正确角度，比普通 `arctan(y/x)` 更稳（不会因为 x=0 或符号而出错）。

**一致性说明**：`TagMatcher2D.draw`（第 635 行）画朝向箭头也用 `result.corners[1] - result.corners[0]`，和这里的 `theta` 用的是同一条边，所以箭头方向 == theta 方向，两者一致。`test_tmpl_grasp_2d.py` 第 250–255 行在算角度差 `delta_theta` 后还做了 `while` 循环把它**归一化到 (-π, π]**，正是 `arctan2` 的输出范围，避免跨 ±π 的差值算错。

### 3.8 `compute_tag_mask`（第 276–301 行）

```python
def compute_tag_mask(tag: apriltag2.Tag2D,
                     mask_img: np.ndarray,
                     ex_ratio: float = 0.3) -> None:
    corners2d = tag.corners

    # 计算外延点
    ex_corners2d = np.zeros((4, 2), dtype=np.float32)
    ex_corners2d[0] = corners2d[0] + (corners2d[0] - corners2d[2]) * ex_ratio
    ex_corners2d[1] = corners2d[1] + (corners2d[1] - corners2d[3]) * ex_ratio
    ex_corners2d[2] = corners2d[2] + (corners2d[2] - corners2d[0]) * ex_ratio
    ex_corners2d[3] = corners2d[3] + (corners2d[3] - corners2d[1]) * ex_ratio

    # 制作掩码
    cv2.fillConvexPoly(mask_img, ex_corners2d.astype(np.int32), 255)
```

**它在干什么**：在 `mask_img`（一张和深度图同尺寸的空白图，初始全 0）上，把 tag 区域涂成 255（白），其余保持 0（黑）。这样后续就能"只保留 tag 区域的深度"。

**逐行讲解**：
- `mask_img` 是**输入输出参数**（函数签名里没有返回值，直接改它）。这种"原地修改"的写法在 numpy/OpenCV 里很常见，但要小心——调用方如果在别处还引用这张图，改动会到处生效。
- 第 293–297 行是**对角外延**：以每个角点 `i` 为起点，沿"指向对角角点 `(i+2)%4`"的方向，往外延伸 `ex_ratio` 倍的距离。比如 `ex_corners2d[0] = corners2d[0] + (corners2d[0] - corners2d[2]) * ex_ratio`，`corners2d[0]-corners2d[2]` 是从右下角指向左上角的向量，乘以 `ex_ratio` 再加到左上角上，就把左上角沿对角线反方向推出去一点。
  - **为什么要外延**：tag 角点本身可能正好落在深度图的洞上，或者边缘深度不准。把掩码稍微放大一圈，能多采到 tag 周围（同一平面上）的有效深度点，拟合平面更稳。详见 6.8 节的 ASCII 图。
- 第 300 行 `cv2.fillConvexPoly(mask_img, ..., 255)`：OpenCV 函数，把"凸多边形"内部填成 255。tag 是方块，四个外延后的角点构成凸四边形，刚好用它填。`ex_corners2d.astype(np.int32)` 把浮点像素坐标转成整数（图像坐标必须是整数）。

### 3.9 `compute_tag_corners3d`（第 304–362 行）

这是整个文件**最核心、最值得仔细读**的函数。

```python
def compute_tag_corners3d(tag: apriltag2.Tag2D,
                          depth_img: np.ndarray,
                          intrinsic: np.ndarray,
                          depth_scale: float,
                          ex_ratio: float = 0.1
                          ) -> np.ndarray:
    # 绘制掩码
    mask_img = np.zeros(depth_img.shape, dtype=np.uint8)
    compute_tag_mask(tag, mask_img, ex_ratio=ex_ratio)

    # 将非掩码区域设为0
    flt_depth_img = depth_img.astype(np.float32) * depth_scale
    flt_depth_img[mask_img == 0] = 0.0

    # 提取掩码区域的点云
    pc = open3d.geometry.PointCloud.create_from_depth_image(
        open3d.geometry.Image(flt_depth_img),
        open3d.camera.PinholeCameraIntrinsic(flt_depth_img.shape[1], flt_depth_img.shape[0],
                                             intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]),
        np.eye(4),
        depth_scale=1.0,
        depth_trunc=0.5
    )

    # 平面拟合
    plane, inliers = pc.segment_plane(distance_threshold=0.002, ransac_n=6, num_iterations=1000)
    logging.info(f"plane equation: {plane}, inliers count: {len(inliers)}, inliers ratio: {len(inliers)/len(pc.points)}")

    # 计算角点的空间坐标( 根据平面方程计算 )
    def compute_pt3d(corner: np.ndarray, intrinsic: np.ndarray, plane: np.ndarray) -> np.ndarray:
        nx = (corner[0] - intrinsic[2]) / intrinsic[0]
        ny = (corner[1] - intrinsic[3]) / intrinsic[1]
        A, B, C, D = plane
        z = -D / (A * nx + B * ny + C)
        x = nx * z
        y = ny * z
        return np.array([x, y, z])
    # end def compute_pt3d

    corners2d = tag.corners
    corners3d = np.zeros((4, 3), dtype=np.float32)
    for i in range(4):
        corners3d[i] = compute_pt3d(corners2d[i], intrinsic, plane)
    # end for

    return corners3d
```

**它在干什么**（串起来）：
1. 在深度图上圈出 tag 区域（带外延）。
2. 把圈内深度换成"米"，圈外清零。
3. 用这片深度生成点云。
4. 用 RANSAC 从点云拟合出 tag 贴的那个**平面**（物体表面）。
5. 拿 tag 的四个 2D 角点，每个角点沿它的射线方向去和这个平面求交（见 1.6），算出四个 3D 角点 `(4, 3)`。

**逐行讲解**：
- 第 323 行：先造一张全 0 的 `mask_img`，尺寸和 `depth_img` 一样。
- 第 324 行：调用上一节的函数把 tag 区域涂成 255。
- 第 327 行 `flt_depth_img = depth_img.astype(np.float32) * depth_scale`：把原始深度（一般是 `uint16` 毫米整数）转成 float 米。例如原始 `500` × `0.001` = `0.5` 米。`depth_scale` 来自 `read_rgbd_params`。
- 第 328 行 `flt_depth_img[mask_img == 0] = 0.0`：掩码外的深度全清零（不要）。
- 第 331–338 行 `create_from_depth_image(...)`：把这片深度变成点云。注意三个参数：
  - `np.eye(4)`：一个单位变换，表示点云直接就在相机坐标系里（不做额外旋转）。
  - `depth_scale=1.0`：因为上面已经手动乘了 depth_scale，这里告诉 Open3D"深度单位已经是米，别再缩放"。
  - **第 337 行 `depth_trunc=0.5`：硬编码把深度 > 0.5 米的点直接丢掉**。这就是用户发现的大坑（第 7 节第 1 点）——如果物体离相机超过半米，这片点云空了，平面拟合退化。
- 第 341 行 `pc.segment_plane(...)`：RANSAC 拟合平面，返回 `plane=[A,B,C,D]` 和 `inliers`（内点索引）。
- 第 342 行打日志：`len(inliers)/len(pc.points)` 是内点占比。但注意：如果 `pc.points` 为空（上面 depth_trunc 把点全截没了），这里会**除以 0 崩溃**（第 7 节第 2 点）。
- 第 345–353 行是**内嵌函数** `compute_pt3d`：实现了 1.6 的射线-平面求交公式。`A,B,C,D = plane` 是 Python 的"元组拆包"，把长度为 4 的数组分别赋给四个变量。第 349 行 `z = -D / (A*nx + B*ny + C)` 正是核心公式（推导见 6.2）。**这里没有检查分母是否为 0**。
- 第 355–359 行：对 4 个角点逐个调用 `compute_pt3d`，填进 `corners3d`。

**业务作用**：`TagMatcher3D.match`（第 767 行）和 `TagMatcher3D.track`（第 869 行）都调用它得到 3D 角点，再喂给 `compute_tag_pose` 建坐标系。它是 3D 抓取"眼睛看懂物体"的第一步。

### 3.10 `compute_tag_pose`（第 365–395 行）

```python
def compute_tag_pose(corners3d: np.ndarray) -> np.ndarray:
    center3d = np.mean(corners3d, axis=0)

    axis_x = corners3d[1] - corners3d[0]
    axis_x = axis_x / np.linalg.norm(axis_x)

    axis_y = corners3d[3] - corners3d[0]
    axis_y = axis_y / np.linalg.norm(axis_y)

    axis_z = np.cross(axis_x, axis_y)
    axis_z = axis_z / np.linalg.norm(axis_z)

    axis_y = np.cross(axis_z, axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)

    R = np.stack([axis_x, axis_y, axis_z], axis=1)  # (3,3)

    T_cam_tag = np.eye(4, dtype=np.float32)
    T_cam_tag[:3, :3] = R
    T_cam_tag[:3, 3] = center3d

    return T_cam_tag
```

**它在干什么**：由 4 个 3D 角点，建立 tag 的**局部坐标系**，输出 4×4 位姿矩阵 `T_cam_tag`（从 tag 坐标系到相机坐标系）。

**坐标系怎么建**（非常关键，对照 6.4 节）：
- 原点 `center3d`：四个角点的平均（几何中心）。
- **X 轴 = 角点1 − 角点0**：即 tag 的上边（从左上到右上）。
- **Y 轴（初始）= 角点3 − 角点0**：即 tag 的左边（从左上到左下）。
- **Z 轴 = X × Y**（叉乘，右手定则得到平面的法向）。
- **然后重算 Y = Z × X**：这一步叫**正交化**。因为 X 和 Y 不一定严格垂直（角点检测有误差、或 tag 不是完美矩形），直接用会得到一个"歪"的坐标系。先有精确垂直的 Z（由 X 和 Y 叉乘），再用 Z 和 X 叉乘出"真正垂直于 X 且落在平面内的 Y"，保证三个轴两两垂直、构成标准正交基。

**逐行讲解**：
- 第 374 行 `np.mean(corners3d, axis=0)`：沿第 0 维（角点维度）求平均，得到 1 个 3 维中心点。
- 第 376–380 行：两条边向量，各自除以自己的长度（`np.linalg.norm`，向量模长）变成**单位向量**。
- 第 382–383 行：`np.cross` 叉乘得 Z，归一化。
- 第 385–386 行：重新叉乘得 Y，归一化（正交化，见上面解释）。
- 第 388 行 `np.stack([axis_x, axis_y, axis_z], axis=1)`：把三个列向量**按列**拼成 3×3 旋转矩阵 `R`。`axis=1` 表示"在第 1 维（列）方向上堆叠"，于是 `R[:,0]=axis_x`、`R[:,1]=axis_y`、`R[:,2]=axis_z`。
- 第 390–392 行：构造 4×4 单位阵，把旋转块和平移（中心）填进去。

**⚠️ 大坑（第 7 节第 7 点）**：`arm_utils.GripperBody.initialize`（第 76–77 行）建立坐标系用的是 `edge_x = corners3d[2]-corners3d[0]`（对角线）、`edge_y = corners3d[1]-corners3d[3]`（另一条对角线）。这套"对角线建法"和本函数"邻边建法"相差 45°。两个模块对同一个 AprilTag 建出的坐标系朝向不一样——夹爪标定用 `GripperBody`，物体定位用 `compute_tag_pose`，若当成一致会出错。

### 3.11 `ImageUndistorter`（第 401–485 行）

```python
class ImageUndistorter:
    """
    图像畸变矫正器
    """
    def __init__(self,
                 intrinsic: List[float],
                 distortion: List[float] = None):
        self.K = np.array([
            [intrinsic[0], 0.0, intrinsic[2]],
            [0.0, intrinsic[1], intrinsic[3]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.D = None
        if distortion is not None and len(distortion) > 0:
            self.D = np.array(distortion, dtype=np.float64)
        # end if
        self._undistort_maps = []  # 畸变校正映射表
    # end def __init__

    def undistort_img(self, src_img: np.ndarray) -> np.ndarray:
        if self.D is None:
            return src_img.copy()
        # end if
        if len(self._undistort_maps) == 0:
            map1, map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=self.K,
                distCoeffs=self.D,
                R=None,
                newCameraMatrix=self.K,
                size=(src_img.shape[1], src_img.shape[0]),
                m1type=cv2.CV_16SC2
            )
            self._undistort_maps = [map1, map2]
            logging.info("undistort rectify maps initialized")
        # end if
        dst_img = cv2.remap(src_img, self._undistort_maps[0], self._undistort_maps[1], interpolation=cv2.INTER_LINEAR)
        return dst_img
    # end def undistort_img

    def undistort_points(self, src_pts: np.ndarray, do_normalize: bool = False) -> np.ndarray:
        if self.D is None:
            return src_pts.copy()
        # end if
        P = self.K if not do_normalize else np.eye(3, dtype=np.float64)
        dst_pts = cv2.undistortPoints(src_pts.reshape(-1, 1, 2), cameraMatrix=self.K, distCoeffs=self.D, P=P)
        return dst_pts.reshape(-1, 2)
    # end def undistort_points
```

**它在干什么**：真实镜头不是完美针孔，会有**畸变**（图像边缘被拉伸/压缩）。去畸变就是把扭曲的像素"掰直"。`undistort_img` 去整张图，`undistort_points` 只去若干坐标点。

**背景：畸变系数 `D`**：通常是 `[k1, k2, p1, p2, k3, ...]`。它描述镜头怎么把理想针孔图像扭曲。OpenCV 的 `initUndistortRectifyMap` 会先算一张"映射表"（map1/map2），记录"输出图的每个像素，应该去输入图的哪个位置取色"。之后 `cv2.remap` 用这张表快速重采样。

**逐行讲解**：
- `__init__`（第 406–430 行）：把 `[fx,fy,cx,cy]` 拼成 3×3 内参矩阵 `K`；把畸变 `D` 存好（None 表示"无畸变"）；`_undistort_maps` 初始化为空列表，用来缓存映射表。
- `undistort_img`（第 432–461 行）：
  - 第 442–443 行：如果 `D is None`，直接返回副本（没有畸变可去）。
  - 第 446–455 行：如果映射表还没算过，就 `cv2.initUndistortRectifyMap(...)` 算一次并缓存到 `self._undistort_maps`。
  - 第 460 行 `cv2.remap(...)`：用缓存的映射表把输入图重采样成去畸变后的图。
- `undistort_points`（第 464–483 行）：`cv2.undistortPoints` 去畸变坐标点。`do_normalize=True` 时把 `P` 设为单位阵，使输出直接是**归一化坐标** `(nx, ny)`（去掉了 fx/cx 影响）。

**⚠️ 大坑（第 7 节第 6 点）**：映射表只在**第一次**调用时按当时的图尺寸 `(src_img.shape[1], src_img.shape[0])` 算并缓存。如果之后换了分辨率的图（比如彩色 1280×720、深度对齐后 640×480），会**静默地用错尺寸的 map**，导致去畸变完全错误。改进见第 7 节。

### 3.12 `TagMatcher2D`（第 488–641 行）

这是一个"类"，封装了 2D 标签匹配的全部逻辑。先讲它的内部小类，再讲方法。

#### 3.12.1 `Config`（第 493–513 行）

```python
@dataclasses.dataclass
class Config:
    intrinsic: List[float]
    distortion: List[float] = None
    tag_family: str = "tag36h11"
    black_border: int = 2
    debug_dir: str = None
```

`@dataclasses.dataclass` 让这个类自动获得构造函数（`__init__`）：你写 `TagMatcher2D.Config(intrinsic=[...], distortion=[...])`，它就自动把参数存成同名属性。**每个字段后面的 `: 类型 = 默认值` 是 dataclass 的标准写法**——这里 `intrinsic` 没有默认值（必填），其余都有默认值。

#### 3.12.2 `Result`（第 515–534 行）

```python
@dataclasses.dataclass
class Result:
    id = int(-1)
    center = np.zeros(2, dtype=np.float32)
    corners: np.ndarray = np.zeros((4, 2), dtype=np.float32)
    pose_2d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
```

**⚠️ 大坑（第 7 节第 4 点，必须重点理解）**：`Config` 的每个字段**都带类型注解**（`: List[float]` 等），所以它们是真正的 dataclass 字段。但 `Result` 的 `id`、`center`、`corners` **没有类型注解**，只有赋值（`id = int(-1)` 这种写法是"先调用 int(-1) 得到 -1 再赋值给类属性"，不是注解）。**在 dataclass 里：只有带注解的才被当成字段，没注解的只是普通 Python 类属性。**

后果：`center` 和 `corners` 是**可变对象**（numpy 数组），且被当作"类属性"——**所有 `Result` 实例共享同一个 `center` 和 `corners` 数组对象**。当前代码在 `match` 里每次都 `result.center = tag.center`（整体重新赋值一个新对象），所以没暴露问题；但一旦有人写 `result.center[0] = 1`（改数组内容而不是换对象），就会污染后续所有实例。详见第 7 节第 4 点。

#### 3.12.3 `__init__`（第 536–558 行）

```python
def __init__(self, config: Config):
    self.intrinsic = config.intrinsic
    self.undistorter = ImageUndistorter(intrinsic=self.intrinsic, distortion=config.distortion)
    self.detector = apriltag2.Detector(tag_family=config.tag_family, black_border=config.black_border)
    self.debug_dir = config.debug_dir
    if self.debug_dir is not None:
        os.makedirs(self.debug_dir, exist_ok=True)
        logging.info(f"debug results will be saved to: {GREEN}{self.debug_dir}{RESET}")
    # end if
    logging.info("TagMatcher2D initialized")
```

把 `Config` 里的参数落到实例属性上，并创建两个工具：`undistorter`（去畸变器）和 `detector`（apriltag2 检测器）。`os.makedirs(..., exist_ok=True)`：如果调试目录不存在就创建，已存在也不报错。

#### 3.12.4 `match`（第 560–616 行）

```python
def match(self, bgr_img: np.ndarray, top_k: int = 0, debug_level: int = 0) -> Tuple[List[Result], str]:
    result_dir = f"{self.debug_dir}/match"
    if self.debug_dir is not None and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
        logging.info(f"match results will be saved to: {result_dir}")
    # end if

    un_img = self.undistorter.undistort_img(bgr_img)

    tag_list = self.detector.detect(un_img, -1)

    if (len(tag_list) == 0 or debug_level >= 1) and os.path.exists(result_dir):
        cv2.imwrite(f"{result_dir}/color.png", un_img)
    # end if

    if len(tag_list) == 0:
        msg = "no tag detected"
        return [], msg
    # end if

    result_list = []
    for tag in tag_list:
        result = self.Result()
        result.id = tag.id
        result.center = tag.center
        result.corners = tag.corners
        result.pose_2d = compute_tag_pose_2d(tag, self.intrinsic)
        result_list.append(result)
    # end for

    # 选取分数最高的 top_k 个实例
    if top_k > 0 and len(result_list) > top_k:
        result_list.sort(key=lambda r: r.pose_2d[0]**2 + r.pose_2d[1]**2)
        result_list = result_list[:top_k]
    # end if

    msg = f"match successful"
    return result_list, msg
```

**逐行讲解**：
- 第 583 行：先去畸变。
- 第 586 行：apriltag2 检测，返回所有找到的 tag 列表。
- 第 588–595 行：没检测到就保存（如开启 debug）并返回空列表 + 消息。
- 第 599–606 行：对每个检测到的 tag，造一个 `Result`，填 id/center/corners，并用 `compute_tag_pose_2d`（3.7 节）算出 2D 位姿 `pose_2d`，收进 `result_list`。
- 第 608–612 行：**top_k 排序（大坑，第 7 节第 5 点）**：注释写"分数最高的 k 个"，但实际排序键是 `pose_2d[0]**2 + pose_2d[1]**2`——也就是**离光轴（归一化坐标原点）最近的优先**。这和"分数最高"完全不是一回事，而且**没有按 tag id 过滤**——场景里若有多个 tag，会优先抓离相机中心最近的那个，而不是指定的那个。

**被谁调用**：`test_tmpl_grasp_2d.py` 的 `do_grasp` 第 338 行 `result_list, msg = matcher.match(rgb_img, top_k=1)`，拿到 `result_list[0].pose_2d` 后喂给 `compute_delta_end_pose` 做视觉伺服。

#### 3.12.5 `draw`（第 618–640 行）

```python
def draw(self, bgr_img: np.ndarray, result_list: List[Result]):
    for result in result_list:
        corners = result.corners.astype(np.int32)
        cv2.polylines(bgr_img, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        dir = result.corners[1] - result.corners[0]
        cv2.arrowedLine(bgr_img, tuple(result.center.astype(np.int32)),
                        tuple((result.center + dir).astype(np.int32)),
                        color=(0, 255, 255), thickness=1)
    # end for
```

在图上画出检测框（绿色多边形）和朝向箭头（青色），方便人眼检查。箭头方向和 3.7 节的 `theta` 用同一条边 `corners[1]-corners[0]`，一致。

### 3.13 `TagMatcher3D`（第 644–876 行）

3D 版，结构和 2D 版几乎一样，但多了深度处理、`compute_tag_corners3d`、`compute_tag_pose`，以及 `track` 方法。同样先讲内部类。

#### 3.13.1 `Config`（第 649–672 行）

```python
@dataclasses.dataclass
class Config:
    intrinsic: List[float]
    depth_scale: float
    distortion: List[float] = None
    tag_family: str = "tag36h11"
    black_border: int = 2
    debug_dir: str = None
```

比 2D 版多了 `depth_scale`（深度缩放系数，见 1.3）。注意 `intrinsic` 和 `depth_scale` 必填。

#### 3.13.2 `Result`（第 674–692 行）

```python
@dataclasses.dataclass
class Result:
    id = int(-1)
    center = np.zeros(2, dtype=np.float32)
    corners: np.ndarray = np.zeros((4, 2), dtype=np.float32)
    T_cam_tag: np.ndarray = np.eye(4, dtype=np.float32)
```

**同样有第 7 节第 4 点的 dataclass 注解坑**（`id`、`center` 无注解）。这里 `T_cam_tag` 有注解（是真正的字段），但 `id`、`center`、`corners` 没有。

#### 3.13.3 `__init__`（第 694–719 行）

```python
def __init__(self, config: Config):
    self.intrinsic = config.intrinsic
    self.depth_scale = config.depth_scale
    self.undistorter = ImageUndistorter(intrinsic=self.intrinsic, distortion=config.distortion)
    self.detector = apriltag2.Detector(tag_family=config.tag_family, black_border=config.black_border)
    self.tag_size = None  # 标签边长( 米 ), 根据识别结果计算得到
    self.debug_dir = config.debug_dir
    if self.debug_dir is not None:
        os.makedirs(self.debug_dir, exist_ok=True)
        logging.info(f"debug results will be saved to: {GREEN}{self.debug_dir}{RESET}")
    # end if
    logging.info("TagMatcher3D initialized")
```

比 2D 版多了 `self.depth_scale` 和 `self.tag_size`。**`self.tag_size` 初始为 None，只在 `match` 里被赋值（见下）**——这是第 7 节第 8 点的隐患：`track` 依赖它，但若没先 `match` 就直接 `track`，`tag_size` 是 None 会崩。

#### 3.13.4 `match`（第 721–786 行）

```python
def match(self, bgr_img: np.ndarray, depth_img: np.ndarray, top_k: int = 0, debug_level: int = 0) -> Tuple[List[Result], str]:
    result_dir = f"{self.debug_dir}/match"
    if self.debug_dir is not None and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
        logging.info(f"match results will be saved to: {result_dir}")
    # end if

    un_img = self.undistorter.undistort_img(bgr_img)
    tag_list = self.detector.detect(un_img, -1)

    if (len(tag_list) == 0 or debug_level >= 1) and os.path.exists(result_dir):
        cv2.imwrite(f"{result_dir}/color.png", un_img)
    # end if

    if len(tag_list) == 0:
        msg = "no tag detected"
        return [], msg
    # end if

    result_list = []
    for tag in tag_list:
        result = self.Result()
        result.id = tag.id
        result.center = tag.center
        result.corners = tag.corners
        corners3d = compute_tag_corners3d(tag, depth_img, self.intrinsic, self.depth_scale)
        result.T_cam_tag = compute_tag_pose(corners3d)
        result_list.append(result)

        if self.tag_size is None:
            self.tag_size = np.linalg.norm(corners3d[0] - corners3d[2]) / math.sqrt(2)
            logging.info(f"tag size estimated to be: {self.tag_size:.4f} m")
        # end if
    # end for

    # 选取分数最高的 top_k 个实例
    if top_k > 0 and len(result_list) > top_k:
        result_list.sort(key=lambda r: r.T_cam_tag[2, 3])  # 按深度分数排序
        result_list = result_list[:top_k]
    # end if

    msg = f"match successful"
    return result_list, msg
```

**逐行讲解**：
- 第 746–767 行：去畸变 → 检测 → 对每个 tag 算 3D 角点 `compute_tag_corners3d` → 建位姿 `compute_tag_pose` → 得到 `T_cam_tag` 填进 `Result`。
- 第 771–774 行：**估算 tag 边长** `tag_size`。用角点 0 和角点 2 的对角线长度除以 `√2`（正方形对角线 = 边长×√2），所以边长 = 对角线/√2。存到 `self.tag_size` 供 `track` 用。
- 第 778–782 行：**top_k 排序（大坑，第 7 节第 5 点）**：按 `T_cam_tag[2, 3]`（即 tag 中心在相机坐标系的 z，也就是"离相机多深"）排序，**离相机最近的优先**。同样没有按 id 过滤。

**被谁调用**：`test_tmpl_grasp_3d.py` 的 `match()` 辅助函数（第 188 行）调用 `matcher.match(bgr_img=color_img, depth_img=depth_img, top_k=1)`，拿到 `result_list[0].T_cam_tag` 作为 `cur_T_cam_model`（物体相对相机的位姿），再喂给 `compute_ready_pose`。

#### 3.13.5 `track`（第 788–874 行）

```python
def track(self, bgr_img: np.ndarray, depth_img: np.ndarray, init_T_cam_tag: np.ndarray, debug_level: int = 0) -> Tuple[np.ndarray, str]:
    result_dir = f"{self.debug_dir}/track"
    if self.debug_dir is not None and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
        logging.info(f"track results will be saved to: {result_dir}")
    # end if

    corners3d = np.zeros((4, 3), dtype=np.float32)
    half_size = self.tag_size / 2
    corners3d[0] = np.array([-half_size, -half_size, 0], dtype=np.float32)
    corners3d[1] = np.array([half_size, -half_size, 0], dtype=np.float32)
    corners3d[2] = np.array([half_size, half_size, 0], dtype=np.float32)
    corners3d[3] = np.array([-half_size, half_size, 0], dtype=np.float32)

    corners3d = corners3d @ init_T_cam_tag[:3, :3].T + init_T_cam_tag[:3, 3]

    # 计算角点在图像中的投影位置
    proj_corners2d = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        x, y, z = corners3d[i]
        u = self.intrinsic[0] * (x / z) + self.intrinsic[2]
        v = self.intrinsic[1] * (y / z) + self.intrinsic[3]
        proj_corners2d[i] = [u, v]
    # end for

    # 计算投影的最小外接矩形( 用于后续位姿细化的初始掩码 )
    min_x, min_y = np.min(proj_corners2d, axis=0)
    max_x, max_y = np.max(proj_corners2d, axis=0)

    # 放大矩形以包含更多的像素
    rect_w = (max_x - min_x) * 2.0
    rect_h = (max_y - min_y) * 2.0
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    min_x = max(0, center_x - rect_w / 2)
    max_x = min(bgr_img.shape[1] - 1, center_x + rect_w / 2)
    min_y = max(0, center_y - rect_h / 2)
    max_y = min(bgr_img.shape[0] - 1, center_y + rect_h / 2)

    # 畸变矫正
    un_img = self.undistorter.undistort_img(bgr_img)

    # 根据投影的最小外接矩形裁剪图像
    crop_bgr_img = un_img[int(min_y):int(max_y), int(min_x):int(max_x)]
    tag_list = self.detector.detect(crop_bgr_img, -1)

    if (len(tag_list) == 0 or debug_level >= 1) and os.path.exists(result_dir):
        vis_img = un_img.copy()
        cv2.rectangle(vis_img, (int(min_x), int(min_y)), (int(max_x), int(max_y)),
                      color=(0, 0, 255), thickness=1)
        cv2.imwrite(f"{result_dir}/proj_area.png", vis_img)
    # end if

    if len(tag_list) == 0:
        msg = "no tag detected in cropped image"
        return None, msg
    # end if

    # 恢复标签坐标到原图坐标系
    tag = tag_list[0]
    tag.center += np.array([min_x, min_y])
    tag.corners += np.array([min_x, min_y])

    # 定位
    corners3d = compute_tag_corners3d(tag, depth_img, self.intrinsic, self.depth_scale)
    T_cam_tag = compute_tag_pose(corners3d)

    msg = "track successful"
    return T_cam_tag, msg
```

**它在干什么**（这是"跟踪"，不是"从头检测"）：上一帧已经知道物体在 `init_T_cam_tag` 附近，这一帧就不用在全图里找，而是**根据上一帧位姿，把 tag 大概会出现的图像区域裁剪出来**，只在这个小窗口里检测。这样更快、更稳，也避免误检别的 tag。

**逐行讲解**：
- 第 812–817 行：在**tag 自己的局部坐标系**里，按已知（或估算的）`tag_size` 造 4 个角点：`(-h,-h,0)`、`(h,-h,0)`、`(h,h,0)`、`(-h,h,0)`，h = 边长的一半。这是一个边长为 `tag_size` 的方块，平躺在 z=0 平面上。
- 第 819 行 `corners3d = corners3d @ init_T_cam_tag[:3, :3].T + init_T_cam_tag[:3, 3]`：把这个方块从"tag 坐标系"变换到"相机坐标系"。`R.T` 是旋转转置（这里 `init_T_cam_tag[:3,:3]` 已是 T_cam_tag 的旋转，乘以 `.T` 是把 tag 局部点变换到相机系的标准做法：`p_cam = R @ p_tag + t`；写成矩阵乘法 `corners3d @ R.T` 等价于对每个点做 `R @ p`，因为 `corners3d` 的每行是一个点）。
- 第 822–828 行：把 3D 角点用针孔模型（1.2）投影回图像，得到 `proj_corners2d`。
- 第 831–842 行：求投影的包围盒，再**放大 2 倍**（rect_w/h = 原宽高 × 2）作为裁剪窗口，并用 `max(0, ...)` / `min(图宽-1, ...)` 夹在图像范围内。
- 第 845–849 行：去畸变后，按窗口裁剪出小图，在小图上检测 tag。
- 第 863–866 行：**恢复坐标到原图**：因为检测是在"裁剪小图"上做的，角点/中心坐标都是相对小图左上角的，要加回 `min_x, min_y` 才是原图坐标。
- 第 869–870 行：用恢复后的 tag 重新算 3D 角点和位姿。

**⚠️ 多个大坑（第 7 节第 8、9 点）**：
1. 第 813 行 `half_size = self.tag_size / 2`：`self.tag_size` 只在 `match` 里赋值。若直接调 `track` 而没先 `match`，`tag_size` 是 None → 报错。
2. 第 848 行 `int(min_y):int(max_y)`：用 `int()` 截断成整数像素，会有 sub-pixel（亚像素）误差。
3. 第 865–866 行 `tag.center += ...`：这是**原地修改** `apriltag2` 返回的 `tag` 对象里的数组（numpy 的 `+=` 原地改）。这污染了 detector 返回的对象，若后续还引用它会有副作用。

**被谁调用**：`test_tmpl_grasp_3d.py` 的 `track()` 辅助函数（第 227 行），在 `do_grasp` 的"迭代细化"循环里反复调用，逐步细化物体位姿。

---

## 4. Python 基础语法速查

本文件用到的语法，每条给最小示例。

**`@dataclasses.dataclass`（装饰器）**：加在类定义前，自动生成 `__init__` 等。字段必须写成 `名字: 类型 = 默认值`。

```python
import dataclasses
@dataclasses.dataclass
class Point:
    x: float = 0.0      # 有注解 → 是字段
    y: float = 0.0
p = Point(1.0, 2.0)    # 自动有了 __init__
```

**嵌套类**：类里面再定义类。本文件的 `TagMatcher2D.Config` 就是"在 `TagMatcher2D` 里面定义的 `Config` 类"。用 `外层.内层` 访问。

```python
class Outer:
    class Inner:
        a: int = 1
o = Outer.Inner()       # 创建嵌套类实例
```

**类属性 vs 实例属性**：写在 `class` 下面、方法外面的叫"类属性"（所有实例共享）；写在 `__init__` 里 `self.xxx = ...` 的叫"实例属性"（每个对象一份）。

```python
class C:
    shared = []         # 类属性，所有实例共享同一列表（危险！）
    def __init__(self):
        self.own = []   # 实例属性，每个对象独立
```

> 第 7 节第 4 点讲的 dataclass 坑，本质就是：没注解的"类属性"被所有实例共享。

**默认参数**：函数定义里 `def f(a, b=5)`，`b` 不传就用 5。注意：默认参数只在函数**定义时**求值一次，所以**不要用可变对象（list/dict）作默认参数**（会被反复共享）。本文件用 `None` 作默认再内部判断，是正确写法。

**`assert` 断言**：`assert 条件, "报错信息"`。条件假就抛 `AssertionError`。用于"这里不该发生"的检查。

**`lambda` 排序键**：`list.sort(key=lambda r: r.x)` 表示"按每个元素的 `x` 属性排序"。`lambda` 是匿名小函数，`lambda 参数: 表达式`。

```python
pts = [{'v': 3}, {'v': 1}, {'v': 2}]
pts.sort(key=lambda d: d['v'])   # 按 v 升序 → [1, 2, 3]
```

**列表推导式**：`[表达式 for 变量 in 可迭代]` 一行生成列表。

```python
squares = [i*i for i in range(5)]   # [0, 1, 4, 9, 16]
```

**`np.stack` / `np.vstack`**：`np.stack([a, b], axis=0)` 沿新轴堆叠；`np.vstack([a, b])` 沿行方向（第 0 轴）竖着拼。

```python
import numpy as np
a = np.array([1,2,3]); b = np.array([4,5,6])
np.stack([a,b], axis=0)   # 形状 (2,3)
np.vstack([a,b])          # 形状 (2,3)
```

**`np.where` 返回行列**：`vs, us = np.where(mask == 255)` 返回满足条件元素的"行索引数组"和"列索引数组"（总是两个数组，对应 y 和 x）。

```python
img = np.array([[0,255],[255,0]])
vs, us = np.where(img == 255)   # vs=[0,1], us=[1,0]
```

**`np.divide` 的 `out` / `where`**：`np.divide(a, b, out=res, where=mask)` 只在 `mask` 为真处做除法，结果写进 `res`（避免除零）。见 3.4 节。

**`astype` 类型转换**：`arr.astype(np.float32)` 把数组元素类型转换（不修改原数组，返回新数组）。`np.uint16` 是 16 位无符号整数。

**`np.clip`**：`np.clip(x, lo, hi)` 把 x 限制在 `[lo, hi]`。常用于把余弦/正弦值夹在 [-1,1] 防止 `arccos` 出 nan。

**`np.rint`**：四舍五入取整（返回 float，需再 `.astype` 成整数）。

**`enumerate`**：`for i, v in enumerate(lst)` 同时拿到下标和值。

**`zip`**：`for a, b in zip(list1, list2)` 把多个序列按位置一一配对遍历。

```python
for i, v in enumerate(['a','b']):
    print(i, v)            # 0 a / 1 b
for x, y in zip([1,2],[3,4]):
    print(x, y)            # 1 3 / 2 4
```

**`*args` 变参**：`def f(*args)` 收集任意多个位置参数为元组。本文件主程序里 `threading.Thread(target=run, args=(...))` 就是把一堆参数打包成元组传给线程函数。

**私有方法下划线约定**：以单下划线开头（如 `_undistort_color_img` in arm_utils）表示"这是内部用的，别在外面随便调"。Python 不强制，只是约定。本文件 `ImageUndistorter._undistort_maps` 的 `_` 也表示"内部缓存，别直接碰"。

**文档字符串 docstring**：函数/类下面第一对三重引号 `"""..."""` 里写的说明，可用 `help(函数)` 查看。本文件几乎每个函数都有。

---

## 5. 输入输出规范

每个函数对输入数据格式有约定，调用方（比如 `test_tmpl_grasp_2d.py` / `test_tmpl_grasp_3d.py` / `arm_utils.py`）必须按此准备数据。

| 数据 | 约定格式 | 说明 | 由谁准备 |
|---|---|---|---|
| `rgb_img` / `bgr_img` | **CV_8UC3**：`uint8`、高×宽×3（BGR 顺序，OpenCV 默认） | 彩色图。OpenCV 读图是 BGR 不是 RGB，注意通道顺序 | `CamNode` 的 RGB 话题回调 |
| `depth_img`（传给 `rgbd_/depth_to_point_cloud`） | **CV_32FC1**：`float32`、高×宽、单位**米** | 已经乘过 `depth_scale` 转成米的浮点深度 | `CollisionDetector` 里手动 `astype(float32)*depth_scale` |
| `depth_img`（传给 `compute_tag_corners3d`） | 一般是 **CV_16UC1**：`uint16`、高×宽、单位**原始计数** | 函数内部第 327 行会 `* depth_scale` 转米 | `CamNode` 深度话题（原始整数） |
| `depth_img_list`（传给 `depth_mean_filter`） | **CV_16UC1**：`uint16` 列表 | docstring 明说输入是整数深度；**不要传米制 float**（否则转 uint16 变 0） | 多帧深度图 |
| `intrinsic` | `List[float]` 四元组 `[fx, fy, cx, cy]` | 来自 `read_cam_params` / `read_rgbd_params` | `cam_params.json` |
| `depth_scale` | `float`，米/计数 | 来自 `read_rgbd_params` | `cam_params.json` |
| `distortion` | `List[float]`，如 `[k1,k2,p1,p2,k3]` | OpenCV 畸变格式；`None` 或空表示无畸变 | `cam_params.json` |
| `debug_dir` | 字符串路径或 `None` | 不为 None 时，函数会在其下建 `match/`、`track/` 子目录，存 `color.png`、`proj_area.png` 等 | 调用方传 `os.path.join(root_dir,'results','debug','grasp_3d')` |

**`debug_dir` 的目录结构**（以 3D 为例，见 `test_tmpl_grasp_3d.py` 第 779 行）：

```
results/debug/grasp_3d/
├── match/          ← matcher.match 时创建
│   └── color.png   ← 去畸变后的彩色图（debug_level>=1 或没检测到时）
└── track/          ← matcher.track 时创建
    └── proj_area.png ← 投影裁剪窗口可视化（debug_level>=1 或没检测到时）
```

**调用方准备数据的典型代码**（`test_tmpl_grasp_3d.py` 第 721–781 行）：

```python
intrinsic, distortion, depth_scale = read_rgbd_params(cam_params_path)
config = TagMatcher3D.Config(
    intrinsic=intrinsic,
    depth_scale=depth_scale,
    distortion=distortion,
    debug_dir=os.path.join(root_dir, 'results', 'debug', 'grasp_3d')
)
matcher = TagMatcher3D(config)
# 主循环里：
color_img, depth_img = frames[0][0], frames[0][1]   # 来自 CamNode
result_list, msg = matcher.match(bgr_img=color_img, depth_img=depth_img, top_k=1)
```

注意 `match` 收到的 `depth_img` 是**原始整数深度**（CV_16UC1），由相机节点直接给，函数内部会自己乘 `depth_scale`。这点和 `rgbd_to_point_cloud` 要求"已经换算成米的 float32"正好相反——**同一个仓库里深度图的"单位约定"不统一，使用时务必看清每个函数要的是哪种**。

---

## 6. 核心数学：一步一步算给你看

### 6.1 反投影：从 (u,v,depth) 到 (x,y,z)

已知像素 `(u, v)` 和深度 `Z`（米），内参 `fx=900, fy=900, cx=640, cy=360`。

公式（1.2 节）：
```
X = (u - cx) / fx * Z
Y = (v - cy) / fy * Z
```

取一个真实例子：`u=730, v=450, Z=0.5` 米。

```
X = (730 - 640) / 900 * 0.5 = 90 / 900 * 0.5 = 0.1 * 0.5 = 0.05 米
Y = (450 - 360) / 900 * 0.5 = 90 / 900 * 0.5 = 0.1 * 0.5 = 0.05 米
Z = 0.5 米（已知）
```

所以相机坐标系下这个点是 `(0.05, 0.05, 0.5)`——在相机右下方 5 厘米、前方 0.5 米处。注意 `u > cx` 对应 `X > 0`（相机坐标系通常 X 向右为正），`v > cy` 对应 `Y > 0`（OpenCV 图像 Y 向下为正，所以 `v > cy` 即图像下方对应相机 Y 正方向，这是 OpenCV 相机系的约定）。

### 6.2 射线与平面求交 z = -D/(A·nx+B·ny+C) 的推导

平面方程：`A·x + B·y + C·z + D = 0`，`(A,B,C)` 是单位法向量。

一个像素 `(u, v)` 对应的射线方向（归一化坐标）是 `(nx, ny, 1)` 的倍数，射线上任一点可写成：

```
P(z) = (nx·z,  ny·z,  z)        z 是深度，待求
```

把它代入平面方程：

```
A·(nx·z) + B·(ny·z) + C·(z) + D = 0
```

把含 z 的提出来：

```
z·(A·nx + B·ny + C) + D = 0
```

移项：

```
z·(A·nx + B·ny + C) = -D
```

两边除以 `(A·nx + B·ny + C)`：

```
        -D
z = ───────────────
     A·nx + B·ny + C
```

这就是第 349 行的公式。求出 z 后：

```
x = nx · z
y = ny · z
```

**数值例子**：平面是竖直墙面，法向量 `(1, 0, 0)`（即 A=1, B=0, C=0），D = -0.3（墙面在 x=0.3 米处）。入射方向 `nx=0.5, ny=0.2`（射线指向右前方）。

```
z = -D / (A·nx + B·ny + C) = -(-0.3) / (1·0.5 + 0 + 0) = 0.3 / 0.5 = 0.6
x = 0.5 · 0.6 = 0.3   ← 正好等于墙面位置，合理
y = 02 · 0.6 = 0.12
```

得到交点 `(0.3, 0.12, 0.6)`，x=0.3 正好落在墙面上，验证正确。

> 分母 `A·nx + B·ny + C` = 法向量·射线方向（忽略了射线方向的 z 分量 1 的系数）。当射线**平行于**平面时这个值为 0，z 趋于无穷——代码第 349 行没有防护（第 7 节第 2 点相关）。

### 6.3 用三个角点定平面法向量：叉乘

点云平面拟合（`segment_plane`）给出平面；但在 `arm_utils._proj_rect_3d`（第 236–241 行）里，是直接用矩形三个点算平面法向量的。方法：**两个边向量的叉乘 = 法向量**。

设平面上三点 `p0, p1, p2`：

```
v1 = p1 - p0
v2 = p2 - p0
normal = v1 × v2        （叉乘）
```

**叉乘（cross product）**定义：对向量 `a=(a1,a2,a3)`、`b=(b1,b2,b3)`，

```
a × b = (a2·b3 - a3·b2,  a3·b1 - a1·b3,  a1·b2 - a2·b1)
```

叉乘的结果**垂直于 a 也垂直于 b**，所以垂直于它们张成的平面——正是我们想要的平面法向量。

**数值例子**：`p0=(0,0,0)`, `p1=(1,0,0)`, `p2=(0,1,0)`（XY 平面上的三点）。

```
v1 = (1,0,0),  v2 = (0,1,0)
normal = v1 × v2 = (0·0-0·1, 0·0-1·0, 1·1-0·0) = (0, 0, 1)
```

法向量 `(0,0,1)`——确实是 XY 平面的法向（朝 z 轴）。然后 `D = -dot(normal, p0) = 0`，平面方程 `z = 0`，正确。

### 6.4 用角点建坐标系：边向量与正交化

`compute_tag_pose`（第 376–386 行）建 tag 坐标系。以四个 3D 角点 `c0,c1,c2,c3`（左上、右上、右下、左下）为例：

```
        c1 ─────── c2
        │          │
        │   tag    │
        │          │
        c0 ─────── c3
```

- **X 轴** = `c1 - c0`（上边，指向右）。
- **Y 轴（初）** = `c3 - c0`（左边，指向下）。
- **Z 轴** = `X × Y`（叉乘，朝外，右手定则）。
- **Y 轴（重算）** = `Z × X`。

**为什么要重新算一次 Y（正交化）**：理想情况下 tag 是完美矩形，X 和 Y 严格垂直。但角点检测有噪声，测得的两个边可能不垂直。如果直接用不垂直的 X、Y 当坐标轴，建出的坐标系是"斜的"，后续位姿全错。

正交化的思路：
1. 先归一化 X（`axis_x /= norm`）。
2. 用 X 和 Y 叉乘得到 Z（Z 一定垂直于 X，因为叉乘性质；也近似垂直于 Y）。
3. 再用 Z 和 X 叉乘得到 Y' = `Z × X`。这个 Y' **一定垂直于 X**（叉乘保证），且**落在平面内**（垂直于 Z）。于是 X、Y'、Z 三者两两垂直，构成标准正交基。

```
X 固定 ─┐
        ├─ 叉乘 → Z（垂直 X）
Z × X  ─┘
   │
   └─→ Y'（既垂直 X，又在平面内）
```

`GripperBody.initialize`（第 82–91 行）用的是同样的"先叉乘 Z、再 Z×X 重算 Y"正交化套路，只是它用的边是**对角线**（见第 7 节第 7 点）。

### 6.5 旋转矩阵的迹求夹角

旋转误差公式：`cosθ = (trace(R) - 1) / 2`（第 169 行、第 356 行、arm_utils 多处）。

**为什么？** 一个绕单位轴 `n=(nx,ny,nz)` 转 `θ` 角的旋转矩阵是：

```
R = cosθ·I + (1-cosθ)·(n·nᵀ) + sinθ·[n]×
```

取对角线三项相加（迹 trace = 主对角线之和）：

```
trace(R) = R[0][0] + R[1][1] + R[2][2]
```

把上面 R 的对角元相加：
- `cosθ·I` 贡献 `3·cosθ`
- `(1-cosθ)·(n·nᵀ)` 的对角元是 `(1-cosθ)·(nx²+ny²+nz²) = (1-cosθ)·1`（因为 n 是单位向量，nx²+ny²+nz²=1）
- `sinθ·[n]×` 是反对称矩阵，对角线全 0，贡献 0

所以：

```
trace(R) = 3cosθ + (1-cosθ) = 3cosθ + 1 - cosθ = 1 + 2cosθ
```

解出：

```
cosθ = (trace(R) - 1) / 2
θ = arccos( (trace(R) - 1) / 2 )
```

**数值例子**：R 是绕 z 轴转 60° 的矩阵，理论 trace = 1 + 2·cos60° = 1 + 2·0.5 = 2.0。

```
cosθ = (2.0 - 1)/2 = 0.5
θ = arccos(0.5) = 60°   ✓
```

**为什么要 `np.clip`**：浮点误差可能让 `cosθ` 算成 `1.0000000002`。`arccos(1.0000000002)` 数学上是未定义的，计算库会返回 `nan`，整个误差变废。所以先 `np.clip(cosθ, -1.0, 1.0)` 把它夹回 1.0，得 `arccos(1.0)=0`，安全。

### 6.6 射影变换 H = K_dst · R · K_src⁻¹

`compute_projective_transformation` 第 245 行。

**目标**：把 src 相机拍的图，投影成"在 dst 相机视角"下的图。两相机只差旋转 R（无平移）。

设 src 图上一点（齐次坐标）`p_src = (u, v, 1)ᵀ`。

1. 用 src 内参逆，把它变成 src 相机下的射线方向：
   ```
   ray_src = K_src⁻¹ · p_src
   ```
2. 旋转到 dst 相机坐标系：
   ```
   ray_dst = R · ray_src = R · K_src⁻¹ · p_src
   ```
3. 用 dst 内参投影回像素（注意这里 K_dst 是"带主点偏移"的完整内参，见 3.6 节，dst 的主点我们自己选过）：
   ```
   p_dst = K_dst · ray_dst = K_dst · R · K_src⁻¹ · p_src
   ```

于是射影变换：

```
p_dst = (K_dst · R · K_src⁻¹) · p_src
        └──────── H ────────┘
```

即 `H = K_dst · R · K_src⁻¹`。把 `p_src` 乘 H 就得到 `p_dst` 的齐次坐标，再除以第 3 分量还原成像素。

> 注意：本函数里 `K_src` 用 `src_K`，`K_dst` 是重新构造的（主点选成让内容不裁剪）。`R = R_dst_src` 是从 src 到 dst 的旋转。

### 6.7 depth_mean_filter 的逐像素统计（手算示例）

假设有 3 帧 3×3 的深度图（单位：毫米整数，越大越远）。这里 `obs_ratio=0.5`、`n=3`，阈值 `th_ob_cnt = max(1, ceil(0.5·3)) = max(1, 2) = 2`——即"至少 2 帧有值才有效"。

帧 0：
```
10  20   0
10  20  30
 0  20  30
```
帧 1：
```
10   0  20
10  20  30
10  20   0
```
帧 2：
```
10  20  20
 0  20  30
10   0  30
```

**对每个像素统计 obs_count 和 depth_sum：**

以像素 (0,0)（第 0 行第 0 列）为例：三帧都是 10 → obs_count=3，depth_sum=30 → 有效 → mean = 30/3 = 10。
像素 (0,1)：值 [20, 0, 20] → obs_count=2（2 帧非 0），sum=40 → 有效（2>=2）→ mean=40/2=20。
像素 (0,2)：值 [0, 20, 20] → obs_count=2，sum=40 → mean=20。
像素 (1,2)：值 [30,30,30] → mean=30。
像素 (2,0)：值 [0,10,10] → obs_count=2, sum=20 → mean=10。
像素 (2,1)：值 [20,20,0] → obs_count=2, sum=40 → mean=20。
像素 (2,2)：值 [30,0,30] → obs_count=2, sum=60 → mean=30。
像素 (1,0)：值 [10,10,0] → obs_count=2, sum=20 → mean=10。
像素 (1,1)：值 [20,20,20] → mean=20。

最终均值图（全部有效，因为每格至少 2 帧非 0）：
```
10  20  20
10  20  30
10  20  30
```
对比单帧，那些"孤零零的 0 洞"被有效邻帧填上了——这就是该函数的价值：多帧融合去噪补洞。若某像素只有 1 帧非 0（< 阈值 2），则被置 0（视为不可信），例如假设某格三帧是 [0,5,0]，obs_count=1<2 → mean=0。

### 6.8 对角外延 ex_ratio 的几何含义

`compute_tag_mask`（第 293–297 行）把每个角点沿对角线方向往外推 `ex_ratio` 倍。为什么是"对角"而不是"邻边"？因为沿对角外延能让整个四边形**均匀放大**（各边都外扩），而沿邻边外延会只推一个方向。

ASCII 示意（`ex_ratio=0.3` 大约把角点推出去 30% 的半对角线长）：

```
原始 tag（corners0..3）:          外延后（ex_corners0..3）:

   c0 ┌──────┐ c1                 c0'  ┌────────┐  c1'
      │      │            →          ┌─┘        └─┐
      │ tag  │                       │   tag     │
      │      │                       └─┐        ┌─┘
   c3 └──────┘ c2                 c3'  └────────┘  c2'

   c0' = c0 + (c0 - c2) * 0.3     （沿 c2→c0 方向外推）
```

`(c0 - c2)` 是从右下 c2 指向左上 c0 的向量，乘以 0.3 再加回 c0，等于把 c0 沿"远离中心"的方向推出去。四个角点都这样推，整块区域就被等比例放大了一圈。

**目的**：tag 边缘的深度常不可靠（边缘糊、有洞）。外延后掩码多罩住 tag 周围一圈**同一平面**的区域，拟合平面时能用到更多内点，结果更稳。但 `ex_ratio` 也不能太大，否则罩到不属于该平面的背景，污染平面。

---

## 7. 这段代码里的坑与改进建议

按"现象 / 根因 / 改法"三列呈现，重点问题后附可直接粘贴的改进代码。

### 7.1 问题总览表

| # | 位置 | 现象 | 根因 | 改法 |
|---|---|---|---|---|
| 1 | `compute_tag_corners3d` 第 337 行 | 夹爪/物体离相机 > 0.5m，掩码点云被截断成空，平面拟合退化甚至报错 | `depth_trunc=0.5` 硬编码 | 改为按 tag 深度自适应或参数传入 |
| 2 | 第 341–342 行 | 点云为空时 `len(pc.points)` 为 0，日志里除零崩溃 | 平面拟合后未检查 inliers 数量与除零 | 拟合前检查点云非空、拟合后检查 inliers 比例 |
| 3 | `compute_locate_error` 第 165–178 行 | `sym_tfs` 为 `(0,)` 空数组时抛 `UnboundLocalError`；位置/角度误差耦合 | for 不执行导致 `delta_pos` 未赋值；pos 只在角度更新时记录 | 用 `sym_tfs.reshape(-1,4,4)`；每轮独立算 pos 与 rot 取综合最小 |
| 4 | `TagMatcher2D.Result`/`TagMatcher3D.Result` 第 521/524/527、680/683/686 行 | 有人写 `result.center[0]=1` 会污染所有实例 | 字段无类型注解，变成共享类属性（可变对象） | 给所有字段加注解；或不用 dataclass |
| 5 | `TagMatcher2D.match` 第 610 行 / `TagMatcher3D.match` 第 780 行 | 注释说"分数最高"，实际是"离光轴最近/离相机最近"，且多 tag 会抓错 | 排序键与注释不符、无 id 过滤 | 改成真正的分数排序 + 按 `id` 过滤 |
| 6 | `ImageUndistorter.undistort_img` 第 446–458 行 | 换分辨率图像后静默用错尺寸 map | map 只按首图尺寸缓存 | 按 `(w,h)` 做 key 缓存 |
| 7 | `compute_tag_pose` 第 376–380 行 vs `GripperBody.initialize` 第 76–77 行 | 两个模块对同 tag 建的坐标系差 45° | 一个用邻边、一个用对角线 | 统一建法，或在文档/接口注明差异 |
| 8 | `TagMatcher3D.track` 第 813 行 | 未先 `match` 直接 `track`，`tag_size` 为 None 崩 | `tag_size` 只在 match 赋值 | match 里初始化，或 track 自带估算 |
| 9 | `track` 第 831–866 行 | sub-pixel 误差；`tag.center +=` 原地改 detector 对象 | `int()` 截断；`+=` 原地修改 | 用 `np.round`/浮点裁剪；复制后再改 |
| 10 | `depth_mean_filter` docstring 第 108/111 行 | 若传入米制 float 深度，转 `uint16` 变 0 | 单位/类型约定不清，期望原始整数 | docstring 明确"输入为原始整数深度" |
| 11 | `compute_projective_transformation` 第 231–236 行 | 退化情况（宽高为 0）未处理；假设已去畸变 | 未校验包围盒、未文档化前提 | 加校验；文档声明需先去畸变 |
| 12 | `compute_tag_pose_2d` 第 270 行 | 跨 ±π 的角度差算错（下游已处理，但易踩） | `theta ∈ (-π, π]`，直接相减会跳变 | 下游用归一化（test 已做，本函数可加注释） |

### 7.2 改进代码

**问题 1 + 2：`compute_tag_corners3d` 的截断与除零**

```python
def compute_tag_corners3d(tag, depth_img, intrinsic, depth_scale, ex_ratio=0.1,
                          depth_trunc=None):
    mask_img = np.zeros(depth_img.shape, dtype=np.uint8)
    compute_tag_mask(tag, mask_img, ex_ratio=ex_ratio)

    flt_depth_img = depth_img.astype(np.float32) * depth_scale
    flt_depth_img[mask_img == 0] = 0.0

    # 用 tag 区域的深度中位数自适应截断上限（避免硬编码 0.5m）
    if depth_trunc is None:
        valid = flt_depth_img[mask_img != 0]
        depth_trunc = float(np.median(valid)) * 1.5 if valid.size > 0 else 1.0
    # end if

    pc = open3d.geometry.PointCloud.create_from_depth_image(
        open3d.geometry.Image(flt_depth_img),
        open3d.camera.PinholeCameraIntrinsic(flt_depth_img.shape[1], flt_depth_img.shape[0],
                                             intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]),
        np.eye(4), depth_scale=1.0, depth_trunc=depth_trunc)

    if len(pc.points) == 0:
        logging.error("masked point cloud is empty, cannot fit plane")
        return None
    # end if

    plane, inliers = pc.segment_plane(distance_threshold=0.002, ransac_n=6, num_iterations=1000)
    if len(inliers) < 0.3 * len(pc.points):
        logging.warning(f"plane inlier ratio too low: {len(inliers)}/{len(pc.points)}")
    # end if
    ...
```

**问题 3：`compute_locate_error` 空数组与耦合**

```python
def compute_locate_error(expected_T_cam_model, actual_T_cam_model, sym_tfs=None):
    if sym_tfs is None:
        sym_tfs = np.array([np.eye(4)], dtype=np.float32)
    sym_tfs = np.asarray(sym_tfs, dtype=np.float32).reshape(-1, 4, 4)  # 兼容 (0,) 与 (N,4,4)
    if sym_tfs.shape[0] == 0:
        return float('nan'), float('nan')

    best_pos, best_rot, best_total = 1e9, 1e9, 1e9
    exp_R, exp_t = expected_T_cam_model[:3, :3], expected_T_cam_model[:3, 3]
    for sym_tf in sym_tfs:
        T = actual_T_cam_model @ sym_tf
        dR = exp_R @ T[:3, :3].T
        c = np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)
        rot = abs(np.arccos(c))
        pos = np.linalg.norm(exp_t - T[:3, 3])
        total = pos + rot * 0.05   # 位置(米)+角度(弧度)加权，按综合最小取
        if total < best_total:
            best_total, best_pos, best_rot = total, pos, rot
    # end for
    return best_pos * 1000.0, best_rot * 180.0 / np.pi
```

**问题 4：给 Result 字段加注解**

```python
@dataclasses.dataclass
class Result:
    id: int = -1
    center: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    corners: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((4, 2), dtype=np.float32))
    pose_2d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
```

> 关键点：带注解后，dataclass 才会把它当字段，且用 `default_factory` 让**每个实例拿到独立的可变对象**，不再共享。这是修复"共享可变默认值"的标准写法。

**问题 5：top_k 按 id 过滤 + 真实分数**

```python
# 2D 版：先按 id 过滤，再取离光轴最近（语义明确）
if target_id is not None:
    result_list = [r for r in result_list if r.id == target_id]
if top_k > 0 and len(result_list) > top_k:
    result_list.sort(key=lambda r: r.pose_2d[0]**2 + r.pose_2d[1]**2)
    result_list = result_list[:top_k]

# 3D 版同理
if target_id is not None:
    result_list = [r for r in result_list if r.id == target_id]
if top_k > 0 and len(result_list) > top_k:
    result_list.sort(key=lambda r: r.T_cam_tag[2, 3])
    result_list = result_list[:top_k]
```

**问题 6：`ImageUndistorter` 按尺寸缓存 map**

```python
def undistort_img(self, src_img):
    if self.D is None:
        return src_img.copy()
    h, w = src_img.shape[:2]
    if (w, h) not in self._undistort_maps:
        map1, map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=self.K, distCoeffs=self.D, R=None,
            newCameraMatrix=self.K, size=(w, h), m1type=cv2.CV_16SC2)
        self._undistort_maps[(w, h)] = (map1, map2)
    # end if
    map1, map2 = self._undistort_maps[(w, h)]
    return cv2.remap(src_img, map1, map2, interpolation=cv2.INTER_LINEAR)
```

> 注意 `_undistort_maps` 要改成字典（`self._undistort_maps = {}`），见 `__init__`。

**问题 7：坐标系建法统一（建议在 `compute_tag_pose` 与 `GripperBody` 间二选一并注释）**

若要让 `compute_tag_pose` 和 `GripperBody` 一致，可统一成"对角线"或"邻边"。但**更稳妥的做法是保持各自语义、在文档和接口名里注明**，避免悄悄改动导致已标定数据失效。在 `compute_tag_pose` 上方加注释：

```python
# 注意：本函数用邻边(c1-c0, c3-c0)建 X/Y；
# GripperBody.initialize 用对角线(c2-c0, c1-c3)建轴，二者朝向相差 45°。
# 物体定位用本函数，夹爪标定用 GripperBody，不要混用同一套朝向假设。
```

**问题 8：`track` 自带 tag_size 估算**

```python
# 在 track 开头，若尚未估算则先粗估
if self.tag_size is None:
    # 用上一帧 init_T_cam_tag 的尺度 + 投影面积粗估，或要求调用方保证先 match
    logging.warning("tag_size is None, call match() before track() or estimate it")
```

更稳妥：把 `match` 里的估算逻辑抽成 `_estimate_tag_size(corners3d)` 私有方法，`track` 在拿到 `compute_tag_corners3d` 结果后也调用一次。

**问题 9：`track` 浮点裁剪与避免原地修改**

```python
# 用四舍五入代替 int() 截断
y0, y1 = int(np.round(min_y)), int(np.round(max_y))
x0, x1 = int(np.round(min_x)), int(np.round(max_x))
crop_bgr_img = un_img[y0:y1, x0:x1]

# 恢复坐标时复制，不污染 detector 返回对象
tag = tag_list[0]
tag.center = tag.center + np.array([min_x, min_y])      # 复制后再加
tag.corners = tag.corners + np.array([min_x, min_y])
```

**问题 10：`depth_mean_filter` 文档与类型双保险**

```python
def depth_mean_filter(depth_img_list, obs_ratio=0.5):
    """
    输入: depth_img_list —— 原始整数深度图列表 (CV_16UC1, uint16, 单位=计数)
    注意: 本函数期望原始整数深度, 不要传入已乘 depth_scale 的米制 float,
          否则 .astype(np.uint16) 会把 <1 的浮点截断成 0。
    """
    # 加一道类型校验
    assert depth_img_list[0].dtype == np.uint16, "expected CV_16UC1 uint16 raw depth"
    ...
```

**问题 11：`compute_projective_transformation` 退化保护**

```python
dst_w = int(np.ceil(u_max - u_min)) + 1
dst_h = int(np.ceil(v_max - v_min)) + 1
if dst_w <= 0 or dst_h <= 0:
    logging.error("degenerate projection (zero-size), check R_dst_src and src corners")
    return None, None, None
# end if
# 另：文档声明调用方必须先 undistort
```

---

## 8. 一句话总结

> **`vision_utils.py` 把"彩色图 + 深度图 + 相机内参"翻译成"贴着物体的 AprilTag 在相机坐标系下的 4×4 位姿矩阵"：先去畸变找 tag 的 2D 角点，再用深度把角点反投成 3D 并拟合出物体表面平面，最后由角点建出 tag 坐标系；它同时附带去畸变、点云生成、对称定位误差、掩码外延、射影变换等抓取前处理工具，被 `test_tmpl_grasp_2d.py`/`test_tmpl_grasp_3d.py` 当作"眼睛到大脑"的翻译官调用。**

**最该记住的三件事：**

1. **核心数学是"射线与平面求交"** `z = -D/(A·nx+B·ny+C)`（第 349 行），它让 tag 角点即使落在深度洞上也能借平面补出 3D 坐标。
2. **两个 45° 坐标系坑**：`compute_tag_pose`（邻边建系）和 `GripperBody.initialize`（对角线建系）对同一个 tag 朝向差 45°，物体定位与夹爪标定不要混用同一套朝向假设。
3. **dataclass 字段必须带注解**：`Result` 里的 `id`/`center`/`corners` 因为没写 `: 类型`，成了所有实例共享的可变类属性——想加注解修复，就用 `default_factory`。

**排错清单（结果不对时按顺序查）：**

```
① 物体离相机 > 0.5m 却匹配失败？→ 第 337 行 depth_trunc 硬编码，改自适应（问题1）
② 多 tag 场景抓错物体？→ match 没按 id 过滤（问题5）
③ 换分辨率图后位姿全乱？→ ImageUndistorter 缓存了错尺寸 map（问题6）
④ 报 UnboundLocalError？→ 传入空 sym_tfs 数组（问题3）
⑤ 深度图一片 0？→ 把米制 float 喂给了期望 uint16 的 depth_mean_filter（问题10）
```

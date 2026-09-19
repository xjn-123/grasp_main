# `calib_camera.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、也没接触过相机标定的同学。
> 目标：读完之后，你能逐行看懂这个文件在干什么，知道每个数字是怎么来的，知道怎么运行它、结果存在哪、出了问题往哪查。
>
> 被分析的源文件：
> `C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\examples\common\src\calib_camera.py`（共 231 行）

---

## 目录

- [0. 先建立一个整体印象](#0-先建立一个整体印象)
- [1. 背景知识：相机标定到底在算什么](#1-背景知识相机标定到底在算什么)
  - [1.1 拍照 = 把三维世界压扁成二维照片](#11-拍照--把三维世界压扁成二维照片)
  - [1.2 内参矩阵 K：fx, fy, cx, cy](#12-内参矩阵-kfx-fy-cx-cy)
  - [1.3 畸变系数 D：k1, k2, p1, p2, k3](#13-畸变系数-dk1-k2-p1-p2-k3)
  - [1.4 标定 = 解方程（反推参数）](#14-标定--解方程反推参数)
  - [1.5 AprilTag 标定板是什么](#15-apriltag-标定板是什么)
  - [1.6 为什么是「至少 10 张图、每张至少 4 个点」](#16-为什么是至少-10-张图每张至少-4-个点)
- [2. 整体流水线](#2-整体流水线)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 文件顶部的说明字符串](#31-文件顶部的说明字符串)
  - [3.2 导入区（第 6–31 行）](#32-导入区第-631-行)
  - [3.3 `sys.path` 那三行（第 21–23 行）](#33-syspath-那三行第-2123-行)
  - [3.4 主函数 `calib_camera`（第 36–112 行）](#34-主函数-calib_camera第-36112-行)
  - [3.5 `if __name__ == "__main__"`（第 121–230 行）](#35-if-__name__--__main__第-121230-行)
- [4. Python 基础语法速查](#4-python-基础语法速查)
- [5. 输入与输出：怎么跑、结果长啥样](#5-输入与输出怎么跑结果长啥样)
- [6. 动手算一遍（数值演练）](#6-动手算一遍数值演练)
- [7. 这段代码里的坑与改进建议](#7-这段代码里的坑与改进建议)
- [8. 一句话总结](#8-一句话总结)

---

## 0. 先建立一个整体印象

用一句大白话说这个脚本：

> **你拿着一块印满黑白方块（AprilTag）的板子，在不同角度拍了一堆照片，丢进这个脚本，它就能算出这台相机的"身份证参数"：焦距、主点、畸变，然后存成一个 `cam_params.json` 文件，供后续的机器人抓取程序使用。**

为什么机器人抓取需要这个？因为机器人的"眼睛"（相机）看到的是**二维像素坐标**（比如"目标在照片的第 320 行、第 240 列"），而机械臂要移动的是**三维空间坐标**（比如"往前 30 厘米、往左 12 厘米"）。要把像素坐标换算成空间坐标，就必须先知道这台相机的内参和畸变——这就是"标定"。

**输入 → 输出：**

| | 内容 |
|---|---|
| 输入 1 | 一个装满了 `.png` 照片的文件夹（`--img_dir`），照片内容都是那块 AprilTag 标定板 |
| 输入 2 | 标定板的物理规格（`--calib_board_info`）：每个 tag 多大、间距多大、几行几列 |
| 输出 | `cam_params.json`，里面有 `intrinsic`（内参）和 `distortion`（畸变）两组数字 |

---

## 1. 背景知识：相机标定到底在算什么

这一段不看代码，纯粹讲道理。理解了这段，后面的代码就只是"把这段道理翻译成 Python"而已。

### 1.1 拍照 = 把三维世界压扁成二维照片

想象你面前有一个点 $P$，它在空间中有一个位置。相机拍下它之后，照片上只会留下一个像素点 $(u, v)$（$u$ 是横向第几列，$v$ 是纵向第几行）。

从三维到二维，要走三步：

```
    三维世界坐标 (Xw, Yw, Zw)        单位：米，标定板上量出来的
            │
            │  ① 外参变换（旋转 R + 平移 t）
            │     含义：标定板此刻摆在哪儿、歪没歪
            ▼
    相机坐标系 (Xc, Yc, Zc)          单位：米，以相机光心为原点
            │
            │  ② 透视投影：Xc/Zc，Yc/Zc
            │     含义：越远的东西在照片上越小
            ▼
    归一化平面坐标 (x, y)            单位：无量纲
            │
            │  ③ 加畸变 + 乘焦距 + 加主点偏移
            ▼
    像素坐标 (u, v)                  单位：像素
```

写成公式（这就是传说中的**针孔相机模型**）：

$$u = f_x \cdot x_{distorted} + c_x$$
$$v = f_y \cdot y_{distorted} + c_y$$

其中 $x = X_c/Z_c,\ y = Y_c/Z_c$。

### 1.2 内参矩阵 K：fx, fy, cx, cy

把上面那个公式塞进一个 3×3 的矩阵里，就是**内参矩阵**（`cameraMatrix`）：

$$
K = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

四个数字的含义，用大白话讲：

| 参数 | 名字 | 大白话 | 典型数值（1280×720 相机） |
|---|---|---|---|
| `fx` | 横向焦距 | 相机"放大倍率"。越大＝看得越远越窄（长焦）；越小＝视野越广（广角） | 900 左右 |
| `fy` | 纵向焦距 | 同上，纵向的。通常和 `fx` 几乎相等 | 900 左右 |
| `cx` | 主点横坐标 | 相机镜头正中心在照片上的位置，理论上应该是图片宽度的一半 | 640 左右 |
| `cy` | 主点纵坐标 | 同上，纵向。理论上是高度的一半 | 360 左右 |

> ⚠️ **重点**：`fx/fy` 的单位是**像素**，不是毫米。它等于"物理焦距 ÷ 每个像素的物理尺寸"。所以同一个镜头装在不同像素密度的传感器上，`fx` 是不一样的。这就是为什么不能抄别人相机参数的原因。

> 单位提醒：本脚本里 3D 点的单位是**米**（`tag_size=0.0245` 即 24.5 毫米），标出来的 `fx` 单位是像素，两者是自洽的。如果 3D 点用毫米，`fx` 数值会变成 1/1000，但投影结果一样——只要**单位统一**就行。

### 1.3 畸变系数 D：k1, k2, p1, p2, k3

真实镜头不是完美针孔，拍出来的照片会有形变。最常见的两种：

- **径向畸变（桶形/枕形）**：照片边缘往外鼓或往里凹。由 `k1, k2, k3` 描述。
- **切向畸变**：镜头装歪了，和成像平面不平行。由 `p1, p2` 描述。

OpenCV 用的畸变公式（施加在归一化坐标 $x, y$ 上）：

$$r^2 = x^2 + y^2$$

$$x_{dist} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1xy + p_2(r^2 + 2x^2)$$
$$y_{dist} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y^2) + 2p_2xy$$

不用背，只要知道：**这 5 个数字描述了"照片被扭曲了多少"**。知道了它们，就能把扭曲的照片"掰直"（去畸变 `cv2.undistort`）。

- `k1` 一般是绝对值最大的那个。`k1 < 0` → 桶形畸变（广角镜头常见）；`k1 > 0` → 枕形畸变。
- `p1, p2` 通常很小（0.001 量级），接近 0 说明镜头装得很正。

### 1.4 标定 = 解方程（反推参数）

正着算我们刚才已经写了：已知 3D 点 → 可以算出它在照片上的 2D 位置。

但标定是**反着来**的：

- **已知**：一堆 3D 点的真实位置（标定板上量出来的，比如"第 5 个 tag 的左上角在板子坐标 (0.032, 0.075, 0)"）
- **已知**：这些点在照片上的像素位置（AprilTag 检测算法找出来的）
- **未知**：`fx, fy, cx, cy, k1, k2, p1, p2, k3`，外加每张照片拍摄时标定板的姿态（旋转 + 平移，每张图 6 个未知数）

标定就是**求解这组未知数**，使得"用参数正着算出来的 2D 点"和"实际检测到的 2D 点"之间的差距最小。

这个"差距"就叫**重投影误差（reprojection error）**，脚本里打印的 `RMS error` 就是它的均方根，单位是**像素**：

$$RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left[(u_i^{计算} - u_i^{检测})^2 + (v_i^{计算} - v_i^{检测})^2\right]}$$

**怎么判断标定好坏（工程经验值）：**

| RMS 值 | 评价 |
|---|---|
| < 0.3 px | 优秀 |
| 0.3 ~ 0.5 px | 很好 |
| 0.5 ~ 1.0 px | 可接受，能用 |
| > 1.0 px | 有问题：照片糊了 / 标定板尺寸填错 / 姿态太单一 / tag 检测出错 |

> 求解过程本身是 OpenCV 在 `cv2.calibrateCamera` 内部完成的：先按张正友法用"单应性矩阵"给一个初值，再用 Levenberg-Marquardt 迭代优化。我们不用自己写，只需要喂对数据。

### 1.5 AprilTag 标定板是什么

AprilTag 就像"二维码的简版"——一个黑白方块图案，自带 ID 编号。好处：

1. **自带 ID**：算法识别出来不光知道"这里有个角点"，还知道"这是第 17 号 tag 的左上角"。这样就能自动和标定板上第 17 号 tag 的 3D 位置配对上，不用人工排序。
2. **角点精度高（亚像素级）**：比人手工点或者普通棋盘格角点稳。
3. **部分遮挡也能认**：拍不全没关系，认出几个算几个。

脚本里用的 `tag36h11` 家族：意思是 "36 位编码、汉明距离 11"，即图案是 6×6 的黑白格，抗误识别能力强。

**标定板的样子（本脚本默认参数：6 行 6 列）：**

```
┌────┬────┬────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │   ← 每个格子是一个 AprilTag
├────┼────┼────┼────┼────┼────┤     边长 tag_size = 24.5 mm
│ 6  │ 7  │ 8  │ 9  │ 10 │ 11 │     格子之间空隙 space_size = 7.5 mm
├────┼────┼────┼────┼────┼────┤     编号从 start_tag_id = 0 开始
│ 12 │ 13 │ 14 │ 15 │ 16 │ 17 │
├────┼────┼────┼────┼────┼────┤
│ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │
├────┼────┼────┼────┼────┼────┤
│ 24 │ 25 │ 26 │ 27 │ 28 │ 29 │
├────┼────┼────┼────┼────┼────┤
│ 30 │ 31 │ 32 │ 33 │ 34 │ 35 │
└────┴────┴────┴────┴────┴────┘
```

### 1.6 为什么是「至少 10 张图、每张至少 4 个点」

这是可以从"未知数个数 vs 方程个数"算出来的，不是拍脑袋：

- **未知数**：4 个内参 + 5 个畸变 = 9 个（全局），加上**每张图 6 个姿态参数**（3 个旋转 + 3 个平移）。
  设图片数为 $n$，则未知数总数 $= 9 + 6n$。

- **方程（约束）**：每个 3D-2D 点对提供 2 个方程（一个 $u$，一个 $v$）。
  设每张图有 $m$ 个点对，则方程总数 $= 2mn$。

- **可解条件**：方程数 ≥ 未知数

$$2mn \ge 9 + 6n$$

代入本脚本的最低要求（每张图 1 个 tag = 4 个角点，$m = 4$）：

$$8n \ge 9 + 6n \quad\Rightarrow\quad 2n \ge 9 \quad\Rightarrow\quad n \ge 4.5 \quad\Rightarrow\quad n \ge 5$$

所以**数学上 5 张图就够了**。那代码里为什么写 10 张？

因为：
1. 上面只是"必要条件"，实际求解还需要**姿态足够多样**。如果 5 张图都是在同一个角度拍的（标定板几乎共面且姿态相近），方程会"退化"——数学上叫秩亏，解出来是垃圾。
2. 检测有噪声，点多一些可以平均掉误差。
3. 工程经验：**10~20 张**、覆盖整个视野、有远有近、有正有斜，是最稳妥的。

> 📌 补充：注意代码第 74 行判断的是 `len(pts3d) < 4`，而 `pts3d` 存的是**角点**（每个 tag 贡献 4 个角点）。所以实际要求是"**每张图至少 1 个 tag**"，而函数文档字符串里写的"每张图像中至少检测到 4 个 tag2d"这个说法和代码不一致（代码是 4 个**点**不是 4 个 **tag**）。这是一个文档与实现的小偏差，见 [第 7 节](#7-这段代码里的坑与改进建议)。

---

## 2. 整体流水线

```
                    ┌─────────────────────────────────┐
                    │  命令行参数（argparse）           │
                    │  --calib_board_info  板子规格     │
                    │  --img_dir           照片目录     │
                    └───────────────┬─────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
   ┌────────────────────────┐              ┌────────────────────────────┐
   │ apriltag2.             │              │ glob.glob("目录/*.png")     │
   │ create_calib_board_3d  │              │ 找出所有 png，排序          │
   │ → tag3d_list           │              └───────────────┬────────────┘
   │   （板子上每个 tag 的   │                              │
   │     4 个角点的 3D 坐标）│                              ▼
   └───────────┬────────────┘              ┌────────────────────────────┐
               │                           │ 逐张读图（灰度）             │
               │                           │ detector.detect() 检测 tag  │
               │                           │ → tag2d_list                │
               │                           │   （照片上每个 tag 的        │
               │                           │     4 个角点的像素坐标）      │
               │                           └───────────────┬────────────┘
               │                                           │
               └──────────────┬────────────────────────────┘
                              ▼
              ┌───────────────────────────────────┐
              │  配对（靠 tag_id 认亲）             │
              │  3D 角点  ──配──  2D 角点          │
              │  pts3d_list / pts2d_list（每张图） │
              └───────────────┬───────────────────┘
                              ▼
              ┌───────────────────────────────────┐
              │  cv2.calibrateCamera()            │
              │  迭代优化，最小化重投影误差         │
              │  → K(3x3), D(5), rvecs, tvecs     │
              └───────────────┬───────────────────┘
                              ▼
              ┌───────────────────────────────────┐
              │  取 fx,fy,cx,cy + 5 个畸变系数     │
              │  写 cam_params.json（上级目录）     │
              └───────────────────────────────────┘
```

**数据结构流转表**（这个表很重要，能帮你理解后面每一层的循环）：

| 变量 | 类型 | 含义 | 形状/举例 |
|---|---|---|---|
| `tag3d_list` | `List[Tag3D]` | 标定板上 36 个 tag 的 3D 信息 | 长度 36，每个含 `id` 和 `corners`(4 个 3D 点) |
| `tag2d_list_list` | `List[List[Tag2D]]` | **外层 = 每张图**，内层 = 这张图检测到的 tag | `[ [tag2d, tag2d...], [tag2d...], ... ]` |
| `pts3d_list` | `List[np.ndarray]` | 每张图对应一组 3D 点 | 每个元素形状 `(N, 3)`，N = 4×tag 数 |
| `pts2d_list` | `List[np.ndarray]` | 每张图对应一组 2D 像素点 | 每个元素形状 `(N, 2)` |
| `K` | `np.ndarray` | 内参矩阵 | `(3, 3)` |
| `D` | `np.ndarray` | 畸变系数 | `(1, 5)` 或 `(5, 1)` |

> 注意命名里的 `list_list`（两层 list）和 `tag2d_list_list`（三层结构）——这是本脚本最容易绕晕的地方。**记住：多一层 list，就多一层"每张图"的循环。**

---

## 3. 逐段代码精读

下面按行号顺序讲。我会先把原代码贴出来，再逐句翻译。

### 3.1 文件顶部的说明字符串

```python
"""
文件说明:
    读取文件夹中的图像( 拍照了标定板 ), 标定相机( 仅适用于针孔相机模型 )
"""
```

**语法点**：三个引号 `"""` 包起来的是**文档字符串（docstring）**。写在文件最开头，就是对整个文件的说明。它不像 `#` 注释那样被完全忽略——Python 会把它存起来，用 `help()` 或 `__doc__` 能读到。写在这里相当于文件的"说明书封面"。

**业务含义**：这句说明点出了两个限制：
1. 输入是"一个文件夹里的多张照片"，不是单张。
2. **只适用于针孔相机模型**——鱼眼镜头不适用（鱼眼要用 `cv2.fisheye.calibrate` 那套，畸变模型完全不同）。如果你拿鱼眼相机跑这个脚本，RMS 会很大，结果不可用。

---

### 3.2 导入区（第 6–31 行）

```python
import argparse       # 第 6 行
import glob           # 第 7 行
import json           # 第 8 行
import logging        # 第 9 行
import os             # 第 10 行
import sys            # 第 11 行
from typing import Dict, List, Tuple   # 第 12 行

import apriltag2      # 第 14 行
import cv2            # 第 15 行
import numpy as np    # 第 16 行
```

**语法点 `import`**：把别人写好的代码"搬"进来用。

**六个标准库**（Python 自带的，无需安装）：

| 模块 | 在这里干什么 |
|---|---|
| `argparse` | 解析命令行参数，让你能写 `python calib_camera.py --img_dir xxx` |
| `glob` | 按通配符找文件，比如 `*.png` 找出所有 png |
| `json` | 读/写 JSON 格式文件 |
| `logging` | 打印带时间戳、级别（INFO/WARNING/ERROR）的日志 |
| `os` | 处理路径、文件名（`os.path.basename` 等） |
| `sys` | 操作 Python 解释器的搜索路径，`sys.exit()` 退出程序 |

**三个第三方库**（需要 `pip install`）：

| 模块 | 在这里干什么 |
|---|---|
| `apriltag2` | AprilTag 的检测器和标定板生成（**注意：这不是官方的 `apriltag` 包，是本工程自己/二次封装的库，在工程目录里没找到源码，属于外部依赖**） |
| `cv2` | OpenCV，计算机视觉万能库。这里的 `cv2.calibrateCamera`、`cv2.imread` 都来自它 |
| `numpy as np` | 数值计算。`as np` 是起别名，之后写 `np.array` 而不是 `numpy.array` |

**`from typing import Dict, List, Tuple`**（第 12 行）：
这三个是**类型标注**用的工具，只在函数签名里当"标签"用，不影响运行。

- `List[int]` = "一个列表，里面都是整数"
- `Tuple[int, int]` = "一个元组，两个整数"
- `Dict` = "字典"

比如 `img_size: Tuple[int, int]` 就是告诉读代码的人：`img_size` 应该传一个 `(宽, 高)` 这样的二元组。Python **不会**强制检查，写错了照样跑，但 IDE 和类型检查工具能帮你提前发现。

```python
# 导入本工程的模块
# 导入本工程的模块          ← 第 20 行：重复了一遍，注释冗余（小瑕疵）
code_dir = os.path.dirname(os.path.realpath(__file__))     # 第 21 行
root_dir = os.path.normpath(f"{code_dir}/../../../")       # 第 22 行
sys.path.append(root_dir)                                  # 第 23 行

from core.utils import (                                   # 第 25–31 行
    BLUE, GREEN, RED, RESET, YELLOW,
)
```

这三行是整个文件里最"工程味"的地方，值得仔细讲。

---

### 3.3 `sys.path` 那三行（第 21–23 行）

**问题背景**：Python 要 `import` 一个模块时，会去 `sys.path` 这个列表里的每个目录挨个找。本脚本想 `from core.utils import ...`，但 `core` 不在脚本所在目录下，而在工程的**根目录**下。

目录结构是这样的：

```
carm_grasp-main/                       ← 工程根目录（root_dir，我们要找的目标）
├── core/
│   └── utils.py                      ← 里面有颜色常量
└── examples/
    └── common/
        └── src/
            └── calib_camera.py       ← 本脚本（code_dir）
```

从 `src` 往上数三级才到根目录：`src → common → examples → 根目录`。所以是 `../../../`。

**逐行翻译：**

```python
code_dir = os.path.dirname(os.path.realpath(__file__))
```

- `__file__` 是 Python 自动定义的变量，值是**当前文件的路径**（可能是相对路径或符号链接）。
- `os.path.realpath(...)` 把它变成**绝对路径**，并解析掉符号链接（更可靠）。
- `os.path.dirname(...)` 去掉最后的文件名，只留**目录**。
- 结果：`code_dir = ".../carm_grasp-main/examples/common/src"`

```python
root_dir = os.path.normpath(f"{code_dir}/../../../")
```

- `f"{code_dir}/../../../"` 是 **f-string**（格式化字符串）：花括号里的变量会被替换成它的值。等价于老式写法 `code_dir + "/../../../"`，但更简洁。
- `os.path.normpath(...)` 把路径里的 `..` 消化掉，规范化成 `.../carm_grasp-main`。
- 结果：`root_dir = ".../carm_grasp-main"`

```python
sys.path.append(root_dir)
```

- 把工程根目录**临时**（只在本次运行期间）加到模块搜索路径里。
- 这样下一行 `from core.utils import ...` 就能找到了。

> 💡 **为什么不用 `sys.path.insert(0, ...)`？**
> `append` 放到列表末尾，`insert(0, ...)` 放到最前面（优先级更高）。放前面有个风险：如果你的工程里有 `core.py` 这种常见名字，可能会意外"抢先"覆盖掉同名的第三方库。这里用 `append` 更温和。

**导入了什么（第 25–31 行）：**

```python
from core.utils import (BLUE, GREEN, RED, RESET, YELLOW)
```

在 `core/utils.py` 里，这些常量是这样定义的：

```python
RED    = '\033[91m'   # 让终端之后的文字变红
GREEN  = '\033[92m'
YELLOW = '\033[93m'
BLUE   = '\033[94m'
RESET  = '\033[0m'    # 把颜色恢复默认
```

这些是 **ANSI 转义序列**（终端控制码）。用法是：`红色开始标记 + 文字 + 重置标记`。

```python
print(f"{RED}这行是红的{RESET} 这行恢复正常")
```

如果不加 `RESET`，后面所有输出都会一直是红色——很多新手会踩这个坑。

> ⚠️ Windows 老版 `cmd` 默认不支持 ANSI 颜色，会显示成一串乱码 `[91m`。Windows Terminal / PowerShell 7 / Linux / macOS 终端都正常。

---

### 3.4 主函数 `calib_camera`（第 36–112 行）

这是**真正干标定活**的函数。它不知道照片从哪来、结果存哪——它只负责"给我数据，我给你参数"，非常纯粹。

#### 3.4.1 函数签名（第 36–40 行）

```python
def calib_camera(
    tag3d_list: List[apriltag2.Tag3D],
    tag2d_list_list: List[List[apriltag2.Tag2D]],
    img_size: Tuple[int, int],
) -> Tuple[List[float], List[float]]:
```

**语法点 `def`**：定义一个函数。`def 函数名(参数1, 参数2, ...):`，冒号后面缩进的内容就是函数体。

**三个参数：**

| 参数 | 类型标注 | 人话 |
|---|---|---|
| `tag3d_list` | `List[Tag3D]` | 标定板上所有 tag 的 3D 角点信息（**所有图共用这一份**） |
| `tag2d_list_list` | `List[List[Tag2D]]` | 每张图检测到的 tag 列表的列表（**两层**） |
| `img_size` | `Tuple[int, int]` | 图像尺寸 `(宽, 高)`，单位像素 |

**返回值标注 `-> Tuple[List[float], List[float]]`**：表示返回一个二元组，两个元素都是"浮点数列​表"，即 `(内参, 畸变)`。

**业务含义：为什么 3D 只有一份、2D 是每个图一份？**

因为标定板是**刚体**，它上面第 17 号 tag 的左上角相对板子的位置是**永远固定**的（比如"距板子原点 32mm、75mm"），不随你从哪个角度拍而变化。而 2D 像素位置**每次拍摄都不一样**。所以 3D 数据只需一份，2D 数据每张图一份。

这就是"标定"的物理基础：**同一个物理点，在不同视角下投到不同的像素位置，用这些对应关系反推相机参数。**

#### 3.4.2 文档字符串（第 41–50 行）

标准的 Google 风格注释：说明功能、Args（参数）、Returns（返回值）。工程习惯很好，值得学习。

#### 3.4.3 外层循环：遍历每张图（第 52–83 行）

```python
pts3d_list = []      # 第 52 行：最终给 OpenCV 的 3D 点（每张图一组）
pts2d_list = []      # 第 53 行：最终给 OpenCV 的 2D 点（每张图一组）

for tag2d_list in tag2d_list_list:     # 第 54 行：每张图走一遍
    pts3d = []       # 第 55 行：这一张图的 3D 点
    pts2d = []       # 第 56 行：这一张图的 2D 点
```

**语法点**：`[]` 是**空列表**。`pts3d.append(x)` 往列表末尾加一个元素。

**为什么要两个层级？** 因为 OpenCV 的 `calibrateCamera` 要求：

```
objectPoints = [ 第1张图的点数组,  第2张图的点数组,  ... ]
imagePoints  = [ 第1张图的点数组,  第2张图的点数组,  ... ]
```

**两个列表必须一一对应**：`objectPoints[i][j]` 和 `imagePoints[i][j]` 必须是**同一个物理点**的 3D 位置和 2D 位置。如果错位了，标定结果就是垃圾。本脚本靠下面的 `tag_id` 配对来保证不错位。

#### 3.4.4 内层循环：遍历这张图的每个 tag（第 58–71 行）

```python
for tag2d in tag2d_list:                      # 第 58 行
    tag_id = tag2d.id                         # 第 59 行：这个 tag 的编号
    tag3d = next((t for t in tag3d_list if t.id == tag_id), None)   # 第 60 行
    if tag3d is None:                         # 第 61 行
        logging.warning(                      # 第 62–64 行
            f"{YELLOW}Tag ID {tag_id} detected in 2D but not found in 3D list. Skipping this tag.{RESET}"
        )
        continue                              # 第 65 行
```

**第 60 行是本文件语法密度最高的一行，拆开讲：**

```python
tag3d = next((t for t in tag3d_list if t.id == tag_id), None)
```

它由三块拼成：

**① 生成器表达式** `(t for t in tag3d_list if t.id == tag_id)`

翻译：*"从 tag3d_list 里，把 `id` 等于 `tag_id` 的那个元素挑出来。"*

等价于下面这段啰嗦写法：

```python
result = []
for t in tag3d_list:
    if t.id == tag_id:
        result.append(t)
```
区别在于生成器**不会一次性算出全部结果**，而是边要边给（惰性求值），所以更省内存。

**② `next(生成器, 默认值)`**

`next()` 的意思是"从生成器里取下一个元素"。带了第二个参数 `None` 之后，如果生成器**空了**（没找到匹配的），就返回 `None`，**而不是抛异常**。

所以这一整行 = **"在 3D 列表里找同 id 的 tag，找不到就给我 None"**。

**③ 对比：如果用列表推导式 `[...]` 会怎样？**

```python
# 写法 A（列表推导式）
matches = [t for t in tag3d_list if t.id == tag_id]   # 返回列表
tag3d = matches[0] if matches else None               # 还要再判空

# 写法 B（本脚本，next + 生成器）
tag3d = next((t for t in tag3d_list if t.id == tag_id), None)
```

B 更短，而且**找到第一个就停**，不用遍历完剩下的 35 个 tag，效率略高。

> ⚠️ **性能小评**：这里是**线性查找**，复杂度 O(36)×O(每张图的 tag 数)。数量小完全无所谓。但如果板子上有几百个 tag，应该改成字典：`tag3d_map = {t.id: t for t in tag3d_list}`，一次建表、之后 O(1) 查询。

**第 61–65 行的容错逻辑：**

```python
if tag3d is None:
    logging.warning(f"{YELLOW}...Skipping this tag.{RESET}")
    continue
```

- `is None`：**判空的标准写法**。注意不能用 `== None`，也不能用 `if not tag3d`（因为有些对象"falsy"但不是 None）。
- `logging.warning(...)`：打印黄色警告，但**程序继续跑**。
- `continue`：**跳过本次循环的剩余部分，直接进入下一次循环**。这里就是"这个 tag 不认识，跳过它，处理下一个 tag"。

**为什么会出现"2D 检测到了但 3D 里没有"？** 常见原因：
1. 拍到了板子之外的另一个 AprilTag（比如旁边墙上贴着一个测试 tag）。
2. 你 `--calib_board_info` 填的行列数比实际板子小，导致生成的 3D 板子只有 25 个 tag，而照片里出现了第 30 号。
3. 误检（光照反光造成假阳性）。

这个 `continue` 是很稳的防御性编程。

**第 68–71 行：取 4 个角点**

```python
for i in range(4):
    pts3d.append(tag3d.corners[i])
    pts2d.append(tag2d.corners[i])
```

- `range(4)` 产生 `0, 1, 2, 3` 四个数。
- `tag3d.corners[i]` 是第 i 个角点的 3D 坐标（形如 `[x, y, z]`，单位米）。
- `tag2d.corners[i]` 是第 i 个角点的像素坐标（形如 `[u, v]`）。

**关键点：`corners[0]` 在两个列表里必须是同一个角**（比如都是"左上角"）。AprilTag 库保证了角点顺序一致（一般是左上→右上→右下→左下，逆时针）。如果顺序不一致，标定直接废掉——这是用第三方库必须信任的约定。

**第 74–79 行：这张图够不够格？**

```python
if len(pts3d) < 4:
    logging.warning(f"{YELLOW}Only {len(pts3d)} valid 3D-2D point pairs found... Skipping this image.{RESET}")
    continue
```

- `len(pts3d)` 数一数有几个点。
- 少于 4 个点 → 这张图信息量不够，整张图丢弃（`continue` 跳到下一张）。
- 注意：**4 个点 = 1 个 tag**（因为每个 tag 贡献 4 个角点）。

> 这里就体现了前面提到的"文档说 4 个 tag，代码实际是 4 个点"的不一致。从数学上讲，1 个 tag（4 个点）**确实**可以参与标定（见 [1.6](#16-为什么是至少-10-张图每张至少-4-个点) 的计算），所以代码是对的，是注释写错了。

**第 81–82 行：转成 numpy 数组并收进大列表**

```python
pts3d_list.append(np.array(pts3d))
pts2d_list.append(np.array(pts2d))
```

- `np.array(...)` 把 Python 的普通列表转成 numpy 多维数组。转换后 `pts3d` 形状是 `(N, 3)`（N 行，每行 x,y,z），`pts2d` 是 `(N, 2)`。
- 为什么必须转？因为 OpenCV 的 Python 接口**只认 numpy 数组**（底层是 C++，需要连续内存）。传普通 list 会报错。

#### 3.4.5 图片数量检查（第 85–90 行）

```python
if len(pts3d_list) < 10:
    logging.error(f"{RED}Only {len(pts3d_list)} valid images ... At least 10 are required ... Exiting.{RESET}")
    return None, None
```

- `len(pts3d_list)` = 有多少张图**合格**（即有足够的点）。
- 少于 10 张 → 打印**红色错误**，返回 `(None, None)`。
- `return None, None` 实际上是返回一个**元组** `(None, None)`，Python 会自动打包。调用方用 `intrinsic, distortion = ...` 接收，两个变量都变成 `None`。

**注意**：这里用的是 `logging.error`（红色）而不是 `logging.warning`（黄色）。语义区别：
- warning = "有点问题，但我还能继续"
- error = "出大事了，这部分活干不了了"

而这里函数内**没有** `sys.exit()`，只是返回 None，把"要不要退出"的决定权交给调用者（第 208–211 行处理）。这是很好的分层设计：**工具函数不该擅自终止整个程序**。

#### 3.4.6 初始化 K 和 D（第 92–93 行）

```python
K = np.eye(3)        # 3x3 单位矩阵
D = np.zeros(5)      # 长度 5 的全零向量
```

**语法点：**
- `np.eye(3)` 生成 3×3 单位矩阵（对角线上是 1，其余是 0）：
  ```
  [[1, 0, 0],
   [0, 1, 0],
   [0, 0, 1]]
  ```
- `np.zeros(5)` 生成 `[0., 0., 0., 0., 0.]`

**业务含义**：这两个是传给 `calibrateCamera` 的**初始猜测值**。为什么要给初值？因为标定是一个**迭代优化**过程，需要一个起点。

不过这里给的是"单位矩阵 + 全零"，其实是个很差的起点（相当于猜 `fx=1, cx=0`）。OpenCV 通常会自己用张正友法先算一个初值，所以这个输入基本被忽略——**这更多是满足函数签名要求**。真要严谨，可以这样给一个合理初值：

```python
K = np.array([[img_size[0], 0, img_size[0]/2],
              [0, img_size[1], img_size[1]/2],
              [0, 0, 1]], dtype=np.float64)
```
（猜焦距 ≈ 图片宽度，主点在正中心）

#### 3.4.7 调用标定核心（第 96–102 行）

```python
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=pts3d_list,
    imagePoints=pts2d_list,
    imageSize=img_size,
    cameraMatrix=K,
    distCoeffs=D,
)
```

**这 7 行是整个脚本的灵魂。** 展开讲：

**五个输入：**

| 参数名 | 传进去的 | 含义 |
|---|---|---|
| `objectPoints` | `pts3d_list` | 每张图的 3D 点（世界坐标，单位米） |
| `imagePoints` | `pts2d_list` | 每张图对应的 2D 像素点 |
| `imageSize` | `img_size` | `(宽, 高)`，**OpenCV 的顺序是 (width, height)**，和 numpy 的 `(行, 列)` 相反，这是经典坑 |
| `cameraMatrix` | `K` | 内参初值（会被覆盖） |
| `distCoeffs` | `D` | 畸变初值（会被覆盖） |

**五个返回值：**

| 返回值 | 含义 | 本脚本用了吗 |
|---|---|---|
| `ret` | **RMS 重投影误差**（像素），衡量标定好坏 | ✅ 打印出来 |
| `K` | 标定后的 3×3 内参矩阵 | ✅ 取出 4 个数 |
| `D` | 标定后的畸变系数（5 个） | ✅ 全部取出 |
| `rvecs` | **每张图**的旋转向量（罗德里格斯向量，3 个数） | ❌ 丢弃 |
| `tvecs` | **每张图**的平移向量（3 个数，单位同 3D 点，即米） | ❌ 丢弃 |

> `rvecs/tvecs` 描述的是"拍照那一刻，标定板相对相机在哪、什么姿态"。**手眼标定**（eye-in-hand / eye-to-hand）就非常需要它，但本脚本只做内参标定，所以用 `_` 忽略掉也行：
> ```python
> ret, K, D, _, _ = cv2.calibrateCamera(...)
> ```

**OpenCV 内部干了什么（简化版）：**

1. 对每张图，用一个 3×3 的**单应性矩阵 H** 描述"标定板平面 → 图像平面"的映射（因为标定板是平的，所有 3D 点 z=0，透视退化为单应性）。
2. 从多个 H 里解出内参 K 的**闭式解**（张正友法的经典步骤）。
3. 用这个闭式解当起点，把所有参数（内参 + 畸变 + 所有图的姿态）放一起，用 **Levenberg-Marquardt 算法**迭代优化，最小化所有点的重投影误差平方和。
4. 返回最优解和最终 RMS。

#### 3.4.8 提取结果并打印（第 104–112 行）

```python
intrinsic = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]   # 第 104 行：fx, fy, cx, cy
distortion = D.tolist()                            # 第 105 行：k1, k2, p1, p2, k3
```

**为什么只取 4 个数？** 因为 K 是 3×3 共 9 个数，但有用的只有 4 个：

```
K = [[fx,  0, cx],       位置：K[0,0]=fx   K[0,2]=cx
     [ 0, fy, cy],              K[1,1]=fy   K[1,2]=cy
     [ 0,  0,  1]]              K[2,2]=1（永远是 1，不存）
     ↑
  第 0 行
```

**语法点：`K[0, 0]`** —— numpy 的二维索引用逗号分隔，`K[行, 列]`，从 0 开始数。对比普通 Python 二维列表要写 `lst[0][0]`。

> 严格说，如果相机有**倾斜**（skew），`K[0,1]` 不为 0。OpenCV 默认不加这个自由度，所以忽略它是对的。

**语法点：`D.tolist()`** —— 把 numpy 数组转成 Python 原生的 list。为什么必须转？因为 **JSON 不认识 numpy 的数据类型**，直接 `json.dump` 一个 `np.float64` 会报 `TypeError: Object of type float32 is not JSON serializable`。

> ⚠️ **潜在问题**：`D` 从 `calibrateCamera` 出来后形状通常是 `(1, 5)`（二维），那么 `D.tolist()` 会得到**嵌套列表** `[[k1,k2,p1,p2,k3]]`，写进 JSON 就变成 `"distortion": [[0.1, -0.2, ...]]`，与声明的 `"DistortionFormat": "k1,k2,p1,p2,k3"`（一维）不符。稳妥写法：
> ```python
> distortion = D.ravel().tolist()   # 或 D.flatten().tolist()
> ```
> 详见 [第 7 节](#7-这段代码里的坑与改进建议)。

```python
logging.info(f"Camera calibration RMS error: {ret}")            # 第 107 行
logging.info(f"Camera matrix: {GREEN}{intrinsic}{RESET}")       # 第 108 行
logging.info(f"Distortion coefficients: {GREEN}{distortion}{RESET}")  # 第 109 行
print()                                                          # 第 110 行：打印空行

return intrinsic, distortion                                     # 第 112 行
```

- `logging.info` 打印普通信息。输出格式由 `core/utils.py` 里配置的 `basicConfig` 决定，形如：
  ```
  [09-19 15:30:27.123][INFO][calib_camera.py:107] Camera calibration RMS error: 0.412
  ```
  带时间戳、级别、文件名、行号——比 `print` 好用得多，排查问题时很有价值。
- `print()` 空括号 = 打印一个空行，纯粹为了终端输出好看（分组）。
- 第 112 行返回二元组。

---

### 3.5 `if __name__ == "__main__"`（第 121–230 行）

```python
if __name__ == "__main__":     # 第 121 行
```

**这是 Python 最经典也最容易困惑的一行，务必理解：**

- 每个 Python 文件（`模块`）都有一个内置变量 `__name__`。
- 如果你**直接运行**这个文件（`python calib_camera.py`），Python 会把它的 `__name__` 设成字符串 `"__main__"`。
- 如果你**从别的文件 import 它**（`import calib_camera`），`__name__` 会被设成模块名 `"calib_camera"`。

所以这一行的意思是：

> **"只有当这个文件被当作脚本直接运行时，才执行下面的代码。如果被别人 import，就只提供 `calib_camera()` 函数，不要自动跑起来。"**

好处：本文件既能当命令行工具用，又能当库被复用（比如其他脚本 `from calib_camera import calib_camera` 调用它），而 import 时不会意外地开始标定、读文件、写文件。

#### 3.5.1 argparse：接收命令行参数（第 122–141 行）

```python
parser = argparse.ArgumentParser(description="相机标定")     # 第 122 行

parser.add_argument(                                       # 第 124–129 行
    "--calib_board_info",
    type=str,
    default="[0.0245,0.0075, 6, 6]",
    help="标定板信息 [tag_size, space_size, tag_rows, tag_cols]",
)

parser.add_argument(                                       # 第 131–136 行
    "--img_dir",
    type=str,
    default="/home/i4/桌面/test_location/calib/collect_image/cam0",
    help="保存图像的目录",
)

args = parser.parse_args()                                 # 第 138 行
```

**语法点 `argparse`**：标准库的命令行参数解析器。三步走：

1. `ArgumentParser(...)` 造一个解析器
2. `add_argument(...)` 声明"我接受哪些参数"
3. `parse_args()` 真正去读命令行，把结果放进 `args`

**`add_argument` 的参数：**

| 参数 | 作用 |
|---|---|
| `"--calib_board_info"` | 参数名。命令行里写 `--calib_board_info "[0.0245,0.0075,6,6]"` |
| `type=str` | 收到的值当字符串处理（之后自己 `json.loads` 解析） |
| `default=...` | 命令行不写这个参数时用哪个默认值 |
| `help=...` | 帮助文字，运行 `python calib_camera.py --help` 时显示 |

**使用方式：**

```bash
# 用默认值
python calib_camera.py

# 自定义
python calib_camera.py --img_dir "D:/data/calib/cam0" --calib_board_info "[0.0245,0.0075,6,6]"
```

**两个默认值的含义：**

```python
default="[0.0245,0.0075, 6, 6]"
```

| 位置 | 变量名 | 值 | 含义 |
|---|---|---|---|
| `[0]` | `tag_size` | `0.0245` | 每个 AprilTag 的边长 = **0.0245 米 = 24.5 毫米** |
| `[1]` | `space_size` | `0.0075` | tag 之间的间距 = **7.5 毫米** |
| `[2]` | `rows` | `6` | 6 行 |
| `[3]` | `cols` | `6` | 6 列 |

```python
default="/home/i4/桌面/test_location/calib/collect_image/cam0"
```

这是作者自己机器上的路径（Linux，而且目录名是中文"桌面"）。**你运行时必须换成自己的路径**，否则会报 `No images found`。

```python
calib_board_info = json.loads(args.calib_board_info)   # 第 140 行
img_dir = args.img_dir                                  # 第 141 行
```

**语法点 `json.loads`**：`loads` = **load s**tring，把 JSON 格式的**字符串**解析成 Python 对象。

```python
json.loads("[0.0245,0.0075, 6, 6]")
# → [0.0245, 0.0075, 6, 6]   （Python 列表，元素是 float 和 int）
json.loads('{"a": 1}')
# → {'a': 1}                 （Python 字典）
```

**为什么用 JSON 传数组而不是直接传数字？** 因为一个参数要塞 4 个值。用 JSON 字符串是最省事的做法，用户写的是标准格式，Python 一行就能解析。

> 注意别写错：`json.loads` 处理字符串，`json.load` 处理**文件对象**，两者差一个 `s`。

```python
print()                                                        # 第 143 行
print(f"标定板信息: {BLUE}{calib_board_info}{RESET}")           # 第 144 行
print(f"图像目录: {BLUE}{img_dir}{RESET}")                      # 第 145 行
print()                                                        # 第 146 行
```

把用户传入的参数**回显**出来（蓝色）。这是非常好的调试习惯：**标定结果不对时，第一个要怀疑的就是"参数是不是传错了"**，回显能让你一眼看到脚本实际用的是什么值。

#### 3.5.2 创建 3D 标定板（第 149–155 行）

```python
tag3d_list = apriltag2.create_calib_board_3d(
    tag_size=calib_board_info[0],      # 0.0245
    space_size=calib_board_info[1],    # 0.0075
    rows=calib_board_info[2],          # 6
    cols=calib_board_info[3],          # 6
    start_tag_id=0,                    # 编号从 0 开始
)
```

**业务含义**：**用数学方法"造"出一块虚拟标定板**，算出板子上每个 tag 的 4 个角点在板子坐标系里的 3D 坐标（单位：米）。

它内部大概在做这样的事（简化示意）：

```python
for r in range(rows):          # 行 0~5
    for c in range(cols):      # 列 0~5
        tag_id = start_tag_id + r * cols + c        # 编号 = r*6 + c
        # 这个 tag 左下角（或左上角）在板子坐标系里的位置：
        x0 = c * (tag_size + space_size)
        y0 = r * (tag_size + space_size)
        # 4 个角点：
        corners = [
            [x0,               y0              , 0],   # 角点 0
            [x0 + tag_size,    y0              , 0],   # 角点 1
            [x0 + tag_size,    y0 + tag_size   , 0],   # 角点 2
            [x0,               y0 + tag_size   , 0],   # 角点 3
        ]
        tag3d_list.append(Tag3D(id=tag_id, corners=corners))
```

**三点必须理解：**

1. **所有 z = 0**。标定板是个平面，所以所有点都在 z=0 的平面上。这是张正友标定法的前提（平面标定板）。
2. **坐标系原点**在板子的某个角（取决于库的实现，一般是左上角或左下角），x 轴沿列方向，y 轴沿行方向。具体原点在哪**不影响标定结果**——因为外参 `rvecs/tvecs` 会吸收掉这个差异。
3. **单位必须和真实世界一致**。这里 `0.0245` 表示米，那你用尺子量板子也必须量出 24.5mm。如果实际是 24.5cm 而你写 0.0245，标定出来的 `fx` 会是对的（因为形状自洽）但后续算距离会全错 10 倍。**板子尺寸填错是最常见的标定事故。**

**⚠️ 一个隐藏假设**：`start_tag_id=0` 是**写死**的。如果你的标定板实际是从别的编号开始（比如 `tag36h11` 的第 100 号起），所有 3D-2D 配对都会失败，日志里会刷一大堆 "Tag ID xxx detected in 2D but not found in 3D list"。建议把它也做成命令行参数。

#### 3.5.3 创建检测器（第 158–161 行）

```python
detector = apriltag2.Detector(
    tag_family="tag36h11",
    black_border=2,
)
```

| 参数 | 含义 |
|---|---|
| `tag_family="tag36h11"` | 指定识别哪个家族的 tag。**必须和你实际打印的板子一致**，否则一个都认不出来。常见家族：`tag36h11`（默认，最常用）、`tag25h9`、`tag16h5`、`tagStandard41h12` |
| `black_border=2` | tag 图案外围黑边框的宽度（单位：格子数）。AprilTag 的标准图案外面有一圈黑边，检测器需要知道它有多宽才能正确定位 |

**为什么要先"创建"检测器对象？** 因为检测器内部要预先生成/加载这个家族所有 tag 的模板（可能几百个），这是个耗时操作。**创建一次、复用多次**，比每次检测都重建快得多。所以它被放在循环外面——这是正确的写法。

#### 3.5.4 找图片（第 164–169 行）

```python
img_path_list = glob.glob(f"{img_dir}/*.png")   # 第 164 行
img_path_list.sort()                            # 第 165 行
if len(img_path_list) == 0:                     # 第 166 行
    logging.error(f"{RED}No images found in {img_dir}. Exiting.{RESET}")
    sys.exit(1)                                 # 第 168 行
```

**`glob.glob("目录/*.png")`**：`*` 是通配符，匹配任意字符。所以 `*.png` 匹配所有 png 文件，返回一个**路径字符串的列表**。

```python
glob.glob("D:/data/cam0/*.png")
# → ['D:/data/cam0/0001.png', 'D:/data/cam0/0002.png', ...]
```

**`.sort()`**：按字符串排序。**为什么要排序？** 让处理顺序确定。虽然标定结果与顺序无关，但有序能让每次运行的行为一致，日志也更好读（排查问题时你能知道"卡在第几张"）。

**`sys.exit(1)`**：立即终止程序。括号里的数字是**退出码**：
- `0` = 正常结束
- 非 `0` = 异常结束（这里用 `1`）

退出码是给**调用方**（比如 shell 脚本、CI）看的：`python calib_camera.py && echo "成功"`，失败时 `&&` 后面就不执行了。

> ⚠️ **只找 `.png`**。如果你的照片是 `.jpg`，这个脚本一个都找不到。要么转格式，要么改成：
> ```python
> img_path_list = sorted(glob.glob(f"{img_dir}/*.png") + glob.glob(f"{img_dir}/*.jpg"))
> ```

#### 3.5.5 逐张检测（第 171–201 行）

```python
tag2d_list_list = []                          # 第 171 行：装所有图的检测结果

for img_path in img_path_list:                # 第 173 行
    img_name = os.path.basename(img_path)     # 第 174 行
    img_id = os.path.splitext(img_name)[0]    # 第 175 行

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)   # 第 177 行
    if img is None:                           # 第 178 行
        logging.warning(f"{YELLOW}Failed to read image {img_name}, skipping.{RESET}")
        continue                              # 第 182 行

    tag2d_list = detector.detect(img, -1)     # 第 185 行
    if len(tag2d_list) == 0:                  # 第 186 行
        logging.warning(f"{YELLOW}No tags detected in image {img_name}, skipping.{RESET}")
        continue                              # 第 190 行

    # （第 193–198 行是被注释掉的可视化调试代码，见下方说明）

    tag2d_list_list.append(tag2d_list)        # 第 200 行
```

**第 174–175 行：文件名处理**

```python
os.path.basename("D:/data/cam0/0001.png")     # → "0001.png"
os.path.splitext("0001.png")                  # → ("0001", ".png")
os.path.splitext("0001.png")[0]               # → "0001"
```

- `os.path.basename`：去掉目录，只留文件名。
- `os.path.splitext`：把文件名拆成 `(主名, 扩展名)`，取 `[0]` 就是不带后缀的名字。

> 这两行算出来的 `img_id` **实际上没被使用**（`img_name` 用在日志里了，`img_id` 是"死代码"）。可能是作者打算以后用它做日志编号。无害，但属于冗余。

**第 177 行：读图**

```python
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
```

- `cv2.imread` 读图片，返回一个 numpy 数组。
- 第二个参数 `cv2.IMREAD_GRAYSCALE`：直接读成**灰度图**（单通道）。
  - 灰度图：形状 `(高, 宽)`，每个像素一个 0~255 的值
  - 彩色图：形状 `(高, 宽, 3)`，每个像素 B, G, R 三个值

**为什么用灰度？** AprilTag 本来就是黑白图案，**颜色信息毫无用处**，灰度图只有 1/3 的数据量，检测更快。

**第 178–182 行：读图失败的容错**

```python
if img is None:
    logging.warning(...)
    continue
```

**重要**：`cv2.imread` 读不到图时**不会报错**，而是返回 `None`。这是 OpenCV 的经典陷阱——很多人忘了判空，后面一用 `img.shape` 就崩。这里判空是对的。

读图失败的常见原因：路径有中文/空格、文件损坏、权限不足、路径分隔符写错（Windows 要用 `/` 或 `\\`）。

**第 185 行：检测**

```python
tag2d_list = detector.detect(img, -1)
```

在灰度图里找 AprilTag，返回检测到的 tag 列表，每个 tag 含：
- `.id`：tag 编号
- `.corners`：4 个角点的**亚像素级**像素坐标（浮点，比如 `[321.47, 188.62]`）

第二个参数 `-1`（推测）：限制最多检测多少个 tag，`-1` 表示**不限制**。不同 AprilTag 封装里这个参数含义可能是"最多检测数"或"线程数"，需要查 `apriltag2` 的文档确认（该库在本工程目录内**未找到源码**，是外部依赖）。

**"亚像素"是什么意思？** 普通检测只能确定角点在整数像素位置（比如第 321 列）。亚像素算法利用灰度梯度，能定位到 `321.47`——**精度比一个像素还细**。标定对精度极其敏感，亚像素是标定精度高的关键原因之一。

**第 186–190 行：没检测到就跳过**

如果这张图一个 tag 都没认出来（太糊、太暗、太远、板子出画了），就跳过这张图。

**第 193–198 行：被注释掉的调试代码**

```python
# if True:
#     bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     detector.draw(bgr_img, tag2d_list)
#     cv2.imshow("tag detection", bgr_img)
#     cv2.waitKey(500)
# # end if
```

这是一段**可视化调试开关**，被注释掉了。如果打开：

| 行 | 作用 |
|---|---|
| `cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)` | 灰度图转成 3 通道 BGR 图（因为要画彩色框） |
| `detector.draw(...)` | 在图上画出检测到的 tag 框和 ID |
| `cv2.imshow(...)` | 弹出窗口显示 |
| `cv2.waitKey(500)` | 等待 500 毫秒（0.5 秒），让窗口有时间刷新；无限等待的话要手动按键 |

> 💡 **`if True:` 是一个很实用的调试技巧**：比 `if False:` 只改一个单词就能开关整段代码，比反复注释/取消注释方便。
> 想看检测效果时，把这几行取消注释即可——**排查"为什么标定结果差"时，这一步极其有价值**：你能直观看到 tag 有没有认全、角点画得准不准。

**第 200 行：收集结果**

```python
tag2d_list_list.append(tag2d_list)
```

把这张图的 tag 列表，追加到"所有图"的大列表里。**注意跳过的图不会产生元素**，所以 `tag2d_list_list` 的长度 ≤ 图片总数。

#### 3.5.6 执行标定（第 204–211 行）

```python
img_size = (img.shape[1], img.shape[0])       # 第 204 行
intrinsic, distortion = calib_camera(         # 第 205–207 行
    tag3d_list=tag3d_list,
    tag2d_list_list=tag2d_list_list,
    img_size=img_size,
)
if intrinsic is None or distortion is None:   # 第 208 行
    logging.error(f"{RED}Camera calibration failed. Exiting.{RESET}")
    sys.exit(1)                               # 第 210 行
```

**第 204 行是 OpenCV 的经典坑，重点讲：**

```python
img = cv2.imread(...)      # img 的形状是 (高, 宽) 或 (高, 宽, 通道)
img.shape[0]               # 高（行数 = 纵向像素数）
img.shape[1]               # 宽（列数 = 横向像素数）
```

而 OpenCV 的 `imageSize` 参数要的是 **`(宽, 高)`**！

所以必须是：

```python
img_size = (img.shape[1], img.shape[0])   # ✅ 正确：(宽, 高)
# 而不是
img_size = (img.shape[0], img.shape[1])   # ❌ 错误：(高, 宽) → cx 和 cy 会互换
```

**如果写反了会怎样？** 程序**不报错**，但标定出的 `cx` 和 `cy` 会互换（比如应该是 `(640, 360)` 却变成 `(360, 640)`）。这种错非常隐蔽——RMS 可能看起来还凑合，但后续用这个内参做测量会系统性偏移。**记住：numpy 是 (行,列)，OpenCV 尺寸是 (宽,高)，永远差一个顺序。**

> ⚠️ **这里有个真实隐患**：`img` 是循环里的变量，循环结束后它保留**最后一次成功读图**的值。如果**第一张图就读失败**（`continue` 了），到第 204 行时 `img` 根本没定义过，会抛 `NameError: name 'img' is not defined`。虽然概率低（第一张图一般能读到），但建议在循环外先初始化 `img_size = None`，或在第一张成功读图时就记录尺寸。

**第 205–207 行：调用函数**

注意这里用了**关键字参数**（`tag3d_list=...`）。写成 `calib_camera(tag3d_list, tag2d_list_list, img_size)` 效果一样，但关键字参数**读起来更清楚**，也不怕参数顺序记错。参数多的时候强烈推荐。

**第 208–211 行：检查失败**

`or` 是逻辑"或"。只要有一个是 `None` 就认为失败，打印红色错误并 `sys.exit(1)`。

#### 3.5.7 保存结果（第 214–228 行）

```python
save_path = os.path.join(os.path.dirname(img_dir), "cam_params.json")   # 第 214 行
result_dict = {                                                          # 第 215–222 行
    "camera_type": "Pinhole",
    "IntrinsicFormat": "fx,fy,cx,cy",
    "DistortionFormat": "k1,k2,p1,p2,k3",
    "resolution": [img_size[0], img_size[1]],
    "intrinsic": intrinsic,
    "distortion": distortion,
}

with open(save_path, "w") as f:       # 第 224 行
    json.dump(result_dict, f, indent=4)   # 第 225 行

logging.info(f"Result saved to: {GREEN}{save_path}.{RESET}")   # 第 228 行
```

**第 214 行：保存路径怎么算的？**

```python
os.path.dirname("D:/data/calib/collect_image/cam0")    # → "D:/data/calib/collect_image"
os.path.join("D:/data/calib/collect_image", "cam_params.json")
# → "D:/data/calib/collect_image/cam_params.json"
```

即：**图片目录的上一级目录**下，文件名 `cam_params.json`。

目录结构：
```
.../collect_image/
    ├── cam0/              ← --img_dir 指向这里（照片）
    │   ├── 0001.png
    │   └── 0002.png
    └── cam_params.json    ← 结果存在这里
```

**为什么存在上一级？** 因为 `cam0`、`cam1`、`cam2` 可能是多个相机各自一个子目录，把参数放上级目录，结构清晰。

> ⚠️ **Windows 路径陷阱**：如果 `--img_dir` 写成了带尾斜杠的形式 `"D:/data/cam0/"`，`os.path.dirname` 会返回 `"D:/data/cam0"` 而不是 `"D:/data"`，结果就会**存进 cam0 里面**。建议先 `img_dir = os.path.normpath(img_dir)` 规范化一下。

**第 215–222 行：结果字典**

```python
result_dict = {
    "camera_type": "Pinhole",                        # 相机模型：针孔
    "IntrinsicFormat": "fx,fy,cx,cy",                # 说明 intrinsic 数组里每个位置是什么
    "DistortionFormat": "k1,k2,p1,p2,k3",            # 说明 distortion 数组里每个位置是什么
    "resolution": [img_size[0], img_size[1]],        # 分辨率 [宽, 高]
    "intrinsic": intrinsic,                          # [fx, fy, cx, cy]
    "distortion": distortion,                        # [k1, k2, p1, p2, k3]
}
```

**语法点：字典（dict）** —— 键值对，`{键: 值, 键: 值}`，用 `dict["键"]` 取值。会原样变成 JSON 对象。

> 👍 **`IntrinsicFormat` / `DistortionFormat` 这两个字段写得非常好**。它们的作用是**自描述**：半年后你打开这个 JSON，不用去翻代码就知道 `[912.3, 911.8, 638.2, 361.5]` 里哪个是 fx。很多标定脚本只存一串裸数字，时间一长就分不清顺序了。**这是值得借鉴的好习惯。**

**第 224–226 行：写文件**

```python
with open(save_path, "w") as f:
    json.dump(result_dict, f, indent=4)
```

**语法点：`with` 语句（上下文管理器）**

`with` 的作用是：**保证文件一定会被关闭**，哪怕中间出异常。

```python
# ❌ 不推荐写法
f = open(save_path, "w")
json.dump(result_dict, f)
f.close()          # 如果上一行抛异常，这行执行不到 → 文件句柄泄漏

# ✅ 推荐写法（本脚本）
with open(save_path, "w") as f:
    json.dump(result_dict, f, indent=4)
# 出了缩进范围，Python 自动帮你 close()
```

`open()` 的参数：
- 第一个：文件路径
- `"w"` = **写入模式**（write）。文件不存在就创建，**存在就清空覆盖**。其他模式：`"r"` 读、`"a"` 追加、`"rb"` 二进制读。
- 建议加 `encoding="utf-8"`，Windows 上默认编码可能是 GBK，虽然这里全是 ASCII 无影响，但是好习惯。

`json.dump()` 的参数：
- 第一个：要写的 Python 对象
- 第二个：文件对象
- `indent=4`：缩进 4 个空格，让 JSON **格式化输出**，人类可读。不加的话会写成一整行。

> 注意又是 `dump` vs `dumps`：`dump` 写**文件**，`dumps` 生成**字符串**。和前面的 `load`(文件) / `loads`(字符串) 是同一套规律。

**第 226 行的小瑕疵**：`# end if` 这个闭合注释写错了，应该是 `# end with`（因为它闭合的是 `with` 语句不是 `if`）。无害，但会误导读者。

**第 228 行**：打印绿色成功信息，告诉你文件存哪儿了。

---

## 4. Python 基础语法速查

把本文件用到的语法集中罗列，方便随时回查。

### 4.1 变量与基本类型

```python
x = 10              # 整数 int
y = 0.0245          # 浮点数 float
s = "hello"         # 字符串 str
b = True            # 布尔 bool（注意首字母大写！）
n = None            # 空值 NoneType

lst = [1, 2, 3]     # 列表 list，可增删改
lst.append(4)       # → [1,2,3,4]
len(lst)            # → 4
lst[0]              # → 1（下标从 0 开始！）
lst[-1]             # → 4（负数表示倒数）

tup = (640, 360)    # 元组 tuple，一旦创建不能改
w, h = tup          # 元组解包：w=640, h=360

d = {"a": 1}        # 字典 dict
d["a"]              # → 1
d.get("b", 0)       # → 0（键不存在时返回默认值，不会报错）
```

### 4.2 控制流

```python
# 条件
if 条件:
    ...
elif 另一个条件:
    ...
else:
    ...

# 循环
for 元素 in 列表:
    ...

for i in range(4):        # 0,1,2,3
    ...

while 条件:
    ...

# 循环控制
break       # 立刻跳出整个循环
continue    # 跳过本次，进入下一次

# 布尔运算
a and b     # 都真才真
a or b      # 有一个真就真
not a       # 取反
a is None   # 判断是不是 None（推荐写法）
```

### 4.3 缩进 = 语法

**Python 用缩进（空格）表示代码块，不用 `{}`。** 这是最大的特色：

```python
if True:
    print("属于 if 块")
    print("也属于 if 块")
print("不属于 if 块了")
```

缩进必须**一致**（一般 4 个空格，别混用 Tab）。缩进错了 = 语法错误，或者更糟——逻辑悄悄错了。

### 4.4 f-string（格式化字符串）

```python
name = "相机"
fx = 912.3

f"这是{name}，焦距{fx}"                  # → "这是相机，焦距912.3"
f"{fx:.2f}"                              # → "912.30"（保留 2 位小数）
f"{YELLOW}警告{RESET}"                   # 花括号里可以是变量、表达式
f"{img.shape[1]}x{img.shape[0]}"         # 甚至可以是索引、函数调用
```

比老的写法简洁很多：
```python
"这是%s，焦距%f" % (name, fx)      # 老式
"这是{}，焦距{}".format(name, fx)  # 中庸
f"这是{name}，焦距{fx}"            # 推荐
```

### 4.5 推导式与生成器

```python
# 列表推导式：结果是一个列表
[x * 2 for x in range(5)]                  # → [0, 2, 4, 6, 8]
[x for x in range(10) if x % 2 == 0]       # → [0, 2, 4, 6, 8]（带过滤）
{t.id: t for t in tag3d_list}              # 字典推导式

# 生成器表达式：用圆括号，惰性求值（要一个给一个）
(x * 2 for x in range(5))                  # → 生成器对象
next(上面的生成器, 默认值)                  # 取下一个，取不到返回默认值
```

**本文件第 60 行的拆解：**
```python
next((t for t in tag3d_list if t.id == tag_id), None)
#    ↑生成器：找出 id 匹配的 tag        ↑找不到就返回 None
```

### 4.6 函数

```python
def 函数名(参数1, 参数2=默认值):
    """文档字符串"""
    ...
    return 返回值          # 不写 return 就返回 None

# 调用
函数名(1, 2)                        # 位置参数
函数名(参数1=1, 参数2=2)             # 关键字参数（推荐，可读性好）

# 返回多个值（本质是返回一个元组，然后自动解包）
def f():
    return 1, 2
a, b = f()      # a=1, b=2
```

### 4.7 类型标注（type hints）

```python
def f(x: int, y: str) -> bool:
    ...
```

- **只起说明作用，运行时不检查、不报错**。
- 好处：IDE 能自动补全、帮你发现传错类型；`mypy` 这类工具能静态检查。
- 复杂类型用 `typing` 模块：`List[int]`、`Dict[str, float]`、`Tuple[int, int]`、`Optional[str]`。

### 4.8 模块与 `if __name__ == "__main__"`

```python
import os                    # 导入整个模块，用 os.path.xxx
import numpy as np           # 导入并起别名
from core.utils import RED   # 只导入指定的东西，直接用 RED

if __name__ == "__main__":
    ...                      # 只在"直接运行本文件"时执行
```

### 4.9 numpy 速查

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])   # 二维数组
a.shape                          # → (2, 2)，形状
a[0, 1]                          # → 2，第 0 行第 1 列（逗号分隔维度）
a[0]                             # → [1, 2]，第 0 行

np.eye(3)                        # 3x3 单位矩阵
np.zeros(5)                      # [0,0,0,0,0]
np.ones((2, 3))                  # 2x3 全 1

arr.tolist()                     # numpy 数组 → Python 列表（写 JSON 前必须转）
arr.ravel()                      # 展平成一维
```

### 4.10 常用标准库速查

```python
# os.path
os.path.dirname(p)      # 取目录部分
os.path.basename(p)     # 取文件名部分
os.path.join(a, b)      # 拼接路径（自动处理分隔符，比字符串相加可靠）
os.path.splitext(p)     # 拆成 (主名, 扩展名)
os.path.normpath(p)     # 规范化路径（消化 .. 和 .）
os.path.realpath(p)     # 转成绝对路径，解析符号链接

# glob
glob.glob("dir/*.png")  # 通配符找文件，返回列表

# json
json.loads(字符串)       # 字符串 → Python 对象
json.dumps(对象)        # Python 对象 → 字符串
json.load(文件对象)      # 文件 → Python 对象
json.dump(对象, 文件对象)  # Python 对象 → 文件

# sys
sys.path                # 模块搜索路径列表
sys.path.append(p)      # 添加一个搜索目录
sys.exit(1)             # 退出程序，1 表示异常

# logging
logging.info("普通信息")
logging.warning("警告，黄色")
logging.error("错误，红色")
```

---

## 5. 输入与输出：怎么跑、结果长啥样

### 5.1 运行命令

```bash
# 最简（用默认参数，需要作者那台机器的路径存在）
python calib_camera.py

# 实际使用（换成你自己的路径）
python calib_camera.py \
    --img_dir "D:/data/calib/collect_image/cam0" \
    --calib_board_info "[0.0245,0.0075,6,6]"

# 看帮助
python calib_camera.py --help
```

### 5.2 照片怎么拍（决定标定成败）

这一步比代码更重要。**拍不好照片，代码再对也白搭：**

| 要求 | 说明 |
|---|---|
| 数量 | 建议 15~30 张（代码最低 10 张） |
| 姿态 | 要有**明显的角度变化**：正对、左倾 30°、右倾 30°、俯视、仰视 |
| 距离 | 有远有近，覆盖实际工作距离 |
| 位置 | 覆盖画面**四个角和中心**，畸变主要在边缘，边缘必须有数据 |
| 清晰度 | 不能糊、不能过曝、不能有运动模糊 |
| 完整 | 尽量让板子完整入画，至少露出 4~5 个 tag |
| 刚性 | 板子必须**平整**，不能弯（打印后贴在亚克力板/泡沫板上） |

> ❌ **反面教材**：20 张图都是正面平视、距离一样 → 数学上"退化"，RMS 可能很小但实际不准。
> ❌ **反面教材**：板子只出现在画面中央一小块 → 边缘畸变无数据，`k1/k2` 不可靠。

### 5.3 输出的 `cam_params.json`

```json
{
    "camera_type": "Pinhole",
    "IntrinsicFormat": "fx,fy,cx,cy",
    "DistortionFormat": "k1,k2,p1,p2,k3",
    "resolution": [1280, 720],
    "intrinsic": [912.3456, 911.9821, 638.2341, 361.5523],
    "distortion": [-0.1234, 0.0567, 0.0008, -0.0003, 0.0012]
}
```

| 字段 | 含义 |
|---|---|
| `camera_type` | `Pinhole` 表示针孔模型 |
| `IntrinsicFormat` | 说明 `intrinsic` 数组顺序是 fx, fy, cx, cy |
| `DistortionFormat` | 说明 `distortion` 数组顺序是 k1, k2, p1, p2, k3 |
| `resolution` | `[宽, 高]`。**换分辨率后这组参数就失效了**，必须重新标定 |
| `intrinsic` | `[fx, fy, cx, cy]`，单位像素 |
| `distortion` | `[k1, k2, p1, p2, k3]`，无量纲 |

**怎么快速判断结果对不对：**

| 检查项 | 期望 |
|---|---|
| `fx ≈ fy` | 两者相差应 < 1%，差太多说明有问题 |
| `cx ≈ 分辨率宽 / 2` | 1280 宽的相机，cx 应在 640 ± 50 |
| `cy ≈ 分辨率高 / 2` | 720 高，cy 应在 360 ± 50 |
| `RMS` | < 0.5 px 优秀，< 1.0 px 可用 |
| `k1` | 广角镜头通常 -0.5 ~ -0.1；绝对值 > 1 说明有问题 |

---

## 6. 动手算一遍（数值演练）

用默认参数走一遍，把所有数字算出来。

### 6.1 标定板的物理尺寸

```
tag_size   = 0.0245 m = 24.5 mm
space_size = 0.0075 m =  7.5 mm
rows = cols = 6
```

假设 `space_size` 是 tag 之间的**间隙**（大多数库的语义）：

```
板子宽度 = 6 × 24.5 + 5 × 7.5 = 147 + 37.5 = 184.5 mm
板子高度 = 同上 = 184.5 mm
```

> 如果库把 `space_size` 解释成"相邻 tag 中心的间距（pitch）"，那么：
> 宽度 = 5 × 7.5 + 24.5 = 62 mm
> 这个差异很大！**建议你用尺子量一下实际板子，或者打印一小段代码验证**，这是最容易填错的量。

### 6.2 数据量

```
tag 总数        = 6 × 6 = 36 个
3D 角点总数     = 36 × 4 = 144 个（这是"模板"，不是每次都用）
每张图若检测到 10 个 tag → 3D-2D 点对数 = 40 对
```

### 6.3 约束数 vs 未知数（20 张图，每张 10 个 tag 的情况）

```
未知数 = 4(内参) + 5(畸变) + 6×20(每张图姿态) = 9 + 120 = 129
方程数 = 2 × 40(每图点对数) × 20(图数) = 1600

1600 / 129 ≈ 12.4 倍冗余
```

**冗余度越高，抗噪能力越强。** 这就是为什么建议多拍、多露 tag。

### 6.4 最差情况（刚好卡在代码下限）

```
10 张图 × 每张 1 个 tag = 10 × 4 = 40 个点对
未知数 = 9 + 6×10 = 69
方程数 = 2 × 40 = 80
80 > 69 → 勉强可解，但冗余度只有 1.16 倍
```

**结论**：刚好 10 张、每张 1 个 tag 虽然在数学上"有解"，但**极度脆弱**——任何一个点检测偏了，结果就飘了。所以实际请务必多拍。

### 6.5 RMS 的含义（举例）

假设标定完有 800 个点对，重投影误差平方和 = 128 像素²：

```
RMS = sqrt(128 / 800) = sqrt(0.16) = 0.4 像素
```

意思是：**平均而言，用标定出的参数反推的点，和检测到的点相差约 0.4 个像素**。这是个很好的结果。

---

## 7. 这段代码里的坑与改进建议

按严重程度排列。**前三条是真正会影响结果的。**

| # | 位置 | 问题 | 影响 | 建议改法 |
|---|---|---|---|---|
| 1 | 第 105 行 | `D.tolist()` 可能得到**嵌套列表** `[[...]]`（因为 `D` 形状是 `(1,5)`） | JSON 里 `distortion` 多一层括号，下游读取代码可能解析失败 | `distortion = D.ravel().tolist()` |
| 2 | 第 204 行 | `img` 在循环外使用，若**第一张图读失败**则变量未定义 | `NameError` 崩溃 | 循环外初始化 `img_size = None`，在成功读图处赋值 |
| 3 | 第 214 行 | `os.path.dirname` 遇到**带尾斜杠**的路径（`"D:/cam0/"`）不会上一级 | 结果存错位置 | 先 `img_dir = os.path.normpath(img_dir)` |
| 4 | 第 74 行 vs 文档 | 注释说"至少 4 个 tag"，代码实际检查 4 个**点**（1 个 tag） | 误导读者 | 改成 `if len(pts3d) < 4 * min_tags:` 或修正注释 |
| 5 | 第 154 行 | `start_tag_id=0` **写死** | 板子编号不是从 0 开始时全部配对失败 | 做成命令行参数 |
| 6 | 第 164 行 | 只找 `*.png` | jpg 照片全被忽略 | 加 `*.jpg` / `*.jpeg` |
| 7 | 第 21 行 | 相对路径依赖 `../../../`，**移动文件就失效** | 换个目录放脚本就 import 失败 | 用配置文件或环境变量指定工程根 |
| 8 | 第 175 行 | `img_id` 算了但没用 | 死代码 | 删掉或用于日志 |
| 9 | 第 226 行 | `# end if` 应为 `# end with` | 注释误导 | 改注释 |
| 10 | 第 19–20 行 | 重复了两行一样的注释 | 冗余 | 删一行 |
| 11 | 第 134 行 | 默认路径是**作者机器的中文 Linux 路径** | 直接运行必然失败 | 改成更中性的默认（如空串，强制用户指定） |
| 12 | 第 92–93 行 | `K = np.eye(3)` 初值不合理 | 一般无影响（OpenCV 会自算初值），但极端情况可能收敛慢 | 用图片尺寸构造合理初值 |
| 13 | 第 96 行 | 未处理 `calibrateCamera` 可能的异常 | 某些病态输入会直接抛 OpenCV error | 包 `try/except` |
| 14 | 全局 | 没有**结果合理性校验** | 标定出荒谬结果（如 fx 为负）也不会报警 | 加断言：`fx > 0`，`cx` 在合理范围 |

### 7.1 一个推荐的加固版片段

把上面几个关键点改掉：

```python
# ① 规范化路径（解决坑 #3）
img_dir = os.path.normpath(args.img_dir)

# ② 循环外初始化 img_size（解决坑 #2）
img_size = None

for img_path in img_path_list:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.warning(...)
        continue
    if img_size is None:
        img_size = (img.shape[1], img.shape[0])   # 第一张成功读的图，记下尺寸
    ...

# ③ 展平畸变系数（解决坑 #1）
distortion = D.ravel().tolist()

# ④ 结果合理性校验（解决坑 #14）
fx, fy, cx, cy = intrinsic
assert fx > 0 and fy > 0, "焦距异常，标定失败"
assert 0.3 * img_size[0] < cx < 0.7 * img_size[0], "主点偏离画面中心过多，请检查图像尺寸参数"
```

---

## 8. 一句话总结

**这个脚本做的事，用一条链串起来就是：**

```
拍照（多角度的 AprilTag 标定板）
    → glob 找出所有 png
    → 灰度读图
    → AprilTag 检测，得到每个 tag 的 4 个亚像素角点（2D）
    → 按 tag 编号，把 2D 角点和标定板上预先算好的 3D 角点配对
    → 攒够 ≥10 张图的点对
    → 交给 cv2.calibrateCamera 迭代优化，最小化重投影误差
    → 取出 fx, fy, cx, cy 和 k1, k2, p1, p2, k3
    → 写成带自描述字段的 cam_params.json
```

**三个最需要记住的技术要点：**

1. **numpy 是 `(行, 列)`，OpenCV 的 `imageSize` 是 `(宽, 高)`** —— 永远差一个顺序，写反了不报错但结果是错的。
2. **配对靠 tag ID** —— AprilTag 自带编号，这是自动配对的基础；`start_tag_id` 和 `--calib_board_info` 填错会让配对全军覆没。
3. **RMS 就是重投影误差的均方根，单位像素** —— 它是你判断标定好坏的唯一硬指标，< 0.5 才算好。

**最后一句大实话**：这段代码本身写得相当规范（分层清晰、防御性检查到位、日志分级、结果自描述）。真要出问题，**九成不是代码的锅，而是照片没拍好、或者板子尺寸填错了**。拿到一个差结果时，按这个顺序排查：

```
① 打开被注释掉的可视化（第 193-198 行），看看 tag 认全了没、角点准不准
② 用尺子重新量一遍板子，核对 --calib_board_info
③ 看 RMS 是不是 > 1.0
④ 检查 fx≈fy、cx≈宽/2、cy≈高/2
⑤ 重新拍：更多角度、覆盖画面四角
```

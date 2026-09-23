# `calib_handeye.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、也没接触过机器人标定的同学。
> 源文件：`C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\examples\common\src\calib_handeye.py`（363 行）
> 目录里你把它叫 `calib_handeyes`，工程里实际文件名是**单数** `calib_handeye.py`。
>
> 配套文档：`calib_camera_analysis.md`（相机内参标定）、`calib_gripper_analysis.md`（夹爪标定）、`arm_node_analysis.md`（机械臂 ROS 节点）。

---

## 目录

- [0. 一句话概括](#0-一句话概括)
- [1. 背景知识：手眼标定到底在算什么](#1-背景知识手眼标定到底在算什么)
  - [1.1 先复习：位姿矩阵 T 是什么](#11-先复习位姿矩阵-t-是什么)
  - [1.2 问题的由来](#12-问题的由来)
  - [1.3 两种构型：眼在手 vs 眼在外](#13-两种构型眼在手-vs-眼在外)
  - [1.4 AX=XB 方程的完整推导](#14-axxb-方程的完整推导)
  - [1.5 为什么姿态必须变化（关键！）](#15-为什么姿态必须变化关键)
  - [1.6 需要几组数据](#16-需要几组数据)
- [2. 整体流水线](#2-整体流水线)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 导入区](#31-导入区)
  - [3.2 `calib_handeye()` 主函数](#32-calib_handeye-主函数)
  - [3.3 内嵌函数 `solve_axxb()`](#33-内嵌函数-solve_axxb)
  - [3.4 内嵌函数 `compute_error()`](#34-内嵌函数-compute_error)
  - [3.5 主程序 `__main__`](#35-主程序-__main__)
- [4. 输入输出规范](#4-输入输出规范)
- [5. 数值演练](#5-数值演练)
- [6. 坑与改进建议](#6-坑与改进建议)
- [7. 一句话总结](#7-一句话总结)

---

## 0. 一句话概括

> **相机装在机械臂末端上，它知道"物体在我的视线里的什么位置"，机械臂知道"我的末端在基座里的什么位置"，但这两个坐标系之间的相对关系（相机到底装歪了多少、偏了多少）没人知道。手眼标定就是把这个关系算出来。**

算出来的结果记作 **`T_end_cam`**（从相机坐标系到机械臂末端坐标系的变换）。有了它，就能把"相机看到的物体位置"换算成"机械臂末端该去的位置"——这是抓取的前提。

**输入 → 输出：**

| | 内容 |
|---|---|
| 输入 1 | `cam_params.json`：相机内参 + 畸变（由 `calib_camera.py` 先标定出来） |
| 输入 2 | `arm_pose.json`：采集时每张图对应的机械臂末端位姿 |
| 输入 3 | 一堆拍了标定板的 `.png` 照片 |
| 输入 4 | 标定板规格 `[tag_size, space_size, rows, cols]` |
| 输出 | `calib_handeye.json`，内含 `T_armend_cam`（平移 t + 四元数 q + 旋转矩阵 R） |

**三个标定的先后顺序（很重要，不能乱）：**

```
① calib_camera.py   → cam_params.json    （相机自己的内参：fx,fy,cx,cy + 畸变）
         ↓
② calib_handeye.py  → calib_handeye.json （相机 与 机械臂末端 的关系）
         ↓
③ calib_gripper.py  → gripper_params.json（夹爪尖端 与 相机 的关系）
```

第 ② 步必须用到第 ① 步的结果（要用内参去解 PnP），第 ③ 步必须用到 ② 的结果。

---

## 1. 背景知识：手眼标定到底在算什么

### 1.1 先复习：位姿矩阵 T 是什么

一个 4×4 的**齐次变换矩阵**，把"位置 + 姿态"打包在一起：

$$
T = \begin{bmatrix}
R_{3\times3} & t_{3\times1} \\
0\ \ 0\ \ 0 & 1
\end{bmatrix}
$$

- `R`（左上 3×3）：旋转，描述"歪成什么样"
- `t`（右上 3×1）：平移，描述"在哪儿"

**下标记法是本项目的核心约定，必须看懂：**

`T_base_end` = 从 **end（末端）** 坐标系到 **base（基座）** 坐标系的变换。

读法：**`T_A_B` = 把 B 坐标系里的点，换算到 A 坐标系里。**

**矩阵乘法 = 坐标系接力：**

```python
T_base_cam = T_base_end @ T_end_cam
```

意思是：先把点从 cam 换到 end（`T_end_cam`），再从 end 换到 base（`T_base_end`），得到从 cam 直接到 base。**中间下标相同才能消掉**——这是检查公式对不对的万能方法。

**求逆 = 反过来：**

```python
T_end_base = inv_tf(T_base_end)
```

在 `core/utils.py` 里是这样实现的（比你想象的简单）：

```python
T_inv[:3, :3] = R.T              # 旋转部分转置（正交矩阵的逆 = 转置）
T_inv[:3, 3]  = -R.T @ t         # 平移部分取负再转回去
```

### 1.2 问题的由来

相机（装在机械臂末端上）看到了桌上的一个 AprilTag 标定板，它能算出：

> "标定板在我**相机坐标系**里的位姿是 `T_cam_board`"

机械臂控制器同时能报出：

> "我末端在**基座坐标系**里的位姿是 `T_base_end`"

但这两个数字之间**隔着一层**——相机装在末端上，装的时候既没量角度也没量偏移。这一层就是 `T_end_cam`，**未知、且固定不变**（相机拧上去就不动了）。

手眼标定的任务：**求出这个固定的 `T_end_cam`。**

### 1.3 两种构型：眼在手 vs 眼在外

| | **眼在手（eye-in-hand）** | **眼在外（eye-to-hand）** |
|---|---|---|
| 相机装哪 | 装在机械臂末端，**跟着臂动** | 固定在支架/墙上，**不动** |
| 标定板放哪 | 平放在桌上，**不动** | 抓在末端上，**跟着臂动** |
| 求什么 | `T_end_cam`（相机相对末端） | `T_base_cam`（相机相对基座） |
| 本项目用哪个 | ✅ **用这个** | ❌ 实际没用到（见下文说明） |

**本工程实际上只支持眼在手**，证据有两条：

1. `auto_collect.py` 第 141 行硬编码 `pose_dict["eye_in_hand"] = True`
2. `create_collect_actions.py` 第 572 行：`if T_end_cam is None or not eye_in_hand:` 直接报错退出

所以 `calib_handeye()` 里的 `else` 分支（眼在外）在本工程里**基本是没人走过的路径**。如果你真的要用眼在外，务必先用代码自带的 `compute_error()` 验证误差。

### 1.4 AX=XB 方程的完整推导

这是手眼标定的数学核心，也是这个脚本最值得理解的部分。慢慢来。

**第一步：找到那个"不变量"。**

眼在手时，标定板平放在桌上不动。所以不管机械臂摆到哪个姿势，"标定板相对基座的位置"**永远不变**：

$$T_{base\_board} = \text{const}$$

**第二步：写出从 cam 到 base 的两条路径。**

从标定板走到基座，有两条路可走：

```
board ──T_cam_board──> cam ──T_end_cam──> end ──T_base_end──> base
```

把三段接起来（注意顺序，从右往左读）：

$$T_{base\_board} = T_{base\_end}^{(i)} \cdot T_{end\_cam} \cdot T_{cam\_board}^{(i)}$$

上标 $(i)$ 表示"第 i 次拍照时"的值。其中：

- $T_{base\_end}^{(i)}$：机械臂报出来的，**每次不同**
- $T_{cam\_board}^{(i)}$：相机算出来的，**每次不同**
- $T_{end\_cam}$：**未知、恒定** ← 我们要求它
- $T_{base\_board}$：**未知、但恒定**

**第三步：取两次拍照，让不变量相等。**

因为 $T_{base\_board}$ 恒定，第 i 次和第 j 次算出来必须一样：

$$T_{base\_end}^{(i)} \cdot X \cdot T_{cam\_board}^{(i)} = T_{base\_end}^{(j)} \cdot X \cdot T_{cam\_board}^{(j)}$$

这里 $X = T_{end\_cam}$。

**第四步：整理成 AX = XB 的形式。**

目标是把已知量挪到 X 的两侧。上式左乘 $(T_{base\_end}^{(j)})^{-1}$、右乘 $(T_{cam\_board}^{(i)})^{-1}$：

$$\underbrace{(T_{base\_end}^{(j)})^{-1} T_{base\_end}^{(i)}}_{A} \cdot X = X \cdot \underbrace{T_{cam\_board}^{(j)} (T_{cam\_board}^{(i)})^{-1}}_{B}$$

**得到 $AX = XB$**。其中：

- $A$：只由机械臂位姿算出来，**已知**
- $B$：只由相机观测算出来，**已知**
- $X$：**未知**

这就是传说中的 **AX = XB 问题**（也叫手眼方程）。它是机器人学里的经典问题，有一大堆解法（Tsai、Park、Horaud、Andreff、Daniilidis），OpenCV 全都实现了。

**对照代码里的符号：**

`solve_axxb(T01s, T23s)` 的命名含义是：

| 代码符号 | 本文符号 | 实际含义 |
|---|---|---|
| `T01` | $T_{base\_end}$ | 从 1(末端) 到 0(基座) |
| `T12` | $X = T_{end\_cam}$ | 从 2(相机) 到 1(末端) ← **待求** |
| `T23` | $T_{cam\_board}$ | 从 3(标定板) 到 2(相机) |
| `T01·T12·T23` | $T_{base\_board}$ | 从 3(标定板) 到 0(基座)，**常量** |

所以第 163-165 行：

```python
T01s = np.array(arm_pose_list)   # T_base_end
T23s = np.array(cam_pose_list)   # T_cam_board
T_end_cam = solve_axxb(T01s, T23s)
```

完全对应上面推导的 $A$、$B$、$X$。

### 1.5 为什么姿态必须变化（关键！）

这是**手眼标定最常见的翻车原因**，务必理解。

假设你采集数据时，机械臂只平移不旋转（末端姿态始终朝下，只是位置在变）。那么所有 $T_{base\_end}^{(i)}$ 的旋转部分都一样：

$$R_{base\_end}^{(i)} = R_{base\_end}^{(j)} = R$$

于是：

$$A_{\mathrm{rot}} = R^{-1} R = I$$

即 $A$ 的旋转部分是单位矩阵（推导见下）。

$A$ 的旋转变成了单位阵，**对 X 的旋转部分完全没有约束**——方程退化，解出来的 $X$ 的旋转是垃圾（可能任意值）。

**同理，如果机械臂只绕同一个轴转**（比如每次都只绕竖直轴转不同角度），$A$ 的旋转轴始终相同，那么也只能确定 X 旋转的一部分。

**正确做法：采集时每次都要有明显的、不同方向的旋转。** 让末端像"转手腕"一样在各个方向上倾斜。

> 📌 **一句话**：**平移给不出旋转信息，只有旋转才能标定旋转。** 采数据时，姿态变化要"丰富"——绕不同的轴、有大的角度差。

### 1.6 需要几组数据

| 数量 | 说明 |
|---|---|
| 数学下限 | 2~3 组（$X$ 有 6 个自由度：3 旋转 + 3 平移） |
| 代码要求 | **≥ 4 组**（第 152 行） |
| 工程建议 | **10~20 组**，姿态差异要大 |

为什么代码要求 4？因为 N 组位姿可以配出 $N(N-1)/2$ 个方程对，N=4 有 6 对，在有噪声的情况下才勉强够用。实际请多采。

---

## 2. 整体流水线

```
 ┌──────────────────────────────────────────────────────────┐
 │  输入（4 个命令行参数）                                    │
 │  --cam_param_path    相机内参（来自 calib_camera）         │
 │  --arm_pose_path     机械臂位姿 arm_pose.json             │
 │  --img_dir           照片目录                             │
 │  --calib_board_info  标定板规格                           │
 └───────────────────────────┬──────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
 ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐
 │ read_cam_   │   │ json.load(       │   │ create_calib │
 │ params()    │   │   arm_pose.json) │   │ _board_3d()  │
 │ → K, D      │   │ → pose_dict      │   │ → tag3d_list │
 └──────┬──────┘   └────────┬─────────┘   └──────┬───────┘
        │                   │                    │
        │           ┌───────┴────────────────────┘
        │           │
        ▼           ▼
 ┌────────────────────────────────────────────────────────┐
 │  遍历每张照片（按文件名 id 配对）                        │
 │                                                         │
 │  0000.png ─┬─> cv2.imread 灰度                          │
 │            ├─> detector.detect() → tag2d_list           │
 │            ├─> apriltag2.locate_calib_board()           │
 │            │      → T_cam_board   （放进 cam_pose_dict） │
 │            └─> pose_dict["0000"]                        │
 │                  → ArmWrapper.array_to_matrix()          │
 │                  → T_base_end     （放进 arm_pose_dict） │
 └───────────────────────────┬────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │  calib_handeye()                                        │
 │  ① 按 img_id 配对，凑够 ≥4 组                           │
 │  ② solve_axxb()：cv2.calibrateHandEye(PARK 法) → T_end_cam│
 │  ③ compute_error()：算平均旋转误差(度)/平移误差(mm)      │
 └───────────────────────────┬────────────────────────────┘
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │  存 calib_handeye.json（与 arm_pose.json 同目录）        │
 │  { QuaternionFormat, T_armend_cam: {t, q, R} }          │
 └────────────────────────────────────────────────────────┘
```

**照片文件名与位姿的配对关系**（这是理解主循环的关键）：

```
img_dir/
├── 0000.png   ←→  pose_dict["0000"]  →  T_base_end(0)
├── 0001.png   ←→  pose_dict["0001"]  →  T_base_end(1)
└── 0002.png   ←→  pose_dict["0002"]  →  T_base_end(2)
```

`arm_pose.json` 长这样（由 `auto_collect.py` 生成）：

```json
{
    "0000": [0.32, 0.05, 0.28, 0.001, 0.002, 0.999, 0.001],
    "0001": [0.30, -0.02, 0.31, 0.12, 0.03, 0.99, 0.02],
    "PoseNote": "Meaning: ...; Format: tx,ty,tz,qx,qy,qz,qw",
    "eye_in_hand": true
}
```

- 每个 7 元数组是 `[tx, ty, tz, qx, qy, qz, qw]` —— 注意**四元数顺序是 qx,qy,qz,qw（实部在最后）**
- `PoseNote` 和 `eye_in_hand` 是元数据，不是图片 id

---

## 3. 逐段代码精读

### 3.1 导入区（第 6–27 行）

```python
import argparse      # 命令行参数
import glob          # 按通配符找 png
import json          # 读写 JSON
import logging       # 日志
import os            # 路径处理
import sys           # sys.path / sys.exit

import apriltag2     # AprilTag 检测 + 标定板定位
import cv2           # OpenCV
import numpy as np
import transforms3d  # 四元数 ↔ 旋转矩阵互相转换

code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{code_dir}/../../../")
sys.path.append(root_dir)

from core.arm_wrapper import ArmWrapper
from core.utils import BLUE, GREEN, RED, RESET, YELLOW, inv_tf, read_cam_params
from typing import Dict, List, Tuple
```

几个要点：

- **新增的 `transforms3d`**：专门做旋转表示互换（四元数/旋转矩阵/欧拉角/轴角）。本文件用它把旋转矩阵转成四元数存进 JSON（第 350 行）。
- **`sys.path` 三件套**：和 `calib_camera.py` 一模一样，把工程根目录临时加进模块搜索路径，才能 `from core.xxx import ...`。详见相机标定文档 3.3 节。
- **`inv_tf`**：求 4×4 位姿矩阵的逆（眼在外分支会用到）。
- **`ArmWrapper`**：这里只用它的静态方法 `array_to_matrix()`（把 7 元数组转成 4×4 矩阵），**不连机械臂**——所以这个脚本是纯离线计算的，不需要开机。

### 3.2 `calib_handeye()` 主函数（第 33–193 行）

```python
def calib_handeye(
    arm_pose_dict: Dict[int, np.ndarray],
    cam_pose_dict: Dict[int, np.ndarray],
    eye_in_hand: bool = True,
) -> np.ndarray:
```

**三个参数：**

| 参数 | 类型 | 含义 |
|---|---|---|
| `arm_pose_dict` | `Dict[int, ndarray]` | key=图片 id，value=`T_base_end` (4×4) |
| `cam_pose_dict` | `Dict[int, ndarray]` | key=图片 id，value=`T_cam_board` (4×4) |
| `eye_in_hand` | `bool = True` | 是否眼在手 |

**默认参数 `= True`**：调用时不传就自动用 `True`。这是 Python 的便利特性。

#### 配对逻辑（第 139–157 行）

```python
arm_pose_list = []   # T_base_end
cam_pose_list = []   # T_cam_board
for img_id in arm_pose_dict:
    if img_id in cam_pose_dict:
        arm_pose_list.append(arm_pose_dict[img_id])
        cam_pose_list.append(cam_pose_dict[img_id])
    else:
        logging.warning(f"{YELLOW}No camera pose for image {img_id}, skipping.{RESET}")

if len(arm_pose_list) < 4:
    logging.error(f"{RED}Not enough valid data ... Need at least 4 pairs ...{RESET}")
    return None
```

**语法点：**
- `for img_id in arm_pose_dict:` —— 直接遍历字典时，拿到的是**键（key）**，不是值。想拿值要写 `arm_pose_dict[img_id]`，想拿键值都要写 `for k, v in d.items():`
- `if img_id in cam_pose_dict:` —— `in` 判断**键**是否存在于字典，速度极快（哈希表 O(1)）
- 两个 `append` 一一对应，保证 `arm_pose_list[k]` 和 `cam_pose_list[k]` 属于**同一张图**——这是 AX=XB 成立的命脉

**业务含义**：只有"机械臂位姿有，且相机也成功定位了标定板"的图片才参与计算。任何一边缺失就丢弃这张图。

#### 眼在手分支（第 159–173 行）

```python
if eye_in_hand:
    logging.info(f"Using {GREEN}{len(arm_pose_list)}{RESET} valid pairs ...")
    T01s = np.array(arm_pose_list)      # T_base_end,  shape (N, 4, 4)
    T23s = np.array(cam_pose_list)      # T_cam_board, shape (N, 4, 4)
    T_end_cam = solve_axxb(T01s, T23s)
    err_r, err_p = compute_error(T01s, T23s, T_end_cam)
    logging.info(f"Calibration result (T_end_cam):\n{GREEN}{T_end_cam}{RESET}")
    logging.info(f"Calibration error: rotation(deg) = {GREEN}{err_r:.4f}{RESET}, "
                 f"translation(mm) = {GREEN}{err_p:.4f}{RESET}")
    return T_end_cam
```

- `np.array(列表)` 把 N 个 4×4 矩阵堆成一个 **三维数组**，形状 `(N, 4, 4)`。这是 OpenCV 批量接口要求的格式。
- `:.4f` 是 f-string 的格式化：保留 4 位小数。
- 打印的**误差**是判断成败的唯一硬指标，见 [3.4](#34-内嵌函数-compute_error)。

#### 眼在外分支（第 174–190 行）

```python
else:
    T01s = np.zeros((len(arm_pose_list), 4, 4), dtype=np.float64)
    T23s = np.zeros((len(arm_pose_list), 4, 4), dtype=np.float64)
    for i in range(len(arm_pose_list)):
        T01s[i] = inv_tf(arm_pose_list[i])   # T_end_base
        T23s[i] = inv_tf(cam_pose_list[i])   # T_board_cam
    T_base_cam = solve_axxb(T01s, T23s)
    err_r, err_p = compute_error(T01s, T23s, T_base_cam)
    logging.info(f"Calibration result (T_base_cam):\n{GREEN}{T_base_cam}{RESET}")
    return T_base_cam
```

**语法点：**
- `np.zeros((N, 4, 4))` 预分配一个全零的三维数组，然后用下标 `T01s[i]` 逐个填充。这叫"预分配 + 填充"，比反复 `append` 再转 array 更规范。
- `dtype=np.float64` 显式指定双精度浮点。位姿计算一定要用 float64，float32 的累积误差在这个量级上会看得出来。

**业务含义**：眼在外时，不变量是"标定板相对末端"（板子抓在爪子上，相对末端不动），所以要把两份位姿都**取逆**，再走同一套 AX=XB。

> ⚠️ **提醒**：这个分支在本工程里**从未被真正使用过**（采集脚本硬编码 `eye_in_hand=True`）。如要使用，请务必用 `compute_error()` 打印的误差做验证——误差应该在 1°/2mm 量级，如果到了几十度/几百毫米，说明这个分支的符号/次序需要调整。

### 3.3 内嵌函数 `solve_axxb()`（第 49–76 行）

```python
def solve_axxb(T01s: np.ndarray, T23s: np.ndarray) -> np.ndarray:
    R, t = cv2.calibrateHandEye(
        T01s[:, :3, :3],      # 所有 A 的旋转部分
        T01s[:, :3, 3],       # 所有 A 的平移部分
        T23s[:, :3, :3],      # 所有 B 的旋转部分
        T23s[:, :3, 3],       # 所有 B 的平移部分
        method=cv2.CALIB_HAND_EYE_PARK,   # 这个方法最稳定
    )
    T12 = np.eye(4, dtype=np.float64)
    T12[:3, :3] = R
    T12[:3, 3] = t.ravel()
    return T12
```

**内嵌函数（nested function）**：定义在另一个函数内部的函数。好处是它只在 `calib_handeye()` 里用，不污染全局命名空间；坏处是每次调用外层函数都要重新定义一次（可忽略的开销）。

**这一段的语法密度很高，逐个拆：**

| 表达式 | 含义 |
|---|---|
| `T01s[:, :3, :3]` | 三维数组切片。`:` = 全部，`:3` = 前 3 个。结果形状 `(N, 3, 3)`，即**每个矩阵的左上 3×3 旋转块** |
| `T01s[:, :3, 3]` | 形状 `(N, 3)`，即**每个矩阵的右上 3×1 平移列** |
| `np.eye(4)` | 4×4 单位矩阵 |
| `T12[:3, :3] = R` | 把旋转块填进去 |
| `t.ravel()` | 把 `t` 展平成一维。OpenCV 返回的 `t` 形状是 `(3,1)`，直接赋值给形状 `(3,)` 的位置会报错，`ravel()` 拉平 |

**OpenCV 的五种解法**（`method` 参数）：

| 常量 | 提出者 | 特点 |
|---|---|---|
| `CALIB_HAND_EYE_TSAI` | Tsai & Lenz (1989) | 经典，先解旋转再解平移 |
| `CALIB_HAND_EYE_PARK` | Park & Martin (1994) | **代码选的**，基于李群，数值最稳定 |
| `CALIB_HAND_EYE_HORAUD` | Horaud & Dornaika | 用四元数表示旋转 |
| `CALIB_HAND_EYE_ANDREFF` | Andreff et al. | 线性解法，需要大量数据 |
| `CALIB_HAND_EYE_DANIILIDIS` | Daniilidis | 用对偶四元数，同时解旋转和平移 |

代码注释说"这个方法最稳定"指的是 Park 法，这是实践经验，合理。

> 💡 **调试技巧**：如果标定结果不好，可以把 `method` 换成其他几种试试，对比 `compute_error` 输出的误差，选最小的那个。

### 3.4 内嵌函数 `compute_error()`（第 78–135 行）

**这个函数非常值得学习——它回答了"我怎么知道标定准不准"。**

```python
def compute_error(T01s, T23s, T12) -> Tuple[float, float]:
    assert T01s.shape[0] == T23s.shape[0], (
        f"T01s and T23s must have the same number of poses, got "
        f"{T01s.shape[0]} vs {T23s.shape[0]}"
    )
    N = T01s.shape[0]
    if N < 2:
        return 0.0, 0.0

    T03s = T01s @ T12 @ T23s      # (N,4,4) @ (4,4) @ (N,4,4) → (N,4,4)
```

**核心思想（很巧妙，值得记住）：**

回到那个"不变量"：$T_{base\_board} = T01_i \cdot X \cdot T23_i$。

如果标定是完美的，那么**用同一个 X 对每一组数据算出来的 $T_{base\_board}$ 应该完全一样**。如果不一样，差多少就是误差。

```python
T03s = T01s @ T12 @ T23s
```

这行一次性算出 N 个 $T_{base\_board}$。**numpy 的广播矩阵乘法**：`(N,4,4) @ (4,4)` 会把后两维当矩阵、第一维当批次，结果 `(N,4,4)`；再 `@ (N,4,4)` 是逐批次相乘。非常优雅。

```python
    for i in range(N):
        for j in range(i + 1, N):
            dT = T03s[j] @ np.linalg.inv(T03s[i])    # 两个"不变量"之差

            R = dT[:3, :3]
            cos_angle = (np.trace(R) - 1.0) / 2.0
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            delta_angle = np.arccos(cos_angle)

            delta_position = np.linalg.norm(dT[:3, 3])

            sum_delta_angle += delta_angle
            sum_delta_position += delta_position
            count += 1

    error_r = sum_delta_angle / count * 180.0 / np.pi   # 弧度 → 度
    error_p = sum_delta_position / count * 1000.0        # 米 → 毫米
    return error_r, error_p
```

**逐个语法/数学点解释：**

**① `for j in range(i + 1, N)`**：遍历所有"不重复的数对"（i<j），共 $N(N-1)/2$ 对。避免重复计算 (i,j) 和 (j,i)，也避免 i==j（差为 0）。

**② `dT = T03s[j] @ inv(T03s[i])`**：两个"应该相等"的矩阵之间的相对差异。完美时 `dT = I`（单位阵）。

**③ 从旋转矩阵提取旋转角的公式：**

$$\theta = \arccos\left(\frac{\text{trace}(R) - 1}{2}\right)$$

这是旋转矩阵的经典性质：3×3 旋转矩阵的迹（对角线元素之和）等于 $1 + 2\cos\theta$。

**④ `np.clip(x, -1.0, 1.0)`**：把数值强制限制在 [-1, 1] 区间。**为什么必须加？** 因为浮点误差可能让 `cos_angle` 变成 `1.0000000002`，而 `np.arccos(1.0000000002)` 会得到 `nan`（非数），整个结果就废了。`clip` 是防御性编程的标配。

**⑤ `np.linalg.norm(v)`**：向量的模长（欧几里得长度），即 $\sqrt{x^2+y^2+z^2}$。

**⑥ 单位换算**：`* 180.0 / np.pi` 弧度转度；`* 1000.0` 米转毫米。

**误差怎么判读（工程经验）：**

| 旋转误差 | 平移误差 | 评价 |
|---|---|---|
| < 0.5° | < 2 mm | 优秀 |
| 0.5° ~ 1° | 2 ~ 5 mm | 良好 |
| 1° ~ 3° | 5 ~ 20 mm | 勉强可用，建议重采 |
| > 3° | > 20 mm | 有问题：姿态变化不够 / 内参不准 / tag 检测有误 |

> 📌 **注意这个误差是"自洽性"检查，不是"绝对精度"检查。** 它能发现"数据内部矛盾"（比如有个位姿记错了），但不能发现"系统性偏差"（比如标定板尺寸填错，所有结果都一致地错）。

### 3.5 主程序 `__main__`（第 199–363 行）

#### 参数与回显（第 202–241 行）

```python
parser.add_argument("--cam_param_path", type=str,
                    default="/home/i4/桌面/test_location/calib/collect_image/cam_params.json",
                    help="相机参数文件的路径")
parser.add_argument("--calib_board_info", type=str,
                    default="[0.0245,0.0075 , 6, 6]", ...)
parser.add_argument("--img_dir", type=str,
                    default="/home/i4/桌面/test_location/calib/collect_image_handeye/cam0", ...)
parser.add_argument("--arm_pose_path", type=str,
                    default="/home/i4/桌面/test_location/calib/collect_image_handeye/arm_pose.json", ...)
```

四个参数的默认值都是作者自己机器上的路径，**你必须全部替换**。

> 💡 注意 `"[0.0245,0.0075 , 6, 6]"` 里有个多余的空格，`json.loads` 照样能解析（JSON 对空白不敏感）。

回显部分和 `calib_camera.py` 一样，把关键参数用蓝色打印出来，方便核对。

#### 读取相机参数并组装 K、D（第 245–253 行）

```python
intrinsic, distortion = read_cam_params(cam_param_path)
if intrinsic is None:
    logging.error("Failed to read camera parameters. Exiting.")
    sys.exit(1)

K = np.array([[intrinsic[0], 0, intrinsic[2]],
              [0, intrinsic[1], intrinsic[3]],
              [0, 0, 1]])
D = np.array(distortion) if distortion is not None else None
```

把 `[fx, fy, cx, cy]` 这种"扁平"格式重新组装成 3×3 矩阵 K：

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

**语法点：三元表达式（条件表达式）**

```python
D = np.array(distortion) if distortion is not None else None
```

等价于：

```python
if distortion is not None:
    D = np.array(distortion)
else:
    D = None
```

一行搞定，Python 里非常常用。

> ⚠️ **与相机标定文档的联动**：如果 `cam_params.json` 里的 `distortion` 是嵌套数组 `[[k1,k2,p1,p2,k3]]`（`calib_camera.py` 的 `D.tolist()` 可能产出的形态），这里 `np.array(distortion)` 会变成形状 `(1,5)` 的二维数组。OpenCV 的 `solvePnP` 能接受，但如果变成 `[[...]]` 之外的形态就可能出错。**建议用 `np.array(distortion).ravel()` 保险。**

#### 读取机械臂位姿（第 256–263 行）

```python
pose_dict = json.load(open(arm_pose_path, "r"))
if len(pose_dict) == 0:
    logging.error(f"{RED}No arm pose data found in {arm_pose_path}. Exiting.{RESET}")
    sys.exit(1)
eye_in_hand = pose_dict.get("eye_in_hand", True)  # 默认值为 True
```

**语法点：**
- `json.load(文件对象)` 从文件读（`json.loads` 是从字符串读，差一个 s）
- **`dict.get(key, 默认值)`**：键存在就返回值，不存在就返回默认值，**不会抛异常**。对比 `dict[key]`（不存在会 `KeyError`）。这里很合适——老版本的 json 里可能没有 `eye_in_hand` 字段。
- `len(pose_dict) == 0` 判断空字典

> ⚠️ **小瑕疵**：`json.load(open(...))` 没有用 `with`，文件句柄不会被显式关闭（虽然 CPython 的垃圾回收最终会处理）。建议改成：
> ```python
> with open(arm_pose_path, "r") as f:
>     pose_dict = json.load(f)
> ```

#### 创建标定板与检测器（第 266–278 行）

和 `calib_camera.py` 完全一样，**参数必须一致**（`tag_size`、`space_size`、`rows`、`cols`、`start_tag_id=0`）。

#### 遍历照片（第 284–336 行）

```python
img_path_list = glob.glob(f"{img_dir}/*.png")
img_path_list.sort()
if len(img_path_list) == 0:
    logging.error(f"{RED}No images found in {img_dir}. Exiting.{RESET}")
    sys.exit(1)

for img_path in img_path_list:
    img_name = os.path.basename(img_path)          # "0000.png"
    img_id = os.path.splitext(img_name)[0]         # "0000"

    logging.info(f"Processing image: {BLUE}{img_id}{RESET}")

    if f"{img_id}" not in pose_dict:               # 这张图没有对应的位姿
        logging.warning(f"{YELLOW}No arm pose data for image {img_name}, skipping.{RESET}")
        continue
    arm_pose = pose_dict[f"{img_id}"]

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.warning(f"{YELLOW}Failed to read image {img_name}, skipping.{RESET}")
        continue

    tag2d_list = detector.detect(img, -1)

    cam_pose = apriltag2.locate_calib_board(
        tag2d_list=tag2d_list, tag3d_list=tag3d_list, K=K, D=D
    )
    if cam_pose is None:
        logging.warning(f"{YELLOW}Failed to locate calibration board in image {img_name}, skipping.{RESET}")
        continue

    cam_pose_dict[img_id] = cam_pose
    arm_pose_dict[img_id] = ArmWrapper.array_to_matrix(arm_pose)
    print()
```

**关键理解点：**

**① `img_id` 是整条链的枢纽**

```
文件名 "0000.png" → os.path.splitext → img_id "0000"
                                          ↓
                          pose_dict["0000"] → 机械臂位姿
                                          ↓
                          cam_pose_dict["0000"] → 相机位姿
```

如果照片文件名不是 `0000.png` 这种格式（比如 `IMG_20260919.jpg`），配对会全部失败。

**② 这里为什么不做去畸变？**

对比一下：

| 脚本 | 检测前有没有 `undistort` | 原因 |
|---|---|---|
| `calib_camera.py` | 没有 | 畸变正是要标定的量 |
| **`calib_handeye.py`** | **没有** | 内参/畸变**已知**，直接把 D 传给 `locate_calib_board`，里面的 `solvePnP` 会用畸变模型正确处理带畸变的角点 |
| `calib_gripper.py` | **有** | 后面要用深度图做平面几何，需要像素坐标严格对应，先掰直更省事 |

**两种做法都正确**，只是适用场景不同。

**③ `apriltag2.locate_calib_board()`**

内部做的是 **PnP（Perspective-n-Point）**：已知 N 个 3D 点（标定板上的角点）和它们在照片上的 2D 位置，求相机相对标定板的位姿。返回 `T_cam_board`。

这正好是 `calib_camera.py` 的**逆问题**：
- 相机标定：已知 3D 和 2D，求**内参**
- PnP：已知 3D、2D 和**内参**，求**外参（位姿）**

**④ `ArmWrapper.array_to_matrix(arm_pose)`**

在 `core/arm_wrapper.py` 第 327 行：

```python
@staticmethod
def array_to_matrix(pose: List[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.array(pose[:3])                      # 前 3 个是平移
    q = [pose[6], pose[3], pose[4], pose[5]]            # 转成 [qw,qx,qy,qz]
    T[:3, :3] = transforms3d.quaternions.quat2mat(q)
    return T
```

**注意四元数顺序的重排**：JSON 里存的是 `[tx,ty,tz, qx,qy,qz,qw]`（实部 `qw` 在最后），而 `transforms3d` 要求 `[qw,qx,qy,qz]`（实部在最前）。所以有 `q = [pose[6], pose[3], pose[4], pose[5]]` 这一步**重排**。

**这是最容易出错的地方之一**：不同库用不同的四元数顺序（ROS/OpenCV 用 `xyzw`，transforms3d/scipy 用 `wxyz`）。弄反了旋转结果就完全错了。

`@staticmethod` 是"静态方法"：不需要创建 `ArmWrapper` 实例就能调用（所以这里不连机械臂也能用）。

#### 保存结果（第 346–361 行）

```python
save_path = os.path.join(os.path.dirname(arm_pose_path), "calib_handeye.json")
result_dict = {"QuaternionFormat": "qw,qx,qy,qz"}
R = T_arm_cam[:3, :3]
t = T_arm_cam[:3, 3]
q = transforms3d.quaternions.mat2quat(R).tolist()   # qw,qx,qy,qz
if eye_in_hand:
    result_dict["T_armend_cam"] = {"t": t.tolist(), "q": q, "R": R.tolist()}
else:
    result_dict["T_armbase_cam"] = {"t": t.tolist(), "q": q, "R": R.tolist()}

with open(save_path, "w") as f:
    json.dump(result_dict, f, indent=4)
```

- 保存位置：`arm_pose.json` 所在目录（即采集目录，不是照片目录）
- `"QuaternionFormat": "qw,qx,qy,qz"` —— 自描述字段，和相机标定文档里表扬过的做法一样，很棒
- 同时存了 `t`（3 个平移）、`q`（4 元四元数）、`R`（9 元旋转矩阵）——**冗余但方便**，下游按自己习惯取用
- `.tolist()` 把 numpy 数组转回 Python 列表（JSON 不认 numpy 类型）

---

## 4. 输入输出规范

### 4.1 运行命令

```bash
python calib_handeye.py \
    --cam_param_path   "D:/calib/collect_image/cam_params.json" \
    --calib_board_info "[0.0245,0.0075,6,6]" \
    --img_dir          "D:/calib/collect_image_handeye/cam0" \
    --arm_pose_path    "D:/calib/collect_image_handeye/arm_pose.json"
```

### 4.2 怎么采数据（决定成败）

**这是最容易出问题的环节。** 用 `auto_collect.py` 自动采集时，要让机械臂走一条**姿态丰富**的轨迹：

| 要求 | 说明 |
|---|---|
| 组数 | 10~20 组（最低 4） |
| **旋转必须变化** | ⚠️ 最关键。每次要有明显的、不同轴的旋转，单次旋转建议 15°~45° |
| 距离 | 标定板要完整入画，不要太远（tag 太小检测精度差） |
| 清晰度 | 不能糊、不能过曝 |
| 末端不能撞 | 采集前确认轨迹安全 |

> ❌ **反面教材**：15 组数据，但末端姿态始终朝下，只是位置在桌面不同地方平移 → 数学退化，误差会很大或结果完全错误。

> 💡 **自检方法**：采完看一眼 `arm_pose.json` 里那些四元数，如果它们几乎一样，说明姿态没变化，重采。

### 4.3 输出的 `calib_handeye.json`

```json
{
    "QuaternionFormat": "qw,qx,qy,qz",
    "T_armend_cam": {
        "t": [0.0352, -0.0121, 0.0483],
        "q": [0.7071, 0.0, 0.0, 0.7071],
        "R": [[0.9998, -0.0123, 0.0045],
              [0.0124, 0.9999, -0.0034],
              [-0.0045, 0.0035, 0.9999]]
    }
}
```

| 字段 | 含义 |
|---|---|
| `QuaternionFormat` | 说明 q 的顺序是 **qw, qx, qy, qz**（实部在前） |
| `T_armend_cam.t` | 平移，单位**米**。`[0.035, -0.012, 0.048]` 表示相机原点在末端坐标系里的位置 |
| `T_armend_cam.q` | 四元数 `[qw,qx,qy,qz]` |
| `T_armend_cam.R` | 3×3 旋转矩阵（和 q 等价，冗余存储） |

**怎么判断结果对不对：**

| 检查项 | 期望 |
|---|---|
| 平移量级 | 相机装在末端上，距离一般在 **2~15 cm**。出现几十厘米或几米肯定是错的 |
| 旋转 | 相机一般是朝下的，R 应该接近某个"朝下"的旋转，而不是单位阵（除非相机正好和末端同向） |
| 日志里的误差 | 旋转 < 1°，平移 < 5 mm |

---

## 5. 数值演练

### 5.1 需要多少组？

```
N = 组数
可配出的方程对数 = N(N-1)/2

N=4  → 6 对
N=10 → 45 对
N=20 → 190 对
```

未知数只有 6 个（3 旋转 + 3 平移），但噪声下需要冗余。10 组 45 对约束解 6 个未知数，冗余 7.5 倍，比较稳妥。

### 5.2 误差是怎么算出来的（举例）

假设 N=3 组，算出的三个 `T_base_board` 之间两两差异：

```
dT(0,1): 旋转 0.8°, 平移 1.5 mm
dT(0,2): 旋转 1.2°, 平移 2.1 mm
dT(1,2): 旋转 0.6°, 平移 1.2 mm

平均旋转误差 = (0.8 + 1.2 + 0.6) / 3 = 0.87°
平均平移误差 = (1.5 + 2.1 + 1.2) / 3 = 1.6 mm
```

**这个误差的物理意义**：用标定出的 `T_end_cam`，把不同视角下看到的标定板位置都换算到基座坐标系，它们**应该重合**；实际差了平均 0.87° / 1.6mm。

### 5.3 灵敏感知：姿态变化一点点，结果差多少

假设 `T_end_cam` 的旋转有 1° 误差，机械臂工作半径 0.5 米：

```
末端位置误差 ≈ 0.5 m × tan(1°) ≈ 0.5 × 0.01746 ≈ 8.7 mm
```

**1° 的手眼误差，在 0.5 米外就会导致接近 1 厘米的抓取偏差。** 这就是为什么手眼标定要那么小心——误差是**被力臂放大**的。

---

## 6. 坑与改进建议

按严重程度排序。

| # | 位置 | 问题 | 影响 | 建议 |
|---|---|---|---|---|
| 1 | 采集环节 | **姿态变化不够**导致 AX=XB 退化 | 结果完全错误，且误差指标可能看起来"还行" | 采集时保证每次有 15°~45° 的不同轴旋转 |
| 2 | 第 174-190 行 | 眼在外分支**本工程从未使用**，未经验证 | 若启用可能出错 | 启用前用仿真数据验证；见 `arm_node_analysis.md` 里的另一个相关坑 |
| 3 | 第 256 行 | `json.load(open(...))` 未用 `with` | 文件句柄泄漏 | 改成 `with open(...) as f: pose_dict = json.load(f)` |
| 4 | 第 253 行 | `np.array(distortion)` 未展平 | 若上游写入嵌套数组会出问题 | 改成 `np.array(distortion).ravel()` |
| 5 | 第 53 行 | 文档字符串的 i/j 次序可能有笔误（写了 `T01_j^-1·T01_i` 配 `T23_i·T23_j^-1`） | 读者推导时对不上 | 统一成一种次序：`A = T01_j^-1·T01_i`，`B = T23_j·T23_i^-1` |
| 6 | 第 141 行 | 遍历字典只取 key，依赖 `pose_dict` 里没有非图片键 | 若将来加入别的元数据键，可能误配对 | 显式过滤 `eye_in_hand`/`PoseNote` 这类保留键 |
| 7 | 第 152 行 | 只检查数量 ≥4，不检查**姿态是否足够多样** | 数量够但姿态退化时仍会通过检查 | 加一个"旋转多样性"检查：统计 A 矩阵旋转角的分布 |
| 8 | 第 164 行 | 只支持 `.png` | jpg 采的图全被忽略 | 加 `*.jpg` |
| 9 | 第 271 行 | `start_tag_id=0` 写死 | 板子编号不从 0 开始时全部定位失败 | 做成命令行参数 |
| 10 | 第 205-227 行 | 默认路径全是作者机器的 Linux 路径 | 直接运行必然失败 | 改成更中性的默认值 |
| 11 | 第 339 行 | `calib_handeye` 返回 `None` 时主程序能处理，但 `T_arm_cam[:3,:3]` 若返回非 4×4 会静默出错 | 鲁棒性 | 加 `assert T_arm_cam.shape == (4,4)` |

### 6.1 一个推荐的加固片段

```python
# ① 加姿态多样性检查（在 solve 之前）
def check_pose_diversity(T01s: np.ndarray, min_deg: float = 10.0) -> bool:
    """检查相邻位姿之间的旋转是否足够大"""
    N = T01s.shape[0]
    small_cnt = 0
    for i in range(N - 1):
        dR = T01s[i, :3, :3].T @ T01s[i + 1, :3, :3]
        cos_a = np.clip((np.trace(dR) - 1) / 2, -1, 1)
        if np.degrees(np.arccos(cos_a)) < min_deg:
            small_cnt += 1
    if small_cnt > N // 2:
        logging.warning(f"{YELLOW}{small_cnt}/{N-1} 相邻位姿旋转小于 {min_deg}°, "
                        f"姿态变化不足, 标定结果可能不可靠!{RESET}")
        return False
    return True

# ② 对比多种解法, 选误差最小的
best, best_err = None, 1e9
for m in [cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_TSAI,
          cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_DANIILIDIS]:
    X = solve_axxb_with_method(T01s, T23s, m)
    e_r, e_p = compute_error(T01s, T23s, X)
    if e_r + e_p / 100 < best_err:
        best, best_err = X, e_r + e_p / 100
```

---

## 7. 一句话总结

**用一条链串起来：**

```
机械臂摆 N 个姿态，每个姿态拍一张标定板照片、记一次末端位姿
  → 用相机内参 + AprilTag 解 PnP，得到每张图的 T_cam_board
  → 每对 (i, j) 构造 A = T_base_end_j^-1 · T_base_end_i 、 B = T_cam_board_j · T_cam_board_i^-1
  → cv2.calibrateHandEye 解 AX = XB，得到 X = T_end_cam
  → 用 T01·X·T23 应该恒定这件事，反算出自洽误差（度 / 毫米）
  → 存成带自描述字段的 calib_handeye.json
```

**三个最需要记住的点：**

1. **不变量是 T_base_board（标定板相对基座不动）**——整个推导都是从这一句话长出来的。
2. **姿态必须变化，只平移不行**——平移给不出旋转约束，这是手眼标定第一大翻车原因。
3. **看日志里的误差**：旋转 < 1°、平移 < 5 mm 才算合格；1° 的旋转误差在半米外会放大成近 1 厘米的抓取偏差。

**排查清单（结果不对时按顺序查）：**

```
① 误差是不是 > 3° / > 20mm？→ 大概率姿态变化不够，重采数据
② arm_pose.json 里的四元数是不是几乎一样？→ 姿态没变化
③ cam_params.json 是不是这次相机标定出来的？→ 换过相机/分辨率要重标
④ 标定板尺寸 --calib_board_info 填对了吗？→ 用尺子量
⑤ 打开被注释掉的可视化（第 315-320 行），看 tag 认全没有
```

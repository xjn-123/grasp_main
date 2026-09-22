# `create_tmpl_grasp_3d.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、没接触过线性代数、也不知道 ROS 是什么的同学。
> 目标：读完之后，你能逐行看懂这个文件在干什么，明白 `grasp.json` 和 `ready.json` 里每个字段是怎么来的、
> 为什么 3D 版只要采 2 个状态而 2D 版要采 5 个，也知道怎么把它跑起来、出了问题往哪查。
>
> 被分析的源文件：
> `E:\WORK\arm_disorderly_swap\robot_grasp_test\demo\test_tmpl_grasp_3d\create_tmpl_grasp_3d.py`（共 310 行）
>
> 配套必读依赖：
> `core/arm_utils.py`（`compute_axis_aligned_pose`）、`core/arm_wrapper.py`（`ArmWrapper`）、
> `core/cam_ros_utils.py`（`CamNode`）、`core/common_utils/utils.py`（颜色常量、`KeyboardReader`、
> `read_calib_handeye`、`read_rgbd_params`）、`core/vision_utils.py`（`TagMatcher3D`、`depth_mean_filter`）。
>
> **下游使用者**：`test_tmpl_grasp_3d.py`。本脚本产出的 `grasp.json` + `ready.json` 就是它运行时唯一要读的东西。
>
> **姊妹篇**：`demo/test_tmpl_grasp_2d/create_tmpl_grasp_2d_analysis.md`（2D 版的采集脚本）。
> 两篇对照着看，能一眼看出 2D 与 3D 方案的根本差别。

---

## 目录

- [0. 一句话概括](#0-一句话概括)
- [1. 背景知识：3D 采集和 2D 采集差在哪](#1-背景知识3d-采集和-2d-采集差在哪)
  - [1.1 这个脚本在整套系统里的位置：采集端 vs 运行端](#11-这个脚本在整套系统里的位置采集端-vs-运行端)
  - [1.2 为什么 3D 只要 2 个状态，2D 却要 5 个](#12-为什么-3d-只要-2-个状态2d-却要-5-个)
  - [1.3 `T_cam_model` 到底是什么](#13-t_cam_model-到底是什么)
  - [1.4 `a` 键和 `c` 键：让"谁"朝下](#14-a-键和-c-键让谁朝下)
  - [1.5 深度图为什么要连拍 5 帧再取均值](#15-深度图为什么要连拍-5-帧再取均值)
- [2. 整体结构与数据流](#2-整体结构与数据流)
  - [2.1 主循环：按键分派](#21-主循环按键分派)
  - [2.2 数据/文件依赖图](#22-数据文件依赖图)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 文件头说明与导入区（第 1–34 行）](#31-文件头说明与导入区第-1%E2%80%9334-行)
  - [3.2 命令行参数解析（第 48–96 行）](#32-命令行参数解析第-48%E2%80%9396-行)
  - [3.3 相机参数、匹配器与手眼矩阵（第 98–136 行）](#33-相机参数匹配器与手眼矩阵第-98%E2%80%93136-行)
  - [3.4 主循环与姿态调整键（第 149–209 行）](#34-主循环与姿态调整键第-149%E2%80%93209-行)
  - [3.5 `g` 键：保存抓取状态（第 211–247 行）](#35-g-键保存抓取状态第-211%E2%80%93247-行)
  - [3.6 `r` 键：保存准备状态（第 249–301 行）](#36-r-键保存准备状态第-249%E2%80%93301-行)
  - [3.7 收尾清理（第 305–307 行）](#37-收尾清理第-305%E2%80%93307-行)
- [4. Python 基础语法速查](#4-python-基础语法速查)
- [5. 运行方式与输入输出](#5-运行方式与输入输出)
- [6. 核心数学：一步一步算给你看](#6-核心数学一步一步算给你看)
  - [6.1 `c` 键为什么传个 `T_end_cam` 就能让相机朝下](#61-c-键为什么传个-t_end_cam-就能让相机朝下)
  - [6.2 `depth_mean_filter` 的 `obs_ratio=0.5` 在筛什么](#62-depth_mean_filter-的-obs_ratio05-在筛什么)
  - [6.3 两个 json 在下游是怎么被拼起来求解的](#63-两个-json-在下游是怎么被拼起来求解的)
- [7. 这段代码里的坑与改进建议](#7-这段代码里的坑与改进建议)
- [8. 一句话总结](#8-一句话总结)

---

## 0. 一句话概括

> **这是 3D 抓取的"采模板"工具：你用键盘把机械臂摆到 2 个关键姿态——
> "准备位"（相机清楚看到物体、还没抓）和"抓取位"（夹爪刚好夹住物体），各按一个键，
> 脚本就把当时的末端位姿、夹爪开度、以及物体在相机坐标系里的完整 6 自由度位姿存成两个 json。
> 有了这两份"标准答案"，下游脚本就能在物体随便摆放时，自己算出该把手伸到哪。**

---

## 1. 背景知识：3D 采集和 2D 采集差在哪

### 1.1 这个脚本在整套系统里的位置：采集端 vs 运行端

和 2D 版完全同构，只是模板内容不同：

| | 本脚本 `create_tmpl_grasp_3d.py` | 下游 `test_tmpl_grasp_3d.py` |
|---|---|---|
| 什么时候跑 | **离线**，换物体时手动跑一次 | **在线**，真正抓取时自动跑 |
| 谁在动 | **人**按键盘指挥机械臂 | 程序自己算、自己动 |
| 产出 / 输入 | 写出 `grasp.json` + `ready.json` | 读入这两个 json |
| 核心问题 | "抓取时手在哪、准备时物体在相机里长什么样" | "现在物体摆歪了，抓取位姿该跟着怎么变" |

### 1.2 为什么 3D 只要 2 个状态，2D 却要 5 个

**这是 2D 和 3D 方案最根本的分歧，值得花两分钟想明白。**

2D 版用的是普通彩色相机，**没有深度信息**。它只知道"物体在画面里的哪个位置"，
不知道"物体离相机多远"。所以必须额外采 `near / next_near / far / next_far` 四组数据，
靠"手挪了多少米 ↔ 画面挪了多少格"这个**比值**来间接推算距离——
那四组数据本质上是在**标定一把尺子**。

3D 版用的是 **RGB-D 相机**（彩色 + 深度）。深度图直接告诉每个像素离相机多远，
AprilTag 的四个角点有了深度之后，就能用 PnP 直接解出物体在相机坐标系下的**完整位姿** $T_{cam\_model}$
（3 个位置 + 3 个旋转，共 6 个自由度）。

> 既然能直接"量"出物体在哪，就**不需要那把间接的尺子**了。
> 所以 3D 版只需要两个状态：
> - `ready`：物体在相机里的标准位姿（用于求"物体现在相对标准姿势偏了多少"）；
> - `grasp`：抓取那一刻手的标准位姿（用于把上面的偏差"搬"到手上）。

一句话：**2D 在"猜"距离，所以要多采几组来标定；3D 在"量"距离，所以两组就够。**

### 1.3 `T_cam_model` 到底是什么

`T_cam_model` 是一个 4×4 的齐次变换矩阵，读作"**从 model（物体）到 cam（相机）**"的变换：

$$T_{cam\_model} = \begin{bmatrix} R & \mathbf{t} \\ 0 & 1 \end{bmatrix}$$

- $\mathbf{t}$：物体原点在相机坐标系下的**位置**（米），3 个数；
- $R$：物体坐标系相对相机的**旋转**，3×3。

它同时回答了"物体在相机前方多远、偏左还是偏右、朝哪个方向转"。
`TagMatcher3D` 通过 AprilTag 的四个角点（带深度）+ PnP 解出它，见第 271 行。

本脚本把它存进 `ready.json`（第 294 行），这是整个 3D 模板里**最关键的一个字段**。

### 1.4 `a` 键和 `c` 键：让"谁"朝下

两个键都调用 `compute_axis_aligned_pose`，但"要被掰正的那个东西"不一样：

| 按键 | 调用 | 被对齐的轴 | 通俗说法 |
|---|---|---|---|
| `a` | `compute_axis_aligned_pose(T_base_end, base_axis_idx=-3, obj_axis_idx=3)` | 末端的 +Z 轴 | **让夹爪垂直朝下** |
| `c` | `compute_axis_aligned_pose(T_base_end, base_axis_idx=-3, obj_axis_idx=3, T_end_obj=T_end_cam)` | 相机的 +Z 轴 | **让相机垂直朝下** |

差别就在那个 `T_end_obj` 参数：不传时"物体"默认是末端自己（单位矩阵），
传 `T_end_cam` 后"物体"就变成了**相机**——于是掰的就是相机。详见 6.1 节。

为什么要让相机朝下？因为 3D 匹配依赖深度图，
相机正对着桌面看时深度最均匀、AprilTag 最不容易被透视拉变形，识别精度最高。

### 1.5 深度图为什么要连拍 5 帧再取均值

深度相机（尤其是结构光 / ToF）有几个毛病：

- **边缘和反光面会"没数据"**，深度值直接是 0（无效）；
- 单帧噪声不小，同一平面测出来可能抖几个毫米。

所以脚本在 `r` 键里连拍 5 帧（第 256 行 `frames_num=5`），
再用 `depth_mean_filter` 做**逐像素的时间维均值**：
同一个像素，把 5 帧里"有效的那些"求平均，噪声就被压下去了。

> `g` 键（抓取时）**没做这一步**，只取单帧。这是个不一致，见第 7 节问题 4。

---

## 2. 整体结构与数据流

### 2.1 主循环：按键分派

```
读 5 个命令行参数（相机参数 / 手眼标定 / 彩色话题 / 深度话题 / 模板目录）
  ↓
读 RGB-D 参数 → 建 TagMatcher3D
读手眼标定 → T_end_cam
  ↓
连机械臂、初始化 ROS2（同时订阅彩色 + 深度两个话题）、建键盘读取器
  ↓
┌─ while 循环（每轮 sleep 0.05 秒）──────────────┐
│  读键 → 没按就 continue                         │
│  先取一次 T_base_end / gripper_dist / joints    │
│  ├ q → break                                    │
│  ├ , / . → 夹爪 ±1 mm                           │
│  ├ a → 夹爪朝下      c → 相机朝下                │
│  ├ g → 存 grasp.json（+ 两张图）                 │
│  └ r → 存 ready.json（+ 5 帧均值深度 + T_cam_model）│
└─────────────────────────────────────────────────┘
  ↓
destroy_node() + rclpy.shutdown()
```

### 2.2 数据/文件依赖图

```
输入：
  --cam_params_path      ──→ read_rgbd_params ──→ (intrinsic, distortion, depth_scale)
  --calib_handeye_path   ──→ read_calib_handeye ─→ T_end_cam
  --color_img_topic  ┐
  --depth_img_topic  ┴───→ CamNode ─────────────→ 彩色图 + 深度图（同步）
  机械臂硬件 ────────────→ ArmWrapper ──────────→ T_base_end / gripper_dist / joints

输出（都写在 --tmpl_dir 下）：
  grasp-color.png   grasp-depth.png   grasp.json   ← g 键
  ready-color.png   ready-depth.png   ready.json   ← r 键

下游：
  grasp.json + ready.json ──→ test_tmpl_grasp_3d.py
```

---

## 3. 逐段代码精读

### 3.1 文件头说明与导入区（第 1–34 行）

```python
1   """
2   功能说明: 创建3D抓取模板的 ROS 节点,也可以用于发布机械臂状态
3   模板需要保存以下数据:
4   - 1.末端刚好抓取到物体时的状态, 包含机械臂末端位姿和夹爪距离等信息
5   - 2.处于抓取准备阶段时的状态, 包含机械臂末端位姿、夹爪距离、以及物体在相机坐标系中的位姿
6   """
7
8   import logging
...
12  import argparse
13  import os
14  import sys
15  import time
16  import json
17
18  import cv2
19
21  # 导入本工程的模块
23  code_dir = os.path.dirname(os.path.realpath(__file__))
24  root_dir = os.path.normpath(f'{code_dir}/../../../')
25  sys.path.append(root_dir)
26
27  from core.utils import (
28      GREEN, YELLOW, BLUE, RED, RESET,
29      KeyboardReader, read_calib_handeye, read_rgbd_params
30  )
31  from core.arm_wrapper import ArmWrapper
32  from core.arm_utils import compute_axis_aligned_pose
33  from core.cam_ros_utils import CamNode
34  from core.vision_utils import TagMatcher3D, depth_mean_filter
```

**业务作用**：把工具箱搬进来。和 2D 版几乎一样，区别只在于导入的是 **3D 版**的匹配器和参数读取函数。

**逐行讲解：**

- 第 3-5 行：docstring，明确写了只存**两类**数据（对比 2D 版 docstring 里的 5 类）。
- 第 23-25 行：把工程根塞进 `sys.path`（`demo/test_tmpl_grasp_3d` 往上三级），
  否则 `from core.xxx import ...` 找不到模块。
- 第 27-30 行：从 `core.utils` 导入 5 个**颜色常量**（终端彩色输出，如 `RED` `\033[31m` 之类）、
  `KeyboardReader`（非阻塞读键）、`read_calib_handeye`（读手眼标定）、`read_rgbd_params`（读 RGB-D 相机参数）。
- 第 34 行：`TagMatcher3D`（3D 匹配器，能解出 $T_{cam\_tag}$）和 `depth_mean_filter`（多帧深度均值）。

> **对比 2D 版**：2D 导入的是 `TagMatcher2D` 和 `read_cam_params`（只要内参 + 畸变）；
> 3D 导入的是 `TagMatcher3D` 和 `read_rgbd_params`（内参 + 畸变 + **`depth_scale`**）。
> `depth_scale` 是把深度图原始数值换成"米"的除数，3D 必需。

### 3.2 命令行参数解析（第 48–96 行）

```python
48  if __name__ == '__main__':
49
50      parser = argparse.ArgumentParser()
51
52      parser.add_argument("--cam_params_path", type=str, required=True,
53                          help="相机参数文件的路径, 包含内参和畸变参数")
54
55      parser.add_argument("--calib_handeye_path", type=str, required=True,
56                          help="手眼标定文件的路径, 包含相机与机械臂的位姿关系")
...
58      parser.add_argument("--color_img_topic", type=str, required=True, ...)
61      parser.add_argument("--depth_img_topic", type=str, required=True, ...)
64      parser.add_argument("--tmpl_dir", type=str, required=True, ...)
65
67      args = parser.parse_args()
```

**业务作用**：收 5 个必填参数。

**逐行讲解：**

- 第 55-56 行 **`--calib_handeye_path` 是 3D 版多出来的**（2D 版没有）。
  因为 3D 要把"相机看到的物体位姿"换算到"机械臂坐标系"下，
  必须知道相机和末端之间的固定关系 $T_{end\_cam}$（手眼标定的结果）。
- 第 61-62 行 `--depth_img_topic`：深度图话题。**3D 版必须同时收彩色和深度两个话题**，
  第 129 行把它们一起交给 `CamNode`。
- 第 73-88 行：三个 `if xxx is None` 检查，和 2D 版一样是**死代码**（`required=True` 已保证非 None）。

### 3.3 相机参数、匹配器与手眼矩阵（第 98–136 行）

```python
98      # 读取相机参数
99      intrinsic, distortion, depth_scale = read_rgbd_params(cam_params_path)
100     if intrinsic is None or distortion is None or depth_scale is None:
101         logging.error('read camera parameters failed, exiting')
102         exit(1)
103     # end if
104
105     config = TagMatcher3D.Config(
106         intrinsic=intrinsic,
107         distortion=distortion,
108         depth_scale=depth_scale
109     )
110     matcher = TagMatcher3D(config)
111
112     # 读取手眼标定矩阵
113     print()
114     T_end_cam, _ = read_calib_handeye(calib_handeye_path)
115     if T_end_cam is None:
116         logging.error('read handeye calib failed, exiting')
117         exit(1)
118     # end if
```

**业务作用**：准备"看懂 3D 世界"的三样东西：内参、匹配器、手眼关系。

**逐行讲解：**

- 第 99 行：`read_rgbd_params` 返回**三个**值（2D 版的 `read_cam_params` 返回两个）。
  多出来的 `depth_scale` 是深度图的缩放因子——深度图通常存成 16 位整数，
  真实米数 = 原始值 / `depth_scale`（常见值是 1000，即毫米）。
- 第 100-103 行：三个值**任意一个为 None 就退出**。这个检查比 2D 版严谨 ✓
- 第 105-110 行：`TagMatcher3D.Config` 比 2D 版多一个 `depth_scale` 字段。
- 第 114 行 `T_end_cam, _ = read_calib_handeye(...)`：返回两个值，
  用 `_` 接住第二个**表示"我故意不用它"**（Python 的惯用写法）。
  `T_end_cam` = 相机相对末端的位姿，是后面 `c` 键和下游求解的关键。

### 3.4 主循环与姿态调整键（第 149–209 行）

```python
149     while rclpy.ok():
150
151         time.sleep(0.05)  # 避免占用过多 CPU
152
153         key = keyboard_reader.read_key()
154         if key is None:
155             continue
156         # end if
157
159         if key == 'q':  # 退出
161             logging.info('Quit.')
162             break
163         # end if
164
165         T_base_end = arm.get_pose()  # 获取机械臂末端位姿
166         gripper_dist = arm.get_gripper_dist()  # 获取夹爪距离
167         joints = arm.get_joints()  # 获取机械臂关节角度
168
170         gripper_step = 0.001  # 夹爪每次移动的步长, 单位: m
171         if key == ',':  # 缩小夹爪
172             set_dist = gripper_dist - gripper_step
173             arm.set_gripper_dist(set_dist)
...
175         elif key == '.':  # 放大夹爪
...
182         if key == 'a':
183             target_T_base_end = compute_axis_aligned_pose(T_base_end, base_axis_idx=-3, obj_axis_idx=3)
...
197         if key == 'c':
198             target_T_base_end = compute_axis_aligned_pose(T_base_end, base_axis_idx=-3, obj_axis_idx=3, T_end_obj=T_end_cam)
```

**业务作用**：轮询键盘，按不同键做不同事。

**逐行讲解：**

- 第 151 行 `time.sleep(0.05)`：**每轮开头就睡 50 毫秒**，即每秒最多轮询 20 次。
  注意和 2D 版的差别：2D 版是"**读不到键才睡 0.03**"，3D 版是"**每轮都睡 0.05**"。
  效果差不多，但 3D 版在读键前先睡，会多一点点延迟。
- 第 165-167 行：**每轮都重新读一次**机械臂状态。
  `joints` 读出来**后面没用到**（死变量，见第 7 节问题 6）。
- 第 171-178 行：夹爪 ±1 mm。**提示里写的是 `<` / `>`，代码判断 `,` / `.`**（同 2D 版的 bug）。
- 第 182-194 行 `a` 键：让**夹爪**朝下。
- 第 197-209 行 `c` 键：让**相机**朝下，多传了 `T_end_obj=T_end_cam`（原理见 6.1 节）。

### 3.5 `g` 键：保存抓取状态（第 211–247 行）

```python
212         if key == 'g':
215             logging.info(f"{GREEN}task: save grasp data {RESET}")
216
217             # 获取 RGB-D 图像
218             frames = cam_node.get_frames(do_spin_once=True)
219             if frames is None:
220                 logging.warning('No RGB-D frame available yet.')
221                 continue
222             # end if
223
224             color_img, depth_img = frames[0]
...
231             rgb_path = os.path.join(tmpl_dir, f'grasp-color.png')
232             depth_path = os.path.join(tmpl_dir, f'grasp-depth.png')
233
234             cv2.imwrite(rgb_path, color_img)
235             cv2.imwrite(depth_path, depth_img)
236             logging.info(f'Saved color images to: {rgb_path}')
237
238             # 保存机械臂状态
239             data_dict = {
240                 "T_base_end": T_base_end.tolist(),
241                 "gripper_dist": gripper_dist
242             }
243             file_path = os.path.join(tmpl_dir, f'grasp.json')
244             with open(file_path, 'w') as f:
245                 json.dump(data_dict, f, indent=4)
```

**业务作用**：把"夹爪刚好夹住物体"这一刻的**手部状态**存下来。

**逐行讲解：**

- 第 218 行 `get_frames(do_spin_once=True)`：默认 `frames_num=1`，只取 1 帧。
- 第 224 行 `color_img, depth_img = frames[0]`：`frames[0]` 是第一帧，
  它本身是一个列表 `[彩色图, 深度图]`，**顺序与第 129 行传给 `CamNode` 的话题顺序一致**。
  这个隐式依赖有点脆弱（见第 7 节问题 5）。
- 第 234-235 行：`cv2.imwrite` 存图。
  **注意这里没有做 RGB→BGR 转换**（2D 版做了），如果 ROS 给的是 RGB 图，存出来的颜色会偏（见第 7 节问题 3）。
- 第 239-242 行：`grasp.json` 只有两个字段——
  `T_base_end`（手在哪）和 `gripper_dist`（夹多紧）。**没有物体位姿**，
  因为抓取那一刻夹爪贴着物体，tag 往往被挡住，识别不可靠（和 2D 版同理）。
- 第 240 行 `.tolist()`：numpy → list，给 `json.dump` 让路。

### 3.6 `r` 键：保存准备状态（第 249–301 行）

```python
250         if key == 'r':
253             logging.info(f"{GREEN}task: save ready data {RESET}")
254
256             frames = cam_node.get_frames(do_spin_once=True, frames_num=5)
257             if frames is None:
258                 logging.warning('No RGB-D frame available yet.')
259                 continue
260             # end if
261
262             color_img, depth_img = frames[0]
...
268             # 获取匹配结果
269             depth_img_list = [frame[1] for frame in frames]  # 获取所有帧的深度图像列表
270             depth_img = depth_mean_filter(depth_img_list, obs_ratio=0.5)
271             result_list, msg = matcher.match(color_img, depth_img, top_k=1)
272             if len(result_list) == 0:
273                 logging.warning(f'3D match failed, {msg}')
274                 continue
275             # end if
276             T_cam_model = result_list[0].T_cam_tag
277             if T_cam_model is None:
278                 logging.warning('3D match failed.')
279                 continue
280             # end if
281
284             rgb_path = os.path.join(tmpl_dir, f'ready-color.png')
285             depth_path = os.path.join(tmpl_dir, f'ready-depth.png')
286
287             cv2.imwrite(rgb_path, color_img)
288             cv2.imwrite(depth_path, depth_img)
289
292             data_dict = {
293                 "T_base_end": T_base_end.tolist(),
294                 "T_cam_model": T_cam_model.tolist(),
295                 "gripper_dist": gripper_dist
296             }
297             file_path = os.path.join(tmpl_dir, f'ready.json')
```

**业务作用**：把"相机清楚看到物体、手还没抓"这个标准观察位姿存下来。**这是 3D 模板的核心。**

**逐行讲解：**

- 第 256 行 `frames_num=5`：连拍 5 帧（对比 `g` 键只拍 1 帧）。
- 第 269 行 `[frame[1] for frame in frames]`：列表推导式，
  从 5 帧里各取出深度图（`[1]` 号元素），组成一个新列表。
- 第 270 行 `depth_mean_filter(depth_img_list, obs_ratio=0.5)`：
  逐像素求时间维平均。**返回值覆盖了原来的 `depth_img`**——
  所以第 288 行存的是**滤波后**的深度图（第 287 行存的彩色图仍是第 262 行的第一帧）。
- 第 271 行 `matcher.match(color_img, depth_img, top_k=1)`：3D 版多一个 `depth_img` 参数。
  返回 `(结果列表, 消息)`。
- 第 272-275 行：没识别到就 `continue`，**不写文件**。
  和 2D 版一样，失败时只是打一行黄字，很容易被忽略。
- 第 276 行 `result_list[0].T_cam_tag`：**注意字段名是 `T_cam_tag`**（tag 的位姿），
  但存进 json 时改名成了 `T_cam_model`（第 294 行）。
  含义是"物体（用 tag 代表）相对相机的位姿"。
- 第 292-296 行：`ready.json` 有**三个**字段，比 `grasp.json` 多了 `T_cam_model`。

> **`r` 键必须成功识别到 tag**，否则 `ready.json` 根本不会生成。
> 这是采集时最容易失败的一步：tag 太小、太远、反光、被手挡住都会失败。

### 3.7 收尾清理（第 305–307 行）

```python
305     cam_node.destroy_node()
306     rclpy.shutdown()
307     logging.info('shutdown')
```

**业务作用**：退出 `while` 之后销毁 ROS2 节点、关闭 ROS2。

> 这一段**比 2D 版做得好** ✓ 2D 版 `break` 之后直接结束，没有清理（见 2D 版文档第 7 节问题 4）。

---

## 4. Python 基础语法速查

| 写法 | 含义 | 本文件出现位置 |
|---|---|---|
| `a, b, c = f()` | 一次接住多个返回值（元组解包） | 第 99 行 |
| `T_end_cam, _ = f()` | `_` 表示"这个返回值我不用" | 第 114 行 |
| `[x[1] for x in frames]` | 列表推导式，一行生成新列表 | 第 269 行 |
| `if x is None:` | 判断是否为 `None`（**不是** `== None`） | 第 100、115 行 |
| `exit(1)` | 立即终止程序，1 表示异常退出 | 第 102、117 行 |
| `np.ndarray.tolist()` | numpy → Python list（为了 `json` 能存） | 第 240、294 行 |
| `with open(p, 'w') as f:` | 写文件，用完自动关 | 第 244、298 行 |
| `indent=4` | `json.dump` 格式化缩进，方便人看 | 第 245、299 行 |
| `f'{v:.3f}'` | f-string 里限定小数位数（3 位） | 第 174、228 行 |
| `while rclpy.ok():` | ROS2 还活着就一直循环 | 第 149 行 |

---

## 5. 运行方式与输入输出

**运行命令**（5 个参数都必填）：

```bash
python demo/test_tmpl_grasp_3d/create_tmpl_grasp_3d.py \
    --cam_params_path     config/cam_params.json \
    --calib_handeye_path  config/calib_handeye.json \
    --color_img_topic     /camera/color/image_raw \
    --depth_img_topic     /camera/depth/image_raw \
    --tmpl_dir            demo/test_tmpl_grasp_3d/tmpl/box_01
```

**运行后你会看到**：

```
RGB-D camera parameters file: config/cam_params.json
handeye calib file: config/calib_handeye.json
color image topic: /camera/color/image_raw
depth image topic: /camera/depth/image_raw
grasp template will be saved to: demo/test_tmpl_grasp_3d/tmpl/box_01

use keyboard to control:
  q: 退出程序
  <: 缩小夹爪距离
  >: 放大夹爪距离
  a: 使末端的 z 轴方向与基座的 -z 轴平行
  c: 使相机的 z 轴方向与基座的 -z 轴平行
  g: 保存抓取时的状态
  r: 保存准备阶段的状态
```

**推荐的采集顺序**：

1. 按 `c` 让**相机**垂直朝下（保证深度图质量最好）。
2. 用 `,`/`.` 张开夹爪到足够大。
3. 把机械臂移到能清楚看到物体、但**还没抓**的位置 → 按 `r`。
   **务必看日志里有没有打印 `T_cam_model`**，没打印就是识别失败，这一步没存上。
4. 手动把机械臂移到真正夹住物体的位姿，合拢夹爪 → 按 `g`。
5. 按 `q` 退出。

**产出目录**：

```
tmpl/box_01/
├── grasp-color.png   grasp-depth.png   grasp.json   ← { T_base_end, gripper_dist }
└── ready-color.png   ready-depth.png   ready.json   ← { T_base_end, T_cam_model, gripper_dist }
```

验收方法：
- 打开 `ready-color.png`，确认能清楚看到物体上的 tag；
- 打开 `grasp-color.png`，确认夹爪确实夹住了物体；
- 用文本编辑器打开两个 json，确认 `grasp.json` 有 2 个字段、`ready.json` 有 3 个字段。

---

## 6. 核心数学：一步一步算给你看

### 6.1 `c` 键为什么传个 `T_end_cam` 就能让相机朝下

`compute_axis_aligned_pose` 的内部逻辑是（见 `core/arm_utils/arm_utils.py` 第 471-528 行）：

1. 先算"物体"在基座下的位姿：$T_{base\_obj} = T_{base\_end} \cdot T_{end\_obj}$
2. 取出 $T_{base\_obj}$ 的第 `obj_axis_idx` 列，作为**当前方向** $\mathbf{c}$
3. 目标方向 $\mathbf{t}$ 由 `base_axis_idx` 决定（`=-3` → $[0,0,-1]^{\top}$，竖直向下）
4. 算夹角、构造绕 $\mathbf{c} \times \mathbf{t}$ 的旋转 $\Delta R$，得到新位姿

关键在于第 1 步的 $T_{end\_obj}$：

- **`a` 键不传** → $T_{end\_obj} = I$（单位矩阵），于是 $T_{base\_obj} = T_{base\_end}$，
  被掰的轴就是**末端自己的 +Z 轴**。
- **`c` 键传 `T_end_cam`** → $T_{end\_obj} = T_{end\_cam}$，于是
  $T_{base\_obj} = T_{base\_end} \cdot T_{end\_cam} = T_{base\_cam}$，
  被掰的轴就变成了**相机的 +Z 轴**。

一句话：**`T_end_obj` 这个参数决定了"你到底想让哪个东西的轴朝下"**，
传什么，就把什么当成"物体"来掰。

### 6.2 `depth_mean_filter` 的 `obs_ratio=0.5` 在筛什么

对每个像素，函数做三件事（见 `core/vision_utils/vision_utils.py` 第 100-142 行）：

1. 统计该像素在 5 帧里**深度 > 0 的次数** $n$（深度 0 表示"这一帧没测到"）；
2. 有值的那些求平均：$\bar{d} = (\sum d_i)/n$；
3. **如果 $n < 0.5 \times 5 = 2.5$（即 $n \le 2$），这个像素直接置 0**。

也就是说：**5 帧里至少要有 3 帧测到，才认为这个像素可信**；
只测到 1-2 帧的像素被判为"噪声/边缘"，宁可留空也不要。

这是很实用的策略——深度图边缘（物体轮廓、反光面）最容易出现零星的错误值，
直接丢掉比取平均更安全。

### 6.3 两个 json 在下游是怎么被拼起来求解的

下游 `test_tmpl_grasp_3d.py` 拿到这两个 json 后，核心思路是：
**物体放在桌上不动，所以"物体相对基座"是个常量**。

采集时（`r` 键那一刻），物体相对基座为：

$$T_{base\_model} = T_{base\_end}^{rdy} \cdot T_{end\_cam} \cdot T_{cam\_model}^{rdy}$$

运行时（物体被挪到别处，手在当前位置）：

$$T_{base\_model} = T_{base\_end}^{tgt} \cdot T_{end\_cam} \cdot T_{cam\_model}^{cur}$$

两式相等（同一个常量），解出目标手部位姿：

$$T_{base\_end}^{tgt} = T_{base\_end}^{cur} \cdot T_{end\_cam} \cdot T_{cam\_model}^{cur} \cdot (T_{cam\_model}^{rdy})^{-1} \cdot (T_{end\_cam})^{-1}$$

其中：
- $T_{cam\_model}^{rdy}$ —— 来自 **`ready.json`**（本脚本采的）；
- $T_{base\_end}^{cur}$、$T_{cam\_model}^{cur}$ —— 运行时实时测的；
- $T_{end\_cam}$ —— 手眼标定给的（本脚本用 `--calib_handeye_path` 读进来，用于 `c` 键）。

最后再叠上 `grasp.json` 里那一小段"从 ready 位到 grasp 位"的相对位移，
就得到最终的抓取位姿。

> 所以 **`ready.json` 里的 $T_{cam\_model}$ 是整个链条的锚点**——
> 它采歪了，后面每一步都跟着歪。这也是为什么 `r` 键必须看着日志确认识别成功。

---

## 7. 这段代码里的坑与改进建议

| # | 问题 | 后果 | 建议 |
|---|---|---|---|
| 1 | **按键提示与判断不一致**：提示写 `<` / `>`（第 141-142 行），代码判断 `,` / `.`（第 171、175 行） | **最严重**。照提示按 `<` 完全没反应，用户以为程序卡死 | 提示改成 `,` / `.`，或判断改成 `key in (',', '<')` |
| 2 | 第 27 行 `from core.utils import ...`，但该文件现在在 `core/common_utils/utils.py` | **直接 `ModuleNotFoundError`，脚本起不来** | 改成 `from core.common_utils.utils import ...` |
| 3 | 第 234-235 行 `cv2.imwrite` **没做 RGB→BGR 转换**（2D 版做了） | 若 ROS 给的是 RGB，存出的 png 红蓝通道反了；**更麻烦的是 `depth_img` 也可能被误当彩色处理** | 确认 `CamNode` 返回的颜色顺序，必要时 `cv2.cvtColor(..., cv2.COLOR_RGB2BGR)` |
| 4 | `g` 键取**单帧**深度（第 218 行），`r` 键取 **5 帧均值**（第 256 行） | 两处深度图质量不一致；`grasp-depth.png` 噪声更大 | 统一成 `frames_num=5` + `depth_mean_filter` |
| 5 | 第 224 行 `color_img, depth_img = frames[0]` 隐式依赖话题顺序 | 若第 129 行话题顺序调换，这里会静默把深度图当彩色图 | 让 `CamNode` 返回 dict，按名字取图 |
| 6 | 第 167 行 `joints = arm.get_joints()` 读出来**从未使用** | 无害但是死代码，还多花一次通信 | 删掉，或把它也存进 json（方便复现姿态） |
| 7 | 识别失败只打 `warning` 后 `continue`（第 272-275、277-280 行） | 采了 `r` 但实际没生成 `ready.json`，下游才报错 | 返回成败标志，主循环累计成功数并在退出前汇总打印 |
| 8 | 第 73-88 行三个 `if xxx is None` 是死代码 | 无害但误导 | 删掉或改成检查空字符串 |
| 9 | `g` / `r` 两个分支大量重复（取图、存图、写 json） | 改一处要改两处 | 抽成一个 `save_state(save_dir, prefix, need_match: bool)` 函数 |
| 10 | 没有校验两个 json 是否都生成 | 只采了一个就退出，下游照样跑然后莫名失败 | 退出前检查 `grasp.json` / `ready.json` 是否齐全 |

---

## 8. 一句话总结

> **本脚本是 3D 抓取的"标准照拍摄器"：按 `r` 存下"准备时物体在相机里的完整 6 自由度位姿"，
> 按 `g` 存下"抓取时手在哪、夹爪多紧"。因为深度相机能直接量出物体位置，
> 3D 只需要这 2 个状态，不必像 2D 那样额外采 4 组来标定"画面位移 ↔ 手臂位移"的换算尺子。
> `ready.json` 里的 $T_{cam\_model}$ 是整条求解链的锚点，采集时务必确认日志里真的打印了它。**

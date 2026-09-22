# `create_tmpl_grasp_2d.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、没接触过线性代数、也不知道 ROS 是什么的同学。
> 目标：读完之后，你能逐行看懂这个文件在干什么，明白每个 `state.json` 是怎么被"按一下键"存下来的，
> 也知道怎么把它跑起来、五个模板目录分别该在什么时候按、出了问题往哪查。
>
> 被分析的源文件：
> `E:\WORK\arm_disorderly_swap\robot_grasp_test\demo\test_tmpl_grasp_2d\create_tmpl_grasp_2d.py`（共 281 行）
>
> 配套必读依赖：
> `core/arm_utils.py`（`compute_axis_aligned_pose`）、`core/arm_wrapper.py`（`ArmWrapper`）、
> `core/cam_ros_utils.py`（`CamNode`）、`core/common_utils/utils.py`（颜色常量、`KeyboardReader`、`read_cam_params`）、
> `core/vision_utils.py`（`TagMatcher2D`）。
>
> **下游使用者**：`test_tmpl_grasp_2d.py`。本脚本是"**采模板**"那一半，那个脚本是"**用模板**"那一半——
> 本脚本存下的 `state.json` 就是它运行时唯一要读的东西。两边必须成对使用。

---

## 目录

- [0. 一句话概括](#0-一句话概括)
- [1. 背景知识：采集模板到底在采什么](#1-背景知识采集模板到底在采什么)
  - [1.1 这个脚本在整套系统里的位置：采集端 vs 运行端](#11-这个脚本在整套系统里的位置采集端-vs-运行端)
  - [1.2 为什么要采 5 个状态，而不是 1 个](#12-为什么要采-5-个状态而不是-1-个)
  - [1.3 `obj_pose_2d` 里的三个数是什么](#13-obj_pose_2d-里的三个数是什么)
  - [1.4 为什么用键盘交互，而不用 `input()`](#14-为什么用键盘交互而不用-input)
- [2. 整体结构与数据流](#2-整体结构与数据流)
  - [2.1 主循环：一个"按键分派"的状态机](#21-主循环一个按键分派的状态机)
  - [2.2 数据/文件依赖图](#22-数据文件依赖图)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 文件头说明与导入区（第 1–34 行）](#31-文件头说明与导入区第-1%E2%80%9334-行)
  - [3.2 `save_state`（第 42–97 行）](#32-save_state第-42%E2%80%9397-行)
  - [3.3 命令行参数解析（第 102–138 行）](#33-命令行参数解析第-102%E2%80%93138-行)
  - [3.4 硬件与匹配器初始化（第 140–179 行）](#34-硬件与匹配器初始化第-140%E2%80%93179-行)
  - [3.5 主循环与按键分派（第 181–280 行）](#35-主循环与按键分派第-181%E2%80%93280-行)
- [4. Python 基础语法速查](#4-python-基础语法速查)
- [5. 运行方式与输入输出](#5-运行方式与输入输出)
- [6. 核心数学：一步一步算给你看](#6-核心数学一步一步算给你看)
  - [6.1 `a` 键的"轴对齐"到底在算什么](#61-a-键的轴对齐到底在算什么)
  - [6.2 为什么 `near` 和 `next_near` 必须只差"一小段平移"](#62-为什么-near-和-next_near-必须只差一小段平移)
  - [6.3 数值演练：一对 near 数据怎么算出比值](#63-数值演练一对-near-数据怎么算出比值)
- [7. 这段代码里的坑与改进建议](#7-这段代码里的坑与改进建议)
- [8. 一句话总结](#8-一句话总结)

---

## 0. 一句话概括

> **这是一个"手把手教机械臂记住抓取姿势"的采集工具：你用键盘把机械臂摆到 5 个特定姿态——
> 刚抓到物体、相机离得近、近处再平移一小段、相机离得远、远处再平移一小段——每摆好一个就按一个键，
> 脚本立刻把"末端当时在哪、夹爪开多大、物体在画面里的哪个位置和朝哪个方向"写进一个 `state.json`。
> 这 5 个文件就是后面自动抓取时唯一的"标准参考答案"。**

---

## 1. 背景知识：采集模板到底在采什么

### 1.1 这个脚本在整套系统里的位置：采集端 vs 运行端

整套 2D 抓取分成**两半**，本脚本负责前一半：

| | 本脚本 `create_tmpl_grasp_2d.py` | 下游 `test_tmpl_grasp_2d.py` |
|---|---|---|
| 什么时候跑 | **离线**，换物体 / 换批次时手动跑一次 | **在线**，真正抓取时自动跑 |
| 谁在动 | **人**按键盘指挥机械臂 | 程序自己算、自己动 |
| 产出 / 输入 | 写出 5 个 `state.json` + 图片 | 读入那 5 个 `state.json` |
| 核心问题 | "标准姿势长什么样" | "当前离标准还差多少、该往哪挪" |

打个比方：本脚本是**拍标准照**（正面照、左侧照、右侧照），下游脚本是**拿现在的照片和标准照比对**。
标准照拍歪了，后面怎么比都是歪的——所以别嫌这一步麻烦。

### 1.2 为什么要采 5 个状态，而不是 1 个

只采"抓取那一刻"是不够的，因为运行时机械臂**不是一步跳到抓取位姿**，而是从远处一点点挪过去。
在挪动过程中，它得知道"我现在离物体多远、这个距离下画面动一格等于手臂动多少米"。

这 5 个状态的分工是：

| 子目录 | 按键 | 含义 | 存了什么 |
|---|---|---|---|
| `grasp` | `g` | 手臂**刚好抓到**物体那一刻 | `T_base_end`、`gripper_dist` |
| `near` | `n` | 相机离物体**较近**时的一个状态 | 上面两个 + `obj_pose_2d` |
| `next_near` | `b` | 在 `near` 基础上，末端在桌面 xy 方向**平移一小段** | 同上 |
| `far` | `f` | 相机离物体**较远**时的一个状态 | 同上 |
| `next_far` | `d` | 在 `far` 基础上，末端在桌面 xy 方向**平移一小段** | 同上 |

- `grasp` 是**终点**：手臂要到的地方、夹爪要合到多紧。
- `near` / `next_near` 是一对，**近处**的"画面位移 ↔ 手臂位移"换算关系。
- `far` / `next_far` 是另一对，**远处**的换算关系。

> 为什么每对要**两个**状态？因为要算比值，必须有"移动前"和"移动后"两个快照。
> 只有一个状态就只能知道"物体在哪"，无法知道"手挪一米，物体在画面里挪多少"。详见 6.2 节。

### 1.3 `obj_pose_2d` 里的三个数是什么

`obj_pose_2d` 是一个长度为 3 的列表 `[nx, ny, theta]`：

- `nx, ny`：**归一化坐标**（normalized coordinates），不是像素。
  由像素坐标 $(u, v)$ 经过内参矩阵换算得到：
  $n_x = (u - c_x)/f_x,\quad n_y = (v - c_y)/f_y$
  好处是**换相机、换分辨率都不影响数值**，所以模板可以跨设备复用。
- `theta`：物体在**图像平面里的朝向**（弧度），由 AprilTag 的角点方向算出来。

为什么不用像素？因为像素值里混进了焦距 $f_x, f_y$，同一个物体换个镜头数值就全变了；
归一化之后，"物体在视野中心右边 0.1" 这句话在任何相机下都是同一个意思。

### 1.4 为什么用键盘交互，而不用 `input()`

因为采集过程中**相机画面必须一直在跑**（ROS2 节点要持续 `spin` 收图），
而 `input()` 会**阻塞整个线程**——你一敲回车等输入，画面就冻住了，脚本也拿不到最新图像。

所以代码用 `KeyboardReader`（内部是 `select.select` 轮询 stdin）：
- **不阻塞**：没按键时立刻返回 `None`，主循环继续转，ROS2 继续收图。
- 主循环里 `time.sleep(0.03)`，也就是**每秒轮询约 33 次**，够灵敏又不烧 CPU。

---

## 2. 整体结构与数据流

### 2.1 主循环：一个"按键分派"的状态机

整个脚本没有类，结构非常直白：

```
读命令行参数
  ↓
读相机内参 → 建 TagMatcher2D
  ↓
连机械臂（ArmWrapper）、初始化 ROS2 相机节点（CamNode）、建键盘读取器
  ↓
┌─ while 循环（每秒约 33 次）────────────────┐
│  读一个按键 key                             │
│  ├ 没按      → sleep 0.03，继续             │
│  ├ q         → break，退出                  │
│  ├ a         → 把末端掰成竖直朝下           │
│  ├ , / .     → 夹爪缩小 / 放大 1 mm         │
│  └ g/n/b/f/d → 存一个 state.json            │
└─────────────────────────────────────────────┘
```

### 2.2 数据/文件依赖图

```
输入：
  --cam_params_path   相机内参 json ──→ read_cam_params ──→ TagMatcher2D.Config
  --color_img_topic   ROS2 话题名 ────→ CamNode ──────────→ 实时 RGB 图像
  机械臂硬件 ─────────────────────────→ ArmWrapper ──────→ T_base_end / gripper_dist

输出（写到 --tmpl_dir 下）：
  grasp/state.json     +  grasp/color.png
  near/state.json      +  near/color.png  +  near/tag.png
  next_near/state.json +  ...
  far/state.json       +  ...
  next_far/state.json  +  ...

下游：
  这 5 个 state.json ──→ test_tmpl_grasp_2d.py 的 read_tmpl_grasp_2d()
```

---

## 3. 逐段代码精读

### 3.1 文件头说明与导入区（第 1–34 行）

```python
1   """
2   功能: 创建基于 AprilTag2 的 2D 抓取模板数据, ...
10  """
11
12  import argparse
13  import json
14  import logging
15  import os
16  import sys
17  import time
18
19  import cv2
20  import rclpy
21
24  # 导入本工程的模块
26  code_dir = os.path.dirname(os.path.realpath(__file__))
27  root_dir = os.path.normpath(f"{code_dir}/../../../")
28  sys.path.append(root_dir)
29
30  from core.arm_utils import compute_axis_aligned_pose
31  from core.arm_wrapper import ArmWrapper
32  from core.cam_ros_utils import CamNode
33  from core.utils import BLUE, GREEN, RED, RESET, YELLOW, KeyboardReader, read_cam_params
34  from core.vision_utils import TagMatcher2D
```

**业务作用**：把脚本跑起来需要的"工具箱"全部搬进来。

**逐行讲解：**

- 第 2-9 行：docstring，把 5 个状态要存什么写清楚了。注意这里写的是"基于 **AprilTag2**"，
  但工程现在的 AprilTag 能力已经换成 `pyapriltags`（AprilTag 3），注释没跟着更新。
- 第 12-17 行：Python 标准库。`argparse` 解析命令行参数、`json` 读写模板、
  `logging` 打日志、`os` 拼路径、`sys` 改模块搜索路径、`time` 做延时。
- 第 19-20 行：第三方库。`cv2` 是 OpenCV（存图、转颜色空间），`rclpy` 是 ROS2 的 Python 接口。
- 第 26-28 行：**把工程根目录塞进 `sys.path`**。
  `code_dir` 是本文件所在目录（`demo/test_tmpl_grasp_2d`），
  `/../../../` 往上三级就是工程根 `robot_grasp_test`。
  加了这一行，下面 `from core.xxx import yyy` 才找得到模块——
  否则直接 `python create_tmpl_grasp_2d.py` 会报 `ModuleNotFoundError: No module named 'core'`。
- 第 30-34 行：工程自己的模块。注意第 33 行导入的是 `core.utils`，
  而工程里这个文件现在在 **`core/common_utils/utils.py`**（详见第 7 节问题 5）。

### 3.2 `save_state`（第 42–97 行）

```python
42  def save_state(
43      cam_node: CamNode, arm: ArmWrapper, save_dir: str, matcher: TagMatcher2D = None
44  ):
52      """
53      保存当前状态为模板文件
...
58      """
59
60      # 获取图像
61      imgs = cam_node.get_frames(do_spin_once=True)
62      if imgs is None:
63          logging.error(f"{RED}failed to get images, skip saving tmpl{RESET}")
64          return
65      # end if
66      rgb_img = imgs[0][0]  # 取第一帧第一摄像头图像
67
68      # 获取机械臂末端位姿
69      T_base_end = arm.get_pose()
70
71      # 获取夹爪距离
72      gripper_dist = arm.get_gripper_dist()
73
74      data_dict = {"T_base_end": T_base_end.tolist(), "gripper_dist": gripper_dist}
75
76      if matcher is not None:  # 获取物体位置
77          result_list, msg = matcher.match(rgb_img, top_k=1)
78          if len(result_list) == 0:
79              logging.warning(f"{YELLOW}no match found, skip saving tmpl, {msg}{RESET}")
80              return
81          # end if
82
83          pose_2d = result_list[0].pose_2d
84          data_dict["obj_pose_2d"] = [float(x) for x in pose_2d]
85
86          bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
87          matcher.draw(bgr_img, result_list)  # 绘制匹配结果
88          img_path = os.path.join(save_dir, "tag.png")
89          cv2.imwrite(img_path, bgr_img)
90      # end if
91
92      # 保存模板数据
93      img_path = os.path.join(save_dir, "color.png")
94      cv2.imwrite(img_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
...
97      file_path = os.path.join(save_dir, f"state.json")
98      with open(file_path, "w") as f:
99          json.dump(data_dict, f, indent=4)
```

**业务作用**：**这个函数是整个脚本的核心**——抓一张当前图像、问机械臂要当前位姿和夹爪开度、
（可选地）让 `TagMatcher2D` 找出物体在画面里的位置，最后把这些一起写进 `state.json`。

**逐行讲解：**

- 第 43 行 `matcher: TagMatcher2D = None`：**默认参数**。
  调用时不传 `matcher`，它就是 `None`——正好对应 `grasp` 状态（第 237 行不传）。
- 第 61 行 `cam_node.get_frames(do_spin_once=True)`：向 ROS2 要一帧图。
  `do_spin_once=True` 表示"顺便转一次 ROS2 回调"，这样订阅器才有机会把新消息收进来。
- 第 62-65 行：拿不到图就**打印红色错误并 `return`**，不写文件。
  注意这里是**静默跳过**——采集时如果只瞟一眼日志，很容易漏掉"这个状态其实没存上"。
- 第 66 行 `imgs[0][0]`：`get_frames` 返回的是 `List[List[np.ndarray]]`
  （外层是"第几帧"，内层是"第几个摄像头"），`[0][0]` = 第一帧、第一个摄像头的图。
- 第 69 / 72 行：向机械臂要**当前末端位姿**（4×4 矩阵）和**夹爪开度**（一个数，单位米）。
- 第 74 行 `T_base_end.tolist()`：numpy 数组 → Python 原生 list，
  **因为 `json.dump` 不认识 numpy 类型**，必须转。这是很常见的踩坑点。
- 第 76-90 行：只有传了 `matcher` 才走。
  - 第 77 行 `matcher.match(rgb_img, top_k=1)`：识别 AprilTag，只取最像的那一个。
    返回 `(结果列表, 提示消息)` 两个东西。
  - 第 78-81 行：**一个都没识别到就放弃保存**（只打黄色警告）。
    这意味着采集时如果 tag 被挡住 / 太远 / 太糊，你会得到一个"看起来保存了其实没保存"的结果。
  - 第 84 行 `[float(x) for x in pose_2d]`：又一次把 numpy 标量转成原生 float，同样是给 `json` 让路。
  - 第 86 行 `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)`：ROS 那边给的是 **RGB**，
    而 OpenCV 存盘按 **BGR** 解释，不转的话存出来的图红蓝通道是反的。
  - 第 87-89 行：把识别结果**画到图上**存成 `tag.png`。这个文件是**给人看的验收图**——
    采完模板务必打开看一眼，框画对没画对。
- 第 93-94 行：原图存成 `color.png`（同样要 RGB→BGR）。
- 第 97-99 行：写 `state.json`，`indent=4` 表示**格式化缩进**，方便人直接打开看。

> **重要设计点**：`grasp` 状态**故意不传 `matcher`**（第 237 行）。
> 因为抓的那一刻，夹爪已经贴到物体上了，**相机很可能看不到 tag**（被夹爪挡住），
> 强行要求识别会导致保存失败。而 `grasp` 本来也不需要物体位置——它只要"末端在哪、夹爪多紧"。

### 3.3 命令行参数解析（第 102–138 行）

```python
102  if __name__ == "__main__":
103      parser = argparse.ArgumentParser()
104
105      parser.add_argument(
106          "--cam_params_path",
107          type=str,
108          required=True,
109          help="相机参数文件的路径, 包含内参和畸变参数",
110      )
...
113      parser.add_argument(
114          "--color_img_topic", type=str, required=True, help="RGB 图像的 ROS2 话题名称"
115      )
116
117      parser.add_argument("--tmpl_dir", type=str, required=True, help="模板文件的目录")
118
119      args = parser.parse_args()
120
121      cam_params_path = args.cam_params_path
...
123      if color_img_topic is None:
124          logging.error("Error: color_img_topic is not provided.")
125          exit(0)
126      # end if
```

**业务作用**：把"相机参数在哪、订阅哪个话题、模板存到哪"这三个必填项从命令行收进来。

**逐行讲解：**

- 第 102 行 `if __name__ == "__main__":`：**只有直接运行本文件时才执行**。
  如果别的文件 `import` 它，这段不会跑（所以 `save_state` 可以被复用）。
- 第 105-117 行：三个参数都带 `required=True`，**缺任何一个 argparse 会自动报错并退出**，
  还会打印 `--help` 里写的说明。
- 第 119 行：解析命令行，得到一个对象 `args`，后面用 `args.xxx` 取值。
- 第 123-126 行：`if color_img_topic is None` —— **这行其实是死代码**（详见第 7 节问题 3），
  因为 `required=True` 已经保证了它不可能是 `None`。

### 3.4 硬件与匹配器初始化（第 140–179 行）

```python
141      intrinsic, distortion = read_cam_params(cam_params_path)
142
143      config = TagMatcher2D.Config(
144          intrinsic=intrinsic,
145          distortion=distortion,
146      )
147      matcher = TagMatcher2D(config)
148
149      tmpl_dir = os.path.normpath(tmpl_dir)  # 规范化路径
150      os.makedirs(tmpl_dir, exist_ok=True)
151
152      # 创建机械臂对象
153      arm = ArmWrapper()
154      if not arm.is_connected():
155          logging.error(f"{RED}failed to connect to arm, exiting {RESET}")
156          exit(1)
157      # end if
158
159      # 初始化 ROS2 节点
160      rclpy.init(args=None)
161      cam_node = CamNode([color_img_topic])
162
163      # 创建键盘读取对象
164      keyboard_reader = KeyboardReader()
```

**业务作用**：把三样东西准备好——**匹配器**（知道怎么从图里找 tag）、**机械臂**（能问位姿、能下发动作）、**相机节点**（能实时收图）。

**逐行讲解：**

- 第 141 行 `read_cam_params`：读相机内参 json，返回 `(intrinsic, distortion)` 两个数。
  内参就是前面说的 $f_x, f_y, c_x, c_y$，是算归一化坐标的必需品。
- 第 143-147 行：`TagMatcher2D.Config(...)` 是一个**配置对象**（只装参数的壳子），
  再把它喂给 `TagMatcher2D(config)` 造出真正的匹配器。这种"配置类 + 主体类"的写法在工程里到处都是。
- 第 149-150 行：`os.path.normpath` 把路径里的 `\\` `/` 统一，**并去掉多余的 `.` 和 `..`**；
  `os.makedirs(..., exist_ok=True)` 建目录，`exist_ok=True` 表示"已存在也不报错"。
- 第 153-157 行：连机械臂。**连不上就 `exit(1)`**，这是对的——后面所有按键都要用机械臂。
- 第 160-161 行：初始化 ROS2，建相机节点订阅话题。
  注意 `CamNode([...])` 传的是**列表**，因为它支持同时订阅多个摄像头。
- 第 164 行：建键盘读取器（非阻塞，见 1.4 节）。

### 3.5 主循环与按键分派（第 181–280 行）

```python
166      print(
167          f"use keyboard to control: \n{BLUE}"
168          f"  q: 退出程序\n"
169          f"  a: 使末端的 z 轴方向与基座的 -z 轴平行\n"
170          f"  <: 缩小夹爪之间的距离\n"
171          f"  >: 放大夹爪之间的距离\n"
172          f"  g: 保存抓取时的状态\n"
...
179      )
180
181      while rclpy.ok():
182          key = keyboard_reader.read_key()
183          if key is None:
184              time.sleep(0.03)
185              continue
186          # end if
...
190          if key == "q":  # 退出
191              logging.info("exiting...")
192              break
193          # end if
194
195          if key == "a":  # 调整末端的 z 轴方向, 使它与基座的 z 轴平行
196              logging.info(f"{BLUE}调整末端的 z 轴方向 ...{RESET}")
197
198              T_base_end = arm.get_pose()
199              target_T_base_end = compute_axis_aligned_pose(
200                  T_base_end, base_axis_idx=-3, obj_axis_idx=3
201              )
202              if target_T_base_end is None:
203                  continue
204              # end if
205
206              logging.info(f"try move to new T_base_end:\n{target_T_base_end}")
207              is_ok = arm.set_pose(target_T_base_end)
208              if not is_ok:
209                  logging.warning("failed to move to new T_base_end")
210              # end if
211          # end if
212
213          # 夹爪控制
214          gripper_step = 0.001  # 夹爪每次移动的步长
215          gripper_dist = arm.get_gripper_dist()
216          if key == ",":  # 缩小夹爪
217              set_dist = gripper_dist - gripper_step
218              arm.set_gripper_dist(set_dist)
...
223          elif key == ".":  # 放大夹爪
```

**业务作用**：整个交互层。每秒轮询 33 次键盘，按不同键做不同的事。

**逐行讲解：**

- 第 181 行 `while rclpy.ok():`：ROS2 还在正常运行就一直转。
  如果别处按了 Ctrl-C 或 ROS 被关掉，这个条件会变 `False`，循环自然退出。
- 第 182-186 行：读键。没按（`None`）就 `sleep(0.03)` 再 `continue`——
  `continue` 是"跳过本轮剩下的代码，直接开始下一轮"。
- 第 190-193 行：`q` 退出。`break` 直接跳出 `while`。
- 第 195-211 行：**`a` 键 = 把末端掰成竖直朝下**。
  - 第 199-201 行调用 `compute_axis_aligned_pose(T_base_end, base_axis_idx=-3, obj_axis_idx=3)`。
    - `base_axis_idx=-3`：**目标**方向 = 基座坐标系的 **-Z 轴**（竖直向下）。
    - `obj_axis_idx=3`：**当前**要被掰的那根轴 = 末端的 **+Z 轴**（工具朝前的方向）。
    - 合起来："把末端的 +Z 轴，转到和基座的 -Z 轴平行"，也就是**让夹爪垂直朝下**。
  - 第 202-204 行：如果函数返回 `None`（当前偏得**超过 45°**，函数认为掰不动），就跳过这次动作。
  - 第 207 行 `arm.set_pose(...)`：下发新位姿，机械臂真的会动。
    返回 `is_ok`，失败只打警告（第 208-210 行）——**注意它不会重试**。
- 第 214-229 行：夹爪微调。
  `gripper_step = 0.001` 是 **1 毫米**——按一次动 1 mm，方便精细对准。
  **⚠️ 提示里写的是 `<` / `>`，代码判断的却是 `,` / `.`**（详见第 7 节问题 1）。

后面 5 个保存分支（第 231-278 行）结构完全一样，只是目录名和传入的 `matcher` 不同：

```python
231      if key == "g":  # 保存抓取时的状态
234          save_dir = os.path.join(tmpl_dir, "grasp")
235          os.makedirs(save_dir, exist_ok=True)
236
237          save_state(cam_node, arm, save_dir)          # ← 不传 matcher
238      # end if
239
240      if key == "n":  # 保存相机距离物体较近时的状态
243          save_dir = os.path.join(tmpl_dir, "near")
244          os.makedirs(save_dir, exist_ok=True)
245
246          save_state(cam_node, arm, save_dir, matcher=matcher)   # ← 传 matcher
247      # end if
```

| 按键 | 目录 | 传 matcher？ | 行号 |
|---|---|---|---|
| `g` | `grasp` | ❌ 不传 | 231-238 |
| `n` | `near` | ✅ 传 | 240-247 |
| `b` | `next_near` | ✅ 传 | 249-258 |
| `f` | `far` | ✅ 传 | 260-267 |
| `d` | `next_far` | ✅ 传 | 269-278 |

> 注意这 5 个是**独立的 `if`，不是 `elif`**。因为每次只有一个键被按下，
> 所以不会串；但写成 `if` 意味着每次都要判断 5 次——这里无所谓，可读性优先。

---

## 4. Python 基础语法速查

| 写法 | 含义 | 本文件出现位置 |
|---|---|---|
| `f"..."` | f-string，把 `{}` 里的变量换成值 | 第 57、63、89 行等 |
| `if __name__ == "__main__":` | 只在直接运行本文件时执行 | 第 102 行 |
| `def f(a, b=None):` | 带默认值的参数，调用时可省略 | 第 43 行 |
| `with open(...) as f:` | 用完自动关闭文件 | 第 92、98 行 |
| `np.ndarray.tolist()` | numpy 数组 → Python list（为了 `json` 能存） | 第 68 行 |
| `os.makedirs(p, exist_ok=True)` | 建目录，已存在不报错 | 第 150、235 行 |
| `os.path.join(a, b)` | 拼路径，自动处理分隔符 | 第 88、93 行 |
| `continue` / `break` | 跳过本轮 / 跳出循环 | 第 185、192 行 |
| `type=str, required=True` | argparse：这个参数必须是字符串、且必填 | 第 107-117 行 |
| `# noqa: LOG015` | 告诉 linter "这行我故意这么写，别报警" | 第 57、73 行 |

---

## 5. 运行方式与输入输出

**运行命令**（三个参数都必填）：

```bash
python demo/test_tmpl_grasp_2d/create_tmpl_grasp_2d.py \
    --cam_params_path  config/cam_params.json \
    --color_img_topic  /camera/color/image_raw \
    --tmpl_dir         demo/test_tmpl_grasp_2d/tmpl/box_01
```

**运行后你会看到**：

```
camera parameters file: config/cam_params.json
color image topic: /camera/color/image_raw
grasp_2d template will be saved to: demo/test_tmpl_grasp_2d/tmpl/box_01

use keyboard to control:
  q: 退出程序
  a: 使末端的 z 轴方向与基座的 -z 轴平行
  <: 缩小夹爪之间的距离
  >: 放大夹爪之间的距离
  g: 保存抓取时的状态
  n: 保存相机距离物体较近时的状态
  b: 保存下一个相机距离物体较近时的状态
  f: 保存相机距离物体较远时的状态
  d: 保存下一个相机距离物体较远时的状态
```

**推荐的采集顺序**（照着做不容易出错）：

1. 先把物体放好，按 `a` 让夹爪垂直朝下（**每次大范围移动后都建议按一次**）。
2. 用 `,`/`.` 张开夹爪到足够大。
3. 把相机移到**离物体较近**、能看到 tag 的位置 → 按 `n`。
4. 在 xy 平面平移**一小段**（3–5 cm 就够，别太大）→ 按 `b`。
5. 把相机移到**离物体较远**、仍能看到 tag 的位置 → 按 `f`。
6. 同样平移一小段 → 按 `d`。
7. 手动把机械臂移到**真正能抓住物体**的位姿，合拢夹爪夹住 → 按 `g`。
8. 按 `q` 退出。

**产出目录**：

```
tmpl/box_01/
├── grasp/      state.json  color.png
├── near/       state.json  color.png  tag.png
├── next_near/  state.json  color.png  tag.png
├── far/        state.json  color.png  tag.png
└── next_far/   state.json  color.png  tag.png
```

验收方法：**打开每张 `tag.png`**，确认绿色框正好框住目标 tag。框歪了或框到别的东西，
这个模板就是废的，下游抓取一定偏。

---

## 6. 核心数学：一步一步算给你看

### 6.1 `a` 键的"轴对齐"到底在算什么

`compute_axis_aligned_pose(T_base_end, base_axis_idx=-3, obj_axis_idx=3)` 内部做了四件事：

1. **算出当前方向**：取末端姿态矩阵的第 3 列（+Z 轴）在基座下的指向，记作
   $\mathbf{c}$（代码里 `current_dir`）。
2. **算出目标方向**：基座坐标系的 -Z 轴，即 $\mathbf{t} = [0, 0, -1]^{\top}$（`target_dir`）。
3. **算夹角**：$\theta = \arccos(\mathbf{c} \cdot \mathbf{t})$。
   夹角大于 45° 时函数直接返回 `None`（第 514-517 行），**拒绝动作**——
   这是防止机械臂为了对齐而做出大幅度的危险翻转。
4. **构造旋转**：绕 $\mathbf{c} \times \mathbf{t}$ 这根轴转 $\theta$，得到 $\Delta R$，
   再左乘到原姿态上。

> 为什么是"叉乘当轴"？因为 $\mathbf{c} \times \mathbf{t}$ 同时垂直于当前方向和目标方向，
> 绕它旋转**正好把 $\mathbf{c}$ 扫到 $\mathbf{t}$**，而且走的是最短路径（转的角度就是 $\theta$）。

### 6.2 为什么 `near` 和 `next_near` 必须只差"一小段平移"

下游脚本要算的是这个比值（Jacobian 比值 / 换算汇率）：

$$\text{ratio} = \frac{\lVert \Delta \mathbf{x}_{\mathrm{hand}} \rVert}{\lVert \Delta \mathbf{n}_{\mathrm{img}} \rVert}$$

其中：
- $\Delta \mathbf{x}_{\mathrm{hand}}$ = `next_near` 与 `near` 两个末端位姿的**位置差**（米）；
- $\Delta \mathbf{n}_{\mathrm{img}}$ = 两个 `obj_pose_2d` 的**归一化坐标差**（无量纲）。

这是一个**局部线性近似**：它假设"在这一小段范围内，手挪多少 ↔ 画面挪多少"是成正比的。
所以：

- **平移太大**（比如 30 cm）：透视变化剧烈，线性假设失效，比值失真。
- **平移太小**（比如 2 mm）：两个位姿几乎一样，
  分母 $\lVert \Delta \mathbf{n} \rVert$ 接近 0，**比值的相对误差会爆炸**（除以一个极小的数）。

工程经验：**3–5 cm** 比较合适。

### 6.3 数值演练：一对 near 数据怎么算出比值

假设你采到：

```
near:       T_base_end 平移部分 = [0.400, 0.100, 0.350]    obj_pose_2d = [ 0.020, -0.010, 0.05]
next_near:  T_base_end 平移部分 = [0.430, 0.120, 0.350]    obj_pose_2d = [-0.040,  0.005, 0.05]
```

手挪了多少（只看 x、y，因为是在桌面平面内平移）：

$$\Delta \mathbf{x}_{\mathrm{hand}} = [0.030,\ 0.020] \quad\Rightarrow\quad \lVert \Delta \mathbf{x}_{\mathrm{hand}} \rVert = \sqrt{0.03^2 + 0.02^2} \approx 0.0361\ \text{m}$$

画面里挪了多少：

$$\Delta \mathbf{n} = [-0.060,\ 0.015] \quad\Rightarrow\quad \lVert \Delta \mathbf{n} \rVert = \sqrt{0.06^2 + 0.015^2} \approx 0.0618$$

比值：

$$\text{ratio} = \frac{0.0361}{0.0618} \approx 0.584\ \mathrm{m}$$

（单位是"米 / 每单位归一化坐标"，近似于"画面里挪 1 格，手要挪多少米"）

意思是：**物体在画面里挪 1 个归一化单位，机械臂要挪约 0.584 米**。
下游脚本运行时，看到画面差了多少，乘这个比值就知道该让手挪多少米了。

> 注意 `far` 那一对算出来的比值会**更大**（远处物体在画面里动得慢，同样的手部位移对应的画面位移小）。
> 下游脚本按当前距离在两者之间插值，这就是为什么必须近、远**各采一对**。

---

## 7. 这段代码里的坑与改进建议

| # | 问题 | 后果 | 建议 |
|---|---|---|---|
| 1 | **按键提示与判断不一致**：提示写 `<` / `>`（第 170-171 行），代码判断 `,` / `.`（第 216、223 行） | **最严重**。用户照着提示按 `<`，程序毫无反应，以为卡死了 | 二选一：把提示改成 `,` / `.`，或把判断改成 `key in (",", "<")` 两种都认 |
| 2 | `save_state` 失败时只打日志就 `return`（第 63、79 行） | 采了 5 个键，实际只存了 3 个，下游读到的模板是残缺的 | 返回 `bool` 表示成败，主循环里累计成功数，退出前打印"本次成功保存 x/5" |
| 3 | 第 123-132 行的 `if xxx is None` 是**死代码** | 无害但误导——让人以为参数"可能是 None" | 删掉，或改成检查空字符串 `if not tmpl_dir:` |
| 4 | 退出时（第 192 行 `break`）没有 `rclpy.shutdown()`、没有 `cam_node.destroy_node()` | ROS2 节点资源没释放，偶尔会报 "node not destroyed" | 在 `break` 后补清理，或把清理放进 `try/finally` |
| 5 | 第 33 行 `from core.utils import ...`，但工程里该文件已移到 `core/common_utils/utils.py` | **直接 `ModuleNotFoundError`，脚本根本起不来** | 改成 `from core.common_utils.utils import ...`，或在 `core/utils.py` 里做转发 |
| 6 | 第 31 行 `from core.arm_wrapper import ArmWrapper`，但当前 `core/` 下没有这个文件（它在 `carm_grasp-main/core/`） | 同上，导入失败 | 把 `arm_wrapper.py` 补回 `core/`，或调整 `sys.path` 指向正确的 core |
| 7 | `a` 键动作失败只打警告不重试（第 208-210 行） | 用户以为按了没反应 | 失败时明确提示"请手动调整姿态到接近竖直后再按 a" |
| 8 | 重复代码：5 个保存分支（第 231-278 行）几乎一模一样 | 改一个要改 5 处 | 用字典映射：`KEY2DIR = {"g": ("grasp", False), "n": ("near", True), ...}` 循环处理 |
| 9 | 没有校验 5 个状态是否都采齐 | 采了 3 个就退出，下游照样跑，然后莫名其妙失败 | 退出前检查 5 个目录是否都有 `state.json`，缺哪个就告警 |

---

## 8. 一句话总结

> **本脚本是 2D 抓取的"标准照拍摄器"：人用键盘指挥机械臂摆位，按 5 个键存下 5 组
> （末端位姿 + 夹爪开度 + 物体画面位置），产出的 `state.json` 就是下游自动抓取时唯一的参考答案。
> 采集质量直接决定抓取精度——`tag.png` 一定要逐张验收。**

# `test_tmpl_grasp_3d_romate.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、没接触过线性代数、也不知道 ROS 是什么的同学。
> 目标：读完之后，你能逐行看懂这个文件在干什么——它是怎么"看着"物体、
> 在**多个抓取模板里挑一个最省事的**、利用**物体的对称性**抄近路、
> 一路躲开碰撞把东西抓起来再放好的。
>
> 被分析的源文件：
> `E:\WORK\arm_disorderly_swap\robot_grasp_test\demo\test_tmpl_grasp_3d\test_tmpl_grasp_3d_romate.py`（共 1139 行）
>
> 配套必读依赖：
> `core/utils.py`（颜色常量、`wait_key`、`read_rgbd_params`、`read_handeye_calib`、`inv_tf`）、
> `core/arm_wrapper.py`（`ArmWrapper`）、`core/arm_utils.py`（`GripperBody`、`CollisionDetector`、
> `check_arm_pose`、`TH_ANGLE_Z`、`TH_GRIPPER_HEIGHT`）、`core/arm_ros_utils.py`（`TargetArmNode`）、
> `core/cam_ros_utils.py`（`CamNode`）、`core/vision_utils.py`（`compute_locate_error`）、
> `examples/app/src/client_matching3d.py`（`ClientNode`，外部匹配服务的客户端）。
>
> **姊妹篇（强烈建议对照阅读）**：
> - `test_tmpl_grasp_3d_analysis.md` —— 本文件的**简化版前身**，只支持单个模板、无对称性、无碰撞检测。
> - `create_tmpl_grasp_3d_analysis.md` —— 本文件所读的模板是怎么采出来的。

---

## 目录

- [0. 一句话概括](#0-一句话概括)
- [1. 背景知识：这个"增强版"到底强在哪](#1-背景知识这个增强版到底强在哪)
  - [1.1 与 `test_tmpl_grasp_3d.py` 的关系](#11-与-test_tmpl_grasp_3dpy-的关系)
  - [1.2 为什么要"多个抓取模板"](#12-为什么要多个抓取模板)
  - [1.3 什么是物体的对称性 `sym_tfs`](#13-什么是物体的对称性-sym_tfs)
  - [1.4 什么是"检查位姿"，为什么多此一举](#14-什么是检查位姿为什么多此一举)
  - [1.5 四个位置：detect / check / ready / grasp / place](#15-四个位置detect--check--ready--grasp--place)
  - [1.6 匹配为什么不自己算，而是调服务](#16-匹配为什么不自己算而是调服务)
- [2. 整体结构与数据流](#2-整体结构与数据流)
  - [2.1 `do_grasp` 的 11 步](#21-do_grasp-的-11-步)
  - [2.2 `run` 的循环与缓存复用](#22-run-的循环与缓存复用)
  - [2.3 数据/文件依赖图](#23-数据文件依赖图)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 文件头与导入区（第 1–64 行）](#31-文件头与导入区第-1%E2%80%9364-行)
  - [3.2 `read_tmpl_grasp`（第 70–166 行）](#32-read_tmpl_grasp第-70%E2%80%93166-行)
  - [3.3 `compute_tmpl_ready_pose`（第 169–232 行）](#33-compute_tmpl_ready_pose第-169%E2%80%93232-行)
  - [3.4 `compute_check_state`（第 235–326 行）](#34-compute_check_state第-235%E2%80%93326-行)
  - [3.5 `compute_ready_pose`（第 329–388 行）](#35-compute_ready_pose第-329%E2%80%93388-行)
  - [3.6 `do_grasp`（第 391–834 行）](#36-do_grasp第-391%E2%80%93834-行)
  - [3.7 `run`（第 837–961 行）](#37-run第-837%E2%80%93961-行)
  - [3.8 `__main__`（第 967–1139 行）](#38-__main__第-967%E2%80%931139-行)
- [4. Python 基础语法速查](#4-python-基础语法速查)
- [5. 运行方式与输入输出](#5-运行方式与输入输出)
- [6. 核心数学：一步一步算给你看](#6-核心数学一步一步算给你看)
  - [6.1 目标位姿公式：把"物体偏了多少"搬到手上](#61-目标位姿公式把物体偏了多少搬到手上)
  - [6.2 对称性怎么让机械臂"抄近路"](#62-对称性怎么让机械臂抄近路)
  - [6.3 跟踪：用机械臂自己的位移预测物体现在在哪](#63-跟踪用机械臂自己的位移预测物体现在在哪)
  - [6.4 检查位姿：绕 Z 轴每 15 度试一次，挑关节动得最少的](#64-检查位姿绕-z-轴每-15-度试一次挑关节动得最少的)
- [7. 这段代码里的坑与改进建议](#7-这段代码里的坑与改进建议)
- [8. 一句话总结](#8-一句话总结)

---

## 0. 一句话概括

> **这是整套系统里真正"干活"的那个脚本：它循环不断地
> ① 先把手臂摆到"检测位"看一眼物体在哪，② 挪到"检查位"让相机正对物体再看清楚一点，
> ③ 在预先采好的**多个抓取模板**里挑一个转起来最省劲的（顺便利用**物体的对称性**抄近路），
> ④ 反复微调两次确认没看错，⑤ 检查夹爪会不会撞到东西，⑥ 冲下去抓住、抬起来、送到"放置位"放下。
> 抓完一个立刻回到检测位，接着抓下一个，直到场景里没有物体为止。**

---

## 1. 背景知识：这个"增强版"到底强在哪

### 1.1 与 `test_tmpl_grasp_3d.py` 的关系

两个文件是**同一套思路的不同成熟度**：

| 能力 | `test_tmpl_grasp_3d.py`（简版） | **`test_tmpl_grasp_3d_romate.py`（本文件）** |
|---|---|---|
| 抓取模板 | **只有 1 个** | **多个**（`tmpl_state_list`，按目录 `0/ 1/ 2/ ...` 存放），自动挑最优 |
| 物体对称性 | 不支持 | **支持**（`sym_tfs`，从匹配服务获取） |
| 碰撞检测 | 无 | **有**（`CollisionDetector`，抓之前验两遍） |
| 检测位 / 放置位 | 无（抓完就结束） | **有**（`detect.json` / `place.json`，全自动循环） |
| 定位方式 | 本地 `TagMatcher3D` | **远程服务** `ClientNode`（ROS2 service） |
| 迭代细化 | 无 | **2 次**（`max_refine_cnt = 2`） |
| 位姿合理性检查 | 无 | **有**（`check_arm_pose`，防止末端歪太多 / 撞桌面） |
| 失败后复用缓存 | 无 | **有**（`use_cache` 机制） |

一句话：**简版是"抓一次"的演示，本文件是"能连续干活"的完整流程。**

### 1.2 为什么要"多个抓取模板"

同一个物体，**从左边抓、从右边抓、从上往下抓**都可能成立。
但机械臂当前在某个姿态，转去某个模板可能很别扭（甚至超出关节限位）。

所以采集时可以多采几组（`0/`、`1/`、`2/` …），运行的时候：

> 对每个模板都算一次"我要转多少度才能对上"，**挑转得最少的那个**。

这就是 `compute_ready_pose` 里 `min_delta_angle` 比较的含义（第 379-383 行）。

### 1.3 什么是物体的对称性 `sym_tfs`

很多物体**转一下还是它自己**：

- 圆柱体绕自己的轴转任意角度都一样；
- 长方体转 180° 后看起来一样；
- 正方形截面转 90° 也一样。

这些"转了等于没转"的变换就叫做**对称变换**，存成一串 4×4 矩阵 `sym_tfs`（形状 $N \times 4 \times 4$）。

有什么用？如果物体现在是"歪着"的，但**歪的角度刚好等于某个对称变换**，
那机械臂**根本不用转**——直接按现在这个朝向抓就行。

这就是"抄近路"：本来要手腕转 180°，发现转 180° 后物体长得一样，那就省了这 180°。

本文件在 `compute_tmpl_ready_pose` 里对**每一个**对称变换都算一遍目标位姿，
然后挑旋转量最小的（第 197-229 行）。详见 6.2 节。

### 1.4 什么是"检查位姿"，为什么多此一举

第 2 步（第 456-479 行）专门把机械臂移到一个**"检查位姿"**再去重新定位：

- 相机**光心垂直朝下**（Z 轴指向 $[0,0,-1]^{\top}$）；
- 物体位于相机**正下方** `CHECK_HEIGHT = 0.18` 米处。

为什么要先摆到这个标准姿势？因为第一次定位时，相机可能是斜着看的，
斜看时透视变形大、深度噪声大，定位精度差。
摆正了再看一次，就能把误差压下来——**多花两秒钟，换一个更可信的位姿**。

详见 6.4 节。

### 1.5 四个位置：detect / check / ready / grasp / place

整个流程要跑 5 个位姿，别搞混：

| 位置 | 谁定的 | 干什么 |
|---|---|---|
| `detect`（检测位） | 模板 `detect.json` | 循环起点，站在这儿扫视整个桌面找物体 |
| `check`（检查位） | **运行时现算** | 相机正对物体、距离 0.18 m，重新精确定位 |
| `ready`（预备位） | **运行时现算** | 夹爪张开着、悬在物体正上方的"临门一脚"位置 |
| `grasp`（抓取位） | 模板 `grasp.json` | 真正夹住物体的位置（由 ready 位叠一个固定增量得到） |
| `place`（放置位） | 模板 `place.json` | 抓到之后放东西的地方 |

### 1.6 匹配为什么不自己算，而是调服务

简版用本地的 `TagMatcher3D` 直接在自己进程里算。
本文件改成通过 `ClientNode` 调一个**独立的 ROS2 服务**（第 446、523、657 行）。

好处：
- 匹配（尤其是 3D 点云匹配）很吃 CPU，**单独跑一个服务**不会拖慢机械臂控制；
- 可以**同时挂多个模型**（`--model_name` 参数，第 980 行），服务名带模型名区分；
- 服务那边还能顺便提供**对称性信息**（第 1112 行 `get_symmetric_info`）。

代价：**多了一个外部依赖**。服务没起来，本脚本会等 10 秒然后退出（第 1098-1109 行）。

---

## 2. 整体结构与数据流

### 2.1 `do_grasp` 的 11 步

`do_grasp` 是核心，一共 11 步（日志里打印成 `grasp-step [N]`）：

```
[1]  定位物体（调匹配服务）                         ← 失败则退出程序
[2]  计算检查位姿                                   ← 逆解失败则换下一个
[3]  移动到检查位姿（用关节角 set_joints）
[4]  重新定位物体（这次用"跟踪"，带预测初值）        ← 相机出问题则退出
[5]  计算预备位姿，并在多个模板里挑最优的
[6]  检查夹爪会不会撞到场景                         ← 撞了则换下一个
[7]  移动到预备位姿
[8]  跟踪 + 重算预备位姿  ┐
[9]  再次移动到预备位姿   ┘ 循环 2 次（细化）
[10] 再查一次碰撞（这次按抓取姿态）
[11] 抓取：下降 → 合爪 → 抬高
```

### 2.2 `run` 的循环与缓存复用

```
while rclpy.ok():
    [0]  若不用缓存 → 移动到 detect 位（检测位）
    ↓
    status = do_grasp(...)
    ├ -1 → 退出程序（场景里没东西了 / 相机坏了）
    ├  0 → 失败但可复用缓存 → use_cache = True，继续下一轮
    └  1 → 成功 → use_cache = False，去放置
    ↓
    [-1] 移动到 place 位 → 张开夹爪放下
↓
机械臂回零点，结束
```

`use_cache` 的含义很讲究：

- 上一轮**失败**了（比如看错了、撞了），机械臂**大概率没碰到物体**，
  所以"上一帧的物体位置"和"上一帧的手臂位姿"仍然有效 →
  下一轮**可以直接复用**，省掉"回到检测位再找一遍"的时间（第 915 行）。
- 上一轮**成功**抓走了，或者刚抓完 → 场景变了 → **必须重新回检测位看**（第 919、801 行）。

### 2.3 数据/文件依赖图

```
模板目录 tmpl_dir/
├── detect.json     { T_base_end, joints, gripper_dist }
├── place.json      { T_base_end, joints, gripper_dist }
├── 0/  grasp.json + ready.json     ← 第 1 个抓取模板
├── 1/  grasp.json + ready.json     ← 第 2 个
└── 2/  ...                          ← 可以有任意多个
        ↓ read_tmpl_grasp()

标定数据（路径在代码里写死，见第 1023-1037 行）
├── data/calib/cam_params.json        → intrinsic, distortion, depth_scale
├── data/calib/calib_handeye.json     → T_end_cam
└── data/calib/gripper_body.json      → 夹爪几何（width / thickness / T_cam_gripper）
        ↓
外部服务
└── MATCHING3D_SRV  → cur_T_cam_model、sym_tfs
```

---

## 3. 逐段代码精读

### 3.1 文件头与导入区（第 1–64 行）

```python
1   # -*- coding: utf-8 -*-
2   """
3   功能说明: 基于自研机械臂 CARM 的 3D 抓取( 6 个自由度 )示例 ROS2 节点
4   """
5
6   import rclpy
...
14  import mmengine
15  from typing_extensions import List, Tuple, Dict
16
17  import numpy as np
18  import transforms3d
19
23  code_dir = os.path.dirname(os.path.realpath(__file__))
24  root_dir = os.path.normpath(f'{code_dir}/../../../')
25  sys.path.append(root_dir)
26
27  from core.utils import (
28      GREEN, YELLOW, BLUE, RED, RESET,
29      wait_key, reset_empty_str,
30      read_rgbd_params, read_handeye_calib, inv_tf
31  )
...
53  from examples.app.src.client_matching3d import (
54      ClientNode
55  )  # 同目录下的模块
56
60  CHECK_HEIGHT = 0.18
61  """检查位姿时物体到相机的垂直距离, 单位: 米"""
62
63  COLLISION_DIR = os.path.normpath(f"{root_dir}/results/app/collision")
```

**业务作用**：搬工具箱 + 定义两个本文件专用的常量。

**逐行讲解：**

- 第 1 行 `# -*- coding: utf-8 -*-`：告诉 Python 3 这个文件是 UTF-8 编码（Python 3 默认就是，写上更保险）。
- 第 14 行 `mmengine`：OpenMMLab 的配置文件读写库，第 1042 行用它读 `gripper_body.json`。
- 第 15 行 `from typing_extensions import List, Tuple, Dict`：类型标注用的
  （`List[float]`、`Tuple[...]` 这些写法，见 3.3 节）。
- 第 18 行 `transforms3d`：处理旋转的库，第 263、274 行用它把"轴 + 角"变成旋转矩阵。
- 第 23-25 行：老规矩，把工程根塞进 `sys.path`。
- 第 29 行 `wait_key`：**调试用的"暂停等按键"**——开了 `--debug` 就会在每一步停下来等人敲键，
  方便一步步观察。第 441 行等处大量使用。
- 第 30 行 `read_handeye_calib`：注意**函数名和 `create_tmpl_grasp_3d.py` 里的 `read_calib_handeye` 不一样**
  （同一个功能的两个名字，容易记混）。
- 第 53-55 行 `ClientNode`：匹配服务的客户端。**注意 import 路径是 `examples.app.src...`，
  但本文件在 `demo/test_tmpl_grasp_3d/`**——这个路径能否 import 成功取决于 `root_dir` 下是否真有
  `examples/app/src/`（见第 7 节问题 8）。
- 第 60-61 行 `CHECK_HEIGHT = 0.18`：**检查位姿时物体到相机的垂直距离**，单位米。
  0.18 m 是"不太近（怕撞）也不太远（怕看不清）"的折中。
- 第 63 行 `COLLISION_DIR`：碰撞检测过程的调试图存哪儿。

### 3.2 `read_tmpl_grasp`（第 70–166 行）

```python
70  def read_tmpl_grasp(tmpl_dir: str) -> Dict:
...
79      # 1. 读取相机处于检测位置时的机械臂状态
80      detect_path = os.path.join(tmpl_dir, 'detect.json')
81      if not os.path.exists(detect_path):
82          logging.error(f'file not found: {detect_path}')
83          return None
...
96      place_path = os.path.join(tmpl_dir, 'place.json')
...
111     # 3. 读取抓取模板位姿
112     tmpl_state_list = []
113     tmpl_cnt = 0
114     while True:
115         tmpl_dir_i = os.path.join(tmpl_dir, f'{tmpl_cnt}')
116         if not os.path.exists(tmpl_dir_i):
117             break
118         # end if
119         grasp_path = os.path.join(tmpl_dir_i, 'grasp.json')
120         ready_path = os.path.join(tmpl_dir_i, 'ready.json')
121         if not os.path.exists(grasp_path) or not os.path.exists(ready_path):
122             logging.warning(f'file not found: {grasp_path} or {ready_path}')
123             break
124         # end if
...
131         state_dict['grasp_T_base_end'] = np.array(grasp_data['T_base_end'], dtype=np.float32)
132         state_dict['grasp_gripper_dist'] = grasp_data['gripper_dist']
...
137         state_dict['ready_T_base_end'] = np.array(ready_data['T_base_end'], dtype=np.float32)
138         state_dict['ready_T_cam_model'] = np.array(ready_data['T_cam_model'], dtype=np.float32)
139         state_dict['ready_gripper_dist'] = ready_data['gripper_dist']
140
141         tmpl_state_list.append(state_dict)
142         tmpl_cnt += 1
143     # end while
144
145     if len(tmpl_state_list) == 0:
146         logging.error(f'no valid grasp tmpl found in dir: {tmpl_dir}')
147         return None
```

**业务作用**：把整个模板目录读成一个字典，**特别是把"多个抓取模板"读成列表**。

**逐行讲解：**

- 第 80-93 行：读 `detect.json`（检测位）。**文件不存在就 `return None`**——
  这一点比简版严谨（简版 `test_tmpl_grasp_2d.py` 的 `read_tmpl_grasp_2d` 会直接抛异常）。
- 第 112-143 行：**这是本文件最关键的一段**——用一个 `while True` 循环
  依次找 `0/`、`1/`、`2/` … 子目录，直到某个编号不存在就停。
  每个子目录里必须有 `grasp.json` + `ready.json`，缺一个就**停止**（并只打 warning）。
- 第 131-139 行：每个模板存 5 个字段。
  注意 `ready_T_cam_model`（物体在相机里的位姿）是**挑模板和算目标位姿的核心依据**。
- 第 145-148 行：一个模板都没读到就返回 `None`。

> **注意第 121-124 行的行为**：如果 `0/` 和 `1/` 都有，但 `2/` 里只有 `grasp.json` 缺 `ready.json`，
> 循环会**在 2 这里断掉**——后面的 `3/`、`4/` 就算齐全也**不会被读到**。
> 这是"遇到第一个缺口就停"的设计，采模板时编号必须连续，中间不能有残缺。

### 3.3 `compute_tmpl_ready_pose`（第 169–232 行）

```python
169 def compute_tmpl_ready_pose(sym_tfs: np.ndarray,
170                             T_end_cam: np.ndarray,
171                             ready_T_base_end: np.ndarray,
172                             ready_T_cam_model: np.ndarray,
173                             cur_T_base_end: np.ndarray,
174                             cur_T_cam_model: np.ndarray) -> Tuple[np.ndarray, float]:
...
188     N = sym_tfs.shape[0]  # 对称变换数量
189
190     ready_T_model_cam = inv_tf(ready_T_cam_model)
191     ready_T_end_base = inv_tf(ready_T_base_end)
192     T_cam_end = inv_tf(T_end_cam)
193
194     target_poses = np.zeros((N, 4, 4), dtype=np.float32)  # N*4*4 T_base_end
195     delta_angles = [np.pi * 2] * N  # N 旋转矩阵的轴角表示的旋转部分
196
197     for i in range(N):
198         sym_tf = sym_tfs[i]
199         target_T_base_end = cur_T_base_end @ T_end_cam @ cur_T_cam_model @ sym_tf @ ready_T_model_cam @ T_cam_end
200         target_poses[i] = target_T_base_end
201
202         delta_T_base_end = ready_T_end_base @ target_T_base_end
203
204         # 由旋转矩阵计算轴角表示的旋转部分
205         delta_R = delta_T_base_end[:3, :3]
206         cosine = (np.trace(delta_R) - 1) / 2
207         cosine = np.clip(cosine, -1.0, 1.0)  # 数值稳定性处理
208         angle = abs(np.arccos(cosine))
209
210         # 计算 ready_T_base_end 和 target_T_base_end 之间的 Z 轴夹角
211         ready_z_dir = ready_T_base_end[:3, 2]  # 机械臂末端 Z 轴方向
212         target_z_dir = target_T_base_end[:3, 2]  # 机械臂末端 Z 轴方向
213         cosine = np.dot(ready_z_dir, target_z_dir) / (...)
215         angle_z = np.arccos(cosine)
216
220         if angle_z > TH_ANGLE_Z:
221             angle += np.pi * 2  # 增加一个惩罚值
222         # end if
223
224         delta_angles[i] = angle
225     # end for
226
227     min_idx = int(np.argmin(delta_angles))
228     target_T_base_end = target_poses[min_idx]
229     delta_angle = delta_angles[min_idx]
230
231     return target_T_base_end, delta_angle
```

**业务作用**：**给定某一个模板**，算出"我该把手摆到哪"，并且**在物体所有对称姿态里挑最省劲的那个**。

**逐行讲解：**

- 第 188-192 行：先把要用到的三个逆矩阵算好（`inv_tf` 是工程自己封装的求逆）。
- 第 194-195 行：准备两个"容器"——`target_poses` 存 N 个候选位姿，
  `delta_angles` 存对应的"要转多少度"，**初始值填 $2\pi$（最大值，表示最差）**。
- 第 197-225 行：**对每个对称变换循环一遍**。
  - 第 199 行：核心公式（详见 6.1 节）。
  - 第 202-208 行：算出"从模板的 ready 位姿到目标位姿"差了多少旋转 $\Delta R$，
    用矩阵迹反求角度：$\theta = \arccos\big((\mathrm{tr}(\Delta R) - 1)/2\big)$。
    **这是旋转矩阵的经典性质**：3×3 旋转矩阵的迹等于 $1 + 2\cos\theta$。
  - 第 210-215 行：**另外**算一个"末端 Z 轴偏了多少" $\theta_z$。
  - 第 220-222 行：**如果 Z 轴偏得超过阈值 `TH_ANGLE_Z`，就给角度加 $2\pi$ 当惩罚**。
    这是一个"软约束"——不是直接拒绝，而是让它在排序里排到最后。
    为什么在意 Z 轴？因为夹爪的 Z 轴就是"朝下抓取"的方向，
    Z 轴歪太多说明手腕别扭，容易撞或者超出限位。
- 第 227-229 行：`np.argmin` 找最小的那个角度，返回对应位姿。

### 3.4 `compute_check_state`（第 235–326 行）

```python
251     cur_T_base_cam = cur_T_base_end @ T_end_cam
252     cur_T_base_model = cur_T_base_cam @ cur_T_cam_model
253
254     target_z_dir = np.array([0, 0, -1])   # 期望的 Z 轴方向
255     cur_z_dir = cur_T_base_cam[:3, 2]  # 当前的 Z 轴方向
256     cosine = np.dot(target_z_dir, cur_z_dir)
257     cosine = np.clip(cosine, -1.0, 1.0)
258     angle = np.arccos(cosine)
259
261     # 计算调整后的姿态
262     axis = np.cross(cur_z_dir, target_z_dir)
263     delta_R = transforms3d.axangles.axangle2mat(axis, angle)
264
265     target_T_base_cam = np.eye(4, dtype=np.float32)
266     target_T_base_cam[:3, :3] = delta_R @ cur_T_base_cam[:3, :3]
267     target_T_base_cam[:3, 3] = cur_T_base_model[:3, 3] + np.array([0, 0, height])
268
269     # 计算多个机械臂位姿
270     T_base_end_list = []
271     angle_step = 15.0 * np.pi / 180.0  # 步长
272     T_cam_end = inv_tf(T_end_cam)
273     for angle_yaw in np.arange(0, 2 * np.pi, angle_step):
274         R_z = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]), angle_yaw)
275         T_base_cam = np.eye(4, dtype=np.float32)
276         T_base_cam[:3, :3] = R_z @ target_T_base_cam[:3, :3]
277         T_base_cam[:3, 3] = target_T_base_cam[:3, 3]
278
279         T_base_end = T_base_cam @ T_cam_end
280         T_base_end_list.append(T_base_end)
281     # end for
282
283     # 计算机械臂逆解
284     st = time.time()
285     joints_list = arm.inverse_kinematics(T_base_end_list, [cur_joints] * len(T_base_end_list))
...
294     # 计算最接近当前关节角( 第一个关节除外 )的解
295     min_delta = 1000.0
296     best_idx = -1
297     for idx, joints in enumerate(joints_list):
...
306         delta = 0.0
307         for i in range(1, len(joints)):
308             delta += abs(joints[i] - cur_joints[i])
309         # end for
310         if delta < min_delta:
311             min_delta = delta
312             best_idx = idx
```

**业务作用**：算一个"相机垂直朝下、正对物体上方 `height` 米"的位姿，
而且**绕竖直轴试 24 个朝向**，挑一个机械臂关节动得最少的。

**逐行讲解：**

- 第 251-252 行：把"相机在基座下的位姿"和"物体在基座下的位姿"都算出来。
- 第 254-263 行：和 `compute_axis_aligned_pose` 完全同一套套路——
  叉乘得到旋转轴，点乘得到夹角，合成 $\Delta R$。
- 第 265-267 行：构造目标相机位姿。
  **旋转**用 $\Delta R$ 修正过的，**位置**直接取"物体位置 + 正上方 `height` 米"。
  这样相机就悬在物体正上方、垂直朝下看。
- 第 271-281 行：**绕 Z 轴（竖直轴）每 15° 生成一个候选**，
  $360/15 = 24$ 个。为什么不只算一个？因为绕竖直轴转多少都不影响"垂直朝下看物体"，
  但**对机械臂来说差别巨大**——某个朝向可能要手腕翻转一大圈。
- 第 285 行 `arm.inverse_kinematics(...)`：**逆运动学**（IK）——
  已知末端该在哪，反算 6 个关节各转多少。**注意它一次接收整个列表**（批量求解，更快）。
  第二个参数 `[cur_joints] * len(...)` 是"以当前关节角作为求解初值"，帮助 IK 收敛到附近的解。
- 第 294-314 行：在 24 个解里挑**关节总变化量最小**的。
  注意 `range(1, len(joints))` **从第 1 个关节开始，跳过了第 0 个**（底座回转关节）——
  大概是作者认为底座转多少无所谓，或者底座转动代价另行考虑。

### 3.5 `compute_ready_pose`（第 329–388 行）

```python
351     # 计算最佳模板
352     if tmpl_idx >= 0:  # 已经指定了模板索引
353         grasp_tmpl = tmpl_state_list[tmpl_idx]
...
357         target_T_base_end, _ = compute_tmpl_ready_pose(sym_tfs, ..., cur_T_cam_model)
363         best_target_T_base_end = target_T_base_end
364         best_tmpl_idx = tmpl_idx
365     else:  # 未指定模板索引, 计算最佳模板
366         tmpl_num = len(tmpl_state_list)
367         min_delta_angle = 1000.0
368         for i in range(tmpl_num):
369             grasp_tmpl = tmpl_state_list[i]
...
373             target_T_base_end, delta_angle = compute_tmpl_ready_pose(...)
379             if delta_angle < min_delta_angle:
380                 min_delta_angle = delta_angle
381                 best_tmpl_idx = i
382                 best_target_T_base_end = target_T_base_end
383             # end if
384         # end for
385     # end if
```

**业务作用**：在**所有模板**之间再挑一次最优的（上一层是在对称性之间挑）。

**逐行讲解：**

- 第 352-364 行：**如果调用方已经指定了模板编号**（第一次循环后就用选定的那个，见第 678 行），
  就不再比较，直接算。这样保证细化阶段**不会跳到别的模板去**（否则会来回横跳）。
- 第 365-384 行：未指定时遍历所有模板，比 `delta_angle`，取最小。

> **两层择优**：先在每个模板内部用对称性挑（`compute_tmpl_ready_pose`），
> 再在所有模板之间挑（`compute_ready_pose`）。合起来就是"全局最省劲的那个抓法"。

### 3.6 `do_grasp`（第 391–834 行）

这是最长的函数（440 多行）。按 11 步拆开讲。

```python
426     T_cam_end = inv_tf(T_end_cam)
427
428     max_refine_cnt = 2  # 最大细化次数
429
430     show_locate_err = False  # 是否显示定位误差
...
446     cur_T_cam_model, _ = client_node.locate_model(timeout_sec=20.0,
447                                                   use_cache=use_cache,
448                                                   debug_level=debug_level)
449     if cur_T_cam_model is None:
450         logging.error(f'{RED}match model failed, exit run.{RESET}')
451         return -1  # 已经是使用缓存了,但仍然失败,说明没有物体,可以退出程序
```

**第 1 步（第 438-454 行）定位物体**：调服务拿 `cur_T_cam_model`。
失败就 `return -1`（退出程序）——注释解释了：既然已经用了缓存还失败，说明**场景里真没东西了**。

```python
464     check_height = CHECK_HEIGHT
465     cur_T_base_end = cache_T_base_end  # 这里必须使用处于检测位置时候的末端位姿
466     cur_joints = cache_joints
467     target_T_base_end, target_joints = compute_check_state(...)
473     if target_T_base_end is None:
474         logging.warning(f"{RED}compute check state failed.{RESET} try check next label.")
475         return 0
476     # end if
...
485     if not check_arm_pose(T_base_end=target_T_base_end, ..., th_angle_z=TH_ANGLE_Z,
490                           th_gripper_height=TH_GRIPPER_HEIGHT):
491         logging.warning(f"{RED}arm pose check failed at ready pose.{RESET} try check next label.")
492         return 0
```

**第 2 步（第 456-493 行）算检查位姿**：
- 第 465 行注释很重要：**必须用缓存里的检测位位姿**，不能用 `arm.get_pose()`。
  因为复用缓存时，手臂可能还在别的地方。
- 第 485-490 行 `check_arm_pose`：检查末端角度和高度合不合理（防止手腕歪过头、夹爪插进桌面）。

**第 3-4 步（第 495-531 行）移动 + 重新定位**：

```python
507     is_ok = arm.set_joints(target_joints)
...
520     cur_T_base_end = arm.get_pose()
521     cur_T_end_base = inv_tf(cur_T_base_end)
522     init_T_cam_model = T_cam_end @ cur_T_end_base @ prev_T_base_end @ T_end_cam @ prev_T_cam_model
523     cur_T_cam_model, _ = client_node.locate_model(init_T_cam_model=init_T_cam_model, ...)
```

- 第 507 行用 `set_joints`（直接给关节角），因为第 2 步算的就是关节角。
- 第 522 行：**用机械臂自己的位移，把上一刻的物体位姿"推"到现在**，作为跟踪的初值。
  这样匹配服务只需要在一个很小的范围内搜索，又快又稳（详见 6.3 节）。

**第 5 步（第 533-569 行）算预备位姿 + 挑模板**：

```python
537     target_T_base_end, tmpl_idx = compute_ready_pose(tmpl_state_list=..., tmpl_idx=-1)
...
546     ready_T_cam_model = tmpl_state_list[tmpl_idx]['ready_T_cam_model']
547     ready_T_base_end = tmpl_state_list[tmpl_idx]['ready_T_base_end']
548     grasp_T_base_end = tmpl_state_list[tmpl_idx]['grasp_T_base_end']
549     grasp_gripper_dist = tmpl_state_list[tmpl_idx]['grasp_gripper_dist']
550
551     # 计算从 ready 位姿到 grasp 位姿的增量
552     delta_T_end = inv_tf(ready_T_base_end) @ grasp_T_base_end  # 末端坐标系下的位姿增量
553
555     target_gripper_dist = grasp_gripper_dist + 0.015  # 预备位姿时先稍微放开一点夹爪
```

- 第 552 行 `delta_T_end`：**"从预备位到抓取位"的相对位移**，是**在末端坐标系下**表达的。
  这样后面只要"当前末端位姿 @ 这个增量"就能得到抓取位姿（第 739 行）。
- 第 555 行：预备位时夹爪比抓取时**多张开 15 mm**，方便观察和下探；真正抓时再收。

**第 6 步（第 571-598 行）碰撞检查**：

```python
585     is_obstacled = collision_detector.check(gripper_dist=target_gripper_dist,
586                                             ref_T_base_end=cur_T_base_end,
587                                             target_T_base_end=target_T_base_end,
588                                             ref_bgr_img=frames[0][0],
589                                             ref_depth_img=frames[0][1],
590                                             max_depth_diff=0.01,
591                                             debug_level=debug_level)
```

把夹爪的矩形"投影"到当前视角的深度图上，比较"夹爪应该离相机多远"和"真实场景离相机多远"。
如果夹爪比真实表面还靠后（更深），就判定会撞。`max_depth_diff=0.01` 是 1 cm 的容差。

**第 7 步（第 600-622 行）移动到预备位**：先收紧夹爪（缩小碰撞范围），再 `set_pose`。
`th_pos_err=0.0005` 是**位置到位精度 0.5 mm**——相当严格。

**第 8-9 步（第 641-712 行）迭代细化 2 次**：

每轮都是"再跟踪一次 → 再算预备位姿 → 再移动过去"。
因为第 7 步移动之后，手臂可能有微小误差，物体的相对位姿也跟着变了，
再测一次能收敛得更准。第 678 行传 `tmpl_idx=tmpl_idx` **锁定模板**，避免来回跳。

**第 10-11 步（第 765-833 行）再查碰撞 → 抓取**：

```python
739     target_T_base_end = cur_T_base_end @ delta_T_end
742     target_gripper_dist = grasp_gripper_dist + 0.008  # 刚好比物体宽一点
...
779     is_obstacled = collision_detector.check(gripper_dist=grasp_gripper_dist + 0.004, ..., max_depth_diff=0.015)
...
804     is_ok = arm.set_pose(target_T_base_end, move_line=True)
811     is_ok = arm.set_gripper_dist(grasp_gripper_dist - 0.008)
817     time.sleep(0.3)  # 等待夹爪闭合完成
820     target_T_base_end[2, 3] += 0.1
822     is_ok = arm.set_pose(target_T_base_end, move_line=True)
```

- 第 739 行：当前末端位姿 × 增量 = 抓取位姿。
- 第 804 行 `move_line=True`：**走直线**（而不是关节插值的曲线），保证垂直下探不会划到旁边。
- 第 811 行：闭合夹爪时**比 `grasp_gripper_dist` 再紧 8 mm**（靠弹性夹紧物体）。
- 第 820-822 行：抬高 10 cm 再走，避免拖着物体撞到别的东西。

**返回值语义（第 422-423 行）**：`-1` 退出程序 / `0` 失败但可复用缓存 / `1` 成功。

### 3.7 `run`（第 837–961 行）

```python
852     detect_joints = grasp_tmpl_dict['detect_joints']
...
859     use_cache = False
860     cache_T_base_end = None
861     cache_joints = None
862     while rclpy.ok():
863
864         print(f"\n{GREEN}start loop {RESET}")
865
866         ######## 0. 移动到检测位置 ########
867         if not use_cache:  # 不使用缓存, 需要重新检测
...
892             cache_joints = arm.get_joints()
893             cache_T_base_end = arm.get_pose()
894         # end if
895
896         status = do_grasp(...)
909
911         if status == -1:
912             break
913         elif status == 0:
915             use_cache = True
916             continue
917         elif status == 1:
919             use_cache = False
...
933         target_T_base_end = place_T_base_end.copy()
934         target_T_base_end[2, 3] = arm.get_pose()[2, 3]  # 保持当前高度
935         is_ok = arm.set_pose(target_T_base_end, move_line=True)
...
948         is_ok = arm.set_gripper_dist(place_gripper_dist)
...
957     arm.set_joints(arm.init_joints)
```

**业务作用**：外层大循环，把"找 → 抓 → 放"串起来，直到没东西可抓。

**逐行讲解：**

- 第 867-894 行：`use_cache=False` 时才回检测位；回完之后**把位姿和关节角缓存起来**，
  供 `do_grasp` 第 2 步用（那里强调必须用缓存值）。
- 第 933-935 行：放置时**只改 X/Y 和姿态，高度保持当前**（`[2,3]` 是平移的 z 分量）——
  这样不会先升高再下降，路径更短。
- 第 941 行：再用 `set_joints(place_joints)` 精确对一遍关节角。
  **先 `set_pose` 再 `set_joints` 有点重复**，但如果两者一致则无害。
- 第 957 行：退出循环后**回到初始关节角**（`arm.init_joints`），安全收尾 ✓

### 3.8 `__main__`（第 967–1139 行）

```python
971     parser.add_argument("--color_img_topic", type=str, required=True, ...)
974     parser.add_argument("--depth_img_topic", type=str, required=True, ...)
977     parser.add_argument("--tmpl_dir", type=str, required=True, ...)
980     parser.add_argument("--model_name", type=str, default=None, ...)
983     parser.add_argument("--debug", action='store_true', ...)
...
1023    camera_param_path = os.path.join(root_dir, 'data/calib/cam_params.json')
...
1032    handeye_calib_path = os.path.join(root_dir, 'data/calib/calib_handeye.json')
1033    T_end_cam, _ = read_handeye_calib(handeye_calib_path)
...
1037    gripper_path = os.path.join(root_dir, 'data/calib/gripper_body.json')
1042    gripper_data_dict = mmengine.load(gripper_path)
...
1070    is_ok = arm.set_gripper_dist(0.02)
1076    is_ok = arm.set_gripper_dist(0.07)
...
1083    collision_detector = CollisionDetector(gripper_body=gripper_body, ..., debug_dir=COLLISION_DIR)
...
1093    cam_node = CamNode(img_topic_list=[color_img_topic, depth_img_topic])
1094    arm_node = TargetArmNode()
1095    client_node = ClientNode(model_name=model_name)
1096
1098    wait_cnt = 0
1099    max_wait_cnt = 10
1100    while not client_node.client.service_is_ready():
...
1112    sym_tfs, _ = client_node.get_symmetric_info(timeout_sec=5.0)
1113    if sym_tfs is None:
1114        logging.error('get symmetric info failed. exiting')
1115        exit(1)
```

**业务作用**：读配置、建对象、等服务、拿对称性，最后启动 `run`。

**逐行讲解：**

- 第 980 行 `--model_name`：可选。指定了之后服务名变成 `MATCHING3D_SRV_BASE_NAME/{model_name}`，
  **可以同时跑多个匹配服务**（比如同时抓不同物体）。
  第 1005 行 `reset_empty_str` 把空字符串变成 `None`，避免拼出奇怪的服务名。
- 第 983 行 `action='store_true'`：**开关型参数**，写了 `--debug` 就是 `True`，不写就是 `False`。
- 第 1023/1032/1037 行：**三个标定文件的路径是写死的**（都在 `data/calib/` 下），
  不走命令行参数。好处是简单，坏处是换机器人/换相机就得改代码（见第 7 节问题 7）。
- 第 1070-1080 行：**夹爪先合到 20 mm 再开到 70 mm**，
  注释说是"表明程序已经启动"——一个很实用的**视觉/听觉提示**，让人知道程序跑起来了。
- 第 1098-1109 行：等服务就绪，**最多等 10 秒**（每秒查一次），超时就退出。
- 第 1112-1116 行：拿物体的**对称性信息** `sym_tfs`。拿不到就直接退出——
  因为后面 `compute_tmpl_ready_pose` 必须要它。
- 第 1132-1135 行：`run` 返回后销毁三个节点并 `rclpy.shutdown()` ✓ 清理完整。

---

## 4. Python 基础语法速查

| 写法 | 含义 | 本文件出现位置 |
|---|---|---|
| `-> Tuple[np.ndarray, float]` | 返回值类型标注（不强制） | 第 174 行 |
| `np.zeros((N, 4, 4))` | 造一个 $N \times 4 \times 4$ 的全 0 数组 | 第 194 行 |
| `[x] * N` | 把列表重复 N 次 | 第 195、285 行 |
| `np.argmin(arr)` | 返回最小值的下标 | 第 227 行 |
| `np.clip(v, -1, 1)` | 把值夹在 [-1, 1]，防 `arccos` 出 nan | 第 207、214、257 行 |
| `np.trace(R)` | 矩阵迹（对角线之和），用来反求旋转角 | 第 206 行 |
| `np.cross(a, b)` | 叉乘，得到同时垂直于 a、b 的轴 | 第 262 行 |
| `np.arange(0, 2*np.pi, step)` | 生成等差序列（这里用来遍历 yaw） | 第 273 行 |
| `enumerate(list)` | 同时拿到下标和元素 | 第 297 行 |
| `arr[:3, 2]` | 取前 3 行、第 2 列（即 Z 轴方向） | 第 211、255 行 |
| `while True: ... break` | 无限循环 + 条件跳出（用来扫模板编号） | 第 114-118 行 |
| `action='store_true'` | argparse 的开关型参数 | 第 983 行 |
| `@` | numpy 的矩阵乘法（等价于 `np.matmul`） | 第 199、251 行等 |

---

## 5. 运行方式与输入输出

**前置条件**（缺一个就跑不起来）：

1. 机械臂已上电、能被 `ArmWrapper` 连上；
2. RGB-D 相机在发布 `--color_img_topic` 和 `--depth_img_topic`；
3. **匹配服务已启动**（本脚本只等服务 10 秒）；
4. 模板目录已用 `create_tmpl_grasp_3d.py` 采好，且包含 `detect.json`、`place.json`、
   以及**编号连续的** `0/`、`1/`… 子目录；
5. `data/calib/` 下有 `cam_params.json`、`calib_handeye.json`、`gripper_body.json`。

**运行命令**：

```bash
python demo/test_tmpl_grasp_3d/test_tmpl_grasp_3d_romate.py \
    --color_img_topic  /camera/color/image_raw \
    --depth_img_topic  /camera/depth/image_raw \
    --tmpl_dir         demo/test_tmpl_grasp_3d/tmpl/box_01 \
    --model_name       box_01        # 可选
    --debug                          # 可选，逐步暂停
```

**典型日志长这样**（节选）：

```
start loop
step [0] , move to detect pose
grasp-step [1] , locate model by matching
match T_cam_model: [[...]]
grasp-step [2] , compute check pose
current cam Z dir: [0.02 -0.01 0.99], need adjust to [0 0 -1], angle(deg): 8.13
grasp-step [3] , move arm to check pose
grasp-step [4] , relocate model by tracking
grasp-step [5] , compute ready pose and select best grasp template
selected tmpl idx: 0
grasp-step [6] , check if gripper is obstacled
is gripper obstacled: False
grasp-step [7] , move to ready pose
grasp-step [8-1] , track model
grasp-step [9-1] , move to ready pose again
grasp-step [8-2] , track model
grasp-step [9-2] , move to ready pose again
reached max refine count.
grasp-step [10] , check if gripper is obstacled at grasp pose
grasp-step [11] , move to grasp pose
grasp task success, try next step...
step [-1] , move to place pose
```

**返回值 / 终止条件**：

| 现象 | 含义 |
|---|---|
| 日志出现 `match model failed, exit run.` | 场景里没有可抓的物体了，程序正常退出 |
| `arm pose check failed` / `gripper is obstacled` | 这个物体这次抓不了，换下一个（继续循环） |
| `service not available, exiting` | 匹配服务没起来 |
| `no valid grasp tmpl found` | 模板目录有问题 |

---

## 6. 核心数学：一步一步算给你看

### 6.1 目标位姿公式：把"物体偏了多少"搬到手上

核心在第 199 行。把它写成人话：

$$T_{base\_end}^{tgt} = T_{base\_end}^{cur} \cdot T_{end\_cam} \cdot T_{cam\_model}^{cur} \cdot S \cdot (T_{cam\_model}^{rdy})^{-1} \cdot (T_{end\_cam})^{-1}$$

从右往左读（矩阵乘法从右往左作用）：

1. $(T_{end\_cam})^{-1}$：从**相机**回到**末端**；
2. $(T_{cam\_model}^{rdy})^{-1}$：从"模板里的物体"回到"模板时的相机"——
   也就是**求出"物体相对模板姿态偏了多少"**；
3. $S$：套上对称变换（物体转一下还是它自己，见 6.2 节）；
4. $T_{cam\_model}^{cur}$：把这个偏差**搬到当前相机下**；
5. $T_{end\_cam}$：再搬到末端；
6. $T_{base\_end}^{cur}$：最后搬到基座。

一句话：**先算"物体相对标准姿势偏了多少"，再把这个偏差以当前机械臂为基准重新施加一遍。**

### 6.2 对称性怎么让机械臂"抄近路"

假设物体是个圆柱，绕自己的轴（记为 Z 轴）转任意角度都一样。
那 `sym_tfs` 里就会有一串矩阵 $S_k$，每个都是绕 Z 轴转 $\theta_k$。

没有对称性时：机械臂必须转到让物体"完全对齐"模板姿态，假设要转 170°。
有对称性时：程序发现**转 170° - 180° = -10°** 也能得到"看起来一样"的结果，
于是挑那个只要转 10° 的解。

代码里的体现就是第 197-225 行的循环：**对每个 $S_k$ 都算一遍目标位姿和所需转角**，
最后 `argmin` 取最小。

> 这就是为什么 `sym_tfs` 拿不到就要 `exit(1)`（第 1113-1115 行）——
> 少了它，机械臂可能做一些完全不必要的翻转。

### 6.3 跟踪：用机械臂自己的位移预测物体现在在哪

第 522 行（以及第 656、723 行）这个式子：

$$T_{cam\_model}^{pred} = (T_{end\_cam})^{-1} \cdot (T_{base\_end}^{cur})^{-1} \cdot T_{base\_end}^{prev} \cdot T_{end\_cam} \cdot T_{cam\_model}^{prev}$$

思路：**物体没动，动的是机械臂**（相机装在末端上）。
所以：

1. $T_{cam\_model}^{prev}$：上一刻物体在相机里的位姿；
2. $\cdot\ T_{end\_cam}$：搬到末端坐标系；
3. $\cdot\ T_{base\_end}^{prev}$：搬到**基座**坐标系（这时物体位置就固定下来了）；
4. $\cdot\ (T_{base\_end}^{cur})^{-1}$：用**新的**手臂位姿，回到现在的末端坐标系；
5. $\cdot\ (T_{end\_cam})^{-1}$：回到现在的相机坐标系。

得到的就是"物体现在应该在相机里的哪个位置"——**一个预测值**。
把它作为 `init_T_cam_model` 传给匹配服务，服务只需要在预测值附近小范围搜索，
**又快又不容易跟丢**。这就是 `track`（跟踪）和 `match`（从头匹配）的区别。

### 6.4 检查位姿：绕 Z 轴每 15 度试一次，挑关节动得最少的

第 265-281 行构造了一个"相机悬在物体正上方 0.18 m、垂直朝下"的位姿。
但这个位姿**只确定了"看哪儿"和"朝哪看"，没确定"相机自己绕视线转多少"**——
也就是相机的**滚转（roll）**是自由的。

差别在哪？看这张表：

| 滚转角 | 相机看到的画面 | 机械臂手腕的姿态 |
|---|---|---|
| 0° | 正着 | 舒服 |
| 90° | 侧着（画面转了 90°） | 舒服 |
| 180° | 倒着 | 可能要翻转一大圈 |

反正画面转了也能识别（匹配是几何的，不在乎画面朝向），
所以**哪个让机械臂最省力就用哪个**。

代码就是这么干的：绕 Z 轴每 15° 生成一个候选（24 个），
全部丢给 IK 求逆解，然后挑**关节角变化总和最小**的那个（第 294-314 行）。

---

## 7. 这段代码里的坑与改进建议

| # | 问题 | 后果 | 建议 |
|---|---|---|---|
| 1 | 第 114-124 行扫模板时**遇到第一个残缺就 `break`** | `0/`、`1/` 齐全但 `2/` 缺文件 → `3/`、`4/` 全被跳过，明明有 4 个模板只读到 2 个 | 改成"跳过残缺的继续扫"，并统计跳过数量告警 |
| 2 | 第 207 行 `angle = abs(np.arccos(cosine))` 之后，第 220-222 行**再加 $2\pi$ 惩罚** | 惩罚后角度可能 > $2\pi$，语义上不再"轴角"，只是排序用；阅读时容易误解 | 单独维护一个 `cost` 变量做排序，`angle` 保持纯几何含义 |
| 3 | 第 307 行 `for i in range(1, len(joints))` **跳过第 0 个关节** | 底座回转可能非常大却不计入代价，选出来的解可能要底座转半圈 | 若要保留，加注释说明；否则改成 `range(0, ...)` 或给第 0 关节单独加权 |
| 4 | 第 465 行依赖 `cache_T_base_end` 非 `None` | 首次进入 `do_grasp` 时若 `use_cache=False`，`run` 第 892-893 行刚赋值，没问题；但**单独调用 `do_grasp` 且传 `None`** 会在 `inv_tf` 处崩 | 在函数入口加 `assert` 或显式判空 |
| 5 | 第 1023/1032/1037 行**三个标定路径写死** | 换机器人 / 换相机必须改代码 | 改成命令行参数，默认值指向 `data/calib/` |
| 6 | 第 941 行 `set_pose` 之后又 `set_joints` | 两次下发，若两者不一致会看到手臂"抖一下" | 确认模板里 `place_T_base_end` 与 `place_joints` 一致；不一致时只保留一个 |
| 7 | 第 811 行 `set_gripper_dist(grasp_gripper_dist - 0.008)` | **硬编码 8 mm 的"过盈量"**；物体软硬不同会夹不稳或夹坏 | 提到模板里做成 `grasp_gripper_overlap` 字段 |
| 8 | 第 53 行 `from examples.app.src.client_matching3d import ClientNode` | import 路径用的是 `examples/`，而本文件在 `demo/` 下；依赖 `root_dir` 下真实存在该目录，否则 `ModuleNotFoundError` | 确认目录，或改成相对本文件的路径 |
| 9 | `show_locate_err = False`（第 430 行）**默认关闭** | 定位误差（位置 mm / 角度 deg）默认看不到，排查时很关键 | 加个命令行开关，默认开或在 debug 下自动开 |
| 10 | 第 934 行 `target_T_base_end[2, 3] = arm.get_pose()[2, 3]` 直接改 z | 若放置位比当前低，会水平撞过去 | 改成"先抬到安全高度 → 平移 → 再下降"三段式 |

---

## 8. 一句话总结

> **本文件是整个 3D 抓取系统的"主执行器"：它站在检测位扫视，
> 挪到检查位把相机摆正看清楚，在多个抓取模板里挑一个转起来最省劲的
> （顺便用物体的对称性抄近路），迭代细化两次确认没看错，
> 再验两遍碰撞，然后垂直下探、夹紧、抬高、送到放置位放下——
> 抓完立刻回到检测位接着抓下一个，直到场景里空了为止。**

# `utils.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、也没接触过机器人/相机标定的同学。
> 目标：读完之后，你能逐行看懂 `core/utils.py` 在干什么，知道每个数字是怎么算出来的，知道谁在调用它、出了错往哪查。
> 被分析的源文件：`C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\core\utils.py`（共 371 行）。

---

## 目录

- [0. 一句话概括](#0-一句话概括)
- [1. 背景知识](#1-背景知识)
  - [1.1 什么是位姿矩阵 T（4×4 表格）](#11-什么是位姿矩阵-t4times4-表格)
  - [1.2 四元数是什么、为什么用它存旋转、wxyz 与 xyzw 两种顺序](#12-四元数是什么为什么用它存旋转wxyz-与-xyzw-两种顺序)
  - [1.3 为什么要写"求逆"函数而不是用 np.linalg.inv](#13-为什么要写求逆函数而不是用-nplinalginv)
  - [1.4 手眼标定文件里 T_armend_cam 和 T_armbase_cam 分别是什么](#14-手眼标定文件里-t_armend_cam-和-t_armbase_cam-分别是什么)
  - [1.5 终端彩色打印的 ANSI 转义码是什么](#15-终端彩色打印的-ansi-转义码是什么)
- [2. 模块地图](#2-模块地图)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 文件顶部说明 + logging 自动配置（第 1–11 行）](#31-文件顶部说明--logging-自动配置第-1--11-行)
  - [3.2 颜色常量（第 33–46 行）](#32-颜色常量第-3346-行)
  - [3.3 reset_empty_str（第 51–70 行）](#33-reset_empty_str第-5170-行)
  - [3.4 read_cam_params（第 73–114 行）](#34-read_cam_params第-73114-行)
  - [3.5 read_rgbd_params（第 117–162 行）](#35-read_rgbd_params第-117162-行)
  - [3.6 read_calib_handeye（第 165–207 行）](#36-read_calib_handeye第-165207-行)
  - [3.7 inv_tf（第 210–226 行）](#37-inv_tf第-210226-行)
  - [3.8 transform_delta_pose（第 229–255 行）](#38-transform_delta_pose第-229255-行)
  - [3.9 compute_aligned_pose（第 258–279 行）](#39-compute_aligned_pose第-258279-行)
  - [3.10 timestamp_to_str（第 282–293 行）](#310-timestamp_to_str第-282293-行)
  - [3.11 wait_key（第 296–313 行）](#311-wait_key第-296313-行)
  - [3.12 get_key（第 316–329 行）](#312-get_key第-316329-行)
  - [3.13 KeyboardReader 类（第 335–370 行）](#313-keyboardreader-类第-335370-行)
- [4. Python 基础语法速查](#4-python-基础语法速查)
- [5. 输入输出规范](#5-输入输出规范)
- [6. 核心推导：一步一步算给你看](#6-核心推导一步一步算给你看)
  - [6.1 inv_tf 为什么是 [Rᵀ, -Rᵀt; 0 0 0 1]](#61-inv_tf-为什么是-r-t--r-t-t-0-0-0-1)
  - [6.2 transform_delta_pose 的相似变换推导](#62-transform_delta_pose-的相似变换推导)
  - [6.3 compute_aligned_pose 的推导（独立验证）](#63-compute_aligned_pose-的推导独立验证)
  - [6.4 四元数 [qw,qx,qy,qz] 顺序对照](#64-四元数-qwqxqyqz-顺序对照)
  - [6.5 时间戳 ns → 秒 → 字符串的换算](#65-时间戳-ns--秒--字符串的换算)
  - [6.6 数值演练：手算 inv_tf](#66-数值演练手算-inv_tf)
- [7. 这段代码里的坑与改进建议](#7-这段代码里的坑与改进建议)
- [8. 一句话总结](#8-一句话总结)

---

## 0. 一句话概括

> **`utils.py` 是整套抓取工程的"工具箱"：它负责把 JSON 文件里的相机内参、畸变、手眼标定矩阵读成 4×4 的位姿矩阵 T；提供几个矩阵运算（求逆、坐标变换、对齐位姿）；提供把时间戳转成字符串、把键盘输入读出来、以及把日志彩色打印出来的小工具。它本身不控制机器人，但几乎所有上层脚本（`test_tmpl_grasp_2d.py`、`test_tmpl_grasp_3d.py`、`arm_node.py`）都靠它拿到最基础的数字。**

---

## 1. 背景知识

这一段完全不碰代码，先把"为什么这么算"的底层道理讲清楚。读不懂这里，后面代码里的 `T`、`R`、`t`、`q` 都是天书。

### 1.1 什么是位姿矩阵 T（4×4 表格）

"位姿" = 位置（在哪）+ 姿态（朝哪）。机器人世界里，物体、相机、机械臂末端都有一个自己的"坐标系"（可以理解为每个人身上贴了一个 xyz 坐标轴）。

一个坐标系相对另一个坐标系怎么摆，用一张 **4×4 的表格（矩阵）** 表示，约定如下：

```
T = | R   t |
    | 0 0 0 1 |
```

拆开写就是：

```
T = | r11 r12 r13  tx |
    | r21 r22 r23  ty |
    | r31 r32 r33  tz |
    | 0   0   0    1  |
```

- 左上角 3×3 块叫 `R`（rotation 旋转矩阵）：描述"坐标轴朝向"差了多少。R 的每一列都是单位向量，且互相垂直——这保证了它只旋转、不拉伸。
- 右上角 3×1 列叫 `t`（translation 平移向量）：`tx, ty, tz` 就是"原点偏移了多少米"。
- 最后一行永远是 `0 0 0 1`：这是"齐次坐标"的固定写法，记住它只是占位即可。

**名字约定**：`T_a_b`（下标的顺序）表示"从 b 坐标系到 a 坐标系"的变换。比如 `T_base_end` 是"机械臂末端"相对"机械臂底座"的位姿；`T_end_cam` 是"相机"相对"末端"的位姿。把矩阵写在左边相乘 `T_a_c = T_a_b @ T_b_c`，意思就是"先 b→c，再 a→b，连起来得到 a→c"。这是本文件所有运算的根。

### 1.2 四元数是什么、为什么用它存旋转、wxyz 与 xyzw 两种顺序

旋转矩阵 R 有 9 个数，但真正"自由"的只有 3 个（绕三根轴的转角）。用 9 个数存 3 个信息既浪费又容易算着算着不再正交（出现很丑的"拉伸"）。所以工程里常用**四元数**存旋转：4 个数 `[qw, qx, qy, qz]`。

四元数本质是 `q = cos(θ/2) + sin(θ/2)·(xi + yj + zk)`，其中 `(x,y,z)` 是旋转轴，`θ` 是旋转角。它的好处是：**不会"万向锁"、插值平滑、乘法就能复合旋转**，比欧拉角稳得多。

**致命坑——顺序有两种**：
- `wxyz`：把标量 `qw` 放第一个。
- `xyzw`：把 `qw` 放最后一个。

同样四个数字，顺序不同算出来的旋转天差地别。Python 的 `transforms3d` 库规定四元数是 **`[qw, qx, qy, qz]`（w 在前）**；但 ROS、OpenCV 习惯用 **`[x, y, z, w]`（w 在后）**。本工程里 `read_calib_handeye` 的注释写的是 `[qw,qx,qy,qz]`，而 `arm_wrapper.py` 里机械臂返回的位姿是 `[tx,ty,tz,qx,qy,qz,qw]`（w 在最后）。后文 6.4 节会专门把三处代码摆在一起对照，避免你踩顺序错位的坑。

### 1.3 为什么要写"求逆"函数而不是用 np.linalg.inv

`np.linalg.inv(T)` 是"通用求逆"，对 4×4 矩阵做高斯消元，能算但不聪明。而位姿矩阵 T 有特殊结构（最后一行固定 `0 0 0 1`，左上 R 是正交阵）。利用这个结构，**逆有个解析公式**（见 6.1 节）：

```
T⁻¹ = | Rᵀ   -Rᵀ·t |
      | 0 0 0   1   |
```

其中 `Rᵀ` 是 R 的转置。对正交阵 `Rᵀ = R⁻¹`，所以这就等于"精确求逆"。自己写这个公式：
1. 更快（转置 + 一次矩阵乘，不做消元）；
2. 更稳定（不会因为浮点误差把最后一行算歪）；
3. 强制保证"结果仍是合法位姿矩阵"（最后一行还是 `0 0 0 1`）。

所以 `inv_tf` 就是"针对位姿矩阵特制的求逆"，`tf` 即 transform（变换）。

### 1.4 手眼标定文件里 T_armend_cam 和 T_armbase_cam 分别是什么

"手眼标定"= 算出相机相对机械臂的位置关系。两种安装方式：

- **眼在手（eye-in-hand）**：相机装在机械臂末端上，跟着手臂一起动。这时存的是 `T_armend_cam`（相机相对"末端"的位姿）。
- **眼在外（eye-to-hand）**：相机固定在架子上看着整个手臂。这时存的是 `T_armbase_cam`（相机相对"底座"的位姿）。

二者语义完全不同！`T_armend_cam` 会随手臂移动，`T_armbase_cam` 是固定的。代码第 183–190 行就是靠 JSON 里"有哪个字段"来判断是哪种，并返回 `eye_in_hand` 布尔量。后文 7.5 节会讲：调用方把第二个返回值 `_` 丢掉了，喂错文件会酿成大错。

### 1.5 终端彩色打印的 ANSI 转义码是什么

彩色不是 Python 自带的，而是**终端（命令行窗口）认的一串特殊字符**。以 `\033[91m` 为例：

- `\033` 是 ESC 键的 ASCII 码（十进制 27，八进制 033）；
- `[91m` 是"前景色=亮红"的指令。

打印 `RED + "出错了" + RESET` 时，终端先在出错文字前打开红色，文字后再用 `RESET`（`\033[0m`，复位所有样式）关掉，于是只有中间那句是红的。`\033[92m` 绿、`\033[93m` 黄、`\033[94m` 蓝。注意：往文件里重定向时，这些"指令"不会被解释，会变成一堆难看的乱码（见 7.9 节）。

---

## 2. 模块地图

谁是"输入"、谁是"输出"、谁在"调用"，一张表看全：

| 名称 | 行号 | 输入 | 输出 | 被谁调用 | 一句话作用 |
|---|---|---|---|---|---|
| `reset_empty_str` | 51 | 任意参数（多为路径字符串） | 规整后的参数或 None | `arm_node.py` 等 | 空串变 None，路径变规范 |
| `read_cam_params` | 73 | 相机参数 JSON 路径 | `(intrinsic, distortion)` 或 `(None,None)` | `test_tmpl_grasp_2d.py:597` | 读 RGB 相机内参+畸变 |
| `read_rgbd_params` | 117 | RGB-D 参数 JSON 路径 | `(intrinsic, distortion, depth_scale)` 或 3 个 None | `test_tmpl_grasp_3d.py:721` | 读 RGB-D 相机内参+畸变+深度缩放 |
| `read_calib_handeye` | 165 | 手眼标定 JSON 路径 | `(T 4×4, eye_in_hand)` 或 `(None,None)` | `2d:608` `3d:728` `arm_node:75` | 读手眼标定，转成 4×4 矩阵 |
| `inv_tf` | 210 | T 4×4 | T⁻¹ 4×4 | 几乎全部位姿运算 | 位姿矩阵求逆 |
| `transform_delta_pose` | 229 | `T_a0_a1`, `T_a_b` | `T_b0_b1` | 间接调用（见 6.2） | 把 a 系增量换到 b 系 |
| `compute_aligned_pose` | 258 | 4 个 T | `T_a_b1` | 间接/对齐场景 | 求 1 时刻的对齐位姿 |
| `timestamp_to_str` | 282 | 纳秒整数 | `"HH:MM:SS.sss"` | 调试打印 | 时间戳转可读字符串 |
| `wait_key` | 296 | `debug` 布尔 | True/False | `2d/3d` 每步前 | 调试时等回车 |
| `get_key` | 316 | 无 | 单个按键字符 | `arm_node.py`（经 KeyboardReader） | 无回车读一个键 |
| `KeyboardReader` | 335 | 无 | 读键对象 | `arm_node.py:103` | 非阻塞读键盘类 |

---

## 3. 逐段代码精读

按真实行号分小节，贴关键片段 + 逐步讲解。

### 3.1 文件顶部说明 + logging 自动配置（第 1–11 行）

```python
"""
通用工具函数和类
"""

import logging
# 如果还没有配置 logging, 则配置一个默认的 logging 输出到控制台
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        force=True)
```

- 第 1–3 行是模块文档字符串（`"""..."""`），import 时不会被当作代码执行，只用于说明"这是什么文件"。
- 第 5 行 `import logging`：Python 自带的日志库。比 `print` 好的是它能带时间、带级别（INFO/WARNING/ERROR）。
- 第 7 行 `if not logging.getLogger().hasHandlers()`：判断"根日志器有没有被配置过"。如果没有，第 8–11 行 `logging.basicConfig(...)` 才去配置一个默认输出。**作用**：防止别人先配置了 logging，你又配一遍把格式搞乱。`hasHandlers()` 返回 True/False。
- 第 9 行 `format=...` 里的 `%(asctime)s` 是"事件发生时间"，`%(msecs)03d` 是毫秒（3 位补零），`%(levelname)s` 是级别，`%(filename)s:%(lineno)d` 是"出这行日志的文件:行号"。这些占位符由 logging 在运行时自动填。
- 第 11 行 `force=True`：即便之前别人配置过，也强制用我这版。注意这会把别人的配置覆盖掉。

### 3.2 颜色常量（第 33–46 行）

```python
RED = '\033[91m'
"""在终端开启红色打印"""
GREEN = '\033[92m'
"""在终端开启绿色打印"""
YELLOW = '\033[93m'
"""在终端开启黄色打印"""
BLUE = '\033[94m'
"""在终端开启蓝色打印"""
RESET = '\033[0m'
"""在终端重置打印的颜色"""
```

这 5 行就是 1.5 节说的 ANSI 转义码，定义成全局常量，方便后面写 `f"{GREEN}成功{RESET}"` 这样的彩色日志（见 5 节例子）。注意 `RESET` 必须跟在彩色文字**后面**，否则后面的所有字都会一直带色。

### 3.3 reset_empty_str（第 51–70 行）

```python
def reset_empty_str(param):
    if param is not None:
        if isinstance(param, str):
            if len(param) == 0:
                return None
            elif os.path.exists(param):
                param = os.path.normpath(param)  # 变成规范的路径
    return param
```

**干什么**：把一个"参数"收拾干净。
- 第 60 行 `if param is not None`：`is` 是判断"是不是同一个对象"，`None` 是空值标记。传进来的不是 None 才往下走。
- 第 61 行 `isinstance(param, str)`：判断 param 是不是字符串类型（`isinstance` 是类型检查，见 4 节）。
- 第 62–63 行：如果是空字符串 `""`（长度 0），直接返回 None——因为空串当路径用会出错。
- 第 64–65 行 `elif os.path.exists(param)`：如果这是个真实存在的路径，用 `os.path.normpath` 把它变成规范路径（比如把 `./a/../b` 变成 `b`，去掉多余的 `.` 和 `..`）。
- 第 69 行：前面都没命中（比如 param 不是字符串，或字符串但路径不存在），原样返回。

注意这是"嵌套 if"写法，可读性一般，4 节会讲更清爽的写法。

### 3.4 read_cam_params（第 73–114 行）

```python
def read_cam_params(json_file_path: str) -> Tuple[List[float], List[float]]:
    logging.info(f"Try to read camera parameters from: {GREEN}{json_file_path}{RESET}")
    try:
        with open(json_file_path, 'r') as f:
            param = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read camera parameters from {json_file_path}: {e}")
        return None, None

    if 'serial_number' in param:
        logging.info(f"camera serial number: {GREEN}{param['serial_number']}{RESET}")

    if 'intrinsic' not in param:
        logging.error("JSON file must contain 'intrinsic' field.")
        return None, None

    intrinsic = param['intrinsic']
    if 'distortion' in param:
        distortion = param['distortion']
    else:
        distortion = None
    logging.info(f"intrinsic: {GREEN}{intrinsic}{RESET}")
    logging.info(f"distortion: {GREEN}{distortion}{RESET}")
    return intrinsic, distortion
```

**干什么**：读相机内参 JSON，返回 `(内参列表, 畸变列表)`。
- 第 73 行函数签名里的 `json_file_path: str` 和 `-> Tuple[List[float], List[float]]` 是**类型标注**（见 4 节），告诉读者"参数应是字符串，返回应是一个装了两个 float 列表的元组"。它**不强制**运行时检查，纯属文档。
- 第 86–87 行 `with open(...) as f: param = json.load(f)`：`json.load` 把 JSON 文件解析成 Python 字典。`with` 会在读完自动关文件（见 4 节）。
- 第 89–92 行 `except Exception as e`：如果文件不存在/不是合法 JSON，捕获所有异常，打错误日志，**返回 `(None, None)`**。这就是后面 7.4 节要讲的坑：调用方可能拿到 None 还不检查。
- 第 95 行 `'serial_number' in param`：字典的成员判断，`in` 检查键是否存在。
- 第 99 行 `'intrinsic' not in param`：JSON 必须有 `intrinsic` 字段，没有就报错返回 None。
- 第 105–108 行：畸变可选，没有就设 None。
- 第 113 行 `return intrinsic, distortion`：返回两个值（Python 自动打包成元组）。

**被谁调用**：`test_tmpl_grasp_2d.py:597` `intrinsic, distortion = read_cam_params(...)`。

### 3.5 read_rgbd_params（第 117–162 行）

和 3.4 几乎一样，只多了"深度缩放系数 `depth_scale`"：

```python
    if 'intrinsic' not in param or 'depth_scale' not in param:
        logging.error("JSON file must contain 'intrinsic' and 'depth_scale' fields.")
        return None, None, None
    ...
    depth_scale = float(param['depth_scale'])
    return intrinsic, distortion, depth_scale
```

`depth_scale` 是把"深度图里的整数像素值"乘成"真实米数"的系数（深度相机通常输出毫米或 0.001 米为整数）。**被谁调用**：`test_tmpl_grasp_3d.py:721`。

**坑**：出错时返回 `(None, None, None)` 三个 None，调用方要逐个检查（见 7.4）。

### 3.6 read_calib_handeye（第 165–207 行）

```python
def read_calib_handeye(json_file_path: str) -> Tuple[np.ndarray, bool]:
    if os.path.exists(json_file_path) is False:
        logging.error(f'Hand-eye calibration file not found: {json_file_path}')
        return None, None

    with open(json_file_path, 'r') as f:
        calib_dict = json.load(f)

    if 'T_armend_cam' in calib_dict:
        q = np.array(calib_dict['T_armend_cam']["q"], dtype=np.float32)  # [qw,qx,qy,qz]
        t = np.array(calib_dict['T_armend_cam']["t"], dtype=np.float32)
        eye_in_hand = True
    elif 'T_armbase_cam' in calib_dict:
        q = np.array(calib_dict['T_armbase_cam']["q"], dtype=np.float32)  # [qw,qx,qy,qz]
        t = np.array(calib_dict['T_armbase_cam']["t"], dtype=np.float32)
        eye_in_hand = False
    else:
        logging.error("JSON file does not contain valid hand-eye calibration data.")
        return None, None

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = transforms3d.quaternions.quat2mat(q)
    T[:3, 3] = t

    if eye_in_hand:
        logging.info(f"T_end_cam: \n{GREEN}{T}{RESET}")
    else:
        logging.info(f"T_base_cam: \n{GREEN}{T}{RESET}")
    return T, eye_in_hand
```

**干什么**：把"手眼标定 JSON"变成 4×4 位姿矩阵 T。
- 第 174 行 `os.path.exists(...) is False`：文件不在就返回 `(None, None)`。
- 第 183–190 行：靠"有 `T_armend_cam` 还是 `T_armbase_cam`"决定是眼在手还是眼在外（见 1.4）。`q` 取四元数、`t` 取平移。
- 第 184/188 行注释 `# [qw,qx,qy,qz]`：明确告诉读者，这里四元数顺序是 **w 在前**。这是 6.4 节对照的核心。
- 第 196 行 `np.eye(4, dtype=np.float32)`：造一个 4×4 单位矩阵（对角线 1，其余 0）。
- 第 197 行 `transforms3d.quaternions.quat2mat(q)`：把四元数 q（w 在前）转成 3×3 旋转矩阵 R，塞进 T 左上角 `T[:3,:3]`。
- 第 198 行 `T[:3, 3] = t`：把平移 t 塞进右上 3×1 列。
- 第 200–204 行：眼在手时日志写 `T_end_cam`（相机相对末端），眼在外写 `T_base_cam`。
- 第 206 行 `return T, eye_in_hand`：返回矩阵 + 布尔量。

**被谁调用**：`2d:608`、`3d:728`、`arm_node:75`，都写成 `T_end_cam, _ = read_calib_handeye(...)`——把第二个布尔量 `_` 丢掉（见 7.5 节大坑）。

### 3.7 inv_tf（第 210–226 行）

```python
def inv_tf(T: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), "Input transformation matrix must be of shape 4x4"
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
```

**干什么**：位姿矩阵求逆（公式见 6.1）。
- 第 218 行 `assert`：断言 T 必须 4×4，否则报错停机（见 4 节）。
- 第 220 行 `R = T[:3, :3]`：切出左上 3×3 旋转块。
- 第 221 行 `t = T[:3, 3]`：切出右上平移列。
- 第 222 行 `np.eye(4, dtype=T.dtype)`：造逆矩阵骨架，并**保留 T 的数据类型**（float32/float64，见 7.6 节）。
- 第 223 行 `R.T`：R 的转置（`.T` 是 numpy 转置属性）。
- 第 224 行 `-R.T @ t`：`@` 是矩阵乘法，算出 `-Rᵀ·t`。
- 第 225 行返回。

### 3.8 transform_delta_pose（第 229–255 行）

```python
def transform_delta_pose(T_a0_a1: np.ndarray,
                         T_a_b: np.ndarray) -> np.ndarray:
    """
    ...（docstring 见 6.2 对照）...
    Examples:
        // 已知的变换关系
        T_w_a * T_a_b = T_w_b, T_w_a0 * T_a0_a1 = T_w_a1, T_w_b0 * T_b0_b1 = T_w_b1
        // 已知量: T_a_b, T_w_a0, T_w_a1, T_w_b0,  待求解: T_w_b1
        // 推导
        T_w_a0 * T_a_b = T_w_b0 ,
        T_w_a1 * T_a_b = T_w_b1 -->> T_w_a0 * T_a0_a1 * T_a_b = T_w_b0 * T_b0_b1 = T_w_a0 * T_a_b * T_b0_b1
        T_b0_b1 = inv(T_a_b) * T_a0_a1 * T_a_b  # 这行代码就是函数实现
        T_w_b1 = T_w_b0 * T_b0_b1  # 结果
    """
    T_b0_b1 = inv_tf(T_a_b) @ T_a0_a1 @ T_a_b
    return T_b0_b1
```

**干什么**：已知 a 系下的"位姿增量" `T_a0_a1`，以及固定的 a→b 变换 `T_a_b`，求 b 系下对应的增量 `T_b0_b1`。本质是"相似变换"（见 6.2）。
- 第 253 行就是 docstring 里那句 `inv(T_a_b) * T_a0_a1 * T_a_b`，只是把 `inv` 换成 `inv_tf`，`*` 换成 numpy 的 `@`。

注意 docstring 里用的是 `//` 注释风格（C++ 习惯），Python 里 `#` 才是注释，这些 `//` 实际是 docstring 里的文本，不会被执行（见 7.8 节）。

### 3.9 compute_aligned_pose（第 258–279 行）

```python
def compute_aligned_pose(T_a_b0: np.ndarray,
                         T_b_c: np.ndarray,
                         T_c0_d: np.ndarray,
                         T_c1_d: np.ndarray) -> np.ndarray:
    """
    计算对齐位姿
    已知不变量 T_b_c, 0 时刻的 T_a_b0, T_c0_d, 1 时刻的 T_c1_d,
    求解 1 时刻的 T_a_b1, 使得 T_a_b1 * T_b_c * T_c1_d = T_a_b0 * T_b_c * T_c0_d
    ...
    """
    T_a_b1 = T_a_b0 @ T_b_c @ T_c0_d @ inv_tf(T_b_c @ T_c1_d)
    return T_a_b1
```

**干什么**：已知"0 时刻 a→b 的位姿"和若干中间变换，求"1 时刻 a→b 的位姿"，让前后两端对齐（公式推导见 6.3）。
- 第 277 行是核心：`T_a_b1 = T_a_b0 @ T_b_c @ T_c0_d @ inv_tf(T_b_c @ T_c1_d)`。
- 我会在 6.3 节独立推一遍，验证 docstring 的"使 `T_a_b1 * T_b_c * T_c1_d = T_a_b0 * T_b_c * T_c0_d`"是否成立——结论是**成立**。

### 3.10 timestamp_to_str（第 282–293 行）

```python
def timestamp_to_str(ns: int) -> str:
    dt = datetime.fromtimestamp(ns / 1_000_000_000)
    return dt.strftime("%H:%M:%S.%f")[:-3]
```

**干什么**：把纳秒整数转成 `HH:MM:SS.sss` 字符串。
- 第 290 行 `ns / 1_000_000_000`：纳秒 ÷ 10⁹ = 秒（下划线 `1_000_000_000` 只是方便读，等同 1000000000，见 4 节）。`datetime.fromtimestamp` 把"Unix 秒数"转成"本地时间 datetime 对象"（注意不是 UTC，见 7.7 节）。
- 第 291 行 `dt.strftime("%H:%M:%S.%f")[:-3]`：`strftime` 按格式出字符串，`%f` 是微秒（6 位），`[:-3]` 把最后 3 位切掉，只留毫秒 3 位（见 4 节切片）。

### 3.11 wait_key（第 296–313 行）

```python
def wait_key(debug: bool) -> bool:
    if debug:
        key = input('press, q: break, other: next step \n')
        logging.info(f'pressed: [{key}]')
        if key == 'q':
            return False
    return True
```

**干什么**：调试模式下，停下来等用户敲回车。敲 `q` 返回 False（表示"别继续了"），敲别的返回 True（继续）。
- `input(...)` 是阻塞式读取一行（带回车）。`test_tmpl_grasp_2d/3d` 每个步骤前都调它，方便单步看效果。
- **坑**：在子线程里调 `input()` 有隐患（见 7.3 节）。

### 3.12 get_key（第 316–329 行）

```python
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
```

**干什么**：读**一个**按键（不用按回车），阻塞直到有键。
- 第 321 行 `sys.stdin.fileno()`：标准输入的"文件描述符"编号（通常是 0）。
- 第 322 行 `termios.tcgetattr(fd)`：保存终端当前设置（Linux/macOS 才有这套 API）。
- 第 324 行 `tty.setraw(...)`：把终端设成 raw 模式——每个键立刻送来、不回显、Ctrl+C 也不再打断程序（见 7.2 节）。
- 第 325 行 `sys.stdin.read(1)`：读 1 个字符。
- 第 326–327 行 `finally`：无论是否出错，都把终端设置还原（否则终端会一直乱）。

**致命坑（7.1 节）**：`termios`、`tty`、`select` 这三个模块**在 Windows 上根本不存在**！所以只要 import 本文件，Windows 上直接 ModuleNotFoundError 崩溃。

### 3.13 KeyboardReader 类（第 335–370 行）

```python
class KeyboardReader:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._tcsetattr = termios.tcsetattr
        self._tcgetattr = termios.tcgetattr
        self.old_settings = self._tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self._closed = False
        atexit.register(self._close)

    def _close(self):
        if self._closed:
            return
        try:
            self._tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        finally:
            self._closed = True

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass

    def read_key(self):
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if rlist else None
```

**干什么**：非阻塞读键盘的类（和 get_key 的区别见下）。
- 第 340–348 行 `__init__`：构造时保存终端设置、用 `tty.setcbreak` 设成 cbreak 模式（和 raw 类似，但保留换行/某些信号），注册 `atexit` 退出时自动还原（见 4 节 `atexit`）。
- 第 350–357 行 `_close`：还原终端；`_closed` 防止重复关。
- 第 359–364 行 `__del__`：对象被垃圾回收时也会尝试关（见 4 节 `__del__`）。
- 第 366–368 行 `read_key`：`select.select([sys.stdin],[],[],0)` 是"立刻看 stdin 有没有数据可读"，超时 0 秒。**有键就 `read(1)`，没键就返回 None**（非阻塞）。`rlist` 是"可读的文件列表"，非空即说明有输入。

**被谁调用**：`arm_node.py:103` `keyboard_reader = KeyboardReader()`，然后循环里 `keyboard_reader.read_key()` 读 `q/v/a/c/,/.` 控制机械臂。`arm_node.py` 的帮助文字写 `<`/`>` 但代码实际判断 `,`/`.`——另一个坑（见 7.2 节）。

---

## 4. Python 基础语法速查

每条给最小示例，方便你回看代码时对照。

**模块级代码执行顺序**：`import` 一个模块时，从第一行往下顺序执行。所以第 5–11 行 `if not logging...` 在 import 时就跑了（配好日志），而函数定义只是"登记名字"，不执行函数体。

```python
import logging          # 导入时执行
x = 1                   # 导入时执行
def f(): return x       # 只登记，不执行
```

**if 判断短路**：`A and B` 中若 A 为 False 就不再看 B；`A or B` 中若 A 为 True 就不看 B。第 174 行 `os.path.exists(json_file_path) is False` 没有短路问题，但 `intrinsic is None or depth_scale is None`（3d:722）就是"任一为空就退出"的惯用法。

```python
if a is None or b is None:   # 若 a 是 None，直接走，不浪费算 b
    ...
```

**isinstance**：判断"是不是某类型"。第 61 行 `isinstance(param, str)`。

```python
isinstance(3, int)      # True
isinstance("a", str)    # True
```

**三元 / 多分支**：单行的 `a if 条件 else b` 叫三元；多分支用 `if/elif/else`（见第 51–69、第 51 行串。多分支例子第 188 行 `q if 条件 else ...` 实际是嵌套 if，第 51–69 行也可改写成）：

```python
# 三元
y = "大" if x > 10 else "小"
# 多分支
if a == 1: ...
elif a == 2: ...
else: ...
```

**try/except**："试一试，出错了怎么办"。第 85–92 行捕获文件读取异常。

```python
try:
    f = open("x.json")
except Exception as e:
    print("挂了:", e)
```

**with open**：自动关文件，不怕忘记。第 86–87 行。

```python
with open("a.txt") as f:
    data = f.read()      # 离开 with 块自动 close
```

**json.load vs json.loads**：`json.load(f)` 读**文件对象**；`json.loads(s)` 读**字符串**。上层 `test_tmpl_grasp_2d.py:578` 用 `json.loads(args.detect_pose)` 把命令行字符串解析成列表。

```python
import json
json.load(open("a.json"))     # 文件
json.loads('[1,2,3]')         # 字符串 -> [1,2,3]
```

**类型标注与 typing_extensions**：第 23 行 `from typing_extensions import List, Tuple, Dict`，第 73 行 `-> Tuple[List[float], List[float]]` 只做"文档提示"，**运行时不做检查**。Python 3.9+ 其实内置了 `list`/`tuple`/`dict` 可直接写，这里用 `typing_extensions` 是为了兼容老版本。

**assert**：断言，条件为假就抛 `AssertionError` 并停机。第 218 行 `assert T.shape == (4,4), "..."`。调试好用，但生产环境可用 `-O` 关掉。

**datetime**：处理日期时间。第 290 行 `datetime.fromtimestamp` 把秒数转本地时间；`strftime` 格式化输出。

```python
from datetime import datetime
datetime.now().strftime("%H:%M:%S")   # '14:05:09'
```

**下划线数字字面量**：`1_000_000_000` == `1000000000`，只是让人好数位数（第 290 行）。

**strftime 切片**：`"abc123"[:-3]` 从开头切到倒数第 3 个之前 = `"abc"`（第 291 行把微秒 6 位切成 3 位毫秒）。

```python
s = "14:05:09.123456"
s[:-3]            # '14:05:09.123'
```

**atexit**：注册"程序退出时自动调用的函数"。第 347 行 `atexit.register(self._close)`，保证 KeyboardReader 销毁时还原终端。

```python
import atexit
def bye(): print("bye")
atexit.register(bye)
```

**termios / tty / select**：Linux/macOS 改终端模式、非阻塞读键的底层模块（第 15–17、322、324、367 行）。**Windows 没有**（见 7.1）。

**__del__**：析构函数，对象被回收时调用。第 359 行 KeyboardReader 用它兜底还原终端。

**类与 self**：`class KeyboardReader` 是"模板"，`self` 指代"当前这个实例"。`self.fd = ...` 是给实例存属性；调用 `obj.read_key()` 时 Python 自动把 `obj` 作为 `self` 传入。

```python
class Dog:
    def __init__(self, name):
        self.name = name
    def bark(self):
        print(self.name, "汪")
d = Dog("阿黄"); d.bark()
```

**可选参数默认值**：函数定义时 `def f(a, b=2)`，`b` 不传就用 2。本文件 `wait_key(debug=False)`、`get_frames` 等大量使用。

**元组多返回值**：`return intrinsic, distortion` 实际返回 `(intrinsic, distortion)`，调用处 `a, b = f()` 一次接两个。第 113、161、206 行都用。

**docstring**：函数/类第一行用 `"""..."""` 写的说明，可用 `help(f)` 查看。本文件几乎每个函数都有。

---

## 5. 输入输出规范

### 5.1 相机参数 JSON（`read_cam_params` / `read_rgbd_params` 读）

`cam_params.json` 示例（RGB 相机）：

```json
{
  "serial_number": "ABC123",
  "intrinsic": [fx, fy, cx, cy],
  "distortion": [k1, k2, p1, p2, k3]
}
```

- `intrinsic`：`[fx, fy, cx, cy]`——焦距（像素）和主点（像素）。必需。
- `distortion`：畸变系数，**可省略**（缺省为 None）。顺序 k1,k2,p1,p2,k3（OpenCV 5 参数径向+切向）。
- `serial_number`：可选，只是打印出来好看。

`read_rgbd_params` 额外要求 `depth_scale`：

```json
{
  "serial_number": "D456",
  "intrinsic": [fx, fy, cx, cy],
  "distortion": [k1, k2, p1, p2, k3],
  "depth_scale": 0.001
}
```

### 5.2 手眼标定 JSON（`read_calib_handeye` 读）

眼在手：

```json
{
  "T_armend_cam": {
    "q": [qw, qx, qy, qz],
    "t": [tx, ty, tz]
  }
}
```

眼在外：

```json
{
  "T_armbase_cam": {
    "q": [qw, qx, qy, qz],
    "t": [tx, ty, tz]
  }
}
```

注意 `q` 是 **w 在前**（`[qw,qx,qy,qz]`），第 184/188 行注释明确。返回值是 `T`（4×4 numpy 矩阵）和 `eye_in_hand`（True/False）。

### 5.3 输出数据结构

- `read_cam_params` → `(List[float], List[float])` 或 `(None, None)`
- `read_rgbd_params` → `(List[float], List[float], float)` 或 `(None, None, None)`
- `read_calib_handeye` → `(np.ndarray 4×4, bool)` 或 `(None, None)`
- `inv_tf` / `transform_delta_pose` / `compute_aligned_pose` → `np.ndarray 4×4`
- `timestamp_to_str` → `str`（如 `"14:05:09.123"`）
- `wait_key` → `bool`
- `get_key` / `KeyboardReader.read_key` → `str` 或 `None`

---

## 6. 核心推导：一步一步算给你看

### 6.1 inv_tf 为什么是 [Rᵀ, -Rᵀt; 0 0 0 1]

设 T 把 b 系坐标变到 a 系：

$$p_a = T \cdot p_b = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_b \\ 1 \end{bmatrix} = R x_b + t$$

要反过来从 a 求 b：

$$x_b = R^{-1}(x_a - t) = R^{-1} x_a - R^{-1} t$$

因为 R 是正交旋转阵，**`R⁻¹ = Rᵀ`**，于是：

$$x_b = R^\top x_a - R^\top t$$

写成齐次矩阵：

$$T^{-1} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1 \end{bmatrix}$$

这正是第 223–224 行：`T_inv[:3,:3] = R.T`，`T_inv[:3,3] = -R.T @ t`。

**用 2×2 旋转小例子验证**（把问题降维方便手算）。设旋转 90°：

$$R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix},\quad t = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$$

则

$$R^\top = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix},\quad -R^\top t = -\begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}\begin{bmatrix} 2 \\ 0 \end{bmatrix} = -\begin{bmatrix} 0 \\ -2 \end{bmatrix} = \begin{bmatrix} 0 \\ 2 \end{bmatrix}$$

拿一个 b 系点 `x_b = [1, 0]ᵀ` 试：正向 `x_a = R x_b + t = [0,1]ᵀ + [2,0]ᵀ = [2,1]ᵀ`。反向用公式：`x_b' = Rᵀ x_a - Rᵀ t = [0,1]ᵀ - [0,2]ᵀ... ` 等等，重算：`Rᵀ x_a = [0,1;-1,0]·[2,1] = [1, -2]`，`-Rᵀ t = [0,2]`，相加 = `[1, 0]`。回到原点 `[1,0]`，验证通过。

### 6.2 transform_delta_pose 的相似变换推导

docstring 第 243–251 行给了已知条件：

```
T_w_a * T_a_b = T_w_b          (A)
T_w_a0 * T_a0_a1 = T_w_a1      (B)
T_w_b0 * T_b0_b1 = T_w_b1      (C)
```

已知 `T_a_b, T_a0_a1`，求 `T_b0_b1` 使得 (C) 成立、且两端一致。由 (A) 在 0 时刻：`T_w_a0 * T_a_b = T_w_b0`。由 (B)：`T_w_a1 = T_w_a0 * T_a0_a1`。又由 (A) 在 1 时刻：`T_w_a1 * T_a_b = T_w_b1`，代入：

$$T_w_b1 = T_w_a1 * T_a_b = (T_w_a0 * T_a0_a1) * T_a_b$$

而由 (C) `T_w_b1 = T_w_b0 * T_b0_b1`，且 `T_w_b0 = T_w_a0 * T_a_b`，故：

$$T_w_a0 * T_a_b * T_b0_b1 = T_w_a0 * T_a0_a1 * T_a_b$$

左边同时左乘 `inv(T_w_a0)`（即 `T_w_a0⁻¹`）消掉 `T_w_a0`：

$$T_a_b * T_b0_b1 = T_a0_a1 * T_a_b$$

再左乘 `inv(T_a_b)`：

$$T_b0_b1 = inv(T_a_b) * T_a0_a1 * T_a_b$$

对应代码第 253 行 `inv_tf(T_a_b) @ T_a0_a1 @ T_a_b`。完全吻合。这就是"相似变换"——把 a 系的增量，用固定的 a→b 关系"共轭"到 b 系。

### 6.3 compute_aligned_pose 的推导（独立验证）

docstring 目标：求 `T_a_b1` 使得

$$T_{a\_b1} * T_{b\_c} * T_{c1\_d} = T_{a\_b0} * T_{b\_c} * T_{c0\_d}$$

对等式左边同时左乘 `inv(T_{a\_b1})`、右边同时右乘 `inv(T_{c1\_d})` 不直观；更直接：左乘 `inv(T_{a\_b1})` 得 `T_{b_c} * T_{c1_d} = inv(T_{a\_b1}) * T_{a\_b0} * T_{b_c} * T_{c0_d}`。再左乘 `inv(T_{b_c})`：

$$T_{c1\_d} = inv(T_{b_c}) * inv(T_{a\_b1}) * T_{a\_b0} * T_{b_c} * T_{c0\_d}$$

整理：`inv(T_{a\_b1}) = inv(T_{b_c}) * inv(T_{a\_b0}) * T_{a\_b0}... ` 换个思路，直接解 `T_a_b1`：

从目标式左乘 `inv(T_{a\_b0} * T_{b_c})`、右乘 `inv(T_{c1\_d})`：

$$T_{a\_b1} = T_{a\_b0} * T_{b_c} * T_{c0\_d} * inv(T_{c1\_d}) * inv(T_{b_c})$$

注意 `inv(T_{b_c}) * inv(T_{c1\_d}) = inv(T_{c1\_d} ... )` 不对，正确性质是 `inv(X*Y) = inv(Y)*inv(X)`。写成 `inv(T_{b_c} * T_{c1\_d}) = inv(T_{c1\_d}) * inv(T_{b_c})`。而上面我们要的是 `inv(T_{c1\_d}) * inv(T_{b_c})`，恰好等于 `inv(T_{b_c} * T_{c1\_d})`！

所以：

$$T_{a\_b1} = T_{a\_b0} * T_{b_c} * T_{c0\_d} * inv(T_{b_c} * T_{c1\_d})$$

与代码第 277 行 `T_a_b0 @ T_b_c @ T_c0_d @ inv_tf(T_b_c @ T_c1_d)` **完全一致**。docstring 的公式正确。

### 6.4 四元数 [qw,qx,qy,qz] 顺序对照

三处必须对齐：

1. **`read_calib_handeye` 第 184 行**：`q = np.array(calib_dict['T_armend_cam']["q"], ...)  # [qw,qx,qy,qz]` —— 标定文件里的 q 是 **w 在前**。
2. **第 197 行**：`T[:3,:3] = transforms3d.quaternions.quat2mat(q)` —— `transforms3d` 的 `quat2mat` 要求 `[qw,qx,qy,qz]`（w 在前），和注释一致，正确。
3. **`arm_wrapper.py` 第 339 行**：`q = [pose[6], pose[3], pose[4], pose[5]]  # 转换为 [qw, qx, qy, qz]`，而 `pose` 来自第 156 行机械臂返回 `[tx,ty,tz,qx,qy,qz,qw]`（**w 在最后**）。这里把 `pose[6]`（qw）提到最前，重排成 w 在前，再喂给 `quat2mat`（第 340 行）。顺序处理正确。

**结论**：标定文件 `q`（w 前）→ `quat2mat`（w 前）✓；机械臂 `pose`（w 后）→ 重排成 w 前 → `quat2mat`（w 前）✓。两处都统一成 w 在前，没有冲突。但**命令行参数 `--detect_pose` 格式是 `[tx,ty,tz,qx,qy,qz,qw]`（w 在后，见 2d:553）**，如果你手写这个参数，别把 w 放错位置，否则旋转全错。

### 6.5 时间戳 ns → 秒 → 字符串的换算

$$秒 = \frac{纳秒}{10^9} = \frac{ns}{1\_000\_000\_000}$$

第 290 行 `datetime.fromtimestamp(ns / 1e9)` 得到"本地时间"的 datetime。再 `strftime("%H:%M:%S.%f")` 出 `'HH:MM:SS.abcdef'`（微秒 6 位），`[:-3]` 切到毫秒 3 位，得 `'HH:MM:SS.sss'`。

例：`ns = 1_700_000_000_123_456_789` → `秒 ≈ 1700000000.123` → 对应某个本地时刻 → `'...:00.123'`（具体日期被 `%H:%M:%S` 丢弃，只留时分秒）。

### 6.6 数值演练：手算 inv_tf

取一个具体 T（用整数方便算）：

```
R = | 0 -1  0 |      t = | 1 |
    | 1  0  0 |          | 2 |
    | 0  0  1 |          | 0 |
```

完整 T：

```
| 0 -1  0  1 |
| 1  0  0  2 |
| 0  0  1  0 |
| 0  0  0  1 |
```

按 6.1 公式：`Rᵀ = | 0 1 0; -1 0 0; 0 0 1 |`。算 `-Rᵀ·t`：

```
Rᵀ·t = | 0 1 0; -1 0 0; 0 0 1 | · |1;2;0| = |2; -1; 0|
-Rᵀ·t = |-2; 1; 0|
```

于是：

```
T⁻¹ = | 0  1  0 -2 |
      | -1 0  0  1 |
      | 0  0  1  0 |
      | 0  0  0  1 |
```

验证：用 numpy 算 `T @ T_inv` 应得单位阵。手算取第一行 `[0,-1,0,1]` 乘 T_inv 第一列 `[0,-1,0,0]ᵀ` = `0*0 + (-1)*(-1) + 0*0 + 1*0 = 1`；乘第二列 `[1,0,0,0]ᵀ` = `0*1+(-1)*0+0+1*0=0`；乘第四列 `[-2,1,0,1]ᵀ` = `0*(-2)+(-1)*1+0+1*1 = 0`。符合单位阵。✓

---

## 7. 这段代码里的坑与改进建议

下表"现象 / 根因 / 改法"三列。改法给可直接粘贴的代码片段。

| # | 现象 | 根因 | 改法 |
|---|---|---|---|
| 7.1 | **Windows 上 import 整个工程直接崩溃**（ModuleNotFoundError: termios） | 第 15–17 行 `import termios/tty/select`，这三个是 POSIX 专属，Windows 没有 | 把键盘相关 import 和函数改成"按需导入 + Windows 分支" |
| 7.2 | 程序吞掉 Ctrl+C；帮助文字写 `<`/`>` 但代码判断 `,`/`.` | `KeyboardReader` 用 cbreak，Ctrl+C 变成字符 `\x03` 不再中断；arm_node 帮助与实际按键不一致 | 调用方判断 `key == '\x03'`；修正帮助文字 |
| 7.3 | 子线程里 `wait_key` 用 `input()` 时，Ctrl+C 无法中断、行为异常 | 在 cbreak 终端下 stdin 被改，子线程 input 不安全 | 提供非阻塞替代，或主线程统一读键 |
| 7.4 | 文件坏了仍带着 None 跑，到 TagMatcher 才崩 | `read_cam_params` 返回 None 但 2d:597 没检查 | 调用方显式 `if intrinsic is None: exit` |
| 7.5 | 喂"眼在外"标定文件却被当 `T_end_cam` 用，语义全错 | 2d:608 / 3d:728 把 `eye_in_hand` 用 `_` 丢弃 | 校验 eye_in_hand 与预期一致再继续 |
| 7.6 | float32 矩阵写回 float64 数组有精度损失 | 第 222/196 行 `dtype=T.dtype` / `np.float32` 传播 | 统一用 float64，或显式转换 |
| 7.7 | 多时区机器时间串不对；极大时间戳精度损失 | 第 290 行 `fromtimestamp`（本地时区），`ns/1e9` 除法 | 用 `datetime.utcfromtimestamp` 或 `time.gmtime`；用 `ns * 1e-9` 或 Decimal |
| 7.8 | docstring 用 `//` C++ 注释风格，新手困惑 | 作者习惯，Python 里 `//` 不是注释 | 改成 `#` 或保留为文本示例 |
| 7.9 | 重定向到文件时打印出裸 `\033[91m` 乱码 | ANSI 码在不支持终端/文件里不被解释 | 检测 `sys.stdout.isatty()`，非终端则清空颜色 |

### 7.1 Windows 兼容改法

把第 15–17 行改成条件导入，键盘函数加 Windows 分支（用 `msvcrt.getch`）：

```python
import sys
if sys.platform != "win32":
    import termios, tty, select
else:
    import msvcrt

def get_key():
    if sys.platform == "win32":
        # Windows: msvcrt.getch 读一个键，Ctrl+C 仍是 '\x03'
        return msvcrt.getch().decode(errors="ignore")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
```

`KeyboardReader` 同理，在 Windows 下用 `msvcrt.kbhit()` 非阻塞读：

```python
def read_key(self):
    if sys.platform == "win32":
        return msvcrt.getwch() if msvcrt.kbhit() else None
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.read(1) if rlist else None
```

### 7.2 Ctrl+C 与帮助文字

cbreak 下 `get_key`/`read_key` 不会触发 `KeyboardInterrupt`，Ctrl+C 被读成 `'\x03'`。调用方应：

```python
key = keyboard_reader.read_key()
if key is None:
    continue
if key == '\x03':      # 用户按了 Ctrl+C
    logging.info("Ctrl+C received, exiting.")
    break
if key == 'q':
    break
```

并把 `arm_node.py` 帮助文字里 `<`/`>` 改成实际生效的 `,`/`.`（或反过来把判断改成 `<`/`>`）。

### 7.4 / 7.5 调用方检查

```python
# 7.4：读相机参数必须检查
intrinsic, distortion = read_cam_params(cam_params_path)
if intrinsic is None:
    logging.error("相机参数读取失败，退出")
    exit(1)

# 7.5：眼在手/眼在外必须和预期一致
T_end_cam, eye_in_hand = read_calib_handeye(calib_handeye_path)
if T_end_cam is None:
    exit(1)
if not eye_in_hand:    # 本工程示例都假设眼在手
    logging.error("标定文件是眼在外(T_armbase_cam)，但本脚本按眼在手处理！")
    exit(1)
```

### 7.6 统一 float64

```python
T = np.eye(4, dtype=np.float64)          # 第 196 行
q = np.array(calib_dict[...]["q"], dtype=np.float64)   # 第 184/188 行
t = np.array(calib_dict[...]["t"], dtype=np.float64)
```

### 7.7 时区与精度

```python
from datetime import datetime, timezone
def timestamp_to_str(ns: int) -> str:
    dt = datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)  # 明确 UTC
    return dt.strftime("%H:%M:%S.%f")[:-3]
```
或用 `time.gmtime(ns // 1_000_000_000)` 避免浮点除法精度损失。

### 7.9 非终端去掉颜色

```python
import sys
USE_COLOR = sys.stdout.isatty()
def _c(s): return s if USE_COLOR else ""
RED = _c('\033[91m'); RESET = _c('\033[0m')   # 其余同理
```

---

## 8. 一句话总结

> **`utils.py` 把"JSON 里的相机/标定数字"变成"4×4 位姿矩阵 T"，并提供求逆、坐标变换、时间戳格式化、彩色日志、非阻塞键盘读取等小工具；它是整套抓取工程的地基，但 9 个隐藏坑里最致命的是"Windows 上因 termios/tty/select 不存在而直接 import 失败"，以及"调用方丢弃 `eye_in_hand`、不检查 None 导致语义错乱或延迟崩溃"——用之前务必按第 7 节补上平台分支与参数校验。**

---

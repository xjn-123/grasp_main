# `tmp.py` 逐行详解（零基础版）

> 目标读者：完全没写过 Python、没接触过线性代数、也不知道 ROS 是什么的同学。
> 目标：读完之后，你能看懂"物体的对称性信息"是怎么从一个 json 文件变成一串 4×4 矩阵的，
> 明白 `symmetries_continuous` 和 `symmetries_discrete` 两种写法分别适用于什么物体，
> 也知道本工程当前那份 `symmetric_info.json` 里藏着一个会让整个抓取崩掉的坑。
>
> 被分析的源文件：
> `E:\WORK\arm_disorderly_swap\robot_grasp_test\demo\test_tmpl_grasp_3d\tmp.py`（共 81 行）
>
> ⚠️ **这不是一个能独立运行的脚本**，而是一段**从匹配服务类里摘出来的函数**
> （第一个参数是 `self`，说明它原本是 `ClientNode` 类的方法，
> 对应 `test_tmpl_grasp_3d_romate.py` 第 1112 行调用的 `client_node.get_symmetric_info()`）。
> 文件里没有 `import`，直接 `python tmp.py` 会报 `NameError`。
>
> 配套文件：`demo/test_tmpl_grasp_3d/symmetric_info.json`（本函数要读的那份配置）。
>
> **姊妹篇**：`test_tmpl_grasp_3d_romate_analysis.md` —— 那里用到本函数产出的 `sym_tfs`。

---

## 目录

- [0. 一句话概括](#0-一句话概括)
- [1. 背景知识：对称性信息是什么](#1-背景知识对称性信息是什么)
  - [1.1 它在整个抓取流程里的位置](#11-它在整个抓取流程里的位置)
  - [1.2 连续对称 vs 离散对称](#12-连续对称-vs-离散对称)
  - [1.3 为什么要 `scale` 这个参数](#13-为什么要-scale-这个参数)
- [2. 整体结构与数据流](#2-整体结构与数据流)
- [3. 逐段代码精读](#3-逐段代码精读)
  - [3.1 函数签名与读文件（第 1–17 行）](#31-函数签名与读文件第-1%E2%80%9317-行)
  - [3.2 连续对称：绕轴转一圈（第 19–48 行）](#32-连续对称绕轴转一圈第-19%E2%80%9348-行)
  - [3.3 离散对称：直接给矩阵（第 50–60 行）](#33-离散对称直接给矩阵第-50%E2%80%9360-行)
  - [3.4 两组变换笛卡尔积（第 62–79 行）](#34-两组变换笛卡尔积第-62%E2%80%9379-行)
- [4. Python 基础语法速查](#4-python-基础语法速查)
- [5. 怎么用它、输入输出长什么样](#5-怎么用它输入输出长什么样)
- [6. 核心数学：一步一步算给你看](#6-核心数学一步一步算给你看)
  - [6.1 绕"不经过原点的轴"旋转，平移项为什么是 `offset - R·offset`](#61-绕不经过原点的轴旋转平移项为什么是-offset--roffset)
  - [6.2 数值演练：一个圆柱的对称变换](#62-数值演练一个圆柱的对称变换)
- [7. 这段代码里的坑与改进建议](#7-这段代码里的坑与改进建议)
- [8. 一句话总结](#8-一句话总结)

---

## 0. 一句话概括

> **这个函数把"这个物体转一下还是它自己"这句话，翻译成机械臂能直接用的数学形式：
> 读一个描述对称性的 json，对"连续对称"（比如圆柱绕轴转任意角度）按步长密集采样出一圈旋转矩阵，
> 对"离散对称"（比如长方体转 180°）直接照抄矩阵，最后把两组乘起来，
> 得到一整串 $N \times 4 \times 4$ 的变换矩阵——抓取时机械臂就靠它们"抄近路"。**

---

## 1. 背景知识：对称性信息是什么

### 1.1 它在整个抓取流程里的位置

```
symmetric_info.json
      │
      ↓  read_symmetric_info()   ← 本文件
  sym_tfs (N×4×4)
      │
      ↓  传给 compute_tmpl_ready_pose()
  test_tmpl_grasp_3d_romate.py 里挑"最省劲的抓法"
```

没有它，机械臂可能会为了"把物体转到标准朝向"而做一堆完全没必要的翻转——
因为物体转了某个角度之后**看起来一模一样**，根本不用转。

### 1.2 连续对称 vs 离散对称

| 类型 | 适用物体 | 例子 | json 字段 |
|---|---|---|---|
| **连续** | 绕某根轴转**任意角度**都一样 | 圆柱、圆盘、螺丝、任何回转体 | `symmetries_continuous` |
| **离散** | 只有**特定几个角度**转完一样 | 长方体（180°）、正方体（90° 的倍数） | `symmetries_discrete` |

为什么连续对称要"采样"？因为计算机没法表示"任意角度"，
只能按 `rotation_step`（默认 1°）**转一圈生成 360 个矩阵**来近似。

> 代价：一个连续对称就产生 360 个矩阵。如果物体有两根对称轴，就是 720 个——
> 后面每个模板都要对这 N 个候选各算一遍目标位姿，`compute_tmpl_ready_pose` 的开销会明显上升。

### 1.3 为什么要 `scale` 这个参数

CAD 模型（比如用 SolidWorks 导出的）通常用**毫米**做单位，
而机器人这边一律用**米**。

`scale` 就是把 CAD 里的位移数值换成米的比例因子：
- CAD 单位是毫米 → `scale = 0.001`；
- CAD 单位已经是米 → `scale = 1.0`。

注意：**只缩放平移部分，旋转部分不受影响**（旋转矩阵没有单位）。

---

## 2. 整体结构与数据流

```
read_symmetric_info(json_file_path, scale, rotation_step=1)
  │
  ├─ 读 json（失败返回 None）
  │
  ├─ 若有 "symmetries_continuous"：
  │     对每组 {axis, offset}
  │       └─ 角度从 0° 到 359°，步长 rotation_step
  │            └─ R = 轴角 → 旋转矩阵
  │            └─ t = offset - R·offset      ← 绕不过原点轴的平移补偿
  │            └─ 拼成 4×4，收进 continuous_tf_list
  │
  ├─ 若有 "symmetries_discrete"：
  │     每个 4×4 矩阵直接收进 discrete_tf_list（平移部分 × scale）
  │
  ├─ discrete_tf_list 末尾**追加单位矩阵**（保证至少有一个"不转"的选项）
  │
  └─ 笛卡尔积：对每个离散变换 d、每个连续变换 c，产出 d @ c
        ↓
     np.array(tf_list)   →   N×4×4
```

---

## 3. 逐段代码精读

### 3.1 函数签名与读文件（第 1–17 行）

```python
1   def read_symmetric_info(self, json_file_path: str, scale: float, rotation_step: float = 1) -> np.ndarray:
2       """
3       读取 CAD 模型的对称性信息
4       Args:
5           json_file_path (str): json 文件路径
6           scale (float): 缩放因子( 将位移单位转换到米制单位的比例 )
7           rotation_step (float): 旋转步长( 仅适用于连续旋转 ), 单位: 度
8       Returns:
9           (np.ndarray): 对称变换矩阵 N*4*4, 如果失败则返回 None
10      """
11      try:
12          with open(json_file_path, 'r') as f:
13              data = json.load(f)
14      except Exception as e:
15          logging.error(f"failed to read symmetric info from: '{json_file_path}' \n{e}")
16          return None  # 返回空数组表示读取失败
17      # end with
```

**业务作用**：把 json 读进来；读不到就优雅地返回 `None`（而不是让程序崩掉）。

**逐行讲解：**

- 第 1 行 `self`：说明这是**类的方法**（method），调用时前面要有个对象。
  本文件里没有类定义，所以这段代码是从别处（推测是 `ClientNode`）剪下来的。
- 第 1 行 `rotation_step: float = 1`：**默认步长 1 度**。
  第 36 行 `np.arange(0, 360, 1)` 就是靠它生成 360 个角度。
- 第 11-16 行 `try/except`：**捕获所有异常**。
  文件不存在、json 格式错、编码问题…… 任何一种都会走到 `except`，打日志并返回 `None`。
  这是很稳的写法 ✓（对比 `test_tmpl_grasp_2d.py` 里的 `read_tmpl_grasp_2d` 完全没有保护）。
- 第 16 行注释写"返回空数组"，但代码实际返回的是 `None`——**注释与代码不符**（见第 7 节问题 3）。

### 3.2 连续对称：绕轴转一圈（第 19–48 行）

```python
19      # 读取旋转轴( 仅适用于连续旋转 )
20      continuous_tf_list = []
21      if "symmetries_continuous" in data:
22          symmetries = data.get('symmetries_continuous', [])
23
25          for i, symmetry in enumerate(symmetries):
26              axis = symmetry.get('axis', [])
27              offset = symmetry.get('offset', [])
28
29              # 将轴和偏移量转换为 numpy 数组
30              axis = np.array(axis)
31              offset = np.array(offset) * scale
32              logging.info(f"symmetries_continuous index: {i}, axis: {axis}, offset: {offset}")
33
34              # 生成变换矩阵
35              offset = offset.reshape((3, 1))
36              for angle in np.arange(0, 360, rotation_step):
37                  R = transforms3d.axangles.axangle2mat(axis, np.radians(angle))
38                  t = offset - R.dot(offset)
39
40                  # 组合旋转矩阵和平移向量
41                  tf = np.eye(4)
42                  tf[:3, :3] = R
43                  tf[:3, 3] = t.flatten()
44
45                  continuous_tf_list.append(tf)
46              # end for
47          # end for
48      # end if
```

**业务作用**：对每个"连续对称轴"，按步长转一整圈，生成一串 4×4 矩阵。

**逐行讲解：**

- 第 21 行 `if "symmetries_continuous" in data`：先判断这个键在不在，
  不在就跳过（物体可能只有离散对称）。
- 第 26-27 行 `symmetry.get('axis', [])`：用 `get` 取值，
  **键不存在时返回默认的 `[]`**，不会抛 `KeyError`。
- 第 31 行 `* scale`：**只有 `offset` 乘了 `scale`，`axis` 没有**——
  因为轴是个方向，乘以任何正数都不改变方向（而且马上要被归一化）。
- 第 35 行 `offset.reshape((3, 1))`：把长度 3 的向量变成 $3 \times 1$ 的**列向量**，
  这样第 38 行 `R.dot(offset)` 才能做矩阵乘向量。
- 第 36 行 `np.arange(0, 360, rotation_step)`：生成 `[0, 1, 2, ..., 359]`，共 360 个角度。
  注意 **`np.arange` 不含终点**，所以不会重复生成 0° 和 360°（两者一样）。
- 第 37 行 `transforms3d.axangles.axangle2mat(axis, angle)`：
  **轴角 → 旋转矩阵**。给它"绕哪根轴"和"转多少弧度"，返回 3×3 旋转矩阵。
  内部会先把 `axis` 归一化（除以模长）——**这正是第 7 节那个坑的来源**。
- 第 38 行 `t = offset - R.dot(offset)`：**绕不经过原点的轴旋转时的平移补偿**，详见 6.1 节。
- 第 41-43 行：拼成 4×4 齐次矩阵——左上 3×3 放旋转，右上角 3 个数放平移。

### 3.3 离散对称：直接给矩阵（第 50–60 行）

```python
51      discrete_tf_list = []
52      if "symmetries_discrete" in data:
53          symmetries = data.get('symmetries_discrete', [])
54          for i, symmetry in enumerate(symmetries):
55              tf = np.array(symmetry).reshape((4, 4))
56              tf[:3, 3] *= scale  # 缩放平移向量
57              discrete_tf_list.append(tf)
58          # end for
59      # end if
60      discrete_tf_list.append(np.eye(4))  # 添加单位矩阵作为默认变换
```

**业务作用**：离散对称不用算，json 里直接给现成的 4×4 矩阵。

**逐行讲解：**

- 第 55 行 `np.array(symmetry).reshape((4, 4))`：json 里存的是**扁平的 16 个数**
  （或 4×4 嵌套列表），`reshape((4,4))` 统一整理成矩阵。
- 第 56 行 `tf[:3, 3] *= scale`：只缩放平移部分（第 4 列的前 3 个），
  **不动旋转部分** ✓
- 第 60 行 **无条件追加单位矩阵**：保证 `discrete_tf_list` 至少有一个元素，
  且其中一定包含"不转"这个选项。
  物理意义很清楚：**任何物体转 0° 都等于它自己**，这是最基础的一条对称。

### 3.4 两组变换笛卡尔积（第 62–79 行）

```python
62      # 组合离散和连续变换
63      tf_list = []
64      for disc_tf in discrete_tf_list:
65          if len(continuous_tf_list) > 0:
66              for cont_tf in continuous_tf_list:
67                  tf_list.append(disc_tf @ cont_tf)
68              # end for
69          else:
70              tf_list.append(disc_tf)
71          # end if
72      # end for
73
74      logging.info(f"symmetric transforms num: \033[92m{len(tf_list)}\033[0m")
75
76      # 将变换矩阵列表转换为 numpy 数组
77      tf_array = np.array(tf_list)
78
79      return tf_array
```

**业务作用**：把离散和连续两组变换**两两组合**，得到最终清单。

**逐行讲解：**

- 第 64-72 行：**嵌套循环做笛卡尔积**。
  比如离散有 2 个（含单位阵）、连续有 360 个 → 总共 $2 \times 360 = 720$ 个变换。
  物理上就是"先绕对称轴转任意角度，再套一个离散的翻转"。
- 第 67 行 `disc_tf @ cont_tf`：矩阵乘法，**顺序是"先应用 cont，再应用 disc"**
  （矩阵乘法从右往左作用）。
- 第 70 行：没有连续对称时，离散变换直接使用。
- 第 74 行 `\033[92m ... \033[0m`：**ANSI 转义序列**，让终端输出绿色。
  和工程里 `GREEN` 常量是同一回事，这里写死了没用常量。
- 第 77 行 `np.array(tf_list)`：把 list of (4,4) 堆成 $(N, 4, 4)$ 的三维数组，
  正好是 `compute_tmpl_ready_pose` 期望的形状（那里用 `sym_tfs.shape[0]` 取 N）。

---

## 4. Python 基础语法速查

| 写法 | 含义 | 本文件出现位置 |
|---|---|---|
| `def f(self, ...)` | 类的方法（第一个参数是实例自己） | 第 1 行 |
| `try / except Exception as e` | 捕获所有异常，防止程序崩 | 第 11-16 行 |
| `d.get(k, 默认值)` | 字典取值，键不存在时给默认值 | 第 26、27 行 |
| `if k in d` | 判断键是否在字典里 | 第 21、52 行 |
| `np.arange(0, 360, 1)` | 等差序列，**不含终点** | 第 36 行 |
| `np.radians(deg)` | 角度 → 弧度 | 第 37 行 |
| `v.reshape((3, 1))` | 变成 $3 \times 1$ 列向量 | 第 35 行 |
| `R.dot(v)` | 矩阵乘向量 | 第 38 行 |
| `np.eye(4)` | 4×4 单位矩阵 | 第 41、60 行 |
| `t.flatten()` | 把列向量摊平回一维 | 第 43 行 |
| `a @ b` | 矩阵乘法 | 第 67 行 |
| `\033[92m ... \033[0m` | 终端绿色输出的转义序列 | 第 74 行 |

---

## 5. 怎么用它、输入输出长什么样

**它不能单独运行**（没有 import、没有类）。在完整的服务端里，调用方式大致是：

```python
sym_tfs = self.read_symmetric_info(
    json_file_path='demo/test_tmpl_grasp_3d/symmetric_info.json',
    scale=0.001,        # 若 CAD 用毫米
    rotation_step=1,    # 每 1 度采样一次
)
```

**输入 json 的两种写法**：

```jsonc
{
  // 连续对称：圆柱绕自身 Z 轴转任意角度都一样
  "symmetries_continuous": [
    { "axis": [0, 0, 1], "offset": [0, 0, 0] }
  ],
  // 离散对称：长方体转 180° 后一样
  "symmetries_discrete": [
    [1, 0, 0, 0,
     0, -1, 0, 0,
     0, 0, -1, 0,
     0, 0, 0, 1]
  ]
}
```

**输出**：形状 $(N, 4, 4)$ 的 numpy 数组，`N = (离散数 + 1) × 连续采样数`。
以"1 个连续对称、步长 1°、无离散"为例，$N = 1 \times 360 = 360$。

**配套的 `symmetric_info.json` 长这样**（本工程当前的）：

```json
{
    "symmetries_continuous": [
        { "axis": [0, 0, 0], "offset": [0, 0, 0] }
    ]
}
```

> ⚠️ **这份配置是有问题的**，`axis` 是零向量，详见第 7 节问题 1。

---

## 6. 核心数学：一步一步算给你看

### 6.1 绕"不经过原点的轴"旋转，平移项为什么是 `offset - R·offset`

高中数学里讲旋转，默认都是**绕原点**（或绕经过原点的轴）转。
但现实里的对称轴往往**不经过原点**——比如一个斜放的圆柱，它的轴穿过点 $\mathbf{p}$ 而不穿过原点。

怎么办？分三步：

1. **把整个物体平移**，让轴经过原点：$x_1 = x - \mathbf{p}$
2. **绕原点转**：$x_2 = R\,x_1$
3. **平移回去**：$x' = x_2 + \mathbf{p}$

合起来：

$$x' = R(x - \mathbf{p}) + \mathbf{p} = Rx + \underbrace{(\mathbf{p} - R\mathbf{p})}_{\mathbf{t}}$$

（花括号下面标出来的 $\mathbf{t}$ 就是多出来的平移补偿）

所以那个看似奇怪的 `t = offset - R.dot(offset)`（第 38 行），
**就是"绕不过原点的轴旋转"时多出来的平移补偿** ✓

> 如果轴正好经过原点，$\mathbf{p} = 0$，那 $t = 0 - R\cdot 0 = 0$，退化为纯旋转。
> 这也解释了为什么本工程那份 json 里 `offset` 是 `[0,0,0]` 时平移项恒为 0。

### 6.2 数值演练：一个圆柱的对称变换

假设圆柱的对称轴穿过点 $\mathbf{p} = [0.01, 0, 0]^{\top}$（米），方向沿 Z 轴 $\mathbf{a} = [0,0,1]^{\top}$。
取 `rotation_step = 90`（方便手算），$\theta = 90^\circ$：

绕 Z 轴转 90° 的旋转矩阵：

$$R = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

平移项：

$$R\mathbf{p} = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 0.01 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0.01 \\ 0 \end{bmatrix}$$

$$\mathbf{t} = \mathbf{p} - R\mathbf{p} = \begin{bmatrix} 0.01 \\ 0 \\ 0 \end{bmatrix} - \begin{bmatrix} 0 \\ 0.01 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.01 \\ -0.01 \\ 0 \end{bmatrix}$$

于是这个对称变换的完整 4×4 矩阵是：

$$T = \begin{bmatrix} 0 & -1 & 0 & 0.01 \\ 1 & 0 & 0 & -0.01 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

机械臂看到它就会明白：**"把物体绕那根斜轴转 90°，看起来跟没转一样"**。

---

## 7. 这段代码里的坑与改进建议

| # | 问题 | 后果 | 建议 |
|---|---|---|---|
| 1 | **本工程 `symmetric_info.json` 里 `axis` 是 `[0, 0, 0]`** | **严重**。`axangle2mat` 内部要把轴**除以它的模长**来归一化，零向量会得到 $0/0$ → 结果是 `nan`（或直接抛异常）。**实测：360 个对称变换全部异常**，整个 `sym_tfs` 全是 nan，下游算出来的目标位姿也全是 nan，抓取必然失败 | 把 `axis` 填成真实的对称轴方向（如 `[0,0,1]`）；如果物体**没有**对称性，应该**删掉** `symmetries_continuous` 这个键，让函数只返回单位矩阵 |
| 2 | 第 37 行不校验 `axis` 是否为零向量 / 长度是否为 3 | 配置写错时不会报错，而是静默产出 nan，排查极难 | 入口处加校验：`if np.linalg.norm(axis) < 1e-9: 报错并跳过` |
| 3 | 第 16 行注释写"返回空数组"，代码返回的是 `None` | 误导调用方；调用方若写 `sym_tfs.shape[0]` 会在 `None` 上崩 | 统一成返回 `None`，注释改成"返回 None 表示读取失败" |
| 4 | 连续对称默认步长 1° → 一个轴 360 个矩阵 | 若物体有两个对称轴就是 720 个，`compute_tmpl_ready_pose` 要对**每个模板 × 每个对称**都算一遍，开销不小 | 步长放宽到 5°（72 个）通常够用；或按物体类型给不同步长 |
| 5 | 第 74 行颜色用硬编码 `\033[92m`，没用工程的 `GREEN` 常量 | 风格不统一，某些终端下表现不一致 | 换成 `from core.utils import GREEN` |
| 6 | 没有对 `symmetries_discrete` 里的矩阵做形状校验 | json 里若给了 16 个数以外的内容，`reshape((4,4))` 会抛异常（虽然被外层 `try` 兜住，但会**整体返回 None**，连离散部分也丢了） | `reshape` 前先 `np.array(symmetry).size == 16` 校验 |
| 7 | 本文件是**从类里剪下来的片段**，没有 import 也没有类 | `python tmp.py` 直接报 `NameError: name 'np' is not defined` | 要么放回 `ClientNode` 类里，要么补上 import 并去掉 `self`（做成独立工具函数） |

---

## 8. 一句话总结

> **这个函数是"物体对称性"的翻译官：把 json 里"绕这根轴转任意角度都一样"
> 翻译成几百个 4×4 矩阵，供抓取时挑最省劲的那个姿态用。
> 但当前 `symmetric_info.json` 里 `axis` 写成了零向量 `[0,0,0]`，
> 会让所有对称变换变成 nan——**这份配置必须改**，
> 要么填上真实对称轴，要么直接删掉 `symmetries_continuous` 让物体按"无对称性"处理。**

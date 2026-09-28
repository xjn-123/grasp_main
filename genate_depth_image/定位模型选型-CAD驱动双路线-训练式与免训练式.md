# 定位模型选型：CAD 驱动的两条路线（训练式 / 免训练式）

> **需求范围（已第四次修订）**
>
> 1. 物体**不贴 AprilTag**，改用视觉输出 6D 位姿。
> 2. 两条路线**都需要 CAD 模型**，而且**两条都是神经网络方案，不用几何算法**（ICP / PPF / 模板点云配准这类已排除）。
>    - **路线 A（快）**：CAD → 脚本自动生成训练样本 → 训练模型 → 每帧快速出位姿。
>    - **路线 B（慢）**：有 CAD，但**不做任何训练**，用现成模型直接出位姿，耗时较长。
> 3. 两条路线**都以检测器为共同前置**：先在彩色图上框出（或抠出）物体的大致范围，再做位姿。
> 4. 目标是**复现你以前用过的一个项目**的功能。
>
> **本次修订改了什么（重要）**
>
> 上一版我把路线 B 的主推写成了 **OpenCV 的 PPF + ICP**。你说两条都要用模型，那条作废——PPF/ICP 是纯几何算法，不属于模型方案，**已从主推位置移除**（仅在 5.6 节作为退化场景兜底提一句）。
>
> 路线 B 换成神经网络方案后，最优解变成 **FoundPose**（Meta，ECCV 2024）：它基于冻结的 DINOv2，**官方明确说不需要发布任何权重、不需要任何训练**，只需要 CAD 网格 + 一张分割掩膜。而且它是纯 PyTorch，不用编译 CUDA 扩展。
>
> 但**精度天花板仍然是 FoundationPose**，而它要编译 nvdiffrast / PyTorch3D —— 所以 OS 结论相对上一版**回摆了一半**：选 FoundPose 就留在 Windows，要 SOTA 精度就得装 WSL2。详见第 8 节。
>
> 配套文档：
> - `Fast-FoundationStereo_落地方案.md`（深度图从哪来）
> - `部署环境评估-WSL2与Windows对比.md`（OS 与硬件）

---

## 目录

- [0. 结论先行](#0-结论先行)
- [1. 先把需求和术语对齐](#1-先把需求和术语对齐)
- [2. 共同前置层：检测器](#2-共同前置层检测器)
- [3. 公共地基：CAD 预处理与渲染](#3-公共地基cad-预处理与渲染)
- [4. 路线 A：模型化快速位姿](#4-路线-a模型化快速位姿)
- [5. 路线 B：模型化免训练位姿（慢）](#5-路线-b模型化免训练位姿慢)
- [6. 两条路线的关系：并列 + 互为备份](#6-两条路线的关系并列--互为备份)
- [7. 复现目标项目：需要你确认的 6 个信息](#7-复现目标项目需要你确认的-6-个信息)
- [8. 对 OS 结论的影响](#8-对-os-结论的影响)
- [9. 坑与改进建议](#9-坑与改进建议)
- [10. 四周落地时间表](#10-四周落地时间表)
- [11. 一句话总结](#11-一句话总结)

---

## 0. 结论先行

```
                    彩色图 (RGB)
                         │
                         ▼
        ┌────────────────────────────────┐
        │  共同前置：检测器（给 ROI）      │   ← 两条路线共用
        │  YOLO11-seg / YOLOE / CNOS      │
        │  输出：2D 框 / 掩膜 / 关键点     │
        └────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
   ┌──────────────────┐   ┌──────────────────────┐
   │ 路线 A：训练式     │   │ 路线 B：免训练式       │
   │                  │   │                      │
   │ CAD → 合成数据     │   │ CAD → 渲染模板        │
   │ → 训练网络         │   │ → 现成模型推理        │
   │ → 前向推理         │   │ → 位姿              │
   │                  │   │                      │
   │ YOLO11-pose+PnP   │   │ FoundPose（先做）     │
   │ 或 GDR-Net        │   │ FoundationPose（更准） │
   │                  │   │                      │
   │ 10–80 ms / 帧     │   │ 1–10 s / 次          │
   │ 需要为该物体训练    │   │ 不为该物体训练        │
   │ 平移 2–5 mm       │   │ 平移 1–3 mm           │
   └──────────────────┘   └──────────────────────┘
              │                     │
              └──────────┬──────────┘
                         ▼
                  统一输出 T_cam_obj
```

**四条结论：**

1. **路线 B 先做 FoundPose，别一上来就碰 FoundationPose。** FoundPose（Meta，ECCV 2024）用冻结的 DINOv2 做特征匹配，**官方原话是 "training-free，不要求发布任何权重"**——你只要给一个 CAD 网格和一张掩膜就能出位姿。它是纯 PyTorch，不用编译 CUDA 扩展，**Windows 上就能跑**。FoundationPose 精度更高，但要编译 nvdiffrast / PyTorch3D，代价是要不要装 WSL2。

2. **路线 A 要接受一个现实：几乎所有快速位姿网络的末端都接一个 PnP。** 严格"纯网络直出 6 个数"的方案（DOPE、EfficientPose 那一类）存在，但精度通常明显差一档。工业主流是**网络学 2D-3D 对应 + PnP 闭式解算**——这里的 PnP 是解析求解，跟你排除的 ICP/PPF 那种迭代搜索完全不是一回事。如果你连这一步也要网络化，就得接受精度下降。

3. **两条路线的"慢"和"快"差的是 2 个数量级，但可以共存。** A 是 10–80 ms，B 是 1–10 s。所以 B 不该跑在每帧上，它的正确用法是**新物体 onboarding + 给 A 打标签**（见 6.2）。

4. **检测器必须是两条路线共用的第一级，且最好能免训练。** FoundPose 官方用的输入分割就来自 CNOS（FastSAM + DINOv2）。想让路线 B 真正做到"给个 CAD 就能用"，前置检测可以用 **YOLOE**（开放词汇，零训练）或 **CNOS**，而不是需要训练的 YOLO11-seg。

---

## 1. 先把需求和术语对齐

### 1.1 你要的系统长什么样

| 阶段 | 输入 | 输出 | 回答什么问题 |
|---|---|---|---|
| ① ROI | 彩色图 | 2D 框 / 掩膜 / 类别 | 「画面里有没有这个物体，在哪一片像素」 |
| ② 位姿 | ROI + （深度图） + CAD | `T_cam_obj`（4×4） | 「物体相对相机，位置和朝向量是多少」 |
| ③ 抓取 | `T_cam_obj` + `T_base_cam` | `T_base_end` | 「机械臂摆到哪个位姿能夹住」 |

阶段 ③ 你已有（`test_tmpl_grasp_3d.py` 那套）。这次要解决 ① 和 ②。

注意：**深度图在两条路线里的地位不同**。路线 A 的某些分支（点云配准类）必须有深度；路线 B 里的 FoundPose 是**纯 RGB** 就能出位姿（它渲染的是 RGB-D 模板，但查询时只需要 RGB 图和内参 K）。FoundationPose 则支持 RGB-D 和纯 RGB 两种模式。这意味着**你的深度图质量对 FoundPose 影响较小**——这对反光件是个好消息。

### 1.2 关于「YOLOv11」的命名更正

- Ultralytics 官方正式名是 **YOLO11**（不带 v），模型文件名形如 `yolo11n-seg.pt`、`yolo11n-pose.pt`。
- 「YOLOv11」是社区错误叫法。Ultralytics 主线是 YOLOv3 → v5 → v8 → **YOLO11** → **YOLO26**（2025 年底，官方文档已出现基于 YOLO26 的 YOLOE-26）。
- 要装的是 `pip install -U ultralytics`，不是 `yolov11`。

按任务后缀区分：

| 后缀 | 任务 | 输出 | 本项目用处 |
|---|---|---|---|
| `yolo11n` | 检测 | 框 | 只框范围，信息浪费 |
| `yolo11n-seg` | 实例分割 | 框 + 掩膜 | 给路线 B 提供输入掩膜 |
| `yolo11n-pose` | 关键点 | 2D 关键点 | **路线 A 主推**，配 PnP 直接出位姿 |

### 1.3 检测器在两条路线里的角色

| 路线 | 检测器提供什么 | 官方方案用的什么 |
|---|---|---|
| A | 训练样本里的标注目标；推理时的 ROI | 你自己训的 YOLO11 |
| B | **推理时的必需输入**（FoundPose 要求给一张物体掩膜） | FoundPose 用 **CNOS**（FastSAM + DINOv2，免训练）；FoundationPose 自带分割 |

**FoundPose 必须吃掩膜**——它的输入是「裁剪后的查询图」，不是整图。所以检测器不是可选项。

### 1.4 「免训练」的三层拆解

| 层次 | 问题 | 路线 A | 路线 B（FoundPose） | 路线 B（FoundationPose） |
|---|---|---|---|---|
| L1：检测器要不要训练？ | 换新物体要重训检测器吗？ | **要** | 看选哪个（YOLOE / CNOS 则不用） | 同左 |
| L2：位姿模块要不要为该物体训练？ | | **要** | **不要** | **不要** |
| L3：位姿模块本身有没有权重？ | | 有，你自己训的 | **没有**（用现成的 DINOv2，官方不发布任何权重） | 有，官方预训练权重 |

**FoundPose 是三层里唯一能在 L3 也做到"无"的**——它连自己的权重都没有，完全依赖一个冻结的通用视觉基础模型。这也是它最贴合你「不要训练」这个要求的原因。

### 1.5 CAD 是两条路线的公共地基

两条都要 CAD，但**用法不同**：

```
STEP / IGES / STL（设计给的）
        │
        ▼  trimesh.load()
   三角网格 mesh（单位统一到米）
        │
        ├──▶ 路线 A：采样点云 / 取特征点 → 合成数据渲染 → 训练
        │
        └──▶ 路线 B：渲染几百个视角的 RGB-D 模板 → 建立特征表示
```

**路线 B 多了一个关键环节：渲染器。** FoundPose 要渲染约 800 个视角的 RGB-D 模板，FoundationPose 也要在线渲染姿态假设。**渲染器的可用性是路线 B 的第一个实际门槛**（见 3.2）。

---

## 2. 共同前置层：检测器

### 2.1 三个候选

| 方案 | 要不要训练 | 输出 | 速度（GTX 1650） | Windows | 用在 |
|---|---|---|---|---|---|
| **YOLO11-seg** | 要（合成数据自动造） | 框 + 掩膜 | 15–30 ms | 原生 | 路线 A |
| **YOLOE-seg** | **不要**（文本/视觉提示） | 框 + 掩膜 | 25–40 ms | 原生 | 路线 B（想真零训练） |
| **CNOS**（FastSAM + DINOv2） | **不要** | 框 + 掩膜 | 200–600 ms | 纯 torch，能跑 | 路线 B（FoundPose 官方搭配） |
| GroundingDINO + SAM2 | 不要 | 框 + 掩膜 | 300–800 ms | 能跑但组件多 | 备选，偏慢 |

**路线 B 的前置推荐 CNOS 或 YOLOE**：既然路线 B 的卖点是"不用训练"，前置检测如果还要训就自相矛盾了。CNOS 是 FoundPose 官方验证过的搭配（BOP 2023 的分割结果就是 CNOS 出的），兼容性最好。

### 2.2 YOLOE：开放词汇检测

已核实：YOLOE 进了 `ultralytics` 主包，不需要额外安装。

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-11s-seg.pt")
model.set_classes(["metal bracket", "hex nut"])   # 说出你要找什么
results = model.predict(frame)
mask = results[0].masks.data[0].cpu().numpy()      # 掩膜，喂给路线 B
```

三种模式：文本提示（`*-seg.pt`）/ 视觉提示（给参考图 + 框）/ 无提示（`*-seg-pf.pt`，内置 4585 类词表）。

两个已核实的坑：

- **首次 `set_classes()` 会联网**下载 TorchScript 文本编码器（YOLOE-26 约 254 MB，YOLOE-11 是 `mobileclip_blt.ts`）。离线部署前必须先联网跑一次，或用 `save_prompt_embeddings()` 把提示烘进权重。
- **导出后类别被烘死**，不能再改 `set_classes()`，改类别要重新从 `.pt` 导出。

### 2.3 接口契约

定死这一层，两条路线才能互换：

```python
@dataclass
class Detection:
    cls_id: int              # 类别（决定用哪个 CAD）
    score: float
    bbox: np.ndarray         # (4,) xyxy
    mask: np.ndarray | None  # (H, W) bool，与原图同分辨率
    kpts2d: np.ndarray|None  # (N, 2) 2D 关键点（pose 分支才有）
```

---

## 3. 公共地基：CAD 预处理与渲染

### 3.1 加载与规范化

```python
import trimesh, numpy as np

mesh = trimesh.load("part.stl", force="mesh")   # force="mesh" 防 Scene 对象
if mesh.extents.max() > 1.0:        # STL 常是 mm，差 1000 倍
    mesh.apply_scale(0.001)
mesh.apply_translation(-mesh.centroid)   # 原点移到夹持中心（按你的约定，不一定是质心）
print(mesh.extents, mesh.is_watertight)
```

**物体坐标系尽早定死**：原点建议放在**夹爪实际夹持中心**，不是 CAD 的建模原点（那常在角落）。对称物体（圆柱、方块）必须定义唯一参考方向，否则位姿多解。

### 3.2 渲染器选型（路线 B 的硬门槛）

这是路线 B 最容易卡住的地方，也是我在这一版新加的内容：

| 渲染器 | 跨平台 | 说明 |
|---|---|---|
| **BlenderProc** | 是（Blender 本身跨平台） | **推荐**。用 Cycles/Eevee 离线渲染模板，质量高、可控 |
| BOP toolkit 自带渲染器 | 名义跨平台 | FoundPose 默认用它，基于 OpenGL，**Windows 上可能要处理 GL 上下文** |
| pyrender | 名义跨平台 | Windows 上常需 OSMesa/EGL，容易卡住 |
| Open3D 离屏渲染 | 是 | 能出深度和 mask，但材质光照粗糙，**域差距大** |

**建议**：不要跟 BOP toolkit 的渲染器较劲，**直接用 BlenderProc 渲染模板**，然后自己调用 DINOv2 提取特征。模板渲染这一步和路线 A 的合成数据是同一套工具，一次投入两处复用。

### 3.3 三条几何量必须对

| 量 | 怎么查 | 错了会怎样 |
|---|---|---|
| 单位（米 vs 毫米） | `mesh.extents` 量级 | 平移量差 1000 倍，抓取飞出 workspace |
| 尺寸 | 卡尺量实物核对 | 模板与真实外观对不上 |
| 坐标系原点 | 人为约定 | 位姿数值对但夹爪抓空 |

---

## 4. 路线 A：模型化快速位姿

### 4.1 一个必须接受的现实

你说「不要用几何算法」。这里需要把话说清楚，否则你会一直在找一个不存在的东西：

**现有的快速 6D 位姿网络，绝大多数在末端都接一个 PnP。** 区别在于这个 PnP 是什么性质的：

| 类型 | 是什么 | 是不是"几何算法" |
|---|---|---|
| **PnP 闭式解** | 已知 n 组 2D-3D 对应，直接解一个线性/非线性方程组，一步到位 | 是解析求解，**不是迭代搜索**。等价于"网络输出对应，公式算出位姿" |
| **ICP / PPF** | 在解空间里迭代搜索、投票、收敛 | **是几何算法**，你排除的正是这类 |

所以路线 A 的现实选择是：

- **要精度** → 接受网络 + PnP（PnP 是三段代码，OpenCV 一个函数，耗时 <1 ms）。
- **要纯网络** → 用直接回归 6DoF 的方案，精度通常差一档，且对遮挡更敏感。

我的建议是前者。**PnP 在这里的角色相当于激活函数，不是一个"配准算法"。**

### 4.2 四个候选

| 方案 | 网络学什么 | 末端 | 速度 | 精度 | 训练数据能自动生成 |
|---|---|---|---|---|---|
| **① YOLO11-pose + PnP** | 2D 关键点位置 | PnP | **10–25 ms** | 中 | **能**（3D 关键点投影即可） |
| **② GDR-Net / CDPN / DPOD** | 稠密 2D-3D 对应 | PnP | 30–80 ms | **好** | 能 |
| ③ YOLO6DPose / EfficientPose / DOPE | 直接回归 6 个数 | 无 | 20–50 ms | 中低 | 能 |
| ④ GeoTransformer / PCRNet（学习式点云配准） | 点对应关系 | SVD 闭式解 | 50–150 ms | 好（需深度图） | 能（合成点云） |

**推荐 ① 起步，② 冲精度。**

方案 ① 的妙处在于**训练样本可以完全自动生成**：在 CAD 上手工标一次 3D 关键点（比如物体的 8 个角或几个特征孔），合成数据时用相机内参把它们投影成 2D 点，标签就有了，一个人都不用标注。

```python
# 合成数据里自动生成 2D 关键点标签
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
T_cam_obj = ...                       # 渲染时我们自己设的位姿，已知
kpts3d = np.array([...])               # (N,3) CAD 上标好的关键点
kpts_cam = (T_cam_obj[:3,:3] @ kpts3d.T).T + T_cam_obj[:3,3]
uv = (K @ kpts_cam.T).T
uv = uv[:, :2] / uv[:, 2:3]            # 归一化 → 像素
# uv 就是这张合成图的 2D 关键点标签
```

推理时反过来：网络给出 `uv`，用对应的 `kpts3d` 调 `cv2.solvePnPRansac` 得到 `T_cam_obj`。

### 4.3 合成数据工具与域随机化

| 工具 | 平台 | 结论 |
|---|---|---|
| **BlenderProc** | 跨平台 | **推荐**，自带 BOP writer 出 rgb/depth/mask/位姿 GT |
| NVIDIA Isaac Sim | Linux only | 你这台跑不动 |
| PyBullet / Sapien | 跨平台 | 备选 |

**决定成败的是域随机化，不是网络结构。** 六条必须做：HDRI 背景 ≥50 张、光照（数量/方向/强度/色温，至少一个强方向光）、材质（金属件的粗糙度和金属度最关键）、遮挡 0–50%、相机位姿（覆盖你实际视角范围，别均匀撒满整个球）、噪声与退化（深度加噪+空洞、彩色加噪+运动模糊+JPEG）。

### 4.4 你这台机器上的性能预期（GTX 1650 4GB / 8GB 内存）

| 环节 | 耗时 |
|---|---|
| YOLO11n-pose 推理（640×480） | 15–30 ms |
| `solvePnPRansac`（8 点） | 1–3 ms |
| **合计** | **20–35 ms（30–50 FPS）** |

训练 YOLO11n-pose：`imgsz=640`、`batch=4~8`、开 AMP，几小时能出结果。4 GB 显存够。

---

## 5. 路线 B：模型化免训练位姿（慢）

### 5.1 候选盘点

| 方案 | 年份 | 核心机制 | 要不要权重 | 依赖 | 速度 | Windows |
|---|---|---|---|---|---|---|
| **FoundPose** | ECCV 2024 | DINOv2 patch 特征 + BoW 检索 + kNN 匹配 + PnP-RANSAC | **不需要任何权重** | 纯 torch + 渲染器 | ~1.3–1.7 s | **可行** |
| **FoundationPose** | CVPR 2024 | 大规模预训练 + 渲染姿态假设 + refine 网络 | 官方预训练权重 | nvdiffrast + PyTorch3D | 亚秒–数秒 | **需 WSL2** |
| MegaPose | CVPR 2022 | render-and-compare，粗 + 精两阶段 | 需权重 | PyTorch3D | 数秒 | 需 WSL2 |
| GigaPose | 2024 | MegaPose 的加速版 | 需权重 | PyTorch3D | 比 MegaPose 快 | 需 WSL2 |
| SAM-6D | CVPR 2024 | SAM 分割 + 模板匹配 | 需权重 | 部分需编译 | 慢 | 多半需 WSL2 |

### 5.2 FoundPose 详解（推荐先做这个）

**已核实的事实**（来源：facebookresearch/foundpose README + 论文）：

- 作者：Evin Pınar Örnek 等（Meta + Tomas Hodan），ECCV 2024。
- **"Our training-free method does not require the release of any weights."** —— 官方原话，它连自己的权重都不发布，完全建立在冻结的 DINOv2 之上。
- 前提：**有 CAD 网格 + 一张物体分割掩膜**。查询只需要 RGB 图 + 内参 K（**不需要深度图**）。

**离线 onboarding（每个物体做一次）**：

```
1. 渲染约 800 个视角的 RGB-D 模板（MegaPose 要 9 万+，它少 100 倍）
2. 用 DINOv2 提取每个模板的 14×14 非重叠 patch 描述子，PCA 降维
3. 用模板的深度通道把描述子注册到 3D 点（这样 2D 匹配直接变成 2D-3D 对应）
4. 对所有 patch 描述子做 k-means 得到视觉词表，每个模板用 TF-IDF 加权的 BoW 向量表示
```

**在线推理**：

```
裁剪后的查询图 → 算 BoW 向量 → 检索最相似的 h 个模板
    → kNN 匹配 patch 描述子 → 建立 2D-3D 对应
    → PnP-RANSAC 解位姿 → （可选）featuremetric refinement
```

**两个关键设计，值得记住**：

1. **用 DINOv2 的中间层（第 18 层），不是最后一层。** 浅层偏定位、深层偏语义；物体对称或无纹理时深层语义会歧义，中间层仍保持几何一致的对应。这一条让它在无纹理工业件上明显优于用最后一层特征的方案（如 ZS6D）。
2. **BoW 检索比 MegaPose 的粗估计快 15 倍**，而且只需 800 个模板（内存占用低 25 倍），余弦相似度对遮挡也鲁棒（被遮住的部分仍贡献视觉词）。

**速度**：论文表格里粗位姿约 **1.3–1.7 s/帧**（V100 级 GPU）。你这台 GTX 1650 会更慢，粗估 3–8 s。加上 refinement 更慢。

**两个必须知道的限制**：

- **开源仓库只提供了粗位姿管线，没有 featuremetric refinement。** README 明确写了 "we provide the coarse pose estimation pipeline without the featuremetric refinement stage"。论文里加了 refinement 后效果明显更好，所以开源版精度低于论文数字（官方给的自测：LMO AR 33.7、TUD-L 40.7）。
- 它依赖 **bop_toolkit 和 dinov2 两个 git submodule**，环境搭建要走官方的 conda yaml（`conda_foundpose_gpu.yaml`）。

### 5.3 FoundationPose / MegaPose / GigaPose

**FoundationPose（NVlabs，CVPR 2024）** 是精度天花板，也是"统一 model-based / model-free"的框架。它的价值：

- 不需要为你的物体训练，用官方权重直接跑。
- 支持 RGB-D 和纯 RGB 两种模式。
- 对未见物体泛化好，严重遮挡下也稳。

代价：

- 依赖 **nvdiffrast**（可微分渲染）和 **PyTorch3D**，都要编译 CUDA 扩展。
- **PyTorch3D 官方预编译 wheel 只到 `py39_cu118_pyt200`**，跟你要的 torch 2.6 + cu124 不匹配，Windows 上源码编译要改 torch 头文件（官方 INSTALL.md 明说）。
- 你这台 1650 4 GB 跑起来会比较吃力。

**MegaPose / GigaPose** 是 render-and-compare 路线的代表，同样要 PyTorch3D。GigaPose 是 MegaPose 的加速版。

有意思的一点：**FoundPose 论文里把 MegaPose 当 refinement 用**——两者是互补的，不是替代关系。

### 5.4 对比与选型

| 维度 | FoundPose | FoundationPose |
|---|---|---|
| 是否为物体训练 | 否 | 否 |
| 是否需要任何权重 | **否** | 是（官方预训练） |
| 依赖 | 纯 torch + DINOv2 + 渲染器 | nvdiffrast + PyTorch3D（需编译） |
| 操作系统 | **Windows 可行** | 建议 WSL2 |
| 速度 | 1.3–1.7 s（V100），本机 3–8 s | 亚秒–数秒（GPU 越强越快） |
| 精度 | 中上（开源版无 refine） | **高** |
| 遮挡鲁棒性 | 中（BoW 对遮挡有一定鲁棒性） | **强** |
| 需要深度图 | **否**（纯 RGB 即可） | 可选 |
| 上手时间 | 半天–一天 | 2–3 天（环境）+ 调试 |

### 5.5 为什么它慢，以及怎么用

慢的根源是**"渲染-比对"范式**：要么离线渲染几百个模板再检索匹配，要么在线渲染上百个姿态假设再打分。这是原理决定的，不是参数没调好。

所以路线 B 不该跑在每帧上。正确用法见 6.2。

### 5.6 已排除的几何法（仅作兜底）

OpenCV 的 PPF + ICP（`cv2.ppf_match_3d`）按你的要求已从主推移除。它仍然存在的价值只有一个：**当某个物体几何特征极其规则（比如就是一个带孔的法兰盘），而 FoundPose 的 DINOv2 特征因为无纹理而匹配不到对应时，几何法可能反而更稳。** 真遇到这种情况再回来试，成本只有半天。

---

## 6. 两条路线的关系：并列 + 互为备份

### 6.1 什么时候用哪条

| 场景 | 用哪条 | 原因 |
|---|---|---|
| 产线固定几种零件、要节拍 | **A** | 20–35 ms，跟得上节拍 |
| 新零件刚导入、还没造合成数据 | **B** | 给个 CAD，渲染完模板就能用 |
| 偶尔来的急件 | **B** | 不值得为一次性零件训练 |
| A 在某件上表现不稳，排查 | **B** 做对照 | B 也差 → 数据/渲染问题；B 好 A 差 → 训练问题 |

### 6.2 最实用的用法：用 B 给 A 打标签

这个技巧能省掉大量人工，也是两条路线共存的真正理由：

1. 用真实相机采集几百张图。
2. 用 **FoundPose**（免训练）离线跑出位姿。B 慢无所谓，这是离线批处理。
3. 用多视图一致性过滤掉明显失败的帧（同一个物体不同视角算出的位姿应该自洽）。
4. 剩下的就是**真实场景 + 真实位姿标注**的数据集。
5. 用这份真机数据训练/微调路线 A。

这比纯合成数据训出来的模型更贴真机，而**标注成本接近零**。

### 6.3 统一接口

```python
class PoseEstimator:
    def estimate(self, rgb, depth, K, cls_id) -> tuple[np.ndarray, float]:
        """返回 (T_cam_obj(4x4), 置信度)"""
        ...
```

`PoseEstimatorA`（YOLO11-pose + PnP）和 `PoseEstimatorB`（FoundPose）实现同一个方法，上层自由切换。下游只认 `T_cam_obj`。

---

## 7. 复现目标项目：需要你确认的 6 个信息

| # | 需要确认 | 为什么影响方案 |
|---|---|---|
| **1** | **那个项目叫什么 / 什么形态？** 商业软件（梅卡曼德、大寰、海康、Halcon 3D）还是开源仓库？ | 决定功能清单。商业软件的"深度学习模式 / 模板匹配模式"和学术方案是两套东西 |
| 2 | 它有几个模块？是"训练"和"免训练"两个按钮，还是一条自动流程？ | 决定架构是并列还是级联 |
| 3 | 输入是什么？只给 CAD，还是要先拍几张参考图 / 示教？ | 决定要不要 model-free 能力 |
| 4 | 输出的位姿相对谁？相机、基座还是末端？原点在哪？ | 决定和现有 `T_base_cam` 的对接 |
| 5 | 标定流程是软件自带还是你自己做的？ | 你已有 `calib_handeye.py`，可能能直接复用 |
| 6 | 有没有"示教抓取点"环节（在物体上点一个点表示这样抓）？ | `create_tmpl_grasp_3d.py` 已在做类似的事，可能能对上 |

**如果它是商业 3D 视觉软件**：那它的两个模式大概率是「深度学习（要训练/要标注）」和「模板匹配/特征匹配（导入 CAD 就能用）」。后者的现代等价物就是 **FoundPose 这一类基于基础特征的免训练方法**——这跟你"两条都是模型"的要求正好吻合。

---

## 8. 对 OS 结论的影响

**相对上一版回摆一半。** 上一版因为推荐 PPF+ICP，结论是"两条路线都能留在 Windows"。现在路线 B 换成神经网络方案后：

| 路线 B 的选择 | 操作系统 |
|---|---|
| **FoundPose** | **Windows 可行**（纯 PyTorch + DINOv2；唯一风险是渲染器，用 BlenderProc 规避） |
| FoundationPose / MegaPose / GigaPose / SAM-6D | **需 WSL2**（要编译 nvdiffrast / PyTorch3D） |

路线 A 的结论不变：**Windows 全链路可行**。

所以决策树变成：

```
路线 B 选 FoundPose？
   ├─ 是 → 全程 Windows，不用装 WSL2
   └─ 否（要 SOTA 精度/强遮挡鲁棒）→ 装 WSL2 Ubuntu 22.04
```

考虑到你只有 8 GB 内存，**我建议先走 FoundPose 那条**——先拿到第一个位姿数字，再决定值不值得为 FoundationPose 付环境代价。

---

## 9. 坑与改进建议

| # | 现象 | 根因 | 改法 |
|---|---|---|---|
| 1 | 位姿平移量差 1000 倍 | CAD 是 mm，输出要 m | 加载后检查 `mesh.extents`，>1 就乘 0.001，写成断言 |
| 2 | FoundPose 检索不到正确模板 | 渲染模板与真实外观域差距太大 | 用 BlenderProc 渲染 + 随机光照/背景；模板视角要覆盖实际观察范围 |
| 3 | FoundPose 精度低于论文 | 开源仓库**没有 refinement 阶段** | 已知限制；可外接 MegaPose 做 refinement（论文验证过互补） |
| 4 | BOP toolkit 渲染器在 Windows 报错 | 依赖 OpenGL 上下文 | 改用 BlenderProc 渲染模板，绕开 GL 依赖 |
| 5 | 无纹理金属件匹配不到特征 | DINOv2 最后一层语义歧义 | 确保用**中间层**（第 18 层）特征；这是 FoundPose 的关键设计 |
| 6 | 合成数据训的模型到真机 mAP 崩 | 域随机化不足 | 补 HDRI ≥50、材质随机、噪声、运动模糊；再混 10–20% 真实图 |
| 7 | PnP 结果抖 | 2D 关键点亚像素精度不够 / 误匹配 | 用 RANSAC；增加关键点数量；对称物体做姿态归一化 |
| 8 | 圆柱/方块位姿跳变 | 旋转对称，多解 | CAD 阶段定义唯一参考方向；参考 `compute_locate_error` 里的 `sym_tfs` |
| 9 | YOLOE 首次推理失败 | 首次 `set_classes()` 要联网下文本编码器 | 联网环境先跑一次；或 `save_prompt_embeddings()` 烘进权重 |
| 10 | 抓不准但位姿数值正常 | 物体坐标系原点与夹爪中心不一致 | CAD 阶段显式定义"夹持中心"并写进文档 |
| 11 | 没有真值无法评估 | 不贴 tag 就没有独立参考 | **评估阶段临时贴一个 AprilTag 当真值**，量化完撕掉 |
| 12 | DINOv2 显存爆 | ViT-L/14 在 4 GB 卡上吃力 | 用 ViT-S/14 或 ViT-B/14；或缩小裁剪图尺寸 |

第 11 条再强调一次：不贴 AprilTag 不等于永远不能用 AprilTag。评估阶段临时贴一个是唯一能拿到可信真值的低成本办法，测完撕掉即可。

---

## 10. 四周落地时间表

| 周 | 目标 | 产出 | 环境 |
|---|---|---|---|
| **第 1 周** | CAD 地基 + FoundPose 跑通 | `cad_prep.py`、模板渲染脚本、FoundPose onboarding；**拿到第一个真实位姿数字** | Windows |
| **第 2 周** | 检测器 + 合成数据 | BlenderProc 脚本产出 500–2000 张带 GT 的合成图；训练出 `yolo11n-pose` | Windows |
| **第 3 周** | 路线 A 跑通 | `关键点 → solvePnP → T_cam_obj`；与 FoundPose 对比精度与耗时 | Windows |
| **第 4 周** | 评估与接入 | 临时贴 AprilTag 取真值，量化两条路线误差；封装统一接口接进抓取流程 | Windows |

**第 1 周结束时你就有可判断的东西。** 如果 FoundPose 的误差已满足抓取要求，后面三周都可以推迟。

如果第 4 周发现精度不够，再考虑装 WSL2 上 FoundationPose——那时你已经有了完整的评估基线和对比数据，迁移决策是有依据的，不是赌。

---

## 11. 一句话总结

**两条路线都是模型方案：路线 A 用 CAD 造合成数据训练一个 YOLO11-pose（加 PnP 闭式解）每帧 20–35 ms 出位姿；路线 B 用 FoundPose——基于冻结 DINOv2、官方明确不需要发布任何权重、只要 CAD 加一张掩膜就能出位姿的免训练方法，代价是 1–10 秒。先做 FoundPose 拿到第一个位姿数字，再决定值不值得为 FoundationPose 装 WSL2。**

---

## 附：相关链接

- FoundPose（facebookresearch）：https://github.com/facebookresearch/foundpose
- FoundationPose（NVlabs）：https://github.com/NVlabs/FoundationPose
- MegaPose / GigaPose：https://github.com/agimus-project/megapose
- Ultralytics YOLOE 文档：https://docs.ultralytics.com/models/yoloe
- BlenderProc：https://github.com/DLR-RM/BlenderProc
- BOP 基准与数据集：https://bop.felk.cvut.cz/datasets

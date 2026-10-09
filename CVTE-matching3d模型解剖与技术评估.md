# CVTE matching3d 流水线：模型解剖、2026 替代方案评估、单环境可行性

> 整理日期：2026-09-28
> 依据：你对四个 ONNX 文件与 C++ 符号的解剖结果 + 我核实的官方仓库/榜单数据
> 目的：评估这套方案今天是否仍是优选，以及你能否在一个环境里复现它

---

## 第 0 部分：先给三个结论

**结论一：这套架构的设计思路是对的，而且和 NVIDIA 官方 2026 年的做法几乎一样。**
BOP 2026 榜单上 NVIDIA 提交的 `SAM-FP` 就是"开放词汇分割 + FoundationPose"，和 CVTE 的"YOLO-seg + FoundationPose"是同一个骨架，只是检测器从 SAM3 换成了 YOLO11n-seg（换取速度）。说明 CVTE 选的路线没有过时。

**结论二：有更好的单点模型，但换不换取决于你的显卡，不取决于榜单。**
立体匹配环节出现了 **WAFT-Stereo**（ECCV 2026），三榜第一且快 1.8–6.7 倍；位姿环节出现了 **MAPRPose**（2026），mean AR 76.5 超过 FoundationPose 的 73.4。**但你只有 4GB 显存，这两者的收益你未必吃得到。**

**结论三：五个模型能在一个环境里"加载"，但跑不起来 —— 卡在渲染，不在推理。**
全部权重都是 ONNX，用一套 `onnxruntime-gpu` 就能加载。但 FoundationPose 的 RefineNet 需要**在线渲染 CAD 假设**，这一步不在 ONNX 里，在 CVTE 的 C++ 里。**这才是"一个环境"真正的裂缝**，也是你复现时最大的工程难点。

---

## 第 1 部分：这条流水线在做什么

### 1.1 分层逻辑

```
输入：双目 RGB（+ CAD 模型）
  │
  ├─[1] S²M² → 视差图（+ 遮挡图 + 置信度图）
  │
  ├─[2] YOLO11n-seg → 目标的 2D 掩码
  │
  ├─[3] RDPN6D → 粗位姿（一次前向，快）
  │
  ├─[4] RefineNet → 精位姿（迭代细化，慢但准）
  │
  └─[5] ScoreNet → 给多个候选打分，挑最优
```

**这个"快 + 准"两级定位的设计是整个方案最值得学的地方**：

- 第 3 步 RDPN6D 是**按物体训练**的回归网络，一次前向几十毫秒出粗位姿，但泛化差、精度有限
- 第 4、5 步 FoundationPose 是**类别无关**的，换物体只换 CAD 不用重训，但要迭代渲染、慢

先用快的给出一个差不多的初值，再用慢的在这个初值附近精修 —— 既省了迭代次数，又保证了精度。这是工程上非常经典的"coarse-to-fine"。

### 1.2 为什么精定位权重放在全局目录

你观察到的这个细节很关键：

```
data/obj_14/weights/segmentor.onnx      ← 按物体
data/obj_14/weights/fast_locator.onnx   ← 按物体
data/weights/precise_locator/0.onnx     ← 全局共享
data/weights/precise_locator/1.onnx     ← 全局共享
```

因为 FoundationPose 的 refine/score 网络**吃的是"渲染图 vs 观测图"的对比，不是物体外观本身**。网络学的是"这两张图像不像同一个位姿"这种通用能力，与具体物体无关。所以它一份权重服务所有物体。

**这带来一个巨大的工程好处**：新增物体时，你只需要训一个 segmentor + 一个 fast_locator，精定位部分零成本。

---

## 第 2 部分：五个模型逐个详解

### 2.1 S²M² —— 立体深度（ICCV 2025, Samsung）

| 项 | 值 |
|---|---|
| 论文 | arXiv 2507.13229，ICCV 2025 |
| 作者 | Junhong Min 等，三星电子 |
| 仓库 | github.com/junhong-3dv/s2m2 |
| 架构 | CNN 骨干 → 多分辨率 Transformer（MRT）→ OT/Sinkhorn 初始视差 → 局部 GRU refine + global refine → 4× 上采样 |
| 输出 | **视差 + 遮挡 + 置信度**（三张图） |
| 变体 | S(Ch128,Ntr1, 26.5M) / M(192,2, 80.4M) / L(256,3, 181M) / **XL(384,3, 406M)** |
| CVTE 用的 | **XL**，640×480，TensorRT FP16 |
| ONNX | opset 18，9887 nodes，producer pytorch 2.5.0 |
| 许可 | **三星非商用声明**（research/education only） |

**它和 FFS 的根本分歧在架构范式上：**

| | S²M² | FFS / RAFT-Stereo 系 |
|---|---|---|
| 范式 | **全局匹配** | 迭代局部搜索 |
| 核心机制 | 多分辨率 Transformer 做长程对应 | 代价体 + GRU 逐步修正 |
| 代价体 | **不用 3D 卷积过滤** | 必须（GWC/concat volume） |
| 细化 | 浅层 refine + global refine | 深迭代堆栈（8–32 次） |
| 显存瓶颈 | Transformer 注意力 | 3D 卷积代价体 |

S²M² 的论文核心论点就是：**迭代局部搜索在受限榜单上分数高，但机制上天生缺全局一致性，所以泛化差**。它用全局匹配换泛化，再用多分辨率 Transformer 解决全局匹配"算力吃不下"的老问题。

**为什么 CVTE 选它而不是 FFS？** 我判断是**置信度输出**。机器人抓取需要知道"这个深度值可不可信"，S²M² 联合估计 occlusion 和 confidence，直接给下游用（对应你看到的 `th_occlusion` / `th_confidence` 两个参数）。FFS 不输出这两个。

**一个必须注意的差异**：官方 README 明确写了——

> The publicly released model weights differ from the version used for Middlebury and ETH, but are identical to the one used in Booster benchmark.

也就是说 **HuggingFace 上能下到的权重 ≠ 论文里刷 Middlebury/ETH 榜的那个**。而且为了稳定导出 ONNX，官方把动态注意力细化模块换成了 UNet。你如果自己复现，拿到的是"Booster 版 + UNet 细化"，精度会略低于论文数字。

**速度参考**（官方数据，RTX 5090，FP16，refine_iter=3，640×480）：

| 变体 | torch.compile | TensorRT |
|---|---|---|
| S | 59.8 FPS | **124.0 FPS** |
| M | 32.4 | 66.7 |
| L | 25.8 | 46.6 |
| XL | 16.3 | **26.6** |

### 2.2 YOLO11n-seg + CNOS —— 检测分割

- **模型**：Ultralytics YOLO11n-seg，最小规格的实例分割
- **训练来源**：FoundationPose 仓库里的 `cnos/` 目录（上游是 NAVER 的 CNOS）
- **关键改动**：CNOS 原本用 **SAM + DINOv2** 做检测，这里被换成了 YOLO11n-seg
- **许可**：**AGPL-3.0**（Ultralytics）

**为什么换成 YOLO？** SAM 是 ViT-H 级别的大模型，一张图几百毫秒，机器人实时抓取扛不住。YOLO11n-seg 是 nano 规格，几毫秒出掩码。**用精度换速度，是机器人场景的标准取舍。**

CNOS 本身的方法是"CAD 渲染多视角 → DINOv2 提特征建码本 → 用分割出的区域特征去码本里检索"。CVTE 把检测部分换掉后，剩下的检索逻辑是否保留，需要看 `yolo_segmentor.cpp`。

### 2.3 RDPN6D —— 快速位姿（fast_locator.onnx）

| 项 | 值 |
|---|---|
| 来源 | github.com/AI-Application-and-Integration-Lab/RDPN6D |
| 血统 | 基于 **GDRN / GDRNPP**（Geometry-Guided Direct Regression Network） |
| 输入 | `roi_rgb`（裁剪 RGB）、`roi_xyz`（反投影点云）、`roi_uv`（坐标图） |
| 输出 | `fc_r` 旋转头 + `fc_t` 平移头 |
| 结构 | ResNet 骨干 + `md_pointnet`（点云分支，xyz_emb/conv1-3）+ `up_layer` + 双头 |
| 许可 | MIT |

**它是怎么工作的**：把检测框裁出来的区域，一路走 CNN 提外观特征，一路把深度反投影成点云走 PointNet 提几何特征，融合后**直接回归**出旋转和平移。

**为什么要 `roi_xyz`**：单靠 RGB 回归位姿在弱纹理物体上会崩，点云分支提供了明确的几何约束。这也是为什么**这一步必须依赖深度图** —— 你的两条定位路线都要深度，根因就在这里。

**代价**：它是**按物体训练**的（`obj_14/` 目录）。每加一个物体就要重新采集、渲染、训练一次。这正是 CVTE 要再用 FoundationPose 做第二级的原因——用"按物体训练的快速网络"换速度，用"类别无关的细化网络"换免训练扩展性。

### 2.4 / 2.5 FoundationPose RefineNet + ScoreNet

| 项 | RefineNet (0.onnx) | ScoreNet (1.onnx) |
|---|---|---|
| 对应类 | `RefineNet` | `ScoreNetMultiPair` |
| 输入 | 4 个 160×160×3（uint8/float32 各二） | 多对假设的渲染-观测对 |
| 输出 | 6 维（旋转 + 平移增量） | 1 维 score logit |
| 关键节点 | encodeA / encodeAB / pos_embed / trans_head / rot_head | encoderA / encoderAB / att / att_cross / pos_embed / linear |
| 许可 | **NVIDIA 非商用** | 同 |

**FoundationPose 的完整流程**（这两个网络只是其中两环）：

1. 用粗位姿生成 N 个假设位姿（CVTE 用 RDPN6D 的输出做初值，官方用均匀采样）
2. **为每个假设渲染 CAD 模型**（这步是瓶颈）
3. RefineNet 吃"渲染图 vs 观测图"，输出位姿**增量**
4. 迭代若干次
5. ScoreNet 给每个假设打分，挑分最高的

**⚠️ 第 2 步的渲染不在 ONNX 里。** 这是理解整套系统的关键，也是你复现时最大的坑 —— 详见第 4 部分。

**另一个发现**：`0.pt` / `1.pt` 不是普通 checkpoint，而是 **RKNN 转换用的权重容器**（zip 顶层目录名就是 RefineNet / ScoreNet），`0.rknn` 是 rknn-toolkit2 2.3.2 编译的 **NPU 版（target rk3576）**。说明 CVTE 还有一条**部署到瑞芯微 RK3576 边缘 NPU** 的路线。这佐证了他们很在意边缘部署成本。

---

## 第 3 部分：2026 年有更好的吗

### 3.1 立体深度：WAFT-Stereo 是最大变量

**WAFT-Stereo**（ECCV 2026，Princeton VL，Yihan Wang & Jia Deng，arXiv 2603.24836）：

- **不需要 cost volume** —— 用可变形 warping 取代
- ETH3D（BP-0.5）、Middlebury（RMSE）、KITTI（**所有指标**）三榜第一
- 零样本误差比之前降 **81%**（ETH3D）
- 比同类方法快 **1.8–6.7 倍**
- 仓库：github.com/princeton-vl/WAFT-Stereo，权重在 HuggingFace
- 自带 `profiler.py` 可直接查参数量、MACs、显存

**为什么这对你特别重要**：它去掉 cost volume，意味着**显存大幅下降，且不依赖 TensorRT 优化就有性能**。搜索结果里明确提到这点——"使得在机器人、自动驾驶等实时性要求极高的场景部署高精度立体视觉模型的算力门槛和延迟大幅降低，简化的模型架构也降低了显存占用，使得在边缘设备上运行成为可能，而不依赖 TensorRT 等复杂工程优化"。

**一句话：WAFT-Stereo 是为你这张 4GB 卡量身定做的技术路线。**

⚠️ 但要注意：它的 README 里写了 "You must comply with the NVIDIA license before using it" —— 因为它复用 FoundationStereo 的代码。

**其他 2026 年的竞争者**（SOTA2 榜单数据）：

| 方法 | 主战场 | 备注 |
|---|---|---|
| WAFT-Stereo | ETH3D / Middlebury / KITTI 三榜第一 | 最快，无 cost volume |
| DEFOM-Stereo | KITTI 2015 1.33 D1 | 跨领域最稳（林业 benchmark 里评为最鲁棒） |
| MoCha-Stereo | ETH3D 0.032 bad1.0 | |
| LiteMatch | ETH3D 非遮挡 0.35，零样本 2.01 | 轻量 |
| MonSter / MonSter++ | ETH3D test 0.72 | |
| **FoundationStereo** | **Middlebury 1.1 Bad2.0** | FFS 的"大哥"，2026-07 仍是 Middlebury SOTA |
| BridgeDepth | ETH3D 0.23px EPE 最佳 | 但 Middlebury 崩到 20.03px，泛化差 |

**注意 FoundationStereo 仍在 Middlebury 榜首** —— 你选 FFS 并没有选错，它是 FoundationStereo 的蒸馏加速版。

### 3.2 检测分割：SAM3 带来的免训练可能

NVIDIA 2026-09 提交的 **SAM-FP** 用 **SAM3 做文本提示的开放词汇检测**（conf 阈值 0.10），配 FoundationPose，**完全免训练**。

这意味着：如果你的物体种类经常变，可以**不训 YOLO，直接用文本提示**（"the yellow cup"）。代价是 SAM3 比 YOLO11n-seg 慢很多。

**取舍**：
- 固定几种物体、追求速度 → **YOLO11n-seg**（CVTE 的选择）
- 物体种类多变、追求零训练 → **SAM3**
- 折中 → **YOLOE**（开放词汇 but 快于 SAM）

### 3.3 精定位：MAPRPose 已超过 FoundationPose

BOP 榜单（2026-04 更新，均用 SAM-6D 做分割）：

| 方法 | 年份 | LM-O | T-LESS | Mean AR | 速度 (FPS) | GPU |
|---|---|---|---|---|---|---|
| **MAPRPose** | 2026 | **77** | 70.2 | **76.5** | 1.47 | A100 |
| Co-op (Top-K=5) | 2025 | 73 | 67.8 | 75.9 | 0.14 | 3090 |
| **FreeZeV2** | 2025 | 73.3 | 59.7 | 74.3 | 0.71 | A40 |
| **FoundationPose** | 2024 | 75.6 | 64.6 | 73.4 | **0.03** | A100 |
| Co-op (Coarse) | 2025 | 70 | 65.4 | 72.7 | 1.10 | 3090 |
| SAM-6D | 2024 | 69.9 | 51.5 | 70.4 | 0.23 | 3090 |
| ZeroPose | 2024 | 63.5 | 43 | 62.2 | 0.18 | 3090 |

**两个关键读数**：
1. **MAPRPose 精度最高（76.5），且比 FoundationPose 快 49 倍**（1.47 vs 0.03 FPS）
2. **FoundationPose 慢得离谱** —— 0.03 FPS 意味着一张图 33 秒。原因是它**一次只处理一个假设**，GPU 利用率极低。MAPRPose 用张量批处理一次跑 35 个假设对，GPU 利用率 90%+

BOP23 XYZ-IBD 榜单（更新时间更近）：3PT-Pose 0.816 AP（42s）、SAM-FP 0.765、FRTPose-WAPR.v2 0.744（160s）、FreeZeV2.2 0.636（22.8s）、**FoundationPose(CNOS-SAM) 0.564（42.4s）**

**结论**：FoundationPose 在 2026 年已经不是精度最优，也不是速度最优。它剩下的优势是**生态成熟、有大量工程封装**（CVTE 用的就是这种）。

### 3.4 逐环节替换建议汇总

| 环节 | CVTE 用的 | 2026 更好的选择 | 建议 |
|---|---|---|---|
| 立体深度 | S²M² XL | **WAFT-Stereo**（三榜第一，快 1.8–6.7×，无 cost volume） | ⭐ 值得试，尤其对你 4GB 卡 |
| 检测分割 | YOLO11n-seg | SAM3（免训练）/ YOLOE（折中） | 保持 YOLO，除非物体种类常变 |
| 快速定位 | RDPN6D (GDRN系) | ZebraPose / DLTPose / YOLO11-pose+solvePnP | 保持（你有训练数据） |
| 精定位 | FoundationPose | **MAPRPose**（+3.1 AR，快 49×）/ FreeZeV2.2 | ⭐ 值得关注，但 MAPRPose 代码成熟度待验证 |

---

## 第 4 部分：能否在一个环境里运行（核心问题）

### 4.1 好消息：五个权重全是 ONNX

| 模型 | 格式 | 加载方式 |
|---|---|---|
| stereo_depth_estimator/XL_640x480 | ONNX / .engine | ORT 或 TensorRT |
| segmentor | ONNX | ORT |
| fast_locator | ONNX | ORT |
| precise_locator/0 (RefineNet) | ONNX | ORT |
| precise_locator/1 (ScoreNet) | ONNX | ORT |

**单看推理，一套 `onnxruntime-gpu` 就够了，完全不需要 PyTorch。** CVTE 的 Python 环境里没装 torch 就是证据——推理全在 C++ 里完成。

ONNX 还带来一个巨大好处：**opset 18 + ORT 1.18 是稳定的组合，没有 torch/torchvision/CUDA 的版本锁死问题**。你之前在 FFS 上踩的 torch 2.6/cu124 兼容性坑，在这套架构里不存在。

### 4.2 坏消息：渲染不在 ONNX 里

这是"一个环境"真正的裂缝。FoundationPose 的 RefineNet 需要这样的输入：

```
观测裁剪图 (160×160×3)  ←  从 RGB-D 裁出来
    ＋
渲染图 (160×160×3)      ←  把 CAD 模型按某个假设位姿渲染出来
```

前半部分是普通图像处理。后半部分**需要 GPU 光栅化**：加载 mesh、设置虚拟相机、渲染 RGB + 深度。

CVTE 把它写在 `matching3d::PoseRefiner` 的 C++ 里（支持 ONNX / LibTorch 两种推理后端，但渲染必然是自己实现的）。你**拿不到这部分源码**。

**自己实现的三条路**：

| 方案 | 依赖 | 问题 |
|---|---|---|
| **PyTorch3D** | torch ≤ 2.4.1，需源码编译 | 8GB 显存门槛，编译痛苦，GTX 1650 不合适 |
| **nvdiffrast** | CUDA toolkit + torch 扩展 | 需要 nvcc，你 PATH 里没有；要装 CUDA 11.8/12.6 toolkit |
| **OpenGL / EGL 离屏** | mesa / libEGL / OSMesa | 与深度学习环境无关，可在纯 C++ 或 pyopengl 里做 |

**这就是矛盾所在**：你为了"一个环境"想统一到 ONNX Runtime，但渲染这一步**必须引入 PyTorch 或一套独立的图形栈**。

### 4.3 版本矩阵

CVTE 实际在用的（你 ldd 出来的）：

| 组件 | 版本 | CUDA |
|---|---|---|
| libtorch | 2.2.2 | cu118 |
| TensorRT (C++) | 10.13.0.35 | cuda11.8 |
| ONNX Runtime (C++) | 1.18.0 GPU | — |
| onnxruntime-gpu (py) | 1.18.0 | — |
| tensorrt (py) | 8.6.x | — |
| 驱动 / 硬件 | 595.84 / CUDA 13.2 | RTX 3080 Ti |

而 **S²M² 官方推荐**的是：Python 3.10 + **PyTorch 2.9 + CUDA 12.9 + cuDNN 9.1.0 + TensorRT 10.13.3**。

**这两个不冲突**，因为 CVTE 只拿 S²M² 的 ONNX 权重做推理，不需要官方训练/导出环境。这也说明一件事：**ONNX 作为中间格式，成功隔离了训练环境和部署环境。**

### 4.4 三个可行方案

**方案 A：纯 ONNX Runtime（最干净，推荐你先做这个）**

```
python 3.10
+ onnxruntime-gpu 1.18.0
+ opencv / numpy / open3d（几何前后处理）
+ 渲染：单独用 pyopengl + OSMesa 或 nvdiffrast（独立装）
```

- 优点：无 torch 版本地狱，五个模型共用一套 session
- 缺点：渲染部分要自己造轮子
- 适合：**先把流水线跑通，验证功能**

**方案 B：ORT + torch 双栈（折中）**

```
torch 2.2.2+cu118（与 CVTE 一致）
+ onnxruntime-gpu 1.18.0
+ nvdiffrast（渲染）
```

- 优点：渲染可用现成的 nvdiffrast，网上有 FoundationPose 的参考实现
- 缺点：torch 和 ORT 要共享 CUDA 11.8，版本要对齐；体积大
- 适合：**要复现完整 FoundationPose 效果**

**方案 C：全 TensorRT**

- 优点：最快，CVTE 就是用 TRT FP16
- 缺点：GTX 1650 是 Turing（sm_75），TRT 10.x 对 Turing 的支持要实测；build engine 慢
- 适合：**最后做性能优化时**

---

## 第 5 部分：针对你这张 4GB 卡

### 5.1 显存预算（估算，需实测校准）

| 模型 | 参数量 | FP16 权重 | 估算峰值显存 |
|---|---|---|---|
| S²M² **XL** | 406M | ~812 MB | **2.5–3.5 GB** ⚠️ |
| S²M² **S** | 26.5M | ~53 MB | ~0.5 GB ✅ |
| YOLO11n-seg | ~3M | ~6 MB | ~0.15 GB |
| RDPN6D | ~40M(估) | ~80 MB | ~0.3 GB |
| RefineNet | ~20M(估) | ~40 MB | ~0.2 GB |
| ScoreNet | ~15M(估) | ~30 MB | ~0.2 GB |

**结论很直接**：

- **S²M² XL 单独就要 2.5–3.5GB**，你的 4GB 卡放完它，剩下的模型几乎没有余量 —— 大概率 OOM
- **换成 S 变体，整套约 1.3GB**，能跑
- 或者继续用你已经跑通的 **FFS 半分辨率（411MB）**

### 5.2 我的建议

**不要一上来就照搬 CVTE 的 XL 配置。** 你已经在 WSL2 里把 FFS 跑通了（半分辨率 411MB / 0.26s 每帧 / 有效深度 90.7%），这是你有实测数据支撑的方案。

建议的推进顺序：

| 阶段 | 深度 | 检测 | 精定位 | 状态 |
|---|---|---|---|---|
| **第 1 步（先做完）** | FFS 半分辨率 | YOLO11n-seg | FoundPose（纯 torch，4GB 可跑） | 先跑通整条链路 |
| **第 2 步（再优化）** | 试 S²M²-**S** 或 **WAFT-Stereo** | 保持 | 试 FreeZeV2.2 | 有对比数据后再换 |
| **第 3 步（可选）** | S²M²-L / XL | 保持 | FoundationPose + 自造渲染 | 需要 8GB+ 显卡 |

**关于 FoundationPose**：它需要 PyTorch3D/nvdiffrast 渲染 + 8GB 显存，**在你这张卡上不现实**。先用 **FoundPose**（纯 torch，4GB 可跑，之前调研过）把精定位这一环跑通，等以后有更好的显卡再上 FoundationPose。功能是一样的 —— 都是"换 CAD 免训练"。

---

## 第 6 部分：License 风险清单

这是企业场景下必须摆在桌面上的：

| 组件 | 许可 | 商用风险 |
|---|---|---|
| YOLO11n-seg | **AGPL-3.0** (Ultralytics) | ⚠️ 高 —— 需购买 Ultralytics 商业许可或换模型 |
| FoundationPose 权重 | **NVIDIA 非商用** | ⚠️ 高 —— 明确禁止商用 |
| **S²M²** | **三星非商用**（research/education only） | ⚠️ 高 —— 官方声明"与三星任何官方产品无关" |
| RDPN6D | MIT | ✅ 无 |
| matching3d (CVTE) | MIT，但 **license 2026-07-31 已过期** | ⚠️ 内部库，你已无法使用 |
| WAFT-Stereo | 依赖 FoundationStereo → NVIDIA license | ⚠️ 需确认 |

**你现在的处境其实简化了问题**：CVTE 的 matcher 库 license 过期、`import matcher` 直接失败，所以你**本来就必须自己复现**。这反而给了你一次重新选型的机会——可以顺手把上面几个商用风险高的组件换成友好的：

| 原组件 | 商用友好的替代 |
|---|---|
| YOLO11n-seg (AGPL) | YOLO11n-seg 买商业许可，或换 **RF-DETR / YOLOv8(旧版是GPL?)** |
| S²M² (非商用) | **Fast-FoundationStereo 商用版**（你已经下了 `nvidia/c-fast-foundationstereo`）✅ |
| FoundationPose (非商用) | **FoundPose (CC BY-NC)** 仍非商用；真商用需自研或买授权 |

⚠️ 注意 FoundPose 也是 CC BY-NC（非商用）。**如果目标是商业产品，"免训练精定位"这一环目前没有真正开源商用的成熟方案** —— 这是个行业空白，值得你留意。

---

## 第 7 部分：给你的复现路线

### 立即可做（不依赖任何新硬件）

1. **保持 FFS 作为深度源** —— 你已经跑通，有实测数据
2. **训练 YOLO11n-seg** —— 用 CAD 渲染合成数据（这正是 CNOS 的做法），比手工标注省事得多
3. **用 FoundPose 打通精定位** —— 验证"换 CAD 免训练"这个核心能力
4. **把 RDPN6D 放到最后** —— 它需要按物体训练，是整条链路上成本最高的一环，等前面都通了再做

### 值得试的新东西（低成本高收益）

5. **试 WAFT-Stereo** —— 无 cost volume，显存友好，可能在你这张卡上比 FFS 更快更好。它自带 `profiler.py` 能直接看显存占用，试错成本低
6. **试 S²M²-S 变体** —— 26.5M，官方有 ONNX 导出脚本，可以直接对比 FFS

### 需要条件成熟再做

7. FoundationPose 完整复现 —— 需要 8GB+ 显卡 + 自造渲染管线
8. MAPRPose —— 精度最高但要等代码成熟

---

## 附：你的环境现状确认

| 项 | 状态 |
|---|---|
| CVTE matcher | ❌ license 2026-07-31 过期，`import matcher` 失败 |
| numpy | ✅ 已恢复 1.26.4（你回滚得对） |
| scipy / cv2 / onnxruntime | ✅ 正常 |
| FFS (WSL2) | ✅ 已跑通，半分辨率 411MB / 0.26s |
| 硬件 | i5-9300H / 8GB RAM / **GTX 1650 4GB (Turing sm_75)** |

**一个提醒**：GTX 1650 是 Turing 架构，**不支持 BF16，也不支持 FP8**。选型时看到任何依赖 BF16/FP8 的新模型都要绕开。FP16 是支持的（但你在 FFS 上已经踩过 cuDNN fp16 输出 NaN 的坑，见之前的实测记录）。

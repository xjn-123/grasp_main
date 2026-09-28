# 单环境统一方案：深度 / 检测 / 定位共用一套 torch

> **问题**：能不能只用一个 Python 环境，同时跑「深度图生成（FFS）」+「检测（YOLO）」+「定位（换 CAD 不重训的模型）」？
>
> **答案**：能。而且比你想象的容易——因为真正的硬约束只有一处，而那一处恰好在你的机器上用不上。
>
> 配套文档：
> - `版本选型-torch与CUDA矩阵.md`（四个环境方案的原始推导，本文是它的收敛版）
> - `WSL2部署FFS-完整落地方案.md`（FFS 执行手册）
> - `最终选型总结-推荐顺序与搭建方案.md`（模型选型）

---

## 目录

- [0. 结论先行](#0-结论先行)
- [1. 为什么之前给了四个环境：三个死锁源](#1-为什么之前给了四个环境三个死锁源)
- [2. 关键发现：FFS 根本不锁 torch 版本](#2-关键发现ffs-根本不锁-torch-版本)
- [3. 决定性因素：4GB 显存已经把一半候选排除了](#3-决定性因素4gb-显存已经把一半候选排除了)
- [4. 推荐方案 B：torch 2.6.0 + cu124 + Python 3.12](#4-推荐方案-btorch-260--cu124--python-312)
- [5. 备选方案 A：torch 2.4.1 + cu121 + Python 3.10](#5-备选方案-atorch-241--cu121--python-310)
- [6. 方案 C：不追求单环境，用文件解耦](#6-方案-c不追求单环境用文件解耦)
- [7. 三方案对比表](#7-三方案对比表)
- [8. 验证清单：装完怎么确认每层都活](#8-验证清单装完怎么确认每层都活)
- [9. 坑位表](#9-坑位表)
- [10. 我的建议](#10-我的建议)

---

## 0. 结论先行

**一个环境就够，装这个组合：**

```bash
conda create -n dvp python=3.12 -y          # dvp = depth + vision + pose
conda activate dvp
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

然后三层依次往里装（命令见 [第 4 节](#4-推荐方案-btorch-260--cu124--python-312)）：

| 层 | 装什么 | 状态 |
|---|---|---|
| 深度 | Fast-FoundationStereo | 官方档，直接跑 |
| 检测 | YOLO11-seg / YOLOE（ultralytics） | 只要求 torch ≥ 1.8，无约束 |
| 精定位 | FoundPose（换三个依赖后可跑） | 官方锁 2.3.0+cu117，需改造 |
| 快速定位 | YOLO11-pose + solvePnP + 深度 ICP | 纯 torch + OpenCV，无约束 |

**唯一装不进去的是 PyTorch3D 那一派**（FoundationPose / MegaPose / GigaPose 的 refine 阶段）——但在你这块 **GTX 1650 4GB** 上它们本来就跑不动，所以这个损失是名义上的。

---

## 1. 为什么之前给了四个环境：三个死锁源

上一版文档给出四个 conda 环境，不是因为模型之间"不合"，而是三个具体的技术死锁。逐个拆开看，你会发现**其中两个是可以绕过的**。

### 死锁 ①：PyTorch3D 的官方支持上限是 torch 2.4.1

`pytorch3d/INSTALL.md` 明确列出支持的版本：

> PyTorch 2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0 or 2.4.1.

FFS 官方 Dockerfile 锁的是 **2.6.0**，超出上限。这是真正无法用 pip 调和的一处——PyTorch3D 要编译 CUDA 扩展，版本不匹配就是编不过。

**受影响模型**：FoundationPose（还需 nvdiffrast + kaolin）、MegaPose、GigaPose 的 refine 阶段。

### 死锁 ②：FoundPose 的官方环境绑在 CUDA 11 生态

`conda_foundpose_gpu.yaml` 原文：

```yaml
- python=3.9
- pytorch::pytorch=2.3.0
- pytorch::pytorch-cuda=11.7.0
- faiss-gpu=1.8.0
- pip:
  - xformers==0.0.20
  - cuml-cu11==23.04
```

注意包名里带死的 `cu11`：

- `cuml-cu11==23.04` —— RAPIDS 的 cu11 构建
- `faiss-gpu=1.8.0` —— conda 的 cu11 构建
- `xformers==0.0.20` —— 编译绑定 torch 2.3，ABI 不兼容 2.6

**但这三个都不是算法必需的**，[4.3 节](#43-foundpose-的三个依赖怎么替换)给替换方案。死锁 ② 可绕过。

### 死锁 ③：FoundationPose 是独立的一代

`docker/dockerfile` 锁 `python=3.8` + `torch==2.0.0` + cu118，还要编译 kaolin。跟谁都不共享。

**建议：不要手动编译，直接 `docker pull wenbowen123/foundationpose`。** 这条死锁不解决，绕开它。

---

## 2. 关键发现：FFS 根本不锁 torch 版本

这是本次核实最重要的一条，也是"单环境"能成立的前提。

FFS 的 `requirements.txt` 全文：

```
# === Model & inference ===
timm
einops
omegaconf
scipy
numpy
scikit-image
opencv-contrib-python
imageio
pyyaml
open3d

# === TensorRT (optional — uncomment if needed) ===
# onnx
# onnxruntime-gpu
# tensorrt-cu12
# tensorrt-lean-cu12
# tensorrt-dispatch-cu12
# nvidia-modelopt[torch]
```

**里面没有 torch，没有 flash-attn，没有 triton，没有 xformers，没有自定义 CUDA 扩展。**

全是纯 Python 包或自带预编译 wheel 的包。也就是说：

- `torch==2.6.0` 是**作者在 Dockerfile 里验证过的版本**，不是代码硬性要求
- FFS 前向走的是 `optimize_build_volume='pytorch1'` 这条纯 PyTorch 路径
- 它用到的 torch 特性（autocast fp16、`scaled_dot_product_attention`）在 2.1 之后全都有

**结论**：FFS 对 torch 的实际兼容区间远宽于官方文档暗示的单一版本。上图中那条虚线就是这么来的。

> 唯一需要留意的：`numpy` 没锁版本，而 FFS 代码里可能有 `np.float_` 之类 numpy 1.x 的写法。装完如果报 `AttributeError: np.float_ was removed`，就 `pip install "numpy<2"`。这也是为什么之前给的 WSL 方案里我写了 `"numpy<2"`。

---

## 3. 决定性因素：4GB 显存已经把一半候选排除了

这是我认为你应该走方案 B 的最强理由，比"已经在装了"重要得多。

需要 PyTorch3D 的那些模型，工作方式都是**渲染上百个姿态假设 → 逐个过网络打分**：

| 模型 | 单次要渲染多少假设 | 官方推荐显存 |
|---|---|---|
| FoundationPose | refine 阶段多假设并行 | **≥ 8 GB**（GigaPose README 明写 "at least 8GB VRAM recommended"） |
| MegaPose | 粗估计要遍历几十个模板视角 | ≥ 8 GB |
| FreeZeV2.1 | 三路分割 + RGB-D | 高 |

你的卡是 **GTX 1650 4GB**。这不是"慢一点"的问题，是会 OOM 或者被迫把 batch 降到 1 之后慢到不可用。

**换句话说：死锁 ①（PyTorch3D 上限 2.4.1）在你的机器上不构成实际损失**，因为依赖它的模型你本来就跑不动。

于是覆盖需求简化为：

```
FFS        → 纯 torch，无约束
YOLO11/YOLOE → torch ≥ 1.8，无约束
FoundPose  → 纯 torch + DINOv2，只需替换三个 cu11 依赖
YOLO11-pose + PnP + ICP → 纯 torch + OpenCV，无约束
```

**这四者在 torch 2.6.0 + cu124 上完全一致。** 单环境成立。

---

## 4. 推荐方案 B：torch 2.6.0 + cu124 + Python 3.12

### 4.1 覆盖范围

| 层 | 模型 | 能否装 | 备注 |
|---|---|---|---|
| 深度 | Fast-FoundationStereo | 可用 | 官方档 |
| 深度（备选） | IGEV-Stereo / CREStereo / Fast-ACVNet | 可用 | 纯 torch，版本宽松 |
| 深度（兜底） | OpenCV `cv2.StereoSGBM` | 可用 | 完全不碰 torch |
| 检测 | YOLO11-seg / YOLO11-pose | 可用 | ultralytics |
| 检测（免训） | YOLOE | 可用 | 改提示词，带分割掩膜 |
| 快速定位 | YOLO11-pose 关键点 + solvePnP + 深度 ICP | 可用 | 标签由 CAD 自动投影 |
| 快速定位（强） | ZebraPose / GeoTransformer | 可用 | 纯 torch |
| 精定位 | **FoundPose** | 改造后可用 | 见 4.3 |
| 精定位（快） | OPFormer / PicoPose | 可用 | 纯 torch，0.49 / 0.66 s |
| 精定位（重） | FoundationPose / MegaPose / GigaPose-refine | **不可用** | 需 PyTorch3D，且显存不够 |

### 4.2 完整安装命令

```bash
# ---------- 0. 建议装 miniconda（后面 faiss / 渲染器 conda 更好装） ----------
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# ---------- 1. 建环境 ----------
conda create -n dvp python=3.12 -y
conda activate dvp

# ---------- 2. torch（官方源，别用镜像，cu124 版本常不全） ----------
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# ---------- 3. 深度层 FFS ----------
pip install "numpy<2" timm einops omegaconf scipy scikit-image \
            opencv-contrib-python imageio pyyaml open3d

# ---------- 4. 检测层 ----------
pip install ultralytics

# ---------- 5. 精定位层 FoundPose（改造版依赖，见 4.3） ----------
pip install omegaconf torchmetrics fvcore iopath torchinfo kornia \
            pycocotools scikit-learn scikit-image imageio \
            pyrender pyopengl pyglet triangle pypng vispy glumpy cython
pip install faiss-cpu            # 替代 faiss-gpu=1.8.0

# ---------- 6. WSL2 无头渲染（FoundPose 渲染模板需要） ----------
sudo apt update
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3
echo 'export PYOPENGL_PLATFORM=osmesa' >> ~/.bashrc
source ~/.bashrc

# ---------- 7. 验证 ----------
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

### 4.3 FoundPose 的三个依赖怎么替换

官方 yaml 里这三个包在 torch 2.6 + cu124 上装不了，逐个替换：

| 官方包 | 问题 | 替换 | 理由 |
|---|---|---|---|
| `cuml-cu11==23.04` | cu11 专用 wheel，py3.12 也没有 | **先不装，跑通再说** | RAPIDS 只用于 GPU 加速的聚类/检索。FoundPose 的模板检索规模在几千级，CPU 完全够。若确实需要，试 `pip install cuml-cu12`（新版支持 py3.12），但装上不去就跳过 |
| `faiss-gpu=1.8.0` | conda 的 cu11 构建 | `pip install faiss-cpu` | 同上，检索规模小，CPU faiss 毫秒级 |
| `xformers==0.0.20` | 编译绑定 torch 2.3，ABI 不兼容 2.6 | **直接删掉** | torch 2.x 自带 `F.scaled_dot_product_attention`，DINOv2 会自动走它。xformers 只是可选加速 |

另外两个建议改的：

- `opencv-python==4.5.5.62` 太老（2021 年），和 py3.12 可能编不过 → 装最新的 `opencv-contrib-python`
- `pytorch=2.3.0` → 就用环境里的 2.6.0

**装完先跑官方 demo 验证**，不要先用自定义 CAD：

```bash
cd ~/Depth_Learning && git clone https://github.com/facebookresearch/foundpose
cd foundpose
# 按官方 README 下载 LM-O 的现成产物（模板、特征、检测结果）
# 跑 run_inference.py 出位姿，确认能跑通再导入自己的 CAD
```

### 4.4 失去什么，怎么补偿

方案 B 唯一装不了的是 **MegaPose / FoundationPose 的 refine 阶段**。之前的分析里说过一个关键点：FoundPose 单独用是粗位姿 AR 37.3，配 MegaPose refine 才 55.0。

**但你有深度图，所以不需要 MegaPose：**

```
粗位姿（FoundPose）
   → 用掩膜裁出物体点云
   → 深度 ICP 精修（Open3D，几十毫秒，纯 CPU/GPU 都行）
```

理由：

1. ICP 收敛的前提是初始位姿已经比较接近。FoundPose 的粗位姿（平移误差通常在物体直径 10% 以内）一般够 ICP 收敛。
2. FreeZeV2.1（BOP 2024 冠军，AR 82.1）官方方案**本身就包含深度 ICP**，所以这不是"降级的替代"，是主流做法。
3. 深度 ICP 是 Open3D / 你工程里 `vision_utils.py` 现成的能力，不引入任何新依赖、不碰 torch。

**验证方法**：粗位姿 → ICP → 再跑一次 ICP，看第二次的收敛残差。如果第二次几乎不动（< 1 mm），说明第一次已经收敛到位了。

---

## 5. 备选方案 A：torch 2.4.1 + cu121 + Python 3.10

### 5.1 什么时候才需要它

满足**任意一条**才考虑：

- 换了 ≥ 8 GB 显存的卡，想上 FoundationPose / MegaPose
- 明确需要 GigaPose + MegaPose 那套组合（0.38 s 粗估，榜单最快）
- FFS 在 2.6.0 上跑不通（概率低，但万一）

**不满足就别用**——它唯一的好处是能装 PyTorch3D，代价是 FFS 从"官方验证档"变成"未验证档"。

### 5.2 完整安装命令

```bash
conda create -n dvp241 python=3.10 -y
conda activate dvp241

# torch 2.4.1 + cu121
conda install pytorch=2.4.1 torchvision=0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# PyTorch3D（cu121 有预编译，这条是方案 A 的全部意义）
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install pytorch3d -c pytorch3d -y

# 深度层
pip install "numpy<2" timm einops omegaconf scipy scikit-image \
            opencv-contrib-python imageio pyyaml open3d

# 检测层
pip install ultralytics

# 验证 PyTorch3D
python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)"
python -c "from pytorch3d.renderer import MeshRenderer; print('renderer OK')"
```

### 5.3 风险

| 风险 | 概率 | 后果 | 应对 |
|---|---|---|---|
| FFS 在 2.4.1 上报错 | 低（约 10–20%） | 推理失败 | 报错信息发我，通常是某个 2.5+ 的 API；真不行就把 FFS 单独放一个 venv |
| FoundPose 在 py3.10 上装 cuml 失败 | 中 | 装不上 | 同样跳过 cuml，用 faiss-cpu |
| PyTorch3D conda 装不上（依赖冲突） | 中 | 退回方案 B | `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"` 源码编译，Linux 上通常能过 |

---

## 6. 方案 C：不追求单环境，用文件解耦

这个方案严格说不满足你的要求，但我认为**值得认真考虑**，因为它比强凑单环境稳健得多。

核心观察：**深度生成和位姿估计本来就不需要在同一进程里跑。**

- 深度：每帧都要，输出是一张 `H×W` 的 float32 数组
- 位姿：触发式，输入是 RGB + 深度 + 相机内参

两者的接口就是一个文件：

```python
# 深度侧（环境 1：torch 2.6.0）
np.save('/tmp/depth_m.npy', depth_meter.astype(np.float32))

# 位姿侧（环境 2：任意版本）
depth = np.load('/tmp/depth_m.npy')   # 米制 float32，无效值 0
```

好处：

- 两边 torch 版本彻底独立，互不牵制
- 深度可以缓存、可以离线重算，调试方便
- 将来换深度算法（FFS → SGBM → 别的）只换一个脚本

坏处：多一次磁盘 IO（对 `640×480` float32 = 1.2 MB，可忽略；或用 `/dev/shm` 内存盘做到零成本）。

**如果你最后发现单环境怎么装都有冲突，就退回这个方案**——功能上完全等价。

---

## 7. 三方案对比表

| 维度 | **B（推荐）** 2.6.0 + cu124 | A 2.4.1 + cu121 | C 文件解耦 |
|---|---|---|---|
| 环境数量 | 1 | 1 | 2 |
| FFS | 官方验证档 | 未验证（风险低） | 官方验证档 |
| YOLO11 / YOLOE | 可用 | 可用 | 可用 |
| FoundPose | 需替换 3 个依赖 | 更接近官方档 | 独立环境，官方档 |
| 快速定位（PnP + ICP） | 可用 | 可用 | 可用 |
| PyTorch3D 系 | 不可用 | **可用** | 可在第二环境装 |
| 4GB 显存下的实际覆盖 | **完整** | 完整（用不上的部分也能装） | 完整 |
| 换更大显卡后 | 需重建环境 | 直接可用 | 直接可用 |
| 迁移成本 | 无（已在装） | 重搭约 30 分钟 | 无 |

---

## 8. 验证清单：装完怎么确认每层都活

按顺序跑，任何一步不过就停在那排查。

```bash
conda activate dvp

# ① torch + CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.6.0 True NVIDIA GeForce GTX 1650

# ② 检测层
python -c "
from ultralytics import YOLO
m = YOLO('yolo11n-seg.pt')
print('yolo ok', m.device)
"
# 期望：下载权重并打印设备，不报错

# ③ 深度层 FFS
cd ~/Depth_Learning/FFS/Fast-FoundationStereo
python scripts/run_demo.py \
  --model_dir weights/c-fast-foundationstereo/model_best_bp2_serialize.pth \
  --out_dir ~/Depth_Learning/FFS/output --get_pc 0
# 期望：output 目录下出现视差/深度文件

# ④ 渲染（FoundPose 前置）
python -c "
import pyrender, numpy as np
print('pyrender ok', pyrender.__version__)
"
# 期望：不报 OpenGL 错误。
# 若报 Could not initialize OpenGL → 检查 PYOPENGL_PLATFORM=osmesa 是否生效

# ⑤ 深度 ICP（Open3D，精定位的补偿手段）
python -c "
import open3d as o3d, numpy as np
src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(np.random.rand(500,3))
tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(np.random.rand(500,3))
r = o3d.pipelines.registration.registration_icp(src, tgt, 0.02, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print('icp ok, fitness', r.fitness)
"
```

**第 ① 步的 `True` 是第一天的唯一里程碑。** 后面几步可以慢慢来。

---

## 9. 坑位表

| # | 现象 | 根因 | 改法 |
|---|---|---|---|
| 1 | `AttributeError: np.float_ was removed` | numpy 2.x 移除了 `np.float_` | `pip install "numpy<2"` |
| 2 | `torch.cuda.is_available()` 返回 False 但 `nvidia-smi` 正常 | **在 WSL2 里装了 NVIDIA 驱动** | WSL2 里永远不装驱动，驱动只在 Windows 侧。检查 `dpkg -l \| grep nvidia`，有就删 |
| 3 | FoundPose 报 `No module named cuml` | 按官方 yaml 装但装不上 | 跳过 cuml，用 `faiss-cpu` + `sklearn`。若代码确实调用，用 `pip install cuml-cu12` 试 |
| 4 | `import xformers` 失败或 ABI 报错 | xformers 0.0.20 绑定 torch 2.3 | 卸载 xformers。torch 2.6 自带 SDPA，DINOv2 会自动用 |
| 5 | pyrender 报 `Could not initialize OpenGL` | WSL2 无显示服务 | `sudo apt install libosmesa6-dev` + `export PYOPENGL_PLATFORM=osmesa` |
| 6 | FFS 输出目录报 `FileNotFoundError` | 官方脚本默认 `--out_dir /home/bowen/debug/...`，你机器上没这个路径 | 必须显式传 `--out_dir ~/Depth_Learning/FFS/output` |
| 7 | `K.txt` 深度差 1000 倍但图看起来对 | 基线第二行写成毫米 | 基线单位是**米**。D435 是 0.05 不是 50 |
| 8 | 想在 `dvp` 里装 PyTorch3D 失败 | 2.6.0 超官方上限 2.4.1 | 无解，要么降到 2.4.1（方案 A），要么源码编译（大概率失败） |
| 9 | 走 cu130（你现在 `.venv` 里的 torch 2.14） | CUDA 13 生态支持最差 | **不要用**。虽然 CUDA 13 支持 Turing（sm_75），但 faiss / cuml / PyTorch3D 都没跟上，迟早撞墙 |
| 10 | conda 和 venv 混用导致包找不到 | `conda init` 没执行或 shell 没重载 | `source ~/.bashrc` 后再 `conda activate` |
| 11 | ultralytics 训练时爆显存 | 4GB 卡默认 batch 太大 | `imgsz=640, batch=4, amp=True`；或 `batch=-1` 让它自动探测 |
| 12 | 检测层的合成数据训出来在真机上 mAP 崩 | 域随机化没做 | HDRI 背景 ≥50 张、光照/材质/遮挡/噪声/运动模糊全随机化 |

---

## 10. 我的建议

**走方案 B，现在就把 `dvp` 环境建起来。**

三条理由，按重要性排：

1. **你的 4GB 显存决定了上限。** PyTorch3D 那一派（FoundationPose / MegaPose）在 4GB 上跑不动，所以"装不了它们"不是损失。等你换了 8GB+ 的卡再考虑方案 A，那时候重建一个 conda 环境只要 30 分钟。
2. **FFS 的 2.6.0 是唯一官方验证档。** 用未验证版本出了问题，你分不清是模型本身的毛病还是版本的毛病，排查成本远高于"先用官方档跑通，确认基线正确"。
3. **你已经在装了。** `setup_ffs.sh` 装的就是 torch 2.6.0 + cu124，直接往这个环境里补检测层和精定位层的包即可，零返工。

补充两条操作建议：

- **用 conda 不要继续用 venv。** 单环境要同时装 faiss、渲染器、open3d 这些，conda 的二进制包管理能省掉大量编译时间。你已经有的 `.venv`（torch 2.14 + cu130）保持不动即可。
- **FoundPose 先跑官方 demo 再上自己的 CAD。** 它换依赖后是非官方配置，先用 LM-O 数据集确认管线通了，再排查 CAD 相关的问题。两类问题混在一起会非常难查。

**真正需要升级的其实是内存，不是环境。** 你 8 GB 内存要在 WSL2 里同时跑"FFS 推理 + YOLO 训练 + 渲染模板"，会很紧。加到 16 GB（DDR4 SO-DIMM，100–200 元）是这台机器性价比最高的升级。

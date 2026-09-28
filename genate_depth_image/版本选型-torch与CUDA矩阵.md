# torch / CUDA 版本选型：FFS + 检测 + 定位多模型共存方案

> **这份文档回答一个问题**：考虑到 Fast-FoundationStereo 和后续的检测 / 定位模型，WSL2 里 torch 和 CUDA 到底装哪个版本。
>
> **更新**：如果你只想要**一个环境**（深度 + 检测 + 定位共用一套 torch），直接看
> **`单环境统一方案-深度检测定位共用一套torch.md`** —— 结论是 **torch 2.6.0 + cu124 + Python 3.12 一个环境即可**，
> 因为 FFS 的 `requirements.txt` 不含 torch/flash-attn/triton（不锁版本），而唯一装不进去的 PyTorch3D 系模型
> 在 4GB 显存上本来就跑不动。本份文档保留四个环境的完整推导作为背景资料。
>
> 配套文档：
> - `单环境统一方案-深度检测定位共用一套torch.md` — **收敛结论，先看这份**
> - `WSL2部署FFS-完整落地方案.md` — 怎么装（执行手册），本文档是它的版本决策依据
> - `部署环境评估-WSL2与Windows对比.md` — 为什么用 WSL2
> - `最终选型总结-推荐顺序与搭建方案.md` — 模型选哪个
>
> 本机：i5-9300H / 8 GB 内存 / GTX 1650（Turing, sm_75）/ 驱动 32.0.15.9227 / C 盘剩余 162 GB。

---

## 目录

- [0. 一句话结论](#0-一句话结论)
- [1. 为什么一个环境装不下](#1-为什么一个环境装不下)
- [2. 推荐方案：分环境](#2-推荐方案分环境)
- [3. 完整安装命令](#3-完整安装命令)
- [4. 驱动、CUDA Toolkit、CUDA 运行时：三个不同的东西](#4-驱动cuda-toolkitcuda-运行时三个不同的东西)
- [5. 建立顺序与决策点](#5-建立顺序与决策点)
- [6. 坑与对策](#6-坑与对策)
- [7. 版本速查卡](#7-版本速查卡)
- [附：核实来源](#附核实来源)

---

## 0. 一句话结论

**分环境装，不要试图用一个环境装下所有模型。**

先用这一个（唯一必装的）：

```bash
conda create -n ffs python=3.12 -y
conda activate ffs
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

它覆盖 FFS + YOLO11 + 大部分快速定位模型。等到真要用 FoundPose / FoundationPose 时再开第二个环境。

**CUDA 不用"装"**——torch 的 cu124 wheel 自带 CUDA 运行时。你唯一需要在 WSL2 之外操心的是 **Windows 端的 NVIDIA 驱动**，那个你已经有了（32.0.15.9227，支持 CUDA 11.x 和 12.x 全部版本）。

---

## 1. 为什么一个环境装不下

### 1.1 各模型的版本约束（全部来自官方文件，非推测）

| 组件 | Python | torch | torchvision | CUDA | 硬约束来源 |
|---|---|---|---|---|---|
| **Fast-FoundationStereo** | **3.12** | **2.6.0** | **0.21.0** | **12.4** | 官方 `Dockerfile` / `environment.yml` |
| **Ultralytics**（YOLO11 / YOLOE） | 3.8–3.13 | **>= 1.8.0** | >= 0.9.0 | 任意 | `pyproject.toml`，几乎无限制 |
| **FoundPose** | **3.9** | **2.3.0** | **0.18.0** | **11.7** | `conda_foundpose_gpu.yaml` |
| **PyTorch3D**（GigaPose / MegaPose / FreeZe 依赖） | 3.9 / 3.10 | **2.1.0 – 2.4.1** | 匹配 | 匹配 | 官方 `INSTALL.md` 明确列出支持范围 |
| **FoundationPose** | **3.8** | **2.0.0** | **0.15.1** | **11.8** | `docker/dockerfile`（含 kaolin + nvdiffrast） |

### 1.2 冲突在哪

三处死锁，都不是靠 pip 能调和的：

**① FFS 的 torch 2.6.0 超出了 PyTorch3D 的支持上限（2.4.1）。**

PyTorch3D 的 `INSTALL.md` 逐版本列出了支持的 torch：`2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0 or 2.4.1`。**2.6.0 不在列表里。** 在 torch 2.6 环境里源码编译 PyTorch3D 大概率失败或静默出问题。

**② FoundPose 绑死了 CUDA 11 生态。**

它的 yaml 里有 `faiss-gpu=1.8.0`、`cuml-cu11==23.04`、`pytorch::pytorch-cuda=11.7.0`。`cuml-cu11` 这个名字就写着 cu11。换成 CUDA 12 环境，这几个包要重新找对应版本，属于自己给自己找活。

**③ FoundationPose 的 Python 3.8 + torch 2.0 是独立的一代。**

它还要编译 kaolin（`FORCE_CUDA=1 python setup.py develop`，编译很重）和 nvdiffrast，外加 Eigen 3.4 和 pybind11 v2.10.0。这套东西和前面的任何组合都不共享。

> 有意思的一点：FoundationPose 仓库里的 `requirements.txt` 顶部注释**已经更新**为推荐 `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`，但它的 `docker/dockerfile` 仍然锁在 `torch==2.0.0+cu118`。**README 更新了、镜像没更新**，这种不一致在 NVIDIA 仓库里很常见。以 Dockerfile 为准——那是真正跑通过的组合。

### 1.3 一个"最大公约数"方案，以及为什么不推荐

你可能会想：装 torch **2.4.1 + cu124**，是不是就能同时满足 FFS（需要 2.6）和 PyTorch3D（最高 2.4.1）？

理论上 FFS 在 2.4.1 上大概率能跑（它用的是 `torch.amp.autocast`、`torch.load(weights_only=False)` 这类 2.0 时代就有的 API）。但：

- FFS 官方 Dockerfile 锁死 2.6.0，官方没验证过 2.4.1
- 一旦出问题，你没法区分是"模型的问题"还是"torch 版本的问题"，排查成本远高于多建一个 conda 环境（2 分钟）

**结论：FFS 严格按官方来，其他模型另开环境。** 调优的收益不值得承担这个不确定性。

---

## 2. 推荐方案：分环境

### 2.1 总表

| 优先级 | 环境名 | Python | torch | torchvision | CUDA 运行时 | 承载 | 何时建 |
|---|---|---|---|---|---|---|---|
| **必建** | `ffs` | 3.12 | **2.6.0** | **0.21.0** | **12.4** | FFS、YOLO11 / YOLOE、BlenderProc、Open3D、GeoTransformer、ZebraPose、ICP | **现在** |
| 次 | `foundpose` | 3.9 | **2.3.0** | **0.18.0** | **11.7** | FoundPose、CNOS、BOP toolkit | 第 2–3 周 |
| 可选 | `pose3d` | 3.10 | **2.4.1** | **0.19.0** | 12.1 | GigaPose、MegaPose、FreeZe（需要 PyTorch3D） | 确认 FoundPose 不够时 |
| 可选 | `foundationpose` | 3.8 | **2.0.0** | **0.15.1** | 11.8 | FoundationPose（+ kaolin + nvdiffrast） | 建议走 Docker，别手动装 |

磁盘开销估算：每个环境 5–9 GB（torch wheel 本身就 2.5 GB）。四个全建约 30 GB，C 盘 162 GB 够用。

### 2.2 env A：`ffs`（主力，唯一必建）

这是你现在要建的。它覆盖的范围比你想象的大：

- **FFS**：官方指定组合
- **YOLO11 全系列**（detect / seg / pose）：`ultralytics` 只要求 torch >= 1.8.0
- **YOLOE**（开放词汇，换提示词免训）：同上
- **BlenderProc**（合成数据）：跨平台，对 torch 无强依赖
- **Open3D**（点云 / ICP / TSDF）：pip wheel，不依赖 torch
- **快速定位层的纯 torch 模型**：ZebraPose、GeoTransformer、RCVPose、SurfEmb 等基本都能在 2.6 上跑

也就是说，**检测层 + 快速定位层 + 深度层可以全在 env A 里**，只有精定位那几个需要另开。

### 2.3 env B：`foundpose`

严格按官方 yaml，不要自己改版本。理由是它依赖 `faiss-gpu` / `cuml-cu11` 这类和 CUDA 版本强绑定的包。

一个可以尝试的偷懒做法：**先试试在 env A 里装 `faiss-cpu` 跑 FoundPose**。FoundPose 的计算量主要在 DINOv2 特征提取和模板匹配，faiss 检索那一步用 CPU 也不慢。如果跑通了，就省掉一个环境。跑不通再退回官方 yaml。

### 2.4 env C：`pose3d`

PyTorch3D 的 conda 包 `conda install pytorch3d -c pytorch3d` 是 **Linux only**——WSL2 算 Linux，所以能用。但要注意：`-c pytorch3d` 的预编译包未必有 torch 2.4.1 + cu121 的组合，装之前先看一眼有没有匹配构建。

备选是源码编译，需要 `nvcc`：

```bash
conda install -c nvidia cuda-toolkit=11.8   # 装到 env 里，不是系统级
export CUDA_HOME=$CONDA_PREFIX
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

`--no-build-isolation` 是必须的，否则 pip 会新建一个隔离环境拉一份不匹配的 torch 来编译。

### 2.5 env D：`foundationpose`

**建议走 Docker，不要手动装。** 它需要：Eigen 3.4.0（编译安装）、pybind11 v2.10.0（编译安装）、kaolin（`FORCE_CUDA=1` 源码编译，很慢）、nvdiffrast（源码编译）、以及一堆 apt 包。手动装一整天不一定搞定。

官方镜像：

```bash
docker pull wenbowen123/foundationpose
```

在 WSL2 里装 `docker-ce` + `nvidia-container-toolkit`（**不要装 Docker Desktop**，它常驻吃 1–2 GB 内存，你这台 8 GB 扛不住）。

---

## 3. 完整安装命令

### 3.1 env A：ffs（现在就执行）

```bash
# 建环境
conda create -n ffs python=3.12 -y
conda activate ffs

# torch —— 这一步约 2.5 GB，最慢，用官方源
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 验证（第一天唯一的里程碑）
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.6.0+cu124 True NVIDIA GeForce GTX 1650

# FFS 依赖
pip install timm einops omegaconf scipy scikit-image opencv-contrib-python imageio pyyaml
pip install "numpy<2"        # open3d / 老包对 numpy 2 兼容性不稳，锁 1.x

# 检测层
pip install ultralytics

# 点云 / 几何（这两个不依赖 torch，独立）
pip install open3d trimesh

# 可选：FoundPose 的轻量替代尝试
pip install faiss-cpu
```

> 关于 `numpy<2`：torch 2.6 兼容 numpy 2.x，但 `open3d` 和一些老视觉包在 numpy 2 上会炸。锁 1.x 最省事。如果你后面发现某个包明确需要 numpy>=2（比如 FoundationPose 的 requirements 就写了 `numpy>=2`），那是另一个环境的事，不冲突。

### 3.2 env B：foundpose（第 2–3 周）

```bash
git clone --recurse-submodules https://github.com/facebookresearch/foundpose
cd foundpose
conda env create -f conda_foundpose_gpu.yaml
conda activate foundpose_gpu
```

**必须配环境变量**，否则脚本找不到 BOP toolkit 和 DINOv2：

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh <<'EOF'
#!/bin/sh
export REPO_PATH=/path/to/foundpose
export BOP_PATH=/path/to/bop/datasets
export PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2
EOF
```

把 `/path/to/...` 换成实际路径。这个脚本在 `conda activate foundpose_gpu` 时自动执行。

### 3.3 env C：pose3d（按需）

```bash
conda create -n pose3d python=3.10 -y
conda activate pose3d
conda install pytorch=2.4.1 torchvision=0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c iopath iopath
conda install pytorch3d -c pytorch3d

# 验证
python -c "import pytorch3d; print(pytorch3d.__version__)"
```

如果最后一步报找不到匹配包，改源码编译：

```bash
conda install -c nvidia cuda-toolkit=11.8
export CUDA_HOME=$CONDA_PREFIX
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### 3.4 环境切换备忘

```bash
conda env list              # 看有哪些环境
conda activate ffs          # 切到 FFS
conda deactivate
conda env remove -n xxx     # 删环境（磁盘不够时）
```

**每个终端都要 `conda activate`**。WSL2 不会自动激活，新开一个终端忘了激活就会用到 base 环境（没有 torch），然后报 `ModuleNotFoundError: No module named 'torch'`。

---

## 4. 驱动、CUDA Toolkit、CUDA 运行时：三个不同的东西

这是最容易搞混的地方，值得单独讲。

| 东西 | 装在哪 | 你要不要装 | 说明 |
|---|---|---|---|
| **NVIDIA 驱动** | **Windows 端** | **已有**（32.0.15.9227） | WSL2 通过 `/usr/lib/wsl/lib/libcuda.so` 透传。**绝不能在 WSL2 里再装驱动** |
| **CUDA 运行时**（libcudart / cublas / cudnn） | 随 torch wheel 走 | 自动 | `pip install torch==2.6.0+cu124` 会拉一堆 `nvidia-*` pip 包，这就是运行时 |
| **CUDA Toolkit**（nvcc / 头文件） | 系统或 env 内 | **默认不装** | 只有编译 CUDA 扩展（PyTorch3D / nvdiffrast / kaolin）才需要 |

**关键结论：你的驱动一个版本就够，同时支持 CUDA 11.7、11.8、12.1、12.4。**

`nvidia-smi` 右上角显示的 `CUDA Version: xx.x` 是"这个驱动能支持的最高 CUDA 运行时版本"，不是"你装了 CUDA xx.x"。你的驱动是 32.0.15.9227（570+ 时代），支持上限远超 12.4，所以上面四个 CUDA 版本随便选。

**所以"装 CUDA 12.4"这句话的实际含义是**：装一个自带 12.4 运行时的 torch wheel。不需要 `apt install cuda-toolkit`。

什么时候才真要装 Toolkit：

```bash
# 仅在需要编译扩展时（env C / D）
# 注意仓库名必须是 wsl-ubuntu，用 ubuntu2204 会把驱动一起装进来
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-8    # 只装 toolkit，不装 *-drivers*
```

---

## 5. 建立顺序与决策点

**不要一次把四个环境都建了。** 8 GB 内存下，每多一个环境就多一份维护负担，而且你可能根本用不到后面几个。

```
第 1 天   建 env A (ffs) ──→ torch.cuda.is_available() == True  ← 唯一必达里程碑
           │
第 2–4 天  FFS 跑通 + 深度基准测试
           │
           ├── 深度精度够用 ──→ 继续
           └── 不够 ──→ 先解决深度，别急着上定位模型
           │
第 2 周    YOLO11-seg + BlenderProc 合成数据（env A 内，不新建环境）
           │
第 3 周    ┌─ 试 FoundPose 在 env A + faiss-cpu 能不能跑
           │     ├── 能 ──→ 完事，永远不用建 env B
           │     └── 不能 ──→ 建 env B（官方 yaml）
           │
第 4 周    临时贴 AprilTag 取真值，量化误差
           │
           └── 精度不够 ──→ 才考虑 env C / env D
```

**每个决策点都把"建新环境"往后推。** 建环境本身只有几分钟，但每个环境都是一份要维护的东西——尤其是 env D，它那一堆编译产物换台机器就得重来。

---

## 6. 坑与对策

| # | 现象 | 根因 | 改法 |
|---|---|---|---|
| 1 | `torch.cuda.is_available()` 返回 False | 在 WSL2 里装了 NVIDIA 驱动，顶掉了主机映射 | 卸载 WSL2 内驱动；只保留 Windows 端 |
| 2 | `pip install torch` 找不到 cu124 | 用了国内镜像源，cu124 版本不全 | 用官方源 `download.pytorch.org/whl/cu124` |
| 3 | 装了 `cuda-toolkit` 后反而跑不了 | 用了 `ubuntu2204` 仓库而非 `wsl-ubuntu`，把驱动装进来了 | 只装 `cuda-toolkit-12-4`，不装任何 `*-drivers*` |
| 4 | 新开终端报 `No module named 'torch'` | 忘了 `conda activate` | 每个终端都要激活；或 `echo "conda activate ffs" >> ~/.bashrc` |
| 5 | 在 WSL 里敲 `python` 调到了 Windows 的 Python | `/etc/wsl.conf` 里 `appendWindowsPath=true` | 设为 `false`，然后 `wsl --shutdown` |
| 6 | `conda install pytorch3d` 找不到包 | 预编译包没有匹配你 torch/cuda 的构建；且 conda 包 **Linux only** | 源码编译 + `--no-build-isolation`，先 `conda install -c nvidia cuda-toolkit` 拿到 nvcc |
| 7 | PyTorch3D 编译报找不到 torch 头文件 | 用了 build isolation，pip 拉了另一份 torch | 加 `--no-build-isolation` |
| 8 | torch 2.6 环境里装 PyTorch3D 失败 | 官方只支持到 torch 2.4.1 | 另建 env C（torch 2.4.1），不要在 env A 里硬来 |
| 9 | `import open3d` 后 numpy 报错 | 装了 numpy 2.x | `pip install "numpy<2"` |
| 10 | FoundPose 报找不到 bop_toolkit | `PYTHONPATH` 没配，或 git clone 时没加 `--recurse-submodules` | 重新 clone 带 `--recurse-submodules`，配 `env_vars.sh` |
| 11 | 四个环境建完磁盘不够 | torch wheel 每个 2.5 GB | 优先删 env D；`conda clean -a` 清缓存 |
| 12 | 8 GB 内存下渲染模板时 OOM | WSL2 默认吃一半内存且不还 | `.wslconfig` 设 `memory=5GB` + `swap=8GB` + `autoMemoryReclaim=gradual` |
| 13 | 想用最新 torch（2.9+ / 2.12） | 追求新版本 | **不要**。FFS 锁 2.6.0，新版未验证；老卡上也拿不到新特性 |
| 14 | FoundationPose 在 torch 2.6 上编译失败 | Dockerfile 锁的是 2.0.0+cu118 | 走官方 Docker 镜像，别手动装 |

---

## 7. 版本速查卡

```
=========================================================
  WSL2 版本速查卡（本机：GTX 1650 / 驱动 32.0.15.9227）
=========================================================

  Windows 端驱动        32.0.15.9227（已有，不要动，不要在 WSL2 里再装）
  WSL2 发行版           Ubuntu 22.04
  系统级 CUDA Toolkit    默认不装（需要编译扩展时才装 11.8）

  -------------------------------------------------------
  env A  ffs          ← 现在就建，唯一必建
    python        3.12
    torch         2.6.0
    torchvision   0.21.0
    CUDA 运行时    12.4（wheel 自带）
    命令：
      conda create -n ffs python=3.12 -y
      conda activate ffs
      pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124
    numpy 锁 <2
  -------------------------------------------------------
  env B  foundpose    ← 第 3 周，先试 env A + faiss-cpu
    python 3.9 / torch 2.3.0 / torchvision 0.18.0 / cuda 11.7
    用官方 yaml：conda env create -f conda_foundpose_gpu.yaml
  -------------------------------------------------------
  env C  pose3d       ← 可选（GigaPose / MegaPose / FreeZe）
    python 3.10 / torch 2.4.1 / torchvision 0.19.0 / cuda 12.1
    PyTorch3D 官方支持 torch 上限 = 2.4.1
  -------------------------------------------------------
  env D  foundationpose ← 可选，建议 Docker
    python 3.8 / torch 2.0.0 / torchvision 0.15.1 / cuda 11.8
    docker pull wenbowen123/foundationpose
  -------------------------------------------------------

  第一天的里程碑只有一个：
    python -c "import torch; print(torch.cuda.is_available())"
    → 必须打印 True
=========================================================
```

---

## 附：核实来源

本表的版本号全部取自官方仓库文件，不是凭记忆写的：

| 组件 | 来源文件 |
|---|---|
| Fast-FoundationStereo | `NVlabs/Fast-FoundationStereo` 的 `Dockerfile`、`environment.yml`、`requirements.txt` |
| Ultralytics | `ultralytics/ultralytics` 的 `pyproject.toml`（`torch>=1.8.0`，`requires-python>=3.8`） |
| FoundPose | `facebookresearch/foundpose` 的 `conda_foundpose_gpu.yaml` |
| PyTorch3D | `facebookresearch/pytorch3d` 的 `INSTALL.md`（明确列出支持的 torch 版本区间） |
| FoundationPose | `NVlabs/FoundationPose` 的 `docker/dockerfile`（锁 `torch==2.0.0+cu118`）与 `requirements.txt`（注释已改为推荐 cu124，但镜像未同步） |

如果后续这些仓库更新了，以仓库文件为准——尤其是 PyTorch3D 的支持上限，它是这一整套方案里最容易变的那个约束。

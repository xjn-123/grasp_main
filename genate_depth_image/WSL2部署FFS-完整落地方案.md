# 在 WSL2 中部署 Fast-FoundationStereo：完整落地方案

> **目标**：在这台 Windows 机器上装 WSL2，跑通 Fast-FoundationStereo（下称 FFS），从一对已校正的左右图生成深度图，并评估效果。
>
> **范围**：只做「给一对立体图 → 出深度图 → 评估」。不涉及 ROS2、不涉及相机实时采集、不涉及 `carm_grasp` 的 `utils.py`。
>
> **你的机器（实测）**：i5-9300H 4 核 8 线程 / **8.0 GB 内存** / **GTX 1650 4 GB**（Turing，驱动 32.0.15.9227）/ C 盘剩余 162 GB / 当前未装任何 WSL 发行版。
>
> 配套文档：`Fast-FoundationStereo_落地方案.md`（推理细节、参数、K.txt、坑位）、`部署环境评估-WSL2与Windows对比.md`（选型依据）、**`版本选型-torch与CUDA矩阵.md`（为什么是 torch 2.6.0 + cu124，以及后续检测/定位模型要另开哪些 conda 环境）**。本份是**执行手册**，其余是**参考手册**。

---

## 目录

- [0. 结论先行：三个可以省掉的东西](#0-结论先行三个可以省掉的东西)
- [1. 前置检查（Windows 侧，10 分钟）](#1-前置检查windows-侧10-分钟)
- [2. 安装 WSL2 与 Ubuntu 22.04](#2-安装-wsl2-与-ubuntu-2204)
- [3. 关键配置：.wslconfig 与 wsl.conf](#3-关键配置wslconfig-与-wslconf)
- [4. GPU 透传：驱动在 Windows，toolkit 在 Linux](#4-gpu-透传驱动在-windowstoolkit-在-linux)
- [5. Python 环境：miniconda + torch cu124](#5-python-环境miniconda--torch-cu124)
- [6. 获取代码与权重](#6-获取代码与权重)
- [7. 跑通官方 demo（验证环境）](#7-跑通官方-demo验证环境)
- [8. 实际使用：batch_infer.py](#8-实际使用batch_inferpy)
- [9. 数据准备与跨系统搬运](#9-数据准备与跨系统搬运)
- [10. 你这台机器的性能预期与调参](#10-你这台机器的性能预期与调参)
- [11. 效果评估：怎么知道好不好](#11-效果评估怎么知道好不好)
- [12. 坑位清单](#12-坑位清单)
- [13. 时间线与检查清单](#13-时间线与检查清单)
- [14. 这个环境后续还能干什么](#14-这个环境后续还能干什么)
- [附 A：一键安装脚本](#附-a一键安装脚本)
- [附 B：常用命令速查](#附-b常用命令速查)

---

## 0. 结论先行：三个可以省掉的东西

在动手前先明确，**这三样东西很多人装了，但在你这个目标下完全不需要**。省掉它们，整个部署从「半天」变成「一小时」。

| 通常被认为必需 | 能否省 | 原因 |
|---|---|---|
| **系统级 CUDA Toolkit**（`cuda-toolkit-12-4`，约 4–5 GB） | **能省** | FFS 不编译任何自定义 CUDA 扩展。PyTorch 的 cu124 wheel 自带 CUDA 运行时（libcudart / cublas / cudnn 都打包在 wheel 里）。只有需要 `nvcc` 编译算子、或要用 TensorRT 时才装 |
| **Docker Desktop** | **能省** | 8 GB 内存下 Docker Desktop 常驻 1–2 GB，得不偿失。直接用 conda 环境 |
| **open3d 的可视化** | **能省** | 官方 `run_demo.py` 强制弹点云窗口，但我们要的是深度数组。用第 8 节的脚本绕开 |

**保留了什么**：Windows 端的 NVIDIA 驱动（必需，WSL2 的 GPU 靠它）、WSL2 本身、miniconda、torch 2.6.0+cu124。

### 0.1 版本硬约束

这四个版本号是官方 Dockerfile 里钉死的，**不要改**：

```
Python 3.12
torch 2.6.0
torchvision 0.21.0
CUDA 12.4（由 torch wheel 提供，不单独装）
```

---

## 1. 前置检查（Windows 侧，10 分钟）

### 1.1 确认驱动支持的 CUDA 版本上限

在 **Windows 的 PowerShell** 里执行：

```powershell
nvidia-smi
```

看输出**右上角**那行 `CUDA Version: xx.x`。这个数字是「当前驱动能支持的最高 CUDA 运行时版本」。

- 如果显示 **>= 12.4**：一切正常，继续。
- 如果显示 **< 12.4**：先升级 Windows 端 NVIDIA 驱动到 550 以上（建议直接用 GeForce Experience 或官网下载最新 Game Ready / Studio 驱动）。

> 你现在的驱动是 32.0.15.9227，属于 580+ 时代的新版本号，支持 CUDA 12.4 没有问题。但仍建议跑一次确认，这一步 10 秒。

同时记录下 `nvidia-smi` 里的 **显存总量**（GTX 1650 通常是 4 GB）。

### 1.2 确认磁盘空间

`cuda-toolkit` 不装的话，WSL2 本身 + miniconda + torch 大约占用 **12–15 GB**。C 盘剩余 162 GB，完全够用。

但要注意：**WSL2 的虚拟磁盘默认会动态增长到 1 TB 上限**（新版默认值），不会一开始就吃掉 162 GB，放心。

### 1.3 确认 Windows 版本

```powershell
winver
```

需要 **Windows 10 21H2 / Windows 11** 以上才能用 `wsl --install` 一键安装，并获得 GPU 透传支持。

---

## 2. 安装 WSL2 与 Ubuntu 22.04

### 2.1 一键安装

以**管理员身份**打开 PowerShell（右键开始菜单 → Windows Terminal(管理员)），执行：

```powershell
wsl --install -d Ubuntu-22.04
```

这条命令会自动完成：启用 WSL 功能 → 启用虚拟机平台 → 下载 Ubuntu 22.04 → 设为 WSL2 → 启动。

**如果提示「无法识别」或报错**（常见于较旧的 Win10 或被组策略限制的环境），手动分步执行：

```powershell
# 1. 启用 WSL
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 2. 启用虚拟机平台
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 3. 重启电脑（这一步不能跳）

# 4. 重启后，把 WSL2 设为默认
wsl --set-default-version 2

# 5. 更新 WSL 内核
wsl --update

# 6. 安装 Ubuntu 22.04
wsl --install -d Ubuntu-22.04
```

### 2.2 首次启动与用户创建

安装完成后会自动弹出 Ubuntu 终端，要求你设置：

- **UNIX 用户名**：建议用小写，例如 `x`
- **密码**：输入时**不显示任何字符**（Linux 传统），输完直接回车

之后这个用户自动拥有 sudo 权限。

验证一下挂载情况：

```bash
ls /mnt/c/Users/x/    # 能看到你的 Windows 用户目录，说明 Windows 盘已挂载
```

### 2.3 换 apt 源（国内必做）

默认的 Ubuntu 源在国内很慢。执行：

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
sudo sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
sudo apt update && sudo apt upgrade -y
```

装基础工具：

```bash
sudo apt install -y build-essential curl wget git vim htop
```

---

## 3. 关键配置：.wslconfig 与 wsl.conf

**这一步对你是必需的，不是可选优化。** 8 GB 内存下，WSL2 默认会动态吃掉最多 50%（4 GB），而且不会主动还给 Windows，很容易把主机拖垮。

### 3.1 在 Windows 侧创建 .wslconfig

在 **Windows 的 PowerShell**（不需要管理员）里执行：

```powershell
notepad $env:USERPROFILE\.wslconfig
```

粘贴以下内容（针对 8 GB 内存、4 核 8 线程优化）：

```ini
[wsl2]
# 内存上限：8GB 物理内存，给 WSL2 5GB，给 Windows 留 3GB
memory=5GB

# CPU：4 核 8 线程，给 4 个
processors=4

# 交换分区：内存不够时兜底，防止 OOM 直接杀进程
swap=8GB

# 允许 Windows 访问 WSL2 里的服务（后面取结果、起小服务器要用）
localhostForwarding=true

[experimental]
# 空闲时缓慢把缓存内存还给 Windows（WSL2 默认只增不减）
autoMemoryReclaim=gradual

# 虚拟磁盘稀疏化，只占实际写入的空间
sparseVhd=true
```

保存后，**必须完全关闭 WSL 才生效**：

```powershell
wsl --shutdown
```

然后重新打开 Ubuntu 终端。

验证：

```bash
free -h        # 应看到约 5G 的总内存
nproc          # 应输出 4
```

### 3.2 在 Linux 侧配置 /etc/wsl.conf

```bash
sudo tee /etc/wsl.conf > /dev/null <<'EOF'
[boot]
systemd=true

[interop]
enabled=true
# 不要把 Windows 的 PATH 追加进来，否则 python / pip 可能指向 Windows 版本
appendWindowsPath=false

[network]
generateResolvConf=true
generateHosts=true
EOF
```

`appendWindowsPath=false` 这一条很重要：**如果 Windows 也装了 Python，`appendWindowsPath=true` 会导致在 WSL 里敲 `python` 调到 Windows 的 Python**，然后各种诡异报错。

改完再 `wsl --shutdown`（Windows 侧执行）一次。

---

## 4. GPU 透传：驱动在 Windows，toolkit 在 Linux

### 4.1 原理（理解这个能省掉 90% 的排查时间）

```
┌─ Windows 主机 ─────────────────────┐
│  NVIDIA 驱动（唯一的一份）          │
│  libcuda.so  ← 真实的驱动库        │
└──────────────┬─────────────────────┘
               │ WSL2 自动 bind-mount
┌─ WSL2 Ubuntu ─┴────────────────────┐
│  /usr/lib/wsl/lib/libcuda.so.1     │  ← 驱动从主机映射过来
│  /usr/lib/wsl/lib/nvidia-smi       │
│  torch cu124 wheel 自带的 CUDA 运行时│  ← 我们只装这个
└────────────────────────────────────┘
```

**最重要的一条禁忌**：**绝对不要在 WSL2 里安装 NVIDIA 驱动**（不要 `apt install nvidia-driver-*`、不要跑任何 `.run` 安装器）。

装了的典型症状是：`nvidia-smi` 能显示 GPU，但 `torch.cuda.is_available()` 返回 `False`，或者一跑就 `CUDA error: no CUDA-capable device detected`。原因是 WSL2 里装的内核模块会顶掉主机的驱动映射，而且这个错误极难排查。

### 4.2 验证透传

在 WSL2 的 Ubuntu 终端里：

```bash
# 1. 驱动映射是否存在
ls -l /usr/lib/wsl/lib/libcuda.so.1
# 预期：看到 libcuda.so.1 -> 指向某个 libcuda.so.1.1

# 2. nvidia-smi 能否看到 GPU
nvidia-smi
# 预期：看到 GTX 1650，右上角有 CUDA Version

# 3. 显存情况
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

**如果第 1 步就 `No such file or directory`**：说明 Windows 端驱动没装好或版本太老，回到 1.1 升级驱动，然后 `wsl --shutdown` 重启 WSL。

### 4.3 要不要装系统级 CUDA Toolkit？

按你的目标——**不装**。理由：

- FFS 的推理路径是 `model.forward(..., optimize_build_volume='pytorch1')`，纯 PyTorch 算子，不编译任何 CUDA 扩展
- torch 的 cu124 wheel 自带 `libcudart` / `libcublas` / `libcudnn` / `libcufft` 等运行时库
- `requirements.txt` 里没有任何需要 nvcc 编译的包

**什么时候才装**（到那时再回来执行）：

```bash
# 只在需要 nvcc（编译 nvdiffrast / PyTorch3D）或 TensorRT 时才执行
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version     # 应输出 12.4
```

注意仓库名是 **`wsl-ubuntu`**（WSL 专用），不是 `ubuntu2204`。用错仓库会尝试装驱动，正是我们要避免的。

---

## 5. Python 环境：miniconda + torch cu124

### 5.1 安装 miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

验证：

```bash
conda --version     # 应输出版本号
which python        # 应指向 ~/miniconda3/bin/python
```

**检查 `which python` 这一步别跳过**。如果它指向 `/usr/bin/python` 或某个 `/mnt/c/...` 路径，说明 conda init 没生效或 PATH 被污染。

（可选）换 conda 国内源，加速后续下载：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.2 创建环境并安装 torch

```bash
conda create -n ffs python=3.12 -y
conda activate ffs

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

> 这一步会下载约 2.5 GB（wheel 内含 CUDA 运行时），是整个过程最慢的一步。用官方源（`download.pytorch.org`）下载，不要换国内镜像——国内镜像的 cu124 版本经常不全。

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期输出：

```
2.6.0 True NVIDIA GeForce GTX 1650
```

**如果 `cuda.is_available()` 是 False**，按这个顺序排查：
1. Windows 端 `nvidia-smi` 是否正常 → 不正常就升级驱动
2. WSL2 里 `ls -l /usr/lib/wsl/lib/libcuda.so.1` 是否存在 → 不存在就 `wsl --shutdown` 重启
3. 确认没在 WSL2 里装过驱动

### 5.3 安装 FFS 依赖

先把代码拉下来（下一步会细说），然后：

```bash
pip install timm einops omegaconf scipy numpy scikit-image opencv-contrib-python imageio pyyaml
```

**关于 `open3d`**：官方 `requirements.txt` 里有它，但我们要用的脚本不需要。两种选择：

```bash
# 方案 A（推荐）：装，因为官方 run_demo.py 会 import 它，第 7 节验证环境时需要
pip install open3d

# 方案 B：如果 open3d 装不上或太慢，跳过。第 8 节的 batch_infer.py 不依赖它
```

> `open3d` 有 manylinux wheel，正常能装上（约 100 MB）。如果网络卡，先跳过，第 7 节改用 `batch_infer.py` 验证。

### 5.4 环境自检清单

```bash
conda activate ffs
python - <<'EOF'
import torch, cv2, numpy, yaml, imageio, einops, timm, omegaconf, scipy, skimage
print("torch      ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("device     ", torch.cuda.get_device_name(0))
print("cv2        ", cv2.__version__)
print("numpy      ", numpy.__version__)
print("all imports OK")
EOF
```

---

## 6. 获取代码与权重

### 6.1 克隆仓库

```bash
cd ~
git clone https://github.com/NVlabs/Fast-FoundationStereo.git
cd Fast-FoundationStereo
ls
```

预期看到：`core/`、`scripts/`、`Utils.py`、`docker/`、`requirements.txt`、`demo_data/`。

**重要**：这个目录要放在 WSL2 的 ext4 文件系统里（`~/Fast-FoundationStereo`），**不要放在 `/mnt/c/` 下**——跨文件系统 IO 慢 5–10 倍，加载 2.5 GB 的模型权重时会非常明显。

### 6.2 下载权重（商业版，推荐）

商业版只有一个 67.81 MB 的文件，HF 直链最稳，且许可是 NVIDIA Open Model Agreement（允许商用）。

```bash
conda activate ffs
pip install -U huggingface-hub

# 国内网络加这句（hf-mirror 是 HF 的国内镜像）
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download nvidia/c-fast-foundationstereo \
    model_best_bp2_serialize.pth cfg.yaml \
    --local-dir weights/c-fast-foundationstereo
```

下载完成后检查：

```bash
ls -lh weights/c-fast-foundationstereo/
# 预期：
# model_best_bp2_serialize.pth   约 68 MB
# cfg.yaml                       182 B
```

**直链备份**（如果 huggingface-cli 有问题，用 wget）：

```bash
mkdir -p weights/c-fast-foundationstereo
wget -O weights/c-fast-foundationstereo/model_best_bp2_serialize.pth \
  https://hf-mirror.com/nvidia/c-fast-foundationstereo/resolve/main/model_best_bp2_serialize.pth
wget -O weights/c-fast-foundationstereo/cfg.yaml \
  https://hf-mirror.com/nvidia/c-fast-foundationstereo/resolve/main/cfg.yaml
```

### 6.3 研究版（多档位，可选）

如果你想要更快的档位（`20-30-48` 在你这台机器上更合适），从 Google Drive 下：

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap" -O weights/
```

| 检查点 | valid_iters=8 | valid_iters=4 | 峰值显存 |
|---|---|---|---|
| `23-36-37` | 49.4 ms | 41.1 ms | 653 MB |
| `20-26-39` | 43.6 ms | 37.5 ms | 651 MB |
| `20-30-48` | **38.4 ms** | **29.3 ms** | 646 MB |

（以上是 RTX 3090 @ 640×480 的官方数据。你的 1650 见第 10 节。）

> **两个提醒**：
> 1. `cfg.yaml` 必须和 `.pth` 在**同一个目录**——官方代码是 `open(os.path.dirname(model_dir)/'cfg.yaml')`，找不配套的文件会直接崩。
> 2. `.pth` 是 pickle 的整个模型对象（不是 state_dict），加载用 `weights_only=False`。**只从官方地址下载**，不要用来路不明的权重文件。

### 6.4 目录结构

最终应该是这样：

```
~/Fast-FoundationStereo/
├── core/            # 模型代码
├── scripts/
│   └── run_demo.py
├── Utils.py
├── weights/
│   └── c-fast-foundationstereo/
│       ├── model_best_bp2_serialize.pth
│       └── cfg.yaml
├── demo_data/       # 官方自带 left.png / right.png / K.txt
└── batch_infer.py   # 第 8 节我们自己写的
```

---

## 7. 跑通官方 demo（验证环境）

### 7.1 先跑一次官方脚本

```bash
conda activate ffs
cd ~/Fast-FoundationStereo

python scripts/run_demo.py \
  --model_dir weights/c-fast-foundationstereo/model_best_bp2_serialize.pth \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir ~/ffs_out \
  --get_pc 0 \
  --valid_iters 8
```

### 7.2 WSL2 里的三个必踩问题

跑这一步时你会遇到下面三个问题，**都是已知的、有解的**：

**问题 1：默认输出目录不存在**

`run_demo.py` 里有一行：

```python
os.system(f'rm -rf {args.out_dir} && mkdir -p {args.out_dir}')
```

默认 `out_dir` 是 `/home/bowen/debug/stereo_output`（作者自己机器的路径）。在你的 WSL2 里 `/home/bowen` 不存在，`mkdir -p` 会失败（没权限建 `/home/bowen`），后面写文件时报 `FileNotFoundError`。

**解决**：就是上面命令里的 `--out_dir ~/ffs_out`，**必须显式指定**。

**问题 2：`cv2.imshow` + `waitKey(0)` 卡死**

脚本中段有：

```python
cv2.imshow('disp', resized_vis[:,:,::-1])
cv2.waitKey(0)
```

WSL2 无头环境下没有窗口，`waitKey(0)` 会**永远阻塞等你按一个永远不存在的键**。

**解决办法（临时）**：Ctrl+C 中断即可。视差图其实已经写到 `~/ffs_out/disp_vis.png` 了。

**问题 3：`--get_pc 0` 导致拿不到深度**

官方脚本里 `depth_meter.npy` 只在 `if args.get_pc:` 分支里保存，而这个分支末尾会强制跑：

```python
vis = o3d.visualization.Visualizer()
vis.create_window()
...
vis.run()
```

也就是说——**官方脚本里「只算深度不弹窗」这个选项不存在**。想要深度文件就必须看点云，而 WSL2 里 open3d 弹窗需要 WSLg + OpenGL，很可能直接崩。

**所以第 8 节的脚本才是你实际要用的东西。** 第 7 节这一跑只是为了确认环境和权重没问题。

### 7.3 验证标准

这一步成功的标志：

```bash
ls -lh ~/ffs_out/
# 至少应看到：
# left.png  right.png  disp_vis.png
```

打开 `disp_vis.png` 看一眼（后面第 9.3 节讲怎么在 Windows 里看）——如果是一张左右原图 + 彩色视差图的三联图，且视差图有合理的层次结构（近处暖色、远处冷色），说明**环境和权重都对了**。

日志里还应该看到类似：

```
Start forward, 1st time run can be slow due to compilation
forward done
```

**第一次前向很慢是正常的**（CUDA kernel 初始化、可能的算子选择），之后会快很多。

---

## 8. 实际使用：batch_infer.py

这是你真正要用的脚本。它只做官方脚本的前半段（读图 → pad → 前向 → unpad → 转深度 → 存盘），砍掉所有弹窗和 open3d。

### 8.1 完整代码

在 `~/Fast-FoundationStereo/batch_infer.py` 创建文件：

```python
#!/usr/bin/env python3
"""
FFS 批量推理：只输出深度，不弹窗、不做点云可视化。
用法见 --help。
"""
import os, sys, argparse, logging, time, glob, yaml
import numpy as np
import torch
import imageio
import cv2

FFS_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FFS_ROOT)

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed


def load_model(model_dir, valid_iters=8, max_disp=192):
    """加载模型。cfg.yaml 必须与 .pth 同目录。"""
    cfg_path = os.path.join(os.path.dirname(model_dir), 'cfg.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    model = torch.load(model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    return model, cfg


@torch.no_grad()
def infer_disp(model, left_path, right_path, scale=1.0,
               valid_iters=8, max_disp=192, hiera=False):
    """返回 (视差图 ndarray[H,W], 耗时秒)"""
    img0 = imageio.imread(left_path)
    img1 = imageio.imread(right_path)
    if img0.ndim == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
    if img1.ndim == 2:
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]

    if scale != 1.0:
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
    H, W = img0.shape[:2]

    x0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    x1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(x0.shape, divis_by=32, force_square=False)
    x0, x1 = padder.pad(x0, x1)

    t0 = time.time()
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        if not hiera:
            disp = model.forward(x0, x1, iters=valid_iters,
                                 test_mode=True,
                                 optimize_build_volume='pytorch1')
        else:
            disp = model.run_hierachical(x0, x1, iters=valid_iters,
                                         test_mode=True, small_ratio=0.5)
    torch.cuda.synchronize()
    dt = time.time() - t0

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)
    return disp, dt


def disp_to_depth(disp, K, baseline, remove_invisible=True):
    """视差 -> 深度（米）。公式：depth = fx * B / d"""
    H, W = disp.shape
    disp = disp.astype(np.float32)
    if remove_invisible:
        # 左图上看不到的区域（右侧超出）判为无效
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        disp = np.where((xx - disp) < 0, np.inf, disp)
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = K[0, 0] * baseline / disp
    depth[~np.isfinite(depth)] = 0.0
    return depth


def read_k_txt(path, scale=1.0):
    """读 K.txt：第 1 行是展平的 3x3 内参，第 2 行是基线（米）。"""
    with open(path, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]
    K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float64).reshape(3, 3)
    baseline = float(lines[1])
    if scale != 1.0:
        K = K.copy()
        K[:2] *= scale          # 图像缩放后内参必须同步缩放
    return K, baseline


def save_outputs(out_dir, name, disp, depth):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{name}_depth_meter.npy'), depth)

    # 视差彩色图，方便肉眼检查
    d = disp.copy()
    d[~np.isfinite(d)] = 0
    if d.max() > 0:
        vis = (np.clip(d / d.max(), 0, 1) * 255).astype(np.uint8)
    else:
        vis = np.zeros_like(d, dtype=np.uint8)
    cv2.imwrite(os.path.join(out_dir, f'{name}_disp_vis.png'),
                cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))

    # 深度伪彩（0.2m~2.0m 映射，按你的工作距离调整）
    dep = depth.copy()
    valid = dep > 0
    dep_vis = np.zeros(dep.shape, dtype=np.uint8)
    if valid.any():
        dep_vis[valid] = np.clip((dep[valid] - 0.2) / (2.0 - 0.2) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f'{name}_depth_vis.png'),
                cv2.applyColorMap(dep_vis, cv2.COLORMAP_TURBO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--left_file', required=True, help='文件或通配符')
    ap.add_argument('--right_file', required=True, help='文件或通配符')
    ap.add_argument('--intrinsic_file', required=True, help='K.txt')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--scale', type=float, default=1.0, help='图像缩放，小显存用 0.5')
    ap.add_argument('--valid_iters', type=int, default=8)
    ap.add_argument('--max_disp', type=int, default=192)
    ap.add_argument('--hiera', type=int, default=0, help='1=层次化推理，更快更省显存')
    ap.add_argument('--warmup', type=int, default=1, help='先跑几帧预热，不计入计时')
    args = ap.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    K, baseline = read_k_txt(args.intrinsic_file, args.scale)
    logging.info(f'K=\n{K}\nbaseline={baseline} m')

    model, cfg = load_model(args.model_dir, args.valid_iters, args.max_disp)
    logging.info('model loaded')

    lefts = sorted(glob.glob(args.left_file))
    rights = sorted(glob.glob(args.right_file))
    assert len(lefts) == len(rights) > 0, f'左右图数量不匹配: {len(lefts)} vs {len(rights)}'
    logging.info(f'{len(lefts)} pairs')

    times = []
    for i, (lf, rf) in enumerate(zip(lefts, rights)):
        name = os.path.splitext(os.path.basename(lf))[0]
        disp, dt = infer_disp(model, lf, rf, scale=args.scale,
                              valid_iters=args.valid_iters,
                              max_disp=args.max_disp, hiera=bool(args.hiera))
        if i < args.warmup:
            logging.info(f'[{i}] warmup {dt:.3f}s (不计入统计)')
            continue
        depth = disp_to_depth(disp, K, baseline)
        save_outputs(args.out_dir, name, disp, depth)
        times.append(dt)

        valid_ratio = float((depth > 0).mean())
        logging.info(f'[{i}] {name}: {dt:.3f}s  有效深度 {valid_ratio*100:.1f}%  '
                     f'中位深度 {np.median(depth[depth>0]) if valid_ratio>0 else 0:.3f}m')

    if times:
        logging.info(f'平均 {np.mean(times):.3f}s/帧  峰值显存 '
                     f'{torch.cuda.max_memory_allocated()/1024**2:.0f} MB')


if __name__ == '__main__':
    main()
```

### 8.2 用法

单张：

```bash
conda activate ffs
cd ~/Fast-FoundationStereo

python batch_infer.py \
  --model_dir weights/c-fast-foundationstereo/model_best_bp2_serialize.pth \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir ~/ffs_out
```

批量（通配符）：

```bash
python batch_infer.py \
  --model_dir weights/c-fast-foundationstereo/model_best_bp2_serialize.pth \
  --left_file  ~/data/mb2014/Adirondack/im0.png \
  --right_file ~/data/mb2014/Adirondack/im1.png \
  --intrinsic_file ~/data/mb2014/Adirondack/K.txt \
  --out_dir ~/ffs_out/mb  --scale 0.5 --valid_iters 4
```

低配档位（你这台机器推荐）：

```bash
python batch_infer.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --left_file ... --right_file ... --intrinsic_file ... \
  --out_dir ~/ffs_out \
  --scale 0.5 --valid_iters 4 --hiera 1
```

### 8.3 与官方脚本的三点差异（有意这样写）

| 差异 | 官方 | 本脚本 | 为什么 |
|---|---|---|---|
| 深度保存 | 只在 `--get_pc 1` 分支，且强制弹点云 | 总是保存 `_depth_meter.npy` | 无头环境能跑 |
| 无效值处理 | `disp[invalid]=inf`，然后 `depth = fx*B/disp` 得到 0 | 显式 `~np.isfinite → 0.0` | 语义更清楚：0 = 无效 |
| 计时 | 无 | 带 warmup 和峰值显存统计 | 调参需要真实数字 |

**注意**：本脚本**不做** `--remove_invisible` 之外的后处理，也不做点云去噪。需要的话自己加。

---

## 9. 数据准备与跨系统搬运

### 9.1 输入的三条铁律

**铁律 1：左右图必须已去畸变且极线水平对齐。**

FFS 假设同一个点在左右图中的 y 坐标相同、只在 x 方向有偏移。没做校正的图喂进去，输出是垃圾，而且**不会报错**。

自查方法：在左右图上找一个明显的特征点，比较它的 y 坐标，差应 **< 0.5 px**。

**铁律 2：图像尺寸会被自动 pad 到 32 的整数倍。**

`InputPadder(..., divis_by=32)`。这个 pad 是内部处理的，输出会自动 unpad 回原尺寸，你不用管——但如果原图尺寸本身就是 32 的倍数会略快。

**铁律 3：`--scale` 缩放后，内参必须同步缩放。**

脚本里 `K[:2] *= scale` 已经处理了。但如果你自己改代码，**别忘了 fx、fy、cx、cy 四个数都要乘同一个系数**。

### 9.2 K.txt 格式

只有两行：

```
fx 0 cx 0 fy cy 0 0 1
0.05
```

第 1 行是展平的 3×3 内参（按行展开），第 2 行是**基线，单位是米**。

**最容易错的**：基线写成毫米（比如 `50` 而不是 `0.05`）。错了之后深度差 1000 倍，但**深度图的形状看起来完全正常**，极难发现。建议生成后立刻检查一个已知距离的物体。

### 9.3 数据怎么在 Windows 和 WSL2 之间搬

**原则：代码和正在处理的数据放 ext4（`~/`），只有「搬入搬出」时才碰 `/mnt/c/`。**

**从 Windows 拷进 WSL2**：

```bash
mkdir -p ~/data
cp -r /mnt/c/Users/x/Learn/grasp/你的图目录 ~/data/
```

**从 WSL2 取结果到 Windows（推荐方式）**：在 Windows 文件资源管理器的地址栏输入：

```
\\wsl$\Ubuntu-22.04\home\x\ffs_out
```

可以直接像普通文件夹一样浏览、复制。**不要用 `/mnt/c/` 作为输出目录**——跨文件系统写大量小文件很慢。

### 9.4 没有双目数据怎么办

这是你现在最现实的问题。你现有工程里 `demo/data/benchmark/{d405,g305}/grasp_3d/` 下的 `*-color.png` + `*-depth.png` 是**单目 RGB + 单目深度**，不是左右图对，**喂不进立体匹配**。

三条路：

| 方案 | 做法 | 成本 |
|---|---|---|
| **A. 用公开数据集（推荐先做）** | 下 Middlebury 2014，自带左右图 + GT 视差 + `calib.txt`（里面有 f 和 baseline，正好是 K.txt 需要的两个数） | 零成本，能定量评估 |
| **B. 用 RealSense 在 Windows 抓红外对** | Windows 装 `pyrealsense2`（有官方 wheel），订阅 `infra1/image_rect_raw` 和 `infra2/image_rect_raw` 存 PNG——这两个话题**已经是去畸变 + 极线校正的**，直接能用。基线用 `tf2_echo camera_infra1_optical_frame camera_infra2_optical_frame` 的 `translation.x` 绝对值 | 需要相机，但在 Windows 侧完成，不违反「WSL2 不连相机」 |
| **C. 自搭双目** | 两个全局快门相机 + 硬件同步 + 自己标定 | 2–3 周 |

**建议先走 A**，一天内拿到定量数字，确认环境和脚本都对，再考虑 B。

---

## 10. 你这台机器的性能预期与调参

### 10.1 速度预估

官方数据是 RTX 3090 上的，GTX 1650（Turing，4 GB）大约要乘 **6–10 倍**：

| 配置 | 3090 | 你的 1650（预估） |
|---|---|---|
| `23-36-37` @ iters8, scale 1.0 | 49 ms | **300–500 ms** |
| `20-30-48` @ iters4, scale 1.0 | 29 ms | **180–300 ms** |
| `20-30-48` @ iters4, scale 0.5 | — | **60–120 ms** |
| 加 `--hiera 1` | — | 再快 30–50% |

**这些数字要实跑才算数**——`batch_infer.py` 会打印每帧耗时和峰值显存，跑一次就知道。

### 10.2 显存

官方实测 640×480 峰值只有 **646–653 MB**。你的 4 GB 显存**完全够**，甚至 1280×720 也应该能跑（约 1.4 GB）。

但要注意：**Windows 端也在用这块显卡**。跑推理时关掉浏览器硬件加速、游戏、视频播放器。显存不够的典型症状是：

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

应对：先 `--scale 0.5`，再 `--hiera 1`，再降 `valid_iters`。

### 10.3 内存（8 GB 是真瓶颈）

`.wslconfig` 里给了 5 GB，但 torch 导入 + 模型加载 + 图像本身也要占。建议：

- 跑推理时关掉 Windows 端的 IDE、浏览器
- 批量处理时**分批处理**，不要一次 glob 几百张（图像会在内存里排队）
- 加内存到 16 GB 是这台机器性价比最高的升级（DDR4 SO-DIMM，约 100–200 元）

### 10.4 推荐起始档位

针对你的机器，第一次跑建议：

```bash
--scale 0.5 --valid_iters 4 --hiera 1
```

确认出图正常后，再逐步加到 `scale 1.0 / iters 8` 看精度提升多少。**先确认能跑通，再追求精度。**

---

## 11. 效果评估：怎么知道好不好

### 11.1 有真值时（Middlebury）

```python
import numpy as np
gt = np.load('gt_disp.npy')            # 或读 GT 视差图
pred = np.load('xx_depth_meter.npy')
# 转成同一量纲后比较
mask = (gt > 0) & np.isfinite(pred)
err = np.abs(pred[mask] - gt[mask])
print('MAE', err.mean(), '| bad-2px:', (err > 2).mean())
```

### 11.2 无真值时（你的实际情况）

四种方法，从易到难：

**① 平面靶标 RMS（最实用）**

拿一块平板正对相机，放到 0.3 / 0.5 / 1.0 / 1.5 m 四个位置，每个位置采 5 帧。对每帧深度图：

```python
# 取中央 100x100 区域
patch = depth[cy-50:cy+50, cx-50:cx+50]
patch = patch[patch > 0]
print(f'{patch.mean():.4f} m  |  std {patch.std()*1000:.2f} mm  |  空洞率 {(1-patch.size/10000)*100:.1f}%')
```

`std` 就是该距离下的深度噪声。

**② 已知尺寸物体**：量一个盒子的长宽高，和深度图里量出来的对比。

**③ 左右一致性（不需要任何真值）**：把左右图**交换**再跑一次，得到 `disp_rl`。在对应位置比较 `disp_lr` 和 `disp_rl`，差异大的地方就是不可靠区域。这个方法零成本，但只能发现不一致，不能保证都对。

**④ 时间稳定性**：相机不动，连采 10 帧，看深度值的抖动（std）。

### 11.3 建议验收线

参考你的抓取工作距离（0.3–1.5 m）：

| 距离 | 深度噪声（RMS） | 有效率 |
|---|---|---|
| 0.3 m | < 1 mm | > 95% |
| 0.5 m | < 2 mm | > 92% |
| 1.0 m | < 8 mm | > 90% |
| 1.5 m | < 20 mm | > 85% |

达不到就要回头查输入（校正、基线单位、内参分辨率匹配），**而不是换模型**。

---

## 12. 坑位清单

| # | 现象 | 根因 | 改法 |
|---|---|---|---|
| 1 | `torch.cuda.is_available()` 返回 False | 在 WSL2 里装了 NVIDIA 驱动，顶掉了主机驱动映射 | 卸载 WSL2 内的驱动；只保留 Windows 端驱动 |
| 2 | `nvidia-smi` 在 WSL2 里命令不存在 | 没装 toolkit 时 `nvidia-smi` 在 `/usr/lib/wsl/lib/` 下 | `export PATH=/usr/lib/wsl/lib:$PATH`，或直接用全路径 |
| 3 | 官方 demo 报 `FileNotFoundError` | 默认 `--out_dir` 是 `/home/bowen/debug/...`，你机器上不存在且 `mkdir -p` 没权限 | 显式 `--out_dir ~/ffs_out` |
| 4 | 脚本卡住不动，CPU 占用为 0 | `cv2.waitKey(0)` 在无头环境永远等按键 | Ctrl+C；改用 `batch_infer.py` |
| 5 | `o3d.visualization` 崩溃 | open3d 需要 OpenGL，WSL2 无头环境没有 | 用 `--get_pc 0` 或 `batch_infer.py` |
| 6 | `FileNotFoundError: .../cfg.yaml` | `cfg.yaml` 没和 `.pth` 放同一目录 | 两者必须同级，官方按 `dirname(model_dir)` 找 |
| 7 | 深度数值差 1000 倍 | K.txt 第 2 行基线写成毫米 | 改成米（如 `0.05` 不是 `50`） |
| 8 | 深度图整体形状对但比例错 | 图像缩放了但 `K[:2] *= scale` 没做 | 脚本已处理；自己改代码时别漏 |
| 9 | 输出全是噪点、毫无结构 | 输入图没做极线校正 / 没去畸变 | 检查左右图同一特征点的 y 差 < 0.5 px |
| 10 | `OutOfMemoryError` | 4 GB 显存 + Windows 端占用 | `--scale 0.5` → `--hiera 1` → 关掉 Windows 端图形应用 |
| 11 | 在 WSL 里敲 `python` 调到了 Windows 的 Python | `/etc/wsl.conf` 里 `appendWindowsPath=true` | 设为 `false`，然后 `wsl --shutdown` |
| 12 | 加载模型极慢（几分钟） | 模型放在 `/mnt/c/` 下，跨文件系统 IO | 移到 `~/` 下的 ext4 |
| 13 | 第一次推理特别慢，后面正常 | CUDA kernel 初始化 | 正常，`--warmup 1` 跳过计时 |
| 14 | `.wslconfig` 改了没效果 | 没执行 `wsl --shutdown` | 改完必须完全关闭 WSL 再启动 |
| 15 | 主机越用越卡，内存被吃光 | WSL2 默认不还内存 | `.wslconfig` 加 `autoMemoryReclaim=gradual` |
| 16 | `pip install torch` 找不到 cu124 版本 | 用了国内镜像源，cu124 版本不全 | 用官方源 `download.pytorch.org/whl/cu124` |
| 17 | 装了 `cuda-toolkit` 后反而跑不了 | 用错仓库（`ubuntu2204` 而非 `wsl-ubuntu`），把驱动也装进来了 | 只装 `cuda-toolkit-12-4`，不装任何 `*-drivers*` |

---

## 13. 时间线与检查清单

### 第 1 天上午：装环境（2–3 小时，多数在等下载）

```
[ ] Windows 端 nvidia-smi 确认 CUDA Version >= 12.4
[ ] wsl --install -d Ubuntu-22.04
[ ] 换 apt 源，装基础工具
[ ] 写 .wslconfig，wsl --shutdown，验证 free -h 约 5G
[ ] 写 /etc/wsl.conf（appendWindowsPath=false）
[ ] 验证 ls -l /usr/lib/wsl/lib/libcuda.so.1 和 nvidia-smi
[ ] 装 miniconda，创建 ffs 环境
[ ] pip install torch==2.6.0 torchvision==0.21.0 cu124
[ ] python -c "import torch; print(torch.cuda.is_available())"  → True
```

**这一上午的里程碑就一个数字：`True`。**

### 第 1 天下午：跑通（1–2 小时）

```
[ ] git clone FFS
[ ] huggingface-cli 下载权重到 weights/c-fast-foundationstereo/
[ ] ls -lh 确认 68 MB + cfg.yaml
[ ] pip install timm einops omegaconf scipy scikit-image opencv-contrib-python imageio pyyaml
[ ] python scripts/run_demo.py --out_dir ~/ffs_out --get_pc 0  → 出 disp_vis.png
[ ] 创建 batch_infer.py，跑通，出 _depth_meter.npy
```

### 第 2 天：拿数据（半天）

```
[ ] 下 Middlebury 2014 的几组数据到 ~/data/
[ ] 从 calib.txt 生成 K.txt
[ ] batch_infer.py 批量跑
[ ] 对比 GT 视差，出误差数字
[ ] 记录每帧耗时和峰值显存
```

### 第 3 天：调参与评估

```
[ ] 试不同档位（scale / valid_iters / hiera），填速度-精度表
[ ] 确定你这台机器的最优档位
[ ] 用方案 A/B 拿到自己场景的数据，评估
```

### 之后（按需）

```
[ ] 需要 TensorRT → 装 cuda-toolkit-12-4 + TensorRT（Linux 包，WSL2 可装）
[ ] 需要接相机 → 相机留在 Windows，抓图存盘后拷进 WSL2
[ ] 需要接 ROS2 → ROS2 装在同一个 WSL2 里
```

---

## 14. 这个环境后续还能干什么

WSL2 一旦装好，它就不只是给 FFS 用的。按你之前讨论的定位模型路线，这个环境正好覆盖那些**在 Windows 上装不了**的东西：

| 用途 | 是否需要 WSL2 | 说明 |
|---|---|---|
| FFS 推理 / TensorRT 加速 | **需要**（TensorRT 只有 Linux 包） | 本方案 |
| TAO Toolkit 微调 | **需要** | TAO 基于 Docker，Linux 优先 |
| PyTorch3D / nvdiffrast 编译 | **需要** | 两行 `--no-build-isolation git+...` 就能装 |
| FoundationPose / GigaPose / MegaPose | **需要** | 依赖上面两个 |
| ROS2（完整生态） | **需要** | 第三方包 Linux 优先 |
| YOLO11 / YOLOE / Open3D ICP | 不需要 | Windows 也能跑，两边都行 |

**建议**：环境建好后先 `conda create -n pose python=3.12` 留一个空环境，后面装位姿模型时不用重新折腾底层。

另外，8 GB 内存下**训练基本不可行**（渲染模板 + 提特征都很紧张）。训练建议上云（AutoDL / 恒源云，3090 24 GB 约 1.5–3 元/小时），WSL2 只做推理和数据准备。

---

## 附 A：一键安装脚本

把下面内容存成 `~/setup_ffs.sh`，在 WSL2 里 `bash ~/setup_ffs.sh` 一次跑完第 5–6 节。**建议新手先手动跑一遍理解每步在干什么，熟了再用脚本重建环境。**

```bash
#!/usr/bin/env bash
set -e

echo "=== [1/6] 基础工具 ==="
sudo apt update
sudo apt install -y build-essential curl wget git vim htop

echo "=== [2/6] miniconda ==="
if [ ! -d "$HOME/miniconda3" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
  "$HOME/miniconda3/bin/conda" init bash
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "=== [3/6] conda 环境 ==="
if ! conda env list | grep -q '^ffs'; then
  conda create -n ffs python=3.12 -y
fi
conda activate ffs

echo "=== [4/6] torch cu124（最慢，约 2.5 GB）==="
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== [5/6] 依赖 ==="
pip install timm einops omegaconf scipy numpy scikit-image \
            opencv-contrib-python imageio pyyaml huggingface-hub

echo "=== [6/6] 代码与权重 ==="
cd ~
[ -d Fast-FoundationStereo ] || git clone https://github.com/NVlabs/Fast-FoundationStereo.git
cd Fast-FoundationStereo
# 国内取消下一行注释
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download nvidia/c-fast-foundationstereo \
    model_best_bp2_serialize.pth cfg.yaml \
    --local-dir weights/c-fast-foundationstereo

echo "=== 验证 ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('OK:', torch.cuda.get_device_name(0))"
ls -lh weights/c-fast-foundationstereo/
echo "完成。下一步：cd ~/Fast-FoundationStereo && python scripts/run_demo.py --out_dir ~/ffs_out --get_pc 0"
```

---

## 附 B：常用命令速查

```bash
# ---- WSL 管理（Windows PowerShell 里执行）----
wsl --install -d Ubuntu-22.04      # 安装
wsl --shutdown                      # 完全关闭（改配置后必做）
wsl --list --verbose                # 查看发行版和版本
wsl --update                        # 更新 WSL 内核
wsl --status                        # 查看状态

# ---- 进入 / 退出（Windows PowerShell）----
wsl                                 # 进入默认发行版
wsl -d Ubuntu-22.04                 # 指定发行版
exit                                # 退出

# ---- GPU（WSL2 内）----
nvidia-smi                          # 看 GPU
nvidia-smi --query-gpu=memory.used --format=csv
ls -l /usr/lib/wsl/lib/libcuda.so.1 # 确认驱动映射

# ---- conda ----
conda activate ffs
conda deactivate
conda env list

# ---- 推理 ----
cd ~/Fast-FoundationStereo
python batch_infer.py --model_dir weights/c-fast-foundationstereo/model_best_bp2_serialize.pth \
    --left_file demo_data/left.png --right_file demo_data/right.png \
    --intrinsic_file demo_data/K.txt --out_dir ~/ffs_out

# ---- 文件互访 ----
# Windows 访问 WSL2：资源管理器地址栏输入  \\wsl$\Ubuntu-22.04\home\x
# WSL2 访问 Windows：                      /mnt/c/Users/x/
```

---

## 附 C：许可与链接

| 项目 | 许可 | 说明 |
|---|---|---|
| Fast-FoundationStereo 代码 | NVIDIA 源码许可 | 见仓库 LICENSE |
| `nvidia/c-fast-foundationstereo` 权重 | NVIDIA Open Model Agreement | **允许商用**，推荐 |
| 研究版权重（Google Drive） | 研究用途 | 商用需注意 |

**链接**：

- 仓库：https://github.com/NVlabs/Fast-FoundationStereo
- 商业权重：https://huggingface.co/nvidia/c-fast-foundationstereo
- 国内镜像：https://hf-mirror.com
- 在线试玩（零安装先看效果）：https://huggingface.co/spaces/hugging-apps/fast-foundationstereo
- NVIDIA CUDA on WSL 文档：https://docs.nvidia.com/cuda/wsl-user-guide/

---

## 一句话总结

**装 WSL2 → 只装 torch cu124 不装系统 CUDA → 必须显式指定 `--out_dir` → 用 `batch_infer.py` 而不是官方脚本 → 数据放 `~/` 不放 `/mnt/c`。**

前两步是环境，中间两步是避开 WSL2 特有的坑，最后一步是性能。按第 13 节的清单走，第一天下午就能看到第一张深度图。

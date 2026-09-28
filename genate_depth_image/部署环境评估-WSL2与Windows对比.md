# 部署环境评估：WSL2 还是 Windows？（范围：只测深度图生成效果）

> **范围已收窄**：只做「给一对立体图 → 生成深度图 → 评估效果」。
> **不涉及**：ROS / ROS2、`carm_grasp` 的 `utils.py`（termios 那些）、相机连接与采集。
>
> 在这个范围下，上一版的结论（「装 WSL2」）**不再成立**。下面是修订后的结论。

> **更新（用户已决策）：最终选 WSL2。**
> 本份文档保留为**选型评估**（为什么 Windows 原生也能跑、什么时候才需要 WSL2）。
> **具体怎么装、怎么跑，看执行手册：`WSL2部署FFS-完整落地方案.md`**——包含安装命令、`.wslconfig` 配置、
> GPU 透传验证、权重下载、`batch_infer.py` 完整代码、性能调参、17 条坑位表和三天时间线。
>
> 两份文档的分工：**本份回答「为什么」，那份回答「怎么做」。**
>
> 配套：`Fast-FoundationStereo_落地方案.md`（完整推理/加速/接入流程）。

## 目录

- [0. 结论先行（修订版）](#0-结论先行修订版)
- [1. 本机实测配置](#1-本机实测配置)
- [2. 为什么现在推荐 Windows 原生](#2-为什么现在推荐-windows-原生)
- [3. 官方脚本在 Windows 上的三处阻塞](#3-官方脚本在-windows-上的三处阻塞)
- [4. 最小可行路径：自己写一个推理脚本](#4-最小可行路径自己写一个推理脚本)
- [5. 测试数据从哪来（不接相机）](#5-测试数据从哪来不接相机)
- [6. 怎么评估效果好（没有真值也能做）](#6-怎么评估效果好没有真值也能做)
- [7. Windows 环境安装命令](#7-windows-环境安装命令)
- [8. 什么时候才需要 WSL2](#8-什么时候才需要-wsl2)
- [9. 权重文件完整下载路径](#9-权重文件完整下载路径)
- [10. GTX 1650 性能预期与降档方案](#10-gtx-1650-性能预期与降档方案)
- [11. 决策清单与下一步](#11-决策清单与下一步)
- [12. 范围扩展：加入 YOLO 与位姿模型后](#12-范围扩展加入-yolo-与位姿模型后)

---

## 0. 结论先行（修订版）

**直接在 Windows 上跑，不用装 WSL2。**

我上一版推荐 WSL2，理由是「你的 `utils.py` 用了 termios，整个工程只能跑 Linux」。现在这个理由不成立了——你不动那部分代码。去掉 ROS、去掉相机之后，剩下的活儿是：

> 读两张 PNG → 送进 PyTorch 模型 → 拿回视差图 → 换算成深度 → 保存

这整套在 Windows 原生上跑得很好。我查过 `scripts/run_demo.py` 的源码，它只 import 了 `torch / numpy / cv2 / imageio / omegaconf / yaml / open3d`，**没有 triton，没有自定义 CUDA 扩展**（模型前向用的是 `optimize_build_volume='pytorch1'` 这条纯 PyTorch 路径）。而 `torch==2.6.0+cu124` 有官方 Windows wheel。

**不装 WSL2 的额外好处**：你这台机器只有 8 GB 内存。装了 WSL2 就得切一半给虚拟机，现在是纯推理不需要。

**但官方脚本不能直接用**，有三处 Windows/GUI 相关的阻塞（见第 3 节）。正确做法是写一个几十行的小脚本，只保留「推理 + 存文件」，把弹窗和点云可视化全砍掉。

**最快看到效果的方式**（零安装，5 分钟）：先用 HuggingFace 上的官方在线 demo 传两张图看看——
https://huggingface.co/spaces/hugging-apps/fast-foundationstereo

---

## 1. 本机实测配置

| 项目 | 实测值 | 影响 |
|---|---|---|
| CPU | i5-9300H @ 2.40 GHz（4 核 8 线程） | 次要瓶颈 |
| 内存 | **8.0 GB** | 不装 WSL2 就没压力 |
| GPU | **NVIDIA GeForce GTX 1650**（Turing TU117，sm_75） | 能跑，约 3090 的 1/6～1/10 |
| 驱动 | 32.0.15.9227 | 满足 CUDA 12.4 |
| 磁盘（C:） | 剩余 162.6 GB | 充裕 |
| `nvidia-smi.exe` | 存在 | 可用来验证 |

两个已核实的数字（推翻了上一版的保守估计）：

- **峰值显存只有 646–653 MB**（官方表，640×480，3090 实测）。4 GB 显存绰绰有余。
- **FFS 用的是 fp16 不是 bf16**（`Utils.py` 里 `AMP_DTYPE = torch.float16`）。Turing 及以后都支持，不存在「bf16 把老卡挡在门外」的问题。

---

## 2. 为什么现在推荐 Windows 原生

| 维度 | **Windows 原生**（推荐） | WSL2 |
|---|---|---|
| PyTorch + CUDA | 官方 Windows wheel（cu118 / cu124 / cu126 都有） | 也支持，但要配 `.wslconfig`、装 CUDA Toolkit |
| 模型依赖 | triton、自定义 CUDA 扩展都不需要 | 同 |
| 内存 | 8 GB 全给推理进程 | 要切 4–5 GB 给虚拟机 |
| 安装耗时 | 约 30 分钟 | 1–2 小时（含系统组件、重启） |
| 文件 IO | 原生 | 代码必须放 ext4，不能放 `/mnt/c` |
| 后续扩展（TRT / TAO / ROS2） | 不支持或很麻烦 | 支持 |
| 适合 | **当前阶段：跑推理、看效果** | 后面要做 TRT 加速或微调时 |

**一句话**：现在这个阶段，装 WSL2 纯属给自己找事。等你需要 `trtexec` 或 TAO 微调时再装不迟（第 8 节有触发条件）。

---

## 3. 官方脚本在 Windows 上的三处阻塞

我逐行读了 `scripts/run_demo.py`，Windows 上有三个地方会卡住。这三处**不是模型问题，是脚本写法问题**，所以必须自己写脚本绕开。

### 3.1 `os.system('rm -rf ...')` —— 静默失败

```python
os.system(f'rm -rf {args.out_dir} && mkdir -p {args.out_dir}')
```

Windows 没有 `rm` 也没有 `mkdir -p`。这行会静默失败（命令找不到，但 `os.system` 不抛异常），于是输出目录根本没创建，后面 `imageio.imwrite(f'{args.out_dir}/disp_vis.png', ...)` 直接 `FileNotFoundError`。

**改法**：`os.makedirs(args.out_dir, exist_ok=True)`

### 3.2 `cv2.imshow` + `waitKey(0)` —— 阻塞

```python
cv2.imshow('disp', resized_vis[:,:,::-1])
cv2.waitKey(0)          # 停在这里等你按任意键
```

单张图调试无所谓，批处理时每帧都要手动按键，没法跑。

**改法**：直接 `imageio.imwrite` 存盘，不显示。

### 3.3 `--get_pc 1` 才存深度，但会强制弹 Open3D 窗口 —— 最关键的一处

这是最坑的。看代码结构：

```python
if args.get_pc:
    ...
    np.save(f'{args.out_dir}/depth_meter.npy', depth)     # ← 深度图只在这里保存
    ...
    vis = o3d.visualization.Visualizer()                  # ← 但紧接着就要弹窗
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()                                             # ← 阻塞，等你关窗口
```

也就是说：**想要深度文件就必须开 `get_pc`，而开了 `get_pc` 就一定会弹点云窗口**。官方脚本没有「只算深度不弹窗」的选项。

**改法**：自己写脚本，只取前半段——算 `depth = K[0,0]*baseline/disp` 然后 `np.save`，不碰 open3d。这样连 `open3d` 这个包都不用装（它在 Windows 上偶尔还会因为缺 VC 运行库装不上）。

### 3.4 顺带一提

- `--out_dir` 默认值是 `/home/bowen/debug/stereo_output`，Windows 上**必须显式传**。
- `--remove_invisible 1`（默认）会把左图看不到的区域视差设成 `inf`，换算后深度为 0。做评估时要知道 0 是「无效」不是「距离为零」。
- 官方 README 的安装命令带 `xformers`，但 `run_demo.py` 并不 import 它。**Windows 上装不上就跳过**，不影响推理。

---

## 4. 最小可行路径：自己写一个推理脚本

目标：一个 `infer_depth.py`，做「读图 → 推理 → 存盘」，无弹窗、不依赖 open3d、可批量。

它要做的事，就是从官方脚本里抽这几步（行号对应 `scripts/run_demo.py`）：

| 步骤 | 官方写法 | 说明 |
|---|---|---|
| 1. 加载配置 | 读 `cfg.yaml`，用命令行参数覆盖 | `cfg.yaml` 必须和 `.pth` 同目录且配套 |
| 2. 加载模型 | `torch.load(model_dir, map_location='cpu', weights_only=False)` | 是 pickle 的整个模型对象，不是 state_dict |
| 3. 设参数 | `model.args.valid_iters` / `model.args.max_disp` | 提速靠这两个 |
| 4. 读图 | `imageio.imread` → 取前 3 通道 → `cv2.resize(scale)` | 灰度图会自动 tile 成 3 通道 |
| 5. padding | `InputPadder(shape, divis_by=32)` | 高宽必须是 32 的倍数，pad 完要 `unpad` |
| 6. 前向 | `torch.amp.autocast('cuda', dtype=torch.float16)` 里 `model.forward(img0, img1, iters=..., test_mode=True, optimize_build_volume='pytorch1')` | `'pytorch1'` 这条路径不需要 triton |
| 7. 取视差 | `padder.unpad(disp.float())` → `.cpu().numpy().reshape(H,W).clip(0, None)` | 单位是**像素** |
| 8. 转深度 | `depth = K[0,0] * baseline / disp` | **米**。`K[:2] *= scale`（缩放过图时内参要同步缩放） |
| 9. 存盘 | `np.save(depth_meter.npy)` + 视差彩色图 | 完事 |

**关键细节，别踩**：

- **左右图不能反**。左图必须是左相机拍的（同一物体在左图里位置更靠右）。反了会得到完全错误的视差。
- **输入必须已校正去畸变**。没校正的话输出是垃圾。判断方法：左右图同一特征点的 y 坐标差 < 0.5 px。
- **`K.txt` 两行**：第 1 行是展平的 1×9 内参（`fx 0 cx 0 fy cy 0 0 1`），第 2 行是**基线，单位米**。写成毫米会差 1000 倍，但图看起来「形状是对的」，很容易漏掉。
- **第一次运行会慢**（编译开销）。测速前先 warmup 几帧。
- **无效值**：`disp=0` 或 `inf` → 深度为 0 或 inf。评估前要 mask 掉。

---

## 5. 测试数据从哪来（不接相机）

| 来源 | 有真值？ | 用途 | 地址 |
|---|---|---|---|
| **仓库自带 `demo_data/`** | 无 | 验证环境装没装对 | `git clone` 后自带 `left.png` / `right.png` / `K.txt` |
| **Middlebury 2014** | **有**（GT 视差 PFM + `calib.txt` 含 f 和 baseline） | **定量评估首选**，室内静态、纹理丰富 | https://vision.middlebury.edu/stereo/data/scenes2014/ |
| **Middlebury 2021** | 有 | 同上，更难 | https://vision.middlebury.edu/stereo/data/scenes2021/ |
| **ETH3D** | 有 | 室内外混合，含弱纹理场景 | https://www.eth3d.net/datasets |
| **KITTI 2012/2015** | 有（激光稀疏） | 室外驾驶场景，与你的场景差异大 | https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php |
| 你自己现有的数据 | 部分 | 见下方说明 | — |

**关于你现有的数据**：`carm_grasp-main/demo/data/` 下面有 `d405/` 和 `g305/` 的 `grasp_3d/{grasp,ready}-color.png` + `-depth.png`，但那是**单张彩色图 + 对应的单目深度**，不是左右图对。**不能直接喂给 FFS**——立体匹配必须要两张不同视点的图。

如果一定要用自己的数据又暂时不接相机，只有一条路：**用已有的 color+depth 反向合成一张右图**（depth-image-based rendering：按深度把左图 warp 到右视点）。这能做定性观察，但边缘会有空洞伪影，合成质量受限于原深度质量，**不能用来做精度评估**。建议先用 Middlebury，等有相机了再用真实数据。

---

## 6. 怎么评估效果好（没有真值也能做）

按可信度从高到低：

### 6.1 有真值时（Middlebury / ETH3D）—— 标准指标

```
EPE       = mean(|disp_pred - disp_gt|)              端点误差，单位像素
bad1.0    = 视差误差 > 1.0 px 的像素占比（%）        最常用
bad2.0    = 视差误差 > 2.0 px 的像素占比（%）
D1        = 误差 > 3px 且 > 5% 的占比（KITTI 用的）
```

评估时通常只统计 **GT 有效的像素**（`disp_gt > 0` 且非遮挡区）。Middlebury 的 `calib.txt` 里能直接读到 `cam0` 内参和 `baseline`，正好就是 `K.txt` 需要的两个量。

**参考量级**：FFS 这类模型在 Middlebury 上 bad2.0 通常在个位数百分比；传统 SGBM 在纹理好的 Middlebury 图上也不差，但在弱纹理/无纹理区域会大面积失效。**重点看 FFS 相对于 SGBM 的优势有没有体现在弱纹理和边缘上**——那才是它值钱的地方。

### 6.2 没有真值时 —— 五种替代方法

| 方法 | 怎么做 | 能说明什么 |
|---|---|---|
| **平面靶标 RMS** | 拍一块正对的平板，取中央区域拟合平面，算点到平面距离的标准差 | **最直接**。不需要任何真值，直接反映深度噪声 |
| **已知尺寸物体** | 量一个已知长宽的盒子，看点云里量出来差多少 | 反映系统级的尺度/标定误差 |
| **左右一致性（LR check）** | 把左右图**交换**再跑一次，得到 `disp_rl`；比较 `disp_lr` 和反向映射后的 `disp_rl`，统计不一致像素占比 | 不需要真值，对遮挡和误匹配很敏感 |
| **时间稳定性** | 同一静止场景连拍 N 帧（或同一张图跑 N 次加不同噪声），算每像素深度的标准差 | 反映深度噪声下限。同一张图重复跑应该完全一致，能验证确定性 |
| **边缘质量（目视）** | 看视差彩色图：物体边界是不是清晰、有没有「飞边」；弱纹理区域（白墙、纯色塑料）是不是塌成一片 | 最主观但最直观，能一眼看出「能不能用」 |

### 6.3 建议的验收线

结合你 0.3–1.5 m 的工作距离：

| 指标 | 合格线 | 说明 |
|---|---|---|
| 0.5 m 处平面 RMS | < 2 mm | 抓取够用 |
| 1.0 m 处平面 RMS | < 8 mm | 双目物理极限附近 |
| 有效像素率（非 0/inf） | > 90% | 太低说明匹配失败或 max_disp 不够 |
| 弱纹理区域（纯色件）有效像素率 | > 70% | **这是 FFS 相对传统方法的核心优势区** |
| 单帧耗时（640×480） | < 500 ms | 静态抓取 2 FPS 就够 |

---

## 7. Windows 环境安装命令

全程 PowerShell / cmd，**不需要管理员权限、不需要重启**。

```powershell
# 1. 建虚拟环境（Python 3.12，与官方一致）
#    如果本机 Python 版本不对，先装 3.12
py -3.12 -m venv C:\Users\x\ffs-env
C:\Users\x\ffs-env\Scripts\activate

# 2. 装 PyTorch（官方 wheel，cu124，Windows 版本存在）
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

#    验证：必须打印 2.6.0 True (7, 5)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"

# 3. 拉代码
git clone https://github.com/NVlabs/Fast-FoundationStereo.git
cd Fast-FoundationStereo

# 4. 装依赖（open3d 可以跳过，自己写脚本就不需要）
pip install timm einops omegaconf scipy numpy scikit-image opencv-contrib-python imageio pyyaml

# 5. 下权重（推荐商业版，67.81 MB，直链最稳；国内加 HF_ENDPOINT）
pip install -U huggingface-hub
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download nvidia/c-fast-foundationstereo model_best_bp2_serialize.pth cfg.yaml --local-dir weights\c-fast-foundationstereo

# 6. 先用仓库自带数据跑通
python scripts\run_demo.py --model_dir weights\c-fast-foundationstereo\model_best_bp2_serialize.pth --left_file demo_data\left.png --right_file demo_data\right.png --intrinsic_file demo_data\K.txt --out_dir output --get_pc 0 --valid_iters 4 --max_disp 192
```

> 第 6 步用 `--get_pc 0` 是为了**避开 Open3D 弹窗**（第 3.3 节）。代价是不会存 `depth_meter.npy`，但会存 `disp_vis.png`——先确认这条链路通了，再用自己的脚本出深度。
>
> 如果第 6 步报 `FileNotFoundError`，就是第 3.1 节那个 `rm -rf` 的坑，手动 `mkdir output` 即可。

**依赖精简说明**：`requirements.txt` 里的 `open3d` 只有点云可视化用得到，自己写脚本的话**完全不用装**（Windows 上它还可能因为缺 VC++ 运行库装失败）。

---

## 8. 什么时候才需要 WSL2

现在的答案是「不用装」。出现下面任一情况时再装（Ubuntu 22.04）：

| 触发条件 | 原因 |
|---|---|
| 要用 TensorRT（`trtexec`）加速 | TensorRT 官方只提供 Linux 包 |
| 要用 TAO Toolkit 微调 | TAO 基于 Docker，Linux 优先 |
| 要接 ROS2 的第三方包（realsense-ros 等） | 核心是 Tier 1 但三方包 Linux 优先（见 12.1） |
| 要重新跑 `carm_grasp` 的完整标定/抓取流程 | 就是上一版说的 termios 问题 |
| **要用 FoundationPose / MegaPose / SAM-6D 这类位姿模型** | 依赖 PyTorch3D + nvdiffrast，需源码编译（见 12.1） |

真要装时的关键配置（8 GB 内存机器必做）：

```ini
# C:\Users\x\.wslconfig
[wsl2]
memory=5GB
swap=8GB
processors=4
```

以及**最大的禁忌**：**不要在 WSL2 里装 NVIDIA 驱动**，只装 `cuda-toolkit-12-4`（用 `wsl-ubuntu` 源）。装了驱动会导致 `nvidia-smi` 正常但 `torch.cuda.is_available()` 返回 `False`。

---

## 9. 权重文件完整下载路径

### 9.1 Fast-FoundationStereo — 商业版（推荐）

| 项目 | 内容 |
|---|---|
| HuggingFace | https://huggingface.co/nvidia/c-fast-foundationstereo |
| 许可 | NVIDIA Open Model Agreement（允许商业使用） |
| 文件 | `model_best_bp2_serialize.pth` **67.81 MB** + `cfg.yaml` 182 B |
| NGC（同一份） | https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/fast-foundationstereo/v1.2/file-browser |

直链：

```
https://huggingface.co/nvidia/c-fast-foundationstereo/resolve/main/model_best_bp2_serialize.pth
https://huggingface.co/nvidia/c-fast-foundationstereo/resolve/main/cfg.yaml
```

```bash
huggingface-cli download nvidia/c-fast-foundationstereo \
    model_best_bp2_serialize.pth cfg.yaml --local-dir weights/c-fast-foundationstereo
# 国内：export HF_ENDPOINT=https://hf-mirror.com （PowerShell 里用 $env:HF_ENDPOINT=...）
```

**为什么推荐它**：单个 67.81 MB 文件、直链稳定、许可明确，而且**跑推理与研究版没有区别**。

### 9.2 Fast-FoundationStereo — 研究版（多档位）

| 项目 | 内容 |
|---|---|
| 下载页 | https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link |
| 放置 | 整个文件夹放 `weights/`，如 `./weights/23-36-37/model_best_bp2_serialize.pth` |

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap" -O weights/
```

**档位全表**（RTX 3090，640×480）：

| 检查点 | valid_iters | PyTorch (ms) | TensorRT (ms) | 峰值显存 (MB) |
|---|---|---|---|---|
| `23-36-37` | 8 | 49.4 | 23.4 | 653 |
| `23-36-37` | 4 | 41.1 | 18.4 | 653 |
| `20-26-39` | 8 | 43.6 | 19.4 | 651 |
| `20-26-39` | 4 | 37.5 | 16.4 | 651 |
| `20-30-48` | 8 | 38.4 | 16.6 | 646 |
| `20-30-48` | 4 | **29.3** | **14.0** | 646 |

你这台机器先按 `valid_iters 4` 跑，够快再试 8。

### 9.3 FoundationStereo（父模型，离线精度上限）

想看「这个技术路线的深度质量天花板」时用。慢很多，只适合离线对照。

| 检查点 | 说明 | 位置 |
|---|---|---|
| `23-51-11` | ViT-Large，**最佳** | Google Drive https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing → `pretrained_models/23-51-11/model_best_bp2.pth` |
| `11-33-40` | ViT-Small，更快 | 同上 → `pretrained_models/11-33-40/model_best_bp2.pth` |
| TAO 商业版 | 基于 ViT-Small | https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/foundationstereo |

> 注意：**父模型的权重文件名是 `model_best_bp2.pth`，没有 `_serialize` 后缀**，放 `pretrained_models/` 而不是 `weights/`。别和 FFS 弄混。
> 社区镜像（Google Drive 下不动时）：`huggingface-cli download vitaebin/foundation-stereo-model --local-dir pretrained_models`

### 9.4 两个安全提醒

1. `.pth` 是 **pickle 的整个模型对象**（不是 state_dict），加载走 `torch.load(..., weights_only=False)`，等于执行一段 pickle 代码。**只从上面这些官方地址下载。**
2. **`cfg.yaml` 和 `.pth` 必须配套**，不同检查点不能混用（`max_disp`、`volume_dim`、`hidden_dims` 等对不上会静默出错）。

---

## 10. GTX 1650 性能预期与降档方案

**先给预期，免得跑出来失望。**

| 对比项 | RTX 3090 | GTX 1650（笔记本） | 倍数 |
|---|---|---|---|
| FP16 算力 | ~35.6 TFLOPS | ~5.9 TFLOPS | ~6× |
| 显存带宽 | 936 GB/s | ~128 GB/s | **7×** |

立体匹配的代价体构建是**显存带宽敏感**的，所以实际差距更接近带宽比。经验估计（640×480）：

| 配置 | 预期耗时 | 预期帧率 |
|---|---|---|
| `valid_iters 8` | 300–600 ms | 2–3 FPS |
| `valid_iters 4` | 150–300 ms | 3–6 FPS |
| `--scale 0.5` + `valid_iters 4`（320×240） | 50–120 ms | 8–20 FPS |

**这个速度对你的场景够不够？够。** 你的抓取是「停稳 → 拍照 → 算 → 抓」，一次只要 1–3 帧。**2 FPS 都能跑**，不要被官方的 47 FPS 唬住——那是给移动机器人实时避障的。

**降档优先级**（按效果排序）：

1. `--scale 0.5` —— 代价体像素数变 1/4，提速最明显
2. `--valid_iters 4` —— 迭代减半，线性提速
3. `--max_disp` 从 192 往下调（按 `d_max = f·B/z_min` 算真实需要值，别盲用 192）
4. 最后才考虑 TensorRT（那才需要 WSL2/Linux）

**显存**：640×480 + max_disp 192 峰值约 650 MB，4 GB 毫无压力。但**别把分辨率提到 1280×720**——代价体约 3 倍增长，接近 2 GB，加上 PyTorch 开销会比较紧张。官方也明确说「模型在宽度 < 1000 时表现更好」。

**第一次运行会慢**（官方提示：首次有编译开销）。测速前先 warmup。

---

## 11. 决策清单与下一步

按顺序勾：

- [ ] **（5 分钟，零安装）** HF Space 在线试玩：https://huggingface.co/spaces/hugging-apps/fast-foundationstereo —— 传两张图先看效果
- [ ] **（30 分钟）** Windows 建 venv，装 `torch==2.6.0+cu124`，验证 `torch.cuda.is_available()` 为 True、capability 为 `(7, 5)`
- [ ] **（10 分钟）** 下商业版权重（67.81 MB），确认 `cfg.yaml` 和 `.pth` 同目录
- [ ] **（30 分钟）** 用 `demo_data/` 跑通 `--get_pc 0`，看到 `disp_vis.png`
- [ ] **（半天）** 写自己的 `infer_depth.py`（第 4 节九步），去掉弹窗、去掉 open3d、能批量
- [ ] **（半天）** 下 Middlebury 2014 几组数据，跑定量评估（EPE / bad1.0 / bad2.0）
- [ ] **（1 天）** 跑弱纹理场景对比：FFS vs OpenCV SGBM，看优势体现在哪
- [ ] **根据结果决定**：继续（上 TensorRT / 接相机）还是到此为止

**下一步我可以帮你做的**（说一声就行）：

1. 写那个 `infer_depth.py`（含批量、warmup、计时、无效值 mask、视差彩色图）
2. 写 Middlebury 评估脚本（读 PFM 真值、算 EPE / bad1.0 / bad2.0、出对比表）
3. 写 FFS vs SGBM 的对比脚本（同一组图，两种算法，指标 + 并排可视化）
4. 写平面靶标 RMS 脚本（将来有相机时用，现在可以先备着）

---

## 12. 范围扩展：加入 YOLO 与位姿模型后

### 12.0 结论

后续要上「YOLO 粗定位 + 精定位模型出 6D 位姿」的话，**推荐装 WSL2 作为模型开发主环境**。

但**不用现在就装，也别把 Windows 这套拆掉**。当前阶段（深度图评估）继续在 Windows 跑完，等你第一次真正需要编译 CUDA 扩展时再迁——那时 Ubuntu 22.04 + 一堆 pip 包装好也就 1 小时。

理由一句话：**决定 OS 的不是"Windows 能不能跑 YOLO"（能，而且很顺），而是你想要的 SOTA 位姿模型几乎都依赖自定义 CUDA 扩展，而这些扩展在 Windows 上没有匹配的预编译包。**

> **本节的结论已两次修订，最新以 `定位模型选型-CAD驱动双路线-训练式与免训练式.md` 第 8 节为准。**
>
> 需求最终澄清为：**两条路线都需要 CAD**，一条用 CAD 造合成数据训练后快速出位姿，另一条有 CAD 但不训练、直接出位姿（耗时较长）；两条都以检测器提供 ROI 为共同前置；物体不贴 AprilTag。
>
> 据此结论**再次收窄**：
> - **路线 A（训练式）在 Windows 上全链路可行**（BlenderProc 跨平台、YOLO11/YOLOE、Open3D ICP 全有 Windows 支持）→ 不用装 WSL2。
> - 用户随后**再次纠正：两条路线都必须用神经网络模型，不能用几何算法** → OpenCV 的 PPF + ICP（纯几何）已从推荐中移除，仅作退化场景兜底。路线 B 改为模型方案后：
>   - **选 FoundPose（Meta，ECCV 2024）→ Windows 可行**。它基于冻结的 DINOv2，**官方明确"training-free、不发布任何权重"**，只要 CAD + 一张掩膜就能出位姿；纯 PyTorch，不编译 CUDA 扩展。速度约 1.3–1.7 s（V100 级），本机粗估 3–8 s。
>   - **选 FoundationPose / MegaPose / GigaPose / SAM-6D → 需 WSL2**（要编译 nvdiffrast / PyTorch3D）。
> - 结论：**先走 FoundPose 那条就能一直留在 Windows**；只有追求 SOTA 精度与强遮挡鲁棒性时才装 WSL2。

> 需求最终澄清为：**两条路线都需要 CAD 且都用神经网络模型**，一条用 CAD 造合成数据训练后快速出位姿，另一条有 CAD 但不训练、用现成模型直接出位姿（耗时较长）；两条都以检测器提供 ROI 为共同前置；物体不贴 AprilTag。

---

### 12.1 三档划分：卡你的是依赖，不是模型

| 档位 | 代表模型 | Windows 现状 |
|---|---|---|
| **A 档：纯 pip 友好** | YOLOv8/11/12、YOLO-seg、YOLO-pose、RF-DETR、DINOv2、DepthAnything v2、SuperPoint+LightGlue、FoundPose、OpenCV `solvePnP`、Open3D ICP、Fast-FoundationStereo | 官方 wheel 齐全，CUDA 直接可用。**完全没问题** |
| **B 档：要源码编译 CUDA 扩展** | FoundationPose、MegaPose、SAM-6D、BundleSDF | 见下，是一场 MSVC 版本配对战 |
| **C 档：Linux 专属** | `torch.compile` / Triton 后端、TAO、DeepStream | Windows 上基本残废 |

#### B 档具体卡在哪（FoundationPose 是最典型的例子）

它的 README 里就要求装 **PyTorch3D + nvdiffrast**，两个都要从源码编译：

1. **PyTorch3D 官方预编译 wheel 只到 `py39_cu118_pyt200`**（`dl.fbaipublicfiles.com` 那个 index）。你要用的是 torch 2.6 + cu124，**没有对应 wheel**，只能 `pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git`。
2. Windows 下编译 PyTorch3D 的官方 INSTALL.md 原话：要在 **"x64 Native Tools Command Prompt for VS 2019"** 里跑，而且 *"Depending on the version of PyTorch, changes to some PyTorch headers may be needed before compilation"*——**要手工改 torch 的头文件**。
3. nvdiffrast 在 Windows 上要设 `CUDA_HOME`、`DISTUTILS_USE_SDK=1`、把 `CUDA_HOME\bin` 和 `lib\x64` 加进 PATH，社区常见还要改 `setup.py`。
4. FoundationPose 官方 README 专门挂了一条 *"For setting up on Windows, refer to this"* 的 issue 链接——**说明 Windows 是例外路径，不是常规路径**。

同样的事在 Linux 上是这两行：

```bash
export CUDA_HOME=/usr/local/cuda
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
```

#### C 档：torch.compile 在 Windows 是残的

PyTorch 官方 Windows wheel **不链接 Triton**，`torch._inductor` 基本不可用；`torch.compile` 会静默 fallback 到无优化。第三方 `triton-windows` 能装，但原仓库已归档（迁到 `triton-lang/triton-windows`），维护状态弱。

#### 顺带纠正我上一版的一个说法

我说过「ROS2 on Windows 生态残缺」——**这话偏保守了，更正一下**：按 REP-2000，ROS 2 Jazzy 的 **Windows 10 (VS2019) amd64 确实是 Tier 1**。真正的坑在：只有二进制归档、没有 deb 包，而且**第三方包**（realsense-ros、相机驱动、MoveIt 相关）普遍 Linux 优先。所以"ROS2 核心能跑、周边要自己扛"才是准确表述。

---

### 12.2 但先别急着装——你很可能不需要 FoundationPose

这是本节最实用的一段。你的目标是「输出物体位姿」，而**已知 CAD 的刚体件，最靠谱的方案根本不用神经网络**。按「速度↑ / 依赖复杂度↓」排个阶梯：

| 档 | 方案 | 速度 | 需要什么 | Windows |
|---|---|---|---|---|
| ① | **YOLO-pose / 关键点 → `cv2.solvePnP`** | 毫秒级 | CAD 上人工标几个特征点（角点、孔位） | 完美 |
| ② | **YOLO-seg 掩膜 + 深度 → 裁剪点云 → Open3D ICP** | 10–50 ms | CAD 网格（.ply/.stl） | 完美 |
| ③ | **FoundPose**（DINOv2 特征 + CAD 模板） | 百毫秒级 | CAD 网格，免训练 | 能跑 |
| ④ | **FoundationPose** | 秒级 | CAD 或重建 mesh + B 档依赖 | 要 Linux |

工程上的判断：

- **①② 是工业界的主力**，因为它们快、可控、可解释。你工程里 `vision_utils.py` 已经有平面拟合和点云管线，② 几乎是现成的。
- **FoundationPose 的真正价值是"新物体零样本、无 CAD、严重遮挡"**。如果你每个物体都有 CAD，且视角不恶劣，④ 相对② 的精度优势未必值得那个复杂度。
- **你已经在用 AprilTag 了**——对自建工装/夹具，AprilTag 位姿精度（常 <1 mm）高于上面任何一个方案。视觉位姿是给"没法贴 tag 的物体"准备的，别为了上模型而上模型。

所以我的建议路径是：**先在 Windows 上花一天试①或②**。够用的话，这台机器永远不用装 WSL2。

---

### 12.3 硬件才是真瓶颈（跟选哪个 OS 无关）

| 项目 | 现状 | 对位姿任务的影响 |
|---|---|---|
| 显存 4 GB | GTX 1650 | YOLOv8n/11n 训练和推理都没问题（imgsz 640、batch 4–8、开 AMP）。**FoundationPose 会很吃力**——它要渲染上百个姿态假设再过 refine 网络，大概率要降分辨率或跑不动，别指望实时 |
| 内存 8 GB | 焊死级别的痛点 | 「训练 YOLO + 开 VS Code + 浏览器」一定会爆。**无论 Windows 还是 WSL2** |
| CPU 4 核 8 线程 | i5-9300H | 次要瓶颈 |

两个高性价比动作（按推荐度排序）：

1. **加内存到 16 GB**（DDR4 SO-DIMM，约 100–200 元）。这是这台机器**唯一能显著改善体验的升级**，而且立刻让 WSL2 从"勉强"变成"舒服"。笔记本 CPU 是 BGA 焊死的换不了，但内存通常是插槽可换的。
2. **训练上云**：AutoDL / 恒源云 / 揽睿星舟租 3090 24 GB，约 1.5–3 元/小时。本地只做推理和调试。比琢磨怎么在 4 GB 显存上塞下训练现实得多。

---

### 12.4 推荐的目标架构（三层）

| 层 | 跑什么 | 为什么放这层 |
|---|---|---|
| **Windows** | 相机采集（`pyrealsense2` 有官方 Windows wheel）、数据标注、可视化、文档 | 相机在 WSL2 里是坑（默认内核不带 UVC/V4L2，RealSense 官方不支持） |
| **WSL2 (Ubuntu 22.04)** | 模型训练与推理、需要编译的扩展、TensorRT 导出 | CUDA 透传成熟，Linux wheel 齐全 |
| **机器人主控 (Linux)** | ROS2、实时控制、抓取执行 | 机器人侧本来就是 Linux |

几个实操约定：

- **代码和 venv 放 WSL2 的 ext4 里，不要放 `/mnt/c`**。跨区 IO 慢 5–10 倍，训练数据加载会哭。数据集可以放 `/mnt/c` 只读引用，或者干脆各存一份。
- **绝对不要在 WSL2 里装 NVIDIA 驱动**，只装 `cuda-toolkit-12-4`（第 8 节已强调）。
- 别在 WSLg 里跑 rviz（软件渲染卡死），用 Foxglove Studio 在浏览器里看。

---

### 12.5 现在到底该做什么（不要提前优化）

| 阶段 | 做什么 | 环境 |
|---|---|---|
| **一（现在）** | 把 FFS 深度图评估做完 | **Windows**，不要为了"以后可能用"先装 WSL2 |
| **二（决定要出位姿时）** | 先试① `solvePnP` 或② `ICP`，一天出结果 | **Windows**。够用就永远不用装 WSL2 |
| **三（确认要 FoundationPose / 自己写 CUDA 算子 / 上 TAO+TensorRT）** | 装 WSL2 Ubuntu 22.04 | **WSL2**，约 1 小时 |

**迁移成本现在是低的**：Windows 侧只装了 torch + FFS，重搭约 1 小时。越晚迁越贵（数据集、标注、venv 都在 Windows 上）。但反过来，**过早迁也没有任何收益**。所以按触发条件迁，别提前。

真要装时，`C:\Users\x\.wslconfig` 用第 8 节那套（8 GB 内存机器必做）。

---

## 附：相关链接

| 用途 | 链接 |
|---|---|
| FFS 代码仓库 | https://github.com/NVlabs/Fast-FoundationStereo |
| FFS 论文 | https://arxiv.org/abs/2512.11130 |
| 在线试玩（零安装） | https://huggingface.co/spaces/hugging-apps/fast-foundationstereo |
| 商业权重（HF） | https://huggingface.co/nvidia/c-fast-foundationstereo |
| 商业权重（NGC） | https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/fast-foundationstereo/v1.2/file-browser |
| 研究权重（Drive） | https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link |
| FoundationStereo 仓库 | https://github.com/NVlabs/FoundationStereo |
| FoundationStereo 权重 | https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing |
| Middlebury 2014 数据集 | https://vision.middlebury.edu/stereo/data/scenes2014/ |
| ETH3D 数据集 | https://www.eth3d.net/datasets |
| HF 国内镜像 | https://hf-mirror.com |

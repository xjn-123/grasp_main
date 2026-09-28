# Fast-FoundationStereo 生成深度图：完整落地方案

> 这份文档讲清楚一件事：怎么用 NVIDIA 的 Fast-FoundationStereo（下称 FFS）从一对立体图算出深度图，并接进你现有的 ROS2 抓取工程。
>
> 全文按「先判断要不要做 → 准备输入 → 跑通推理 → 加速 → 微调 → 接入工程 → 验证」的顺序组织，每一节都给出可直接执行的命令和可粘贴的代码骨架。
>
> 前置说明：**当前阶段只做「立体图对 → 深度图 → 评估效果」，不涉及 ROS、不涉及 `carm_grasp` 的 `utils.py`、不接相机。** 在这个范围下建议直接在 **Windows 原生**跑（不用装 WSL2），详见 `部署环境评估-WSL2与Windows对比.md`。
>
> 将来若要接 ROS2 节点、做 TensorRT 加速或 TAO 微调，再切到 Linux / WSL2——那份文档第 8 节给了触发条件。

## 目录

- [0. 一句话概括与三条落地路径](#0-一句话概括与三条落地路径)
- [1. 先确认你真的需要它](#1-先确认你真的需要它)
- [2. 输入从哪来：三条硬件路线](#2-输入从哪来三条硬件路线)
- [3. 环境搭建](#3-环境搭建)
- [4. 输入准备（最容易翻车的部分）](#4-输入准备最容易翻车的部分)
- [5. 路径 A：官方仓库直接推理](#5-路径-a官方仓库直接推理)
- [6. 路径 B：ONNX + TensorRT 加速](#6-路径-bonnx--tensorrt-加速)
- [7. 路径 C：TAO 微调与部署](#7-路径-ctao-微调与部署)
- [8. 接进你现有的 ROS2 工程](#8-接进你现有的-ros2-工程)
- [9. 精度验证与验收线](#9-精度验证与验收线)
- [10. 参数调优手册](#10-参数调优手册)
- [11. 坑位清单](#11-坑位清单)
- [12. 时间线与检查清单](#12-时间线与检查清单)
- [13. 一句话总结](#13-一句话总结)

---

## 0. 一句话概括与三条落地路径

**FFS 是什么**：NVIDIA 在 CVPR 2026 发布的实时零样本立体匹配模型。给它一对**已经过立体校正**的左右图和相机内参，它输出一张**视差图**，你再用 `z = f·B/d` 换算成深度。

**它比传统方法强在哪**：不需要在你的场景上重新训练（零样本泛化），在弱纹理墙面、透明玻璃、反光金属、强曝光这些 SGBM 会崩的地方明显更稳。速度在 RTX 3090 上约 49 ms（TensorRT 后约 21 ms）。

**三条落地路径**，按投入从小到大：

| 路径 | 做什么 | 投入 | 适合 |
|---|---|---|---|
| **A. 官方仓库直接推理** | clone 仓库、下权重、跑 `run_demo.py` | 半天 | 先验证效果，做离线对比 |
| **B. ONNX + TensorRT** | 导出 ONNX，`trtexec` 转引擎，跑 TRT 推理 | 1–2 天 | 要跑进实时链路 |
| **C. TAO 微调 + 部署** | 用自己场景的数据微调，再导出部署 | 1–2 周 | 零样本效果仍不达标，或要压到更小模型 |

**给你的建议**：先走 A，用你现有的 RealSense 红外流跑通并量化效果；效果达标再走 B；只有在 A、B 都不达标时才走 C。

---

## 1. 先确认你真的需要它

在做任何安装之前，先花半天做完这件事——它能直接告诉你后面两周要不要花。

### 1.1 建立基准（必做）

拿一块平整的板子（亚克力板贴一张有纹理的纸，或直接用你标定用的 AprilTag 板），垂直正对相机，放到 **0.3 / 0.5 / 1.0 / 1.5 m** 四个位置，每个位置采 30 帧，然后：

1. 取深度图中央 100×100 区域，拟合一个平面（你 `vision_utils.py` 里已有现成的平面拟合函数），算点到平面的 **RMS** —— 这就是该距离下的深度噪声
2. 统计**空洞率**（深度为 0 或 NaN 的像素占比）
3. 把板子倾斜 30°、45° 再测一遍
4. 换成反光件、黑色件再测一遍

产出一张四行的表。这张表是所有后续决策的依据。

### 1.2 FFS 能改善什么、不能改善什么

| 症状 | FFS 能否改善 | 说明 |
|---|---|---|
| 弱纹理（白墙、纯色塑料件）深度全是洞 | **明显改善** | 单目先验（DepthAnythingV2 蒸馏而来）能提供几何猜测 |
| 透明、半透明件 | **部分改善** | 论文明确提到蒸馏增强了对半透明材质的鲁棒性，但不是根治 |
| 反光金属件 | **部分改善** | 同上，比 SGBM 好很多，但物理上仍受限 |
| 远距离精度不足 | **不改善** | 这是 `Δz = z²·Δd/(f·B)` 的物理限制，换算法救不了，要换基线或焦距 |
| 标定不准导致的系统性偏差 | **不改善** | 输入图没校正好，再好的网络也白搭 |
| 运动模糊 / 卷帘快门变形 | **不改善** | 硬件问题，换全局快门 |

**判断标准**：如果你的基准测试显示深度误差主要来自「弱纹理和反光」，FFS 值得上。如果主要来自「距离太远」或「标定不准」，先解决那两个。

### 1.3 硬件门槛检查

```bash
# 必须要有 NVIDIA GPU，且驱动正常
nvidia-smi

# 记录这几项：
# - GPU 型号
# - 驱动版本（FFS 用 CUDA 12.4，驱动需 >= 550）
# - 显存
```

**显存要求（官方 README 实测表，640×480）：峰值仅 646–653 MB。** 也就是说 4 GB 显存的 GTX 1650 都能跑推理，不需要 3060/3090 那种大显存。真正的门槛是**算力和显存带宽**，不是容量——它会决定帧率，但不会让你跑不起来。

**架构要求**：FFS 用的是 `torch.float16`（`Utils.py` 里 `AMP_DTYPE = torch.float16`），**不是 bf16**，所以 Turing（GTX 16 系 / RTX 20 系）及以后都能跑。想用 TensorRT fp16 加速需要 Turing 架构以上。

> 显存和架构都达标之后，唯一需要担心的是**速度**。详见 `部署环境评估-WSL2与Windows对比.md` 第 10 节的降档方案（`--scale 0.5` → `--valid_iters 4` → 换快检查点 → 降 `max_disp` → TensorRT）。

---

## 2. 输入从哪来：三条硬件路线

### 2.1 路线 α：继续用 RealSense，只换匹配算法（强烈推荐先试）

**这是我在调研中发现的、对你最有价值的一条路。**

RealSense D4xx 系列本质上就是一台**主动红外双目相机**：两个红外成像器 + 一个红外散斑投影器，机内 ASIC 跑立体匹配出深度。而 ROS 驱动把**校正去畸变后的左右红外图**也以话题形式发布了出来：

```
/camera/camera/infra1/image_rect_raw     # 左红外图，已去畸变 + 已校正
/camera/camera/infra2/image_rect_raw     # 右红外图，已去畸变 + 已校正
/camera/camera/infra1/camera_info        # 左相机内参
```

这两个流**正好就是 FFS 需要的输入格式**——已校正、已去畸变、极线水平对齐。也就是说：

> 你不用买任何新硬件，就能把 RealSense 内置的立体匹配算法换成 FFS，看看能不能拿到更好的深度。

而且 FFS 官方明确说明支持「RGB、单色或 IR 立体图（如 RealSense D4XX）」。

**几个实用细节**：

- **散斑投影器可以开关**（`rs2` 的 `emitter_enabled`，ROS 里对应 `stereo_module.emitter_enabled` 参数）。开着时弱纹理场景有人工纹理，效果好；**室外或强光下建议关掉**，因为阳光会淹没散斑反而引入噪声。
- RealSense 的红外流默认是 1280×800，FFS 在**宽度 < 1000** 时表现更好，所以要降采样或用 `--scale 0.5`，或者把流配置成 640×480。
- IR 图是单通道灰度，FFS 支持单色输入。

**收益**：零硬件成本验证 FFS 对你场景的价值。
**代价**：推理在主机 GPU 上跑，延迟从「硬件几乎零延迟」变成 20–50 ms。对抓取这种非极端实时的场景通常够用。

### 2.2 路线 β：自搭双目

如果路线 α 验证有效，且你需要更高精度 / 更长基线 / 全局快门，再考虑自搭：

| 要求 | 为什么 | 不做的后果 |
|---|---|---|
| 全局快门 | 卷帘快门在机械臂运动时图像变形，极线约束失效 | 动起来深度全错 |
| 硬件同步触发 | 左右相机自由曝光 → 亮度/增益不一致 → 代价度量失效 | 深度图全是噪点，最常见的翻车点 |
| 刚性支架（铝/碳纤维） | 基线变化 0.1 mm 就能毁掉标定 | 温度一变精度就飘 |

基线选择公式（从 `Δz = z²·Δd/(f·B)` 反解）：

```
B = z² · Δd / (f · Δz)
```

例：目标 1 m 处 1 mm 精度，f = 1000 px，亚像素精度 Δd ≈ 0.3 px：

```
B = 1.0² × 0.3 / (1000 × 0.001) = 0.3 m
```

**注意这个反直觉的结论**：近距离高精度靠的是**短基线 + 大焦距 + 靠近**，不是长基线。D405（基线 18 mm）在 0.3 m 处能到 2–3 mm，比 D435（基线 50 mm）好一个数量级，它赢在「近」。

对你的工作距离（0.3–1.5 m），建议基线 60–120 mm，焦距尽量长。

### 2.3 路线 γ：离线处理已有数据

如果你已经录了 rosbag 或者存了图像序列，直接抽帧成 PNG 批量跑 FFS，用来做离线对比和评估。这条路线没有实时要求，最容易跑通，适合作为第一步。

---

## 3. 环境搭建

### 3.1 Docker 路线（推荐）

仓库自带 Dockerfile，能避开大部分依赖地狱：

```bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git
cd Fast-FoundationStereo

docker build --network host -t ffs -f docker/dockerfile .
bash docker/run_container.sh
```

需要主机已安装 NVIDIA Container Toolkit（`nvidia-docker2`）。用 `docker run --gpus all` 验证能否在容器里看到 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 3.2 conda 路线

```bash
conda create -n ffs python=3.12 && conda activate ffs
pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

版本是**硬约束**：Python 3.12、torch 2.6.0、torchvision 0.21.0、CUDA 12.4。换版本可能踩到 xformers 编译或算子不匹配的问题。

### 3.3 权重获取

> 完整下载地址、HF 镜像、NGC 目录、目录结构见 `部署环境评估-WSL2与Windows对比.md` 第 9 节。这里只给速查。

**两个来源，二选一**：

| 用途 | 来源 | 许可 |
|---|---|---|
| **商业使用（推荐）** | HuggingFace `nvidia/c-fast-foundationstereo` | NVIDIA Open Model Agreement |
| **研究 / 评估** | 仓库 README 里的 Google Drive 链接 | 研究用途 |

**下载后放到 `weights/` 目录下**，例如 `./weights/23-36-37/model_best_bp2_serialize.pth`。

**最省事的下载方式**（商业版只有一个 67.81 MB 文件，直链比 Google Drive 稳）：

```bash
pip install -U huggingface-hub
# 国内加这句：export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download nvidia/c-fast-foundationstereo \
    model_best_bp2_serialize.pth cfg.yaml \
    --local-dir weights/c-fast-foundationstereo
```

**直链**：

```
https://huggingface.co/nvidia/c-fast-foundationstereo/resolve/main/model_best_bp2_serialize.pth   (67.81 MB)
https://huggingface.co/nvidia/c-fast-foundationstereo/resolve/main/cfg.yaml                        (182 B)
```

**研究版（Google Drive，多档位）**：

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap" -O weights/
```

**检查点档位全表**（RTX 3090，640×480；峰值显存均在 650 MB 左右）：

| 检查点 | valid_iters | PyTorch (ms) | TensorRT (ms) | 峰值显存 (MB) |
|---|---|---|---|---|
| `23-36-37` | 8 | 49.4 | 23.4 | 653 |
| `23-36-37` | 4 | 41.1 | 18.4 | 653 |
| `20-26-39` | 8 | 43.6 | 19.4 | 651 |
| `20-26-39` | 4 | 37.5 | 16.4 | 651 |
| `20-30-48` | 8 | 38.4 | 16.6 | 646 |
| `20-30-48` | 4 | 29.3 | 14.0 | 646 |

`valid_iters` 越小越快、精度略降。算力充裕时先跑 `23-36-37`（精度优先）；算力紧张（如 GTX 1650）直接上 `20-30-48` + `valid_iters 4`。

**两个提醒**：

- `cfg.yaml` 和 `.pth` **必须配套**，不同检查点的配置不能混用。
- 这些 `.pth` 是 pickle 的整个模型对象（不是 state_dict），加载走 `torch.load(..., weights_only=False)`。只从官方地址下载。

### 3.4 环境自检清单

```bash
nvidia-smi                                  # 能看到 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 应为 2.6.0 True
python -c "import cv2, numpy, open3d; print('ok')"
ls weights/                                 # 权重已就位
python scripts/run_demo.py --help           # 脚本能跑
```

---

## 4. 输入准备（最容易翻车的部分）

这一节是整份文档最重要的部分。**绝大多数「FFS 效果不好」的反馈，最后查出来都是输入没准备好。**

### 4.1 铁律：输入必须已校正且已去畸变

FFS 假设左右图的极线是**水平对齐**的——即空间中同一个点在左右图中的 y 坐标相同，只在 x 方向有偏移（视差）。

如果你喂进去的是原始未校正的图像，网络会在错误的几何假设下做匹配，输出完全没有意义。

```
错误做法：  相机原始图 → FFS → 垃圾
正确做法：  相机原始图 → 立体校正（rectify）→ FFS → 视差图 → 深度
```

**怎么判断你的图是否已校正**：找一对图中同一个明显特征点（比如 AprilTag 的某个角点），看它的 y 坐标是否相同。差超过 1 px 就说明没校正好。

如果你用的是 RealSense 的 `infra1/image_rect_raw` 和 `infra2/image_rect_raw`，**它们已经是校正好的**，这一步跳过。

### 4.2 K.txt 文件格式

FFS 需要一个 `--intrinsic_file`，格式是两行文本：

```
fx 0 cx 0 fy cy 0 0 1
baseline_in_meters
```

- **第 1 行**：3×3 内参矩阵按行展平成 9 个数，空格分隔
- **第 2 行**：左右相机的基线距离，**单位是米**

示例（假设 fx=fy=600, cx=320, cy=240, 基线 0.05 m）：

```
600.0 0.0 320.0 0.0 600.0 240.0 0.0 0.0 1.0
0.05
```

**注意**：
- 内参必须对应**校正后**的图像分辨率。如果你把图缩放到 640×480 再喂进去，内参也要按同样比例缩放（fx、fy、cx、cy 全部乘以缩放系数）。
- 基线的单位是米，不是毫米。写错会得到一个差 1000 倍的深度。

### 4.3 从 RealSense 的 camera_info 生成 K.txt

在 ROS2 里，基线有两种获取方式：

**方式一：从 TF 读（最可靠）**

```bash
ros2 run tf2_ros tf2_echo camera_infra1_optical_frame camera_infra2_optical_frame
```

输出的 `translation.x` 的绝对值就是基线（米）。

**方式二：从右相机 camera_info 的 P 矩阵算**

```
baseline = -P[3] / P[0]
```

其中 `P` 是 `sensor_msgs/CameraInfo` 里的投影矩阵（12 个数，行优先）。

**自动生成的脚本骨架**：

```python
#!/usr/bin/env python3
"""订阅 RealSense 的 camera_info，生成 FFS 需要的 K.txt。"""
import sys
import numpy as np


def write_k_txt(path, fx, fy, cx, cy, baseline):
    """按 FFS 约定写两行内参文件。"""
    with open(path, "w") as f:
        f.write(f"{fx} 0.0 {cx} 0.0 {fy} {cy} 0.0 0.0 1.0\n")
        f.write(f"{baseline}\n")


def scale_intrinsic(fx, fy, cx, cy, src_wh, dst_wh):
    """图像缩放后，内参要按同样比例缩放。"""
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    return fx * sx, fy * sy, cx * sx, cy * sy


if __name__ == "__main__":
    # 这三个数从 camera_info 的 K 矩阵拿：K = [fx 0 cx; 0 fy cy; 0 0 1]
    fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
    baseline = 0.05                      # 从 tf2_echo 拿，单位米
    src_wh, dst_wh = (1280, 800), (640, 400)

    fx, fy, cx, cy = scale_intrinsic(fx, fy, cx, cy, src_wh, dst_wh)
    write_k_txt(sys.argv[1] if len(sys.argv) > 1 else "K.txt",
                fx, fy, cx, cy, baseline)
    print("written:", fx, fy, cx, cy, baseline)
```

### 4.4 从自己的双目标定生成 K.txt

如果你走路线 β 自搭双目，需要先做立体标定。推荐用 **Charuco 板**（棋盘格 + AprilTag 混合），它比纯棋盘格好在**部分遮挡也能用**。你工程里已经在用 AprilTag，工具链现成。

采集要求：

- 15–25 对图，覆盖全部工作距离（0.3 / 0.6 / 1.0 / 1.5 m）
- 板子要有明显的**旋转**（绕 X、Y 各 ±30°），也要有平移
- **不能只在近处拍**，每个距离都要有几对

质量检查两条线：

1. 重投影 RMS < 0.5 px
2. **校正后左右图同一特征点的行坐标差 < 0.5 px** —— 这条直接反映校正质量，很多人只看 RMS 就放过，结果输入几何是歪的

标定后的输出里，校正相机模型的内参矩阵就是 K.txt 第 1 行的内容，基线 = 右相机相对左相机平移向量的模长。

### 4.5 尺寸要求

- **图像宽度 < 1000** 时模型表现更好。1280 宽的图建议降采样，用 `--scale 0.5` 或先 resize
- **高和宽必须是 32 的整数倍**（导出 ONNX 时的硬性要求，推理时也建议遵守）
- 常见可行尺寸：640×480、736×480、960×540（注意 480 和 736 都是 32 的倍数）

### 4.6 max_disp 怎么定

`--max_disp` 是代价体能编码的最大视差，默认 192。它必须覆盖你的**最近工作距离**：

```
d_max = f · B / z_min
```

例：f = 600 px，B = 0.05 m，z_min = 0.3 m：

```
d_max = 600 × 0.05 / 0.3 = 100 px    → max_disp 取 128 够用
```

例：f = 1000 px，B = 0.10 m，z_min = 0.2 m：

```
d_max = 1000 × 0.10 / 0.2 = 500 px   → 需要 max_disp=512，会明显变慢且吃内存
```

**默认 192 在大多数场景够用，只有需要感知 < 0.1 m 的近物时才要加大。** 加大意味着更慢、更耗显存。

### 4.7 图像格式

- 推荐**无损 PNG**，不要用 JPEG（压缩伪影会干扰匹配）
- 支持 RGB、单色（灰度）、IR
- **不要左右互换**——左图必须真的是左相机

---

## 5. 路径 A：官方仓库直接推理

### 5.1 推理命令

```bash
python scripts/run_demo.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir output/ \
  --remove_invisible 0 \
  --denoise_cloud 1 \
  --scale 1 \
  --get_pc 1 \
  --valid_iters 8 \
  --max_disp 192 \
  --zfar 100
```

### 5.2 参数全表

| 参数 | 含义 | 建议 |
|---|---|---|
| `--model_dir` | 权重文件路径 | 见 3.3 |
| `--left_file` / `--right_file` | 左右图路径 | **必须已校正去畸变**，不能互换 |
| `--intrinsic_file` | 内参 + 基线文件 | 见 4.2 |
| `--out_dir` | 输出目录 | |
| `--valid_iters` | 精修迭代次数 | 8（精度优先）或 4（速度优先）。**要和检查点匹配** |
| `--max_disp` | 代价体最大视差 | 默认 192，按 4.6 计算 |
| `--scale` | 图像缩放因子 | 宽度 > 1000 时用 0.5 |
| `--get_pc` | 是否输出点云 | 调试时 1，批量时 0 |
| `--denoise_cloud` | 点云降噪 | 1 |
| `--remove_invisible` | 忽略非重叠区域深度 | 视差图边缘通常不可靠，可设 1 |
| `--zfar` | 点云最大深度（米） | 你的场景设 2 或 3 就够，设 100 会让点云可视化很难看 |

### 5.3 批量处理脚本骨架

```python
#!/usr/bin/env python3
"""批量跑 FFS：从 rosbag 抽帧的 PNG 序列生成深度图。"""
import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_dir", required=True)
    ap.add_argument("--right_dir", required=True)
    ap.add_argument("--intrinsic_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_disp", type=int, default=192)
    ap.add_argument("--valid_iters", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_files = sorted(left_dir.glob("*.png"))
    # 关键：左右图必须按文件名一一对应
    pairs = [(f, right_dir / f.name) for f in left_files if (right_dir / f.name).exists()]
    assert pairs, "左右图没有匹配上，检查命名"

    for i, (lf, rf) in enumerate(pairs):
        sub = out_dir / f"{i:06d}"
        cmd = [
            "python", "scripts/run_demo.py",
            "--model_dir", args.model_dir,
            "--left_file", str(lf),
            "--right_file", str(rf),
            "--intrinsic_file", args.intrinsic_file,
            "--out_dir", str(sub),
            "--get_pc", "0",
            "--denoise_cloud", "0",
            "--scale", str(args.scale),
            "--valid_iters", str(args.valid_iters),
            "--max_disp", str(args.max_disp),
        ]
        subprocess.run(cmd, check=True)
        print(f"[{i+1}/{len(pairs)}] {lf.name}")


if __name__ == "__main__":
    main()
```

### 5.4 视差 → 深度

FFS 输出的是**视差图**（单位：像素）。换算成深度：

```
z = fx · B / d
```

- `z`：深度（米）
- `fx`：内参焦距（像素）
- `B`：基线（米）
- `d`：视差（像素）

**注意**：这里用 `fx` 还是 `fy`？视差是水平方向的位移，所以用 **fx**。

```python
import numpy as np


def disparity_to_depth(disp, fx, baseline, min_disp=1e-6):
    """视差图转深度图。

    disp: (H, W) 视差，单位像素，0 或负值表示无效
    fx: 内参焦距（像素）
    baseline: 基线（米）
    返回: (H, W) 深度（米），无效处为 0
    """
    disp = np.asarray(disp, dtype=np.float64)
    depth = np.zeros_like(disp)
    valid = disp > min_disp
    depth[valid] = fx * baseline / disp[valid]
    return depth


def depth_to_point_cloud(depth, fx, fy, cx, cy):
    """深度图转点云（相机坐标系），返回 (N, 3)。"""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid = z > 0
    x = (u[valid] - cx) * z[valid] / fx
    y = (v[valid] - cy) * z[valid] / fy
    return np.stack([x, y, z[valid]], axis=1)
```

### 5.5 一个常被忽略的细节：视差为 0 的处理

`d → 0` 时 `z → ∞`。这些点通常是**无匹配区**（遮挡、超出视差范围、无纹理）。上面代码用 `min_disp` 把它们过滤成 0（无效），这是标准做法。

如果你的下游代码（比如 `depth_mean_filter`）对 0 值敏感，记得在接入时统一约定：**0 表示无效**。

---

## 6. 路径 B：ONNX + TensorRT 加速

当路径 A 跑通、确认效果达标，再上加速。

### 6.1 导出 ONNX

```bash
python scripts/make_single_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output/ \
  --height 480 \
  --width 640 \
  --valid_iters 8 \
  --max_disp 192
```

**硬约束**：`--height` 和 `--width` 必须是 32 的整数倍，且要和实际推理时喂进去的尺寸**完全一致**。尺寸不匹配会直接报错或输出错乱。

### 6.2 转 TensorRT 引擎

```bash
trtexec --onnx=output/fast_foundationstereo.onnx \
        --saveEngine=output/fast_foundationstereo.engine \
        --fp16
```

- `--fp16` 能带来约 2 倍加速，精度损失通常可忽略
- 想更激进可以试 `--int8`，但需要校准集，且立体匹配对数值精度较敏感，不建议一上来就用
- 引擎文件**与 GPU 型号和 TensorRT 版本绑定**，换机器要重新生成

### 6.3 TRT 推理

```bash
python scripts/run_demo_single_trt.py \
  --model_dir output/ \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir output_demo/ \
  --get_pc 1 \
  --remove_invisible 0 \
  --denoise_cloud 1 \
  --zfar 100
```

### 6.4 归一化注意事项

单 ONNX 版本期望**预归一化过的 float32 输入**：

| 项 | 值 |
|---|---|
| 均值 | `[123.675, 116.28, 103.53]` |
| 标准差 | `[58.395, 57.12, 57.375]` |

官方脚本已自动处理。但**如果你要自己写推理封装**（比如塞进 ROS 节点），必须手动做这一步，否则输出会是垃圾。

### 6.5 速度档位选择

| 配置 | PyTorch | TensorRT fp16 | 适用场景 |
|---|---|---|---|
| `23-36-37` / iters=8 | 49.4 ms（~20 FPS） | 23.4 ms（~43 FPS） | 精度优先，推荐默认 |
| `20-30-48` / iters=4 | 29.3 ms（~34 FPS） | 14.0 ms（~71 FPS） | 需要更高帧率 |

对你的抓取场景（不需要 30 FPS 深度），**`23-36-37` + TRT 的 43 FPS 绰绰有余**，建议优先保精度。

---

## 7. 路径 C：TAO 微调与部署

### 7.1 什么时候需要微调

满足以下任一条件才考虑：

- 零样本在你的场景上明显不达标（基准测试对比后确认）
- 你的场景极端特殊（强反光产线、水下、内窥镜等）
- 需要压到更小的模型以适配边缘设备

**不要一上来就微调**——FFS 最大的卖点就是零样本，先验证零样本。

### 7.2 数据集准备

FFS 通过纯文本注释文件读取数据，**每行一个样本，空格分隔**：

| 列数 | 格式 | 用途 |
|---|---|---|
| 2 | `<left> <right>` | 无真值推理 |
| **3** | `<left> <right> <disparity>` | **训练与评估（你用这个）** |
| 4 | `<left> <right> <disparity> <occlusion>` | 带遮挡掩码评估，**仅 Middlebury / Eth3d** |

**注意**：4 列格式只支持 `Middlebury` 或 `Eth3d`，其他数据集（含 `GenericDataset`）用 4 列会报错。训练用 3 列。

真值视差从哪来？三个来源：

1. 用 FFS 或 FoundationStereo 自己生成**伪标签**（官方就是这个思路，见 1.4M Stereo4D 伪标签）
2. 用激光雷达 / 结构光设备采集
3. 用高精度的传统方法（如 PatchMatch）在高质量图上生成

### 7.3 完整 YAML 配置

```yaml
results_dir: /data/result

dataset:
  dataset_name: StereoDataset
  max_disparity: 192
  min_depth: 0.0
  train_dataset:
    data_sources:
      - dataset_name: GenericDataset
        data_file: /data/datasets/mine/train.txt
    batch_size: 1
    workers: 4
    augmentation:
      crop_size: [320, 736]
  val_dataset:
    data_sources:
      - dataset_name: GenericDataset
        data_file: /data/datasets/mine/val.txt
    batch_size: 1
    workers: 4
    augmentation:
      crop_size: [320, 736]

model:
  model_type: FastFoundationStereo
  encoder: vitl
  hidden_dims: [128]
  n_gru_layers: 1
  corr_radius: 4
  corr_levels: 2
  n_downsample: 2
  max_disparity: 192
  valid_iters: 8
  train_iters: 22
  volume_dim: 28
  mixed_precision: false
  gwc_feature_normalize: true
  motion_encoder_widths: [56, 96, 16, 12]
  motion_encoder_final: 48
  gru_hidden: 60
  gru_gating_conv_widths: [100, 168]
  disp_head_input_dim: 60
  disp_head_intermediate: 36
  disp_head_pwconv1_widths: [212, 244]
  mask_widths: [32, 16]
  stem_2_widths: [12, 16]
  spx_2_gru_widths: [16, 12, 16, 24]
  spx_gru_out: 9
  classifier_mid: 14
  cnet_conv04_widths: [60, 48]
  cam_mid_channels: 8
  cost_agg_conv_patch_padding: [0, 0, 0]
  stereo_backbone:
    edgenext_pretrained_path: ""
    depth_anything_v2_pretrained_path: ""
    use_bn: false
    use_clstoken: false

train:
  num_gpus: 1
  num_epochs: 20
  precision: fp32
  pretrained_model_path: /data/checkpoints/model.pth
  optim:
    optimizer: AdamW
    lr: 1.0e-5

export:
  checkpoint: /data/checkpoints/model.pth
  onnx_file: /data/checkpoints/model.onnx
  input_height: 480
  input_width: 736
  opset_version: 17
  batch_size: 1
  valid_iters: 8
  format: onnx

gen_trt_engine:
  onnx_file: /data/checkpoints/model.onnx
  trt_engine: /data/checkpoints/model.engine
  batch_size: 1
  tensorrt:
    data_type: fp16
    workspace_size: 4096
    min_batch_size: 1
    opt_batch_size: 1
    max_batch_size: 1
```

### 7.4 必须精确匹配的参数（bp2 检查点陷阱）

**NVIDIA 文档明确警告**：使用 bp2 检查点时，以下 `model` 参数**必须显式设置**。TAO 不会报错，但会套用不同的 schema 默认值，导致**输出错误**：

| 参数 | 必须设为 | 默认值 | 错了会怎样 |
|---|---|---|---|
| `max_disparity` | `192` | `416` | 视差范围不匹配 |
| `gwc_feature_normalize` | `true` | `false` | **约 7–8% 像素出现负视差** |
| `volume_dim` | `28` | `32` | 特征维度不匹配 |
| `hidden_dims` | `[128]` | `[128, 128, 128]` | 结构不匹配 |
| `n_gru_layers` | `1` | `3` | 结构不匹配 |

这一条值得单独标红——它是那种「跑起来看起来正常，但结果悄悄错了」的坑。

### 7.5 五个任务的执行顺序

```bash
# 1. 训练 / 微调
tao model depth_net train -e spec.yaml -k $KEY

# 2. 评估
tao model depth_net evaluate -e spec.yaml -k $KEY

# 3. 推理（save_raw_pfm: true 保存原始 PFM 视差）
tao model depth_net inference -e spec.yaml -k $KEY

# 4. 导出 ONNX
tao model depth_net export -e spec.yaml -k $KEY

# 5. 生成 TensorRT 引擎
tao deploy depth_net gen_trt_engine -e spec.yaml -k $KEY
```

---

## 8. 接进你现有的 ROS2 工程

这是本方案最省事的地方：**接口不变，只换深度来源**。

### 8.1 改动点全景

你的 `core/cam_ros_utils.py` 里的 `CamNode` 现在订阅 color + depth 两个话题并做时间同步，输出 `(color_img, depth_img)`。改成自研双目后：

```
现在：  CamNode → [color, depth] → vision_utils / arm_utils
改后：  CamNode → [color, left_ir, right_ir] → FFS → depth → 下游（全部不动）
```

下游这些模块**一行都不用改**：

- `vision_utils.depth_mean_filter`
- `vision_utils.compute_tag_corners3d`
- `arm_utils.CollisionDetector`
- `test_tmpl_grasp_2d.py` / `test_tmpl_grasp_3d.py`

因为它们只依赖「一张和彩色图对齐的深度图 + 一个 depth_scale」。

### 8.2 关于 depth_scale 的约定

你工程里 `read_rgbd_params` 会返回一个 `depth_scale`，深度的真实米数 = 原始值 × depth_scale。

- RealSense 内置深度：原始值是 uint16 毫米，depth_scale = 0.001
- **FFS 输出**：建议直接在节点里换算成**米制的 float32**，然后把 depth_scale 设为 **1.0**

这样下游公式不用改，语义也最清晰（避免「到底是毫米还是米」的经典混淆）。

### 8.3 节点骨架

```python
#!/usr/bin/env python3
"""FFS 深度节点骨架：订阅左右校正图，推理，发布米制深度图。

注意：这是骨架，需要按你实际的推理后端（PyTorch / TRT）补全 infer() 函数。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np


class FfsDepthNode(Node):
    def __init__(self):
        super().__init__("ffs_depth_node")
        self.bridge = CvBridge()

        # 左右校正图，用近似时间同步
        self.left = None
        self.right = None
        self.stamp_left = None
        self.stamp_right = None

        self.fx = self.fy = self.cx = self.cy = None
        self.baseline = None

        self.create_subscription(Image, "/camera/camera/infra1/image_rect_raw",
                                 self.cb_left, 10)
        self.create_subscription(Image, "/camera/camera/infra2/image_rect_raw",
                                 self.cb_right, 10)
        self.create_subscription(CameraInfo, "/camera/camera/infra1/camera_info",
                                 self.cb_info, 10)

        self.pub_depth = self.create_publisher(Image, "/stereo/depth", 10)
        self.pub_info = self.create_publisher(CameraInfo, "/stereo/depth_camera_info", 10)

        self.create_timer(0.05, self.tick)   # 20 Hz

    def cb_info(self, msg):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.info_msg = msg

    def cb_left(self, msg):
        self.left = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_left = msg.header.stamp

    def cb_right(self, msg):
        self.right = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.stamp_right = msg.header.stamp

    def tick(self):
        if self.left is None or self.right is None:
            return
        if self.fx is None or self.baseline is None:
            self.get_logger().warn("等待内参与基线")
            return

        # 时间戳配对检查，跨度过大就丢弃
        dt = abs((self.stamp_left.sec + self.stamp_left.nanosec * 1e-9)
                 - (self.stamp_right.sec + self.stamp_right.nanosec * 1e-9))
        if dt > 0.02:
            self.get_logger().warn(f"左右图时间差过大: {dt:.4f}s")
            return

        disp = self.infer(self.left, self.right)      # 待补全：调 FFS
        depth = self.disparity_to_depth(disp)          # 米制 float32

        out = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
        out.header.stamp = self.stamp_left
        out.header.frame_id = self.info_msg.header.frame_id
        self.pub_depth.publish(out)

        self.info_msg.header.stamp = self.stamp_left
        self.pub_info.publish(self.info_msg)

    def disparity_to_depth(self, disp):
        depth = np.zeros(disp.shape, dtype=np.float32)
        valid = disp > 1e-6
        depth[valid] = self.fx * self.baseline / disp[valid]
        return depth

    def infer(self, left, right):
        # TODO: 接入 FFS 推理（PyTorch 直接调用，或 TRT 引擎）
        # 注意：必须自己做归一化（mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]）
        # 注意：输入尺寸必须是导出 ONNX 时的 height/width，且为 32 的倍数
        raise NotImplementedError


def main():
    rclpy.init()
    node = FfsDepthNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

### 8.4 时间戳与同步

- RealSense 的 infra1 / infra2 是**同一时钟硬件触发**的，天然同步，时间差通常在微秒级
- 如果你自搭双目用了软件采集，务必检查时间差，超过 20 ms 在运动场景下会产生不可忽略的误差
- 发布深度图时**沿用左图的时间戳**，这样下游做 TF 变换时不会报「TF 旧数据」

### 8.5 基线的获取（ROS2 里最省事的做法）

用 TF 静态变换：

```bash
ros2 run tf2_ros tf2_echo camera_infra1_optical_frame camera_infra2_optical_frame
```

在节点里等价的做法是用 `tf2_ros.Buffer` 查一次变换，取 `transform.translation.x` 的绝对值。这样基线会随相机型号自动正确，不用硬编码。

---

## 9. 精度验证与验收线

### 9.1 平面靶标法（推荐）

1. 平整板子正对相机，放到 0.3 / 0.5 / 1.0 / 1.5 m
2. 每个位置采 30 帧深度图
3. 取中央 100×100 区域，拟合平面，算 RMS
4. 统计空洞率
5. 倾斜 30°、45° 重复
6. 换反光件、黑色件重复

### 9.2 与 RealSense 自带深度对比

同一时刻同时订阅：

- RealSense 内置深度：`/camera/camera/depth/image_rect_raw`
- FFS 深度：`/stereo/depth`

对同一块靶标区域同时采样，直接对比 RMS。**这是最有说服力的对比**，因为硬件、标定、场景完全相同，唯一变量就是匹配算法。

### 9.3 建议验收线

| 指标 | 目标 | 说明 |
|---|---|---|
| 标定重投影 RMS | < 0.5 px | 输入几何正确性的前提 |
| 校正后行坐标差 | < 0.5 px | 同上，常被忽略 |
| 0.5 m 处深度 RMS | < 2 mm | 抓取场景的合理目标 |
| 1.0 m 处深度 RMS | < 8 mm | |
| 1.5 m 处深度 RMS | < 20 mm | 物理极限附近 |
| 空洞率（平面靶标） | < 5% | |
| 端到端抓取成功率 | 不降低，最好提升 | 最终判据 |

**重要提醒**：最终判据是**抓取成功率**，不是深度 RMS。深度变好但抓取没变好是完全可能的（说明瓶颈不在深度）。

---

## 10. 参数调优手册

| 现象 | 优先调的参数 | 方向 |
|---|---|---|
| 近处物体深度缺失/错误 | `--max_disp` | 调大，覆盖 `d_max = f·B/z_min` |
| 显存溢出 | `--max_disp`、`--scale` | 调小 max_disp，或 scale 降采样 |
| 太慢 | `--valid_iters`、换 `20-30-48` 检查点 | iters 8 → 4 |
| 宽度 > 1000 的图效果差 | `--scale` | 设 0.5 |
| 视差图边缘大面积错乱 | `--remove_invisible` | 设 1 |
| 点云噪点多 | `--denoise_cloud` | 设 1 |
| 大量负视差（TAO 路径） | `gwc_feature_normalize` | 必须为 `true` |
| 深度整体差一个常数倍 | 检查 K.txt 基线单位 | 必须是米 |
| 深度和彩色对不上 | 检查内参是否按缩放比例同步缩放 | 见 4.2 |

---

## 11. 坑位清单

| 现象 | 根因 | 改法 |
|---|---|---|
| 输出视差图完全是噪声 | 输入图未经立体校正 | 必须喂 `image_rect_raw` 这类已校正流，或先做 rectify |
| 深度差 1000 倍 | K.txt 第二行基线写成了毫米 | 基线单位必须是**米** |
| 图和内参不匹配 | 图像缩放了但内参没同步缩放 | fx、fy、cx、cy 全部乘以同样的缩放系数 |
| ONNX 导出报错 | height/width 不是 32 的倍数 | 改成 640×480 或 736×480 这类尺寸 |
| TRT 推理输出垃圾 | 自己做封装时忘了归一化 | 减均值 `[123.675,116.28,103.53]`，除标准差 `[58.395,57.12,57.375]` |
| TAO 输出 7–8% 负视差 | bp2 检查点用了默认 `gwc_feature_normalize: false` | 显式设为 `true`（还有 `max_disparity:192`、`volume_dim:28`、`hidden_dims:[128]`、`n_gru_layers:1`） |
| 换机器后 TRT 引擎加载失败 | 引擎与 GPU 型号 / TRT 版本绑定 | 在目标机上重新 `trtexec` 生成 |
| 室外深度全是洞 | 散斑被阳光淹没 | 关闭红外投影器（`emitter_enabled: false`） |
| 机械臂动起来深度就崩 | 卷帘快门变形 | 换全局快门相机 |
| 温度一变精度就飘 | 支架刚性不足，基线漂移 | 换铝/碳纤维支架，定期重标定 |
| 远处精度不足 | `Δz ∝ z²` 的物理限制 | 换算法救不了，加基线或加焦距 |
| 透明/反光件深度仍错误 | 物理限制 | 改用 AprilTag 定位或力控，别指望深度 |

---

## 12. 时间线与检查清单

### 第 1 天：基准

- [ ] 平面靶标就位，0.3 / 0.5 / 1.0 / 1.5 m 四组数据
- [ ] 算出 RMS 与空洞率，写进文档
- [ ] 判断瓶颈到底是不是深度质量

### 第 2 天：跑通路径 A

- [ ] 环境搭好（Docker 或 conda），`nvidia-smi` 正常
- [ ] 权重下载到 `weights/`
- [ ] 用 RealSense `infra1/infra2/image_rect_raw` 抽几对 PNG
- [ ] 生成正确的 K.txt（内参 + 基线，单位为米）
- [ ] `run_demo.py` 跑通，看到视差图
- [ ] 视差转深度，和 RealSense 自带深度对比

### 第 3–4 天：量化对比

- [ ] 批量跑 30 帧 × 4 个距离
- [ ] 复算基准表，和 RealSense 自带深度逐项对比
- [ ] 决策：效果达标则继续，不达标则回到第 1 天重新定位瓶颈

### 第 5–7 天：路径 B 加速

- [ ] 导出 ONNX（尺寸 32 倍数）
- [ ] `trtexec --fp16` 转引擎
- [ ] TRT 推理跑通，测帧率
- [ ] 确认精度没有明显下降

### 第 2 周：接入 ROS2

- [ ] 写 FFS 深度节点（骨架见 8.3）
- [ ] 发布米制 float32 深度 + 继承 camera_info
- [ ] 确认下游 `depth_mean_filter` / `compute_tag_corners3d` / `CollisionDetector` 无改动可用
- [ ] A/B 抓取测试各 20 次

### 持续

- [ ] 8 小时连续运行，每小时采一次基准，监控温漂
- [ ] 记录每季度的重标定结果

### 可选（仅当零样本不达标）

- [ ] 路径 C：TAO 微调（注意 bp2 参数必须精确匹配）

---

## 13. 一句话总结

**先用你已有的 RealSense 红外校正流（`infra1/infra2/image_rect_raw`）跑通 FFS 的零样本推理，和相机自带深度做同场景对比——零硬件成本、两天出结论；确认有效再上 TensorRT、再接进 ROS2。**

整条链路的核心风险不在模型和代码，而在三件小事：**输入图是否真的校正过、K.txt 的基线单位是不是米、图像缩放后内参有没有同步缩放**。这三处对了，FFS 基本不会让你失望。

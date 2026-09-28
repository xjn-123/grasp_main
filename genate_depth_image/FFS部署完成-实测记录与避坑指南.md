# FFS 部署完成 — 实测记录与避坑指南

> 环境：WSL2 Ubuntu 22.04 + GTX 1650 (4GB, Turing sm_75) + torch 2.6.0+cu124 + Python 3.10
> 完成时间：2026-09-26
> 状态：✅ 已跑通，demo 图深度结果合理（键盘 0.35m / 杯子 0.50m / 背景 1.58m）

---

## 一、最终目录结构

```
/home/xiao/Depth_Learning/FFS/
├── Fast-FoundationStereo/          # 源码（从官方 zip 解压）
│   ├── batch_infer.py              # ★ 无头批量推理（我改过，见下）
│   ├── bench_ffs.py                # ★ 基准测试脚本
│   ├── make_vis.py                 # ★ 生成三联对比图
│   ├── debug_nan*.py               # 排障过程脚本（可删）
│   ├── weights/c-fast/             # ★ 权重（已下载）
│   │   ├── model_best_bp2_serialize.pth   (68 MB)
│   │   └── cfg.yaml
│   ├── demo_data/                  # 官方示例图 + K.txt
│   ├── output/                     # 推理输出
│   └── benchmark_out/benchmark.json
├── venv/                           # Python 3.10 虚拟环境（torch 2.6.0+cu124）
├── bench.sh                        # 一键基准 + 出图
├── run_ffs_demo.sh                 # 一键单图推理
└── dl_weights.sh / probe_net.sh    # 下载与网络探测脚本
```

---

## 二、权重下载（重要：官方 GitHub 权重在 Google Drive，WSL 里连不上）

**实际采用方案**：HuggingFace 镜像 `hf-mirror.com` 上的官方商用版模型，免登录、未设门禁：

```
https://hf-mirror.com/nvidia/c-fast-foundationstereo
```

文件名与代码预期完全一致（`model_best_bp2_serialize.pth` + `cfg.yaml`），直接可用。
体积仅 68MB（bp2 = 结构剪枝版），比 Google Drive 上的研究版 23-36-37（~1GB）小得多。

> 网络实测：WSL 内 `drive.google.com` / `huggingface.co` / `raw.githubusercontent.com` 全部不通，
> `github.com` / `hf-mirror.com` / `pypi.org` 通。以后下 NVIDIA 模型优先试 hf-mirror。

---

## 三、踩过的坑（按排查顺序）

### 坑 1：`Missing key normalize`（ConfigAttributeError）

HF 上 c-fast 的权重和 `cfg.yaml` 是旧版配置格式，**缺少 `normalize` 字段**，而 master 代码的
`build_gwc_volume_optimized_pytorch1()` 会读 `self.args.normalize`。

**修复**：`batch_infer.py` 的 `patch_cfg()` 会自动用 `OmegaConf.update(..., force_add=True)`
补齐缺失字段，`normalize=True`（与代码默认值及旧版硬编码行为一致）。

### 坑 2（最隐蔽）：fp16 全输出 NaN

现象：任何配置下视差图 100% 是 NaN。逐步排查：

| 测试 | 结果 |
|---|---|
| fp16（默认 AMP） | ❌ 全 NaN |
| bf16 | ❌ 全 NaN |
| fp32 | ✅ 正常 |
| fp16 + 关 SDPA flash/mem-efficient 后端 | ❌ 仍全 NaN |
| fp16 + 关闭 fp16 归约 | ❌ 仍全 NaN |
| **fp16 + 关闭 cuDNN** | ✅ **正常** |
| fp32 + cuDNN ON | ✅ 正常 |

用 forward hook 定位到**首个 NaN 出现在 `cnet.conv04.0.conv`（Conv2d）**。
结论：**cuDNN 9.1.0 在 Turing（sm_75）上的 fp16 卷积实现有 bug，会产出 NaN**。

### 坑 3：关 cuDNN 会爆显存

`cudnn.enabled=False` 后 conv3d 走 PyTorch 原生回退，全分辨率要 **4.41 GB > 4GB 显存** → OOM。

### 结论（本卡最优配置）

| 配置 | 稳态耗时 | 显存 | 数值 |
|---|---|---|---|
| **fp32 + cuDNN ON（推荐）** | **0.94 s/帧** | 1267 MB | ✅ 正确 |
| fp32 + cuDNN ON + iters=4 | 0.79 s/帧 | 1267 MB | ✅ 正确 |
| fp32 + cuDNN ON + 半分辨率 | **0.26 s/帧 (3.8fps)** | 411 MB | ✅ 正确 |
| fp16 + cuDNN ON | — | 1055 MB | ❌ 全 NaN |
| fp16 + cuDNN OFF | 4.51 s/帧 | 3123 MB | ✅ 但又慢又吃显存 |

> 反直觉但真实：在这张卡上 **fp32 比「fp16+无cuDNN」快 5 倍**。别碰 fp16。

---

## 四、基准测试结果（fp32，官方 demo 图 960×540）

| 配置 | 稳态耗时 | FPS | 显存 | 有效深度 | 中位深度 |
|---|---|---|---|---|---|
| 全分辨率 iters=8 max_disp=192 | 0.939s | 1.06 | 1267MB | 90.7% | 0.507m |
| 全分辨率 iters=4 | 0.790s | 1.27 | 1267MB | 90.7% | 0.507m |
| 全分辨率 iters=8 max_disp=416 | 1.564s | 0.64 | 2533MB | 90.7% | 0.507m |
| **半分辨率 iters=8（推荐）** | **0.257s** | **3.89** | **411MB** | 90.7% | 0.506m |
| 半分辨率 iters=4 | 0.235s | 4.25 | 391MB | 90.6% | 0.506m |
| 全分辨率 hiera | 1.221s | 0.82 | 1284MB | 90.7% | 0.507m |

- `max_disp=192` 与 `416` 深度结果一致（demo 场景近距），192 更省显存。
- 半分辨率显存只占 411MB，**给后续 YOLO + 定位模型留了充足空间**（单环境方案关键前提）。

---

## 五、日常使用命令

```bash
# 进环境
cd /home/xiao/Depth_Learning/FFS/Fast-FoundationStereo
source ../venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 单对图推理（出 depth_meter.npy + 可视化 png）
python batch_infer.py \
  --model_dir weights/c-fast/model_best_bp2_serialize.pth \
  --left_file 你的左图.png --right_file 你的右图.png \
  --intrinsic_file 你的K.txt --out_dir output/ \
  --precision fp32 --valid_iters 8 --max_disp 192 --scale 1.0

# 批量（通配符）
python batch_infer.py --left_file 'data/left_*.png' --right_file 'data/right_*.png' ...

# 跑基准
bash ../bench.sh

# 小显存/要速度
python batch_infer.py ... --scale 0.5 --valid_iters 4
```

**K.txt 格式**（与你既有约定一致）：第 1 行展平的 3×3 内参，第 2 行基线（**米**）。

---

## 六、对后续「单环境统一方案」的影响

1. **torch 2.6.0+cu124 + py3.10 结论维持有效**，无需改版本。
2. FFS 占用 411MB（半分辨率）~1267MB（全分辨率），4GB 卡上与 YOLO11 + FoundPose 共存可行，
   但**推理时要串行调用、用完即 `torch.cuda.empty_cache()`**。
3. `batch_infer.py` 已是无头版（无 `cv2.imshow` / 无 open3d 依赖），可直接被后续管线 import：
   `from batch_infer import load_model, infer_disp, disp_to_depth, read_k_txt`
4. open3d 是**可选依赖**（`Utils.py` 里 try/except），不做点云可视化就不用装。
   若后续要 `cloud.ply` 输出，用阿里云镜像装：`pip install open3d -i https://mirrors.aliyun.com/pypi/simple/`

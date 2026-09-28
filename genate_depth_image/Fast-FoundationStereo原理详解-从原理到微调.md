# Fast-FoundationStereo 完全解析：从原理到微调

> 面向初学者。读完后你应该能回答：这个模型在算什么、每个模块为什么存在、数据是怎么流动的、以及想让它适配你自己的相机时该走哪条路。
>
> 配套文档：`FFS部署完成-实测记录与避坑指南.md`（环境搭建与实测数据）
> 代码版本：`Fast-FoundationStereo` master（CVPR 2026, NVIDIA）
> 本文所有形状数字均为**本机实测**（输入 960×540，GTX 1650，fp32），非纸面推算。

---

## 写在前面：这份文档怎么读

| 你现在的状态 | 建议读法 |
|---|---|
| 完全没接触过立体匹配 | 第 1 部分逐字读，第 2、3 部分先跳过公式看框图 |
| 知道视差但没读过深度学习立体匹配 | 第 1 部分扫读，第 2、3 部分精读 |
| 只想做微调 | 直接跳第 5 部分，但先看第 2.2 节的参数量分布 |
| 想改网络结构 | 第 3 部分 + 附录 C 源码地图 |

贯穿全文的一个类比：**立体匹配 = 在右图里为左图的每一个像素"找对象"**。
模型做的所有事情，都是为了让这个"找对象"的过程又快又准。

---

# 第 1 部分 先建立直觉

## 1.1 双目视觉怎么算出深度

两个相机水平并排，光心相距 **B**（基线，baseline），焦距都是 **f**。同一个三维点 P 在左右图上分别成像在 xL 和 xR。

```
            左相机          右相机
              O1 ----B----- O2
               \           /
                \         /
                 \       /
                  \     /
                   P (真实物体)

左图像上 P 的横坐标:  xL
右图像上 P 的横坐标:  xR
```

定义 **视差（disparity）** `d = xL - xR`（注意是左减右，所以通常是正数）。

由相似三角形可得那个著名公式：

```
                 f × B
    depth  =   ─────────
                   d
```

**视差越大 → 物体越近；视差为 0 → 无穷远。** 这就是全部了。剩下的问题只有一个：**怎么求出每个像素的 d？**

> 你的 `K.txt` 里第二行的 `0.063` 就是基线 B（米），第一行的 `754.67` 就是焦距 f（像素）。
> 代码里就是这么算的：`depth = K[0,0] * baseline / disp`（`batch_infer.py::disp_to_depth`）。

## 1.2 为什么这件事难

朴素做法：拿左图一个 5×5 的小窗口，在右图同一行上从左到右滑，找最像的那块。这就是传统块匹配（BM/SGBM）。它在真实世界里会崩，原因是：

| 场景 | 为什么会崩 |
|---|---|
| 白墙、纯色桌面 | 窗口里没有纹理，滑到哪都一样，匹配不出唯一解 |
| 反光金属、玻璃 | 左右图看到的内容不一样（镜面反射不满足朗伯假设） |
| 重复纹理（栅栏、瓷砖） | 有多个同样好的匹配，选错 |
| 遮挡 | 左图看得见的点，右图上被挡住了，根本没有对应点 |
| 透明物体 | 看到的是背后的东西 |

传统算法只能在**局部小窗口**里看，没有"常识"。人是怎么做的？你会认出"这是一个塑料杯""这是键盘"，然后利用对物体形状、材质、典型尺寸的理解来推断深度。**把这种"常识"注入网络，就是立体匹配基础模型（stereo foundation model）的核心动机。**

## 1.3 FFS 是谁，从哪来

```
FoundationStereo (2025, NVIDIA)
    └─ 精度极强、零样本泛化强，但慢（几百 ms/帧）
            │
            │  三大加速手段：
            │  ① 知识蒸馏：把笨重的混合骨干压成一个轻量学生
            │  ② 分块神经架构搜索（NAS）：在延迟预算内自动搜代价过滤结构
            │  ③ 结构化剪枝：把迭代优化模块里的冗余整体切掉
            ↓
Fast-FoundationStereo (2026 CVPR, NVIDIA)   ← 本项目
    └─ 快 10 倍以上，零样本精度接近老师，8090 上 23 ms/帧
```

论文里还有一条关键工程：**自动伪标签流水线**——从互联网视频数据集 Stereo4D 里自动筛出 140 万对真实双目图并生成标签，用来补合成数据的不足。这就是为什么它"零样本"泛化好。

**对你最重要的结论**：FFS 的设计目标是**零样本泛化**。也就是说，**你可能根本不需要微调**。这一点第 5 部分会展开。

---

# 第 2 部分 整体架构

## 2.1 三段式流水线

```
        左图 (H,W,3)                    右图 (H,W,3)
            │                               │
            └───────────────┬───────────────┘
                            │
        ┌───────────────────▼────────────────────┐
        │  ① 特征提取  Feature（共享权重）        │   ← EdgeNeXt 学生骨干
        │     输出 1/4, 1/8, 1/16, 1/32 四层金字塔 │      9.39M 参数 (53%)
        └───────────────────┬────────────────────┘
                            │
        ┌───────────────────▼────────────────────┐
        │  ② 代价体构建 + 代价过滤                 │
        │     GWC(8) + Concat(24) = 32 通道       │   ← 6.68M 参数 (38%)
        │     → 3D 沙漏 hourglass                 │
        │     → 视差维度 Transformer               │
        │     → classifier → softmax → 初始视差    │
        └───────────────────┬────────────────────┘
                            │  初始视差 d₀ (1/4 分辨率)
        ┌───────────────────▼────────────────────┐
        │  ③ 迭代优化  convGRU × N 次             │   ← 0.83M 参数 (4.7%)
        │     每次：几何编码体查表 → GRU → 预测    │
        │     一个残差 Δd，累加到当前视差上         │
        └───────────────────┬────────────────────┘
                            │
                     凸上采样 ×4
                            │
                    最终视差 (H,W) ──→ depth = f·B/d
```

一句话概括：**先粗粗地全局搜一遍得到初始视差，再局部反复打磨。**

前两步负责"别找错"（全局视野 + 语义常识），第三步负责"磨精细"（每次只预测一个小残差，稳）。

## 2.2 实测参数量分布

`model_card.md` 写的是 14.6M，**本机实测 17.65M**（口径差异不影响理解）：

| 模块 | 参数量 | 占比 | 作用 |
|---|---:|---:|---|
| `feature` | 9,392,808 | **53.2%** | 特征提取骨干 |
| ├ `feature.stages` | 5,278,776 | 29.9% | EdgeNeXt 四个 stage |
| ├ `feature.deconv32_16` | 2,621,440 | 14.8% | 1/32→1/16 上采样融合 |
| └ `feature.deconv16_8` | 1,155,072 | 6.5% | 1/16→1/8 上采样融合 |
| `cost_agg`（hourglass） | 6,675,216 | **37.8%** | 代价体过滤 |
| `update_block` | 833,229 | 4.7% | 迭代优化 GRU |
| `context_zqr_convs` | 442,752 | 2.5% | **剪枝残余，forward 中从未被调用** |
| `cnet` | 217,944 | 1.2% | 上下文网络 |
| `corr_feature_att` | 73,948 | 0.4% | 代价体语义门控 |
| 其余（上采样/注意力/分类头） | ~19,000 | ~0.1% | — |
| **合计** | **17,654,857** | 100% | |

> **值得注意**：`context_zqr_convs` 占了 44 万参数却从未参与前向。这是结构化剪枝留下的"空壳"——`__init__` 里建了，剪枝后 `forward` 不再引用。你在做模型瘦身时可以安全删掉它。

## 2.3 一次前向的完整数据流（真实 shape）

输入 960×540 的 demo 图，pad 到 544×960（32 的倍数），`max_disp=192`：

| 阶段 | 张量 | 形状 (B,C,H,W) 或 (B,C,D,H,W) | 说明 |
|---|---|---|---|
| 输入 | `image1` | (1, 3, 544, 960) | pad 后的 RGB，0-255 |
| 归一化 | `normalize_image` | (1, 3, 544, 960) | ImageNet mean/std |
| 特征 | `features[0]` | (1, **224**, 136, 240) | 1/4 分辨率，主特征 |
| | `features[1]` | (1, 192, 68, 120) | 1/8 |
| | `features[2]` | (1, 320, 34, 60) | 1/16 |
| | `features[3]` | (1, 304, 17, 30) | 1/32 |
| 上采样分支 | `stem_2x` | (1, 16, 272, 480) | 1/2 分辨率，供凸上采样用 |
| 降维 | `proj_cmb` | (1, **12**, 136, 240) | 224→12，为拼接体做准备 |
| 代价体 | `gwc_volume` | (1, **8**, **48**, 136, 240) | 分组相关，8 组 |
| | `concat_volume` | (1, **24**, 48, 136, 240) | 12×2 拼接 |
| | `comb_volume` | (1, **32**, 48, 136, 240) | 8+24 |
| | 经 `corr_stem` | (1, 32, 48, 136, 240) | **Identity，被剪掉了** |
| 过滤 | 经 `cost_agg` | (1, **28**, 48, 136, 240) | 32→28（volume_dim） |
| 初始视差 | `classifier` → logits | (1, **1**, 48, 136, 240) | 每个像素 48 个视差候选的打分 |
| | `disparity_regression` | (1, 1, 136, 240) | 加权平均 → d₀ |
| 上下文 | `cnet` | (1, **60**, 136, 240) | 拆成 net(60) / inp(48) 两路 |
| 迭代 | `update_block` ×8 | (1, 60, 136, 240) | 隐藏态 60 通道 |
| 上采样 | `upsample_disp` | (1, 1, 544, 960) | 凸上采样 ×4 |
| 输出 | `disp_up` | (1, 1, 544, 960) → unpad (540, 960) | 最终视差 |

**D = 48 是怎么来的？** `max_disp // 4 = 192 // 4`。因为代价体建在 **1/4 分辨率**的特征图上，视差搜索范围也相应缩小到 1/4。这是立体匹配省显存的标准操作。

---

# 第 3 部分 逐模块拆解

下面按数据流顺序讲。每个模块回答三件事：**它干什么 → 为什么需要它 → 关键实现**。

## 3.1 `normalize_image` —— 喂给网络前的标配

`core/foundation_stereo.py:32`

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
return (img/255.0 - mean) / std
```

就是 ImageNet 标准化。**为什么必须做**：骨干网络（EdgeNeXt）是在 ImageNet 预训练的，输入分布必须和训练时一致，否则特征全废。

> 注意：导出 ONNX 时会把这一步剥掉（readme 里说 ONNX 版需要"预归一化"输入）。自己集成时别重复归一化。

## 3.2 `Feature` —— EdgeNeXt 学生骨干

`core/extractor.py:37`（9.39M 参数，占一半以上）

```python
model = timm.create_model('edgenext_small', pretrained=True)
self.stem   = model.stem      # 48 通道
self.stages = model.stages    # 4 个 stage → [48, 96, 160, 304]
```

**它干什么**：把两张图分别编码成多尺度特征金字塔。注意代码里是 `torch.cat([image1, image2], dim=0)` 后一次前向，再按 batch 切回左右——**左右图共享同一套权重**，这是 Siamese 结构，保证左右特征在同一语义空间里可比。

**为什么用 EdgeNeXt**：FoundationStereo 老师用的是"混合骨干"（CNN + DepthAnything ViT），效果强但太重。FFS 用知识蒸馏把它压成单一的 EdgeNeXt 小网络。EdgeNeXt 本身是为移动端设计的，用了深度可分离卷积 + 转置注意力。

**FPN 式融合**：光有金字塔不够，浅层特征有细节、深层有语义。代码用三级反卷积自顶向下融合：

```python
x16 = self.deconv32_16(x32, x16)   # 1/32 上采样后与 1/16 拼接
x8  = self.deconv16_8(x16, x8)
x4  = self.deconv8_4(x8, x4)
x4  = self.conv4(x4)               # 通道整到 224
```

**1/4 特征为什么是 224 通道？** `chans[0]*2 + vit_feat_dim = 48*2 + 128`。那个 128 是 `vitl` 的特征维度的一半（`DepthAnythingFeature.model_configs['vitl']['features']//2 = 256//2`），是蒸馏时对齐老师特征的残留设计。

> 源码里还有个 `else` 分支用 `forward_intermediates`，那是给 ViT 骨干用的；本 checkpoint 走 `hasattr(self,'stem')` 的 EdgeNeXt 分支。

## 3.3 代价体：GWC + Concat 双通道

`core/submodule.py:377` 和 `:492`

**代价体（cost volume）是什么？** 一个五维张量 `(B, C, D, H, W)`。可以理解成：**对每个像素，把"假设视差为 d 时左右特征有多匹配"这个打分，在 d = 0..D-1 上排成一列。** D 就是视差候选数。

FFS 用了两种代价体，**拼在一起**：

### (a) GWC Volume（Group-Wise Correlation，分组相关）—— 8 通道

```python
def build_gwc_volume_optimized_pytorch1(ref, target, maxdisp, num_groups, normalize=True):
    ref_volume    = ref.unsqueeze(2).expand(B, C, maxdisp, H, W)
    padded_target = F.pad(target, (maxdisp-1, 0, 0, 0))
    unfolded      = padded_target.unfold(3, W, 1)        # 滑窗取出所有位移版本
    target_volume = torch.flip(unfolded, [3]).permute(0,1,3,2,4)
    if normalize:
        ref_volume    = F.normalize(ref_volume.float(), dim=2)     # ← 就是那个缺失的 normalize
        target_volume = F.normalize(target_volume.float(), dim=2)
    cost_volume = (ref_volume * target_volume).sum(dim=2)           # 逐组点积
```

把 224 通道分成 8 组，每组 28 通道，**组内做点积**得到 1 个相似度值 → 8 组 = 8 通道。

**为什么分组而不是整体点积？** 整体点积只给 1 个数，信息量太低；逐通道点积给 224 个数，太冗余。分组是折中——每组对应一种"语义子空间"的相似度，既紧凑又有区分度。

> **这里就是我们在部署时踩坑的地方**：`normalize` 参数。HF 上的 c-fast 权重是旧格式，缺这个字段，直接跑会报 `ConfigAttributeError: Missing key normalize`。我们的 `batch_infer.py::patch_cfg()` 已自动补 `True`。

### (b) Concat Volume（拼接体）—— 24 通道

```python
left_tmp  = self.proj_cmb(features_left[0])    # 224 → 12
right_tmp = self.proj_cmb(features_right[0])   # 224 → 12
concat_volume = build_concat_volume_optimized_pytorch1(left_tmp, right_tmp, 48)
```

不做点积，**直接把左右特征在不同位移下拼起来**（12+12=24 通道），把"像不像"的判断交给后面的网络去学。

**为什么两种都要？** 这是 RAFT-Stereo 以来的经典设计：
- 相关体（GWC）是**手工设计的相似度**，收敛快、显式
- 拼接体是**原始特征**，保留网络自己学习匹配函数的自由度

两者互补。最终 `comb_volume = 32 通道`。

## 3.4 `corr_stem` —— 被剪掉的融合层

`core/foundation_stereo.py:163`

源码里它应该是一个 Conv3d 把 32 通道映射到 `volume_dim=28`。但**本 checkpoint 里它是 `nn.Identity()`**：

```
corr_stem     = Identity
cost_agg 输入 = 32 通道
```

这是分块 NAS 搜索的结果——搜出来"这一块不要更好"。所以 32 通道直接喂给了 hourglass，由 hourglass 的第一层完成降维/升维。

**这说明了什么**：读论文代码时不要假设每层都在。剪枝后的真实结构和 `__init__` 里写的不一样，**要以 checkpoint 为准**。

## 3.5 `hourglass` —— 3D 沙漏代价过滤

`core/foundation_stereo.py:41`（6.68M 参数，第二大模块）

代价体是"半成品"：逐像素独立算的匹配打分，充满噪声。hourglass 的作用是**聚合空间上下文和视差维度上下文**，让打分变可靠。

剪枝后的真实结构（实测）：

```
输入 (1, 32, 48, 136, 240)
  ├─ conv1 = Identity      ← 被剪
  ├─ conv2 = Identity      ← 被剪
  ├─ conv3: 32 → 168       ← 编码器最深一级（NAS 搜出的宽度）
  ├─ conv3_up: 168 → 112   ← 3D 反卷积上采样
  ├─ agg_0: 112            ← 拼接后融合
  ├─ conv2_up: 112 → 56
  ├─ agg_1: 56
  ├─ conv1_up: 56 → 28
  └─ conv_out: 28 → 28
输出 (1, 28, 48, 136, 240)
```

**3D 卷积的意义**：2D 卷积看的是"邻居像素"，3D 卷积看的是"邻居像素 **且** 邻居视差"。后者至关重要——它让网络能学到"视差曲面应该是分段平滑的"这种先验。

注意其中 `Conv3dNormActReduced` 的巧妙设计：

```python
self.conv1 = Conv3d(C_in, hidden, kernel=(1, 3, 3))       # 只在空间维度卷
self.conv2 = Conv3d(hidden, C_out, kernel=(17, 1, 1))     # 只在视差维度卷
```

把 3×3×3 的 3D 卷积**分解**成"空间卷积 + 视差卷积"，参数量从 27 降到 9+17=26 且更高效。这叫**可分离 3D 卷积**，是省显存的关键技巧。

## 3.6 `FeatureAtt` —— 语义门控

`core/submodule.py:513`

```python
def forward(self, cv, feat):
    feat_att = self.feat_att(feat).unsqueeze(2)   # 从图像特征算一个通道注意力
    cv = torch.sigmoid(feat_att) * cv              # 乘到代价体上
    return cv
```

**干什么**：用图像语义特征去**调制**代价体的通道。比如网络认出"这片是天空"，就压低某些通道的响应。

这是 FFS 把"常识"注入匹配的关键手段之一——hourglass 的每一级都插了一个 `FeatureAtt`，共 5 个（`feature_att_8/16/32/up_16/up_8`）+ 1 个 `corr_feature_att`。

## 3.7 `CostVolumeDisparityAttention` —— 视差维度上的 Transformer

`core/submodule.py:581`

```python
x = cv.permute(0,3,4,2,1).reshape(B*H*W, D, C)   # 把每个像素的 D 个视差候选当成序列
x = self.pos_embed0(x)                            # 视差位置编码
for layer in self.sa:                             # 4 层 Transformer
    x = layer(x)
x = x.reshape(B,H,W,D,C).permute(0,4,3,1,2)
```

**这是理解 FFS 精度的关键。** 把每个像素的 48 个视差候选看成一条长度 48 的序列，用 Transformer 做**自注意力**。这意味着：网络可以推理"如果真实视差是 30，那么 29 和 31 的打分应该低，而 45 那个峰值多半是重复纹理造成的假峰"——**在视差维度上做全局推理**，而不是局部平滑。

位置编码是必需的：Transformer 本身不知道候选 0 和候选 47 谁大谁小。

`max_len = max_disp//16`（192//16=12），是给patch化后的序列长度用的。

## 3.8 `classifier` + `disparity_regression` —— 初始视差

`core/foundation_stereo.py:171`、`:506`

```python
logits = self.classifier(comb_volume).squeeze(1)   # (B, 1, D, H, W) → 每个视差候选一个打分
prob   = F.softmax(logits, dim=1)                  # 在视差维度上 softmax
init_disp = disparity_regression(prob, max_disp//4)
```

`disparity_regression` 就是**用 softmax 概率对视差候选做加权平均**：

```python
disp_values = torch.arange(0, maxdisp)
return torch.sum(prob * disp_values, 1, keepdim=True)
```

**为什么用加权平均而不是 argmax？** argmax 不可导（没法训练），加权平均可导，而且天然给出亚像素精度的连续视差。这个技巧叫 **soft argmin**，是立体匹配深度学习的基石之一。

到这里得到的是一个**粗糙但全局**的初始视差 d₀。

## 3.9 `cnet` / `CAM` / `SAM` —— 上下文与双注意力

`core/extractor.py:11`、`core/submodule.py:607`、`:626`

```python
cnet_list = self.cnet(features_left[0], features_left[1], features_left[2])
net_list = [torch.tanh(x[0]) for x in cnet_list]     # → GRU 隐藏态初值 (60 通道)
inp_list = [torch.relu(x[1]) for x in cnet_list]     # → 每轮喂给 GRU 的上下文 (48 通道)
inp_list = [self.cam(x) * x for x in inp_list]       # 通道注意力增强
att      = [self.sam(x) for x in inp_list]           # 空间注意力图
```

三个小模块，来自 Selective-IGEV：

| 模块 | 干什么 |
|---|---|
| `cnet` | 从左图特征生成 GRU 的初始隐藏态和每轮输入 |
| `CAM`（Channel Attention Enhancement） | 通道注意力：哪些**特征通道**重要。用 avg+max 双池化 |
| `SAM`（Spatial Attention Extractor） | 空间注意力：图像上**哪些位置**重要。用 avg+max 沿通道维拼成 2 通道再卷积 |

**SAM 的输出 `att` 后面有妙用**——见 3.11。

> 注意 `cnet` 实际只用了 `conv04` 这一个 ModuleList，构造函数里那些 `c08/c16` 参数没用上（剪枝痕迹）。

## 3.10 `Combined_Geo_Encoding_Volume` —— 几何编码体

`core/geometry.py:7`

这是**迭代阶段的信息来源**。每一轮迭代，GRU 都需要知道"以当前视差为中心，附近的匹配情况如何"。

它准备两类信息，各建 2 层金字塔（`corr_levels=2`）：

```python
# ① 几何编码体：从过滤后的代价体里，按当前视差做 1D 采样
geo_volume = bilinear_sampler1d(geo_volume, dx + disp/2**i)

# ② all-pairs correlation：从原始特征算的全局相似度里采样
init_corr = bilinear_sampler1d(init_corr, coords/2**i - disp/2**i + dx)
```

- **①** 用的是 **过滤后**的代价体（含语义上下文），但分辨率粗
- **②** 用的是**原始特征**算的全对相关（`torch.einsum('aijk,aijh->ajkh', ...)`），没有过滤但保真

两者拼接 → 每轮 GRU 的输入。通道数：

```
corr_levels × (2×corr_radius + 1) × (volume_dim + 1)
=   2       ×      (2×4+1)=9      ×     (28+1)=29
= 522
```

**正好等于 `BasicMotionEncoder.convc1` 的输入通道 522**（实测权重 shape `(56, 522, 1, 1)`）。这个数字对上了，说明理解正确。

`dx = [-4,-3,...,+4]`（9 个偏移），意味着每轮只看当前视差附近 ±4 的邻域，配合金字塔实现"由粗到细"。

## 3.11 `update_block` —— 选择性卷积 GRU

`core/update.py:84`（0.83M 参数）

```python
for itr in range(iters):          # 默认 8 次
    disp = disp.detach()          # ← 关键：截断梯度，避免 8 轮展开爆显存
    geo_feat = geo_fn(disp, coords, dx=self.dx)
    net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
    disp = disp + delta_disp      # 只预测残差
```

**为什么不一次性预测？** 因为一次预测大数值不稳定。改成"每轮只预测一个小残差 Δd"，8 轮累加。这是**迭代残差细化**（RAFT 开创），可以用很少的参数达到很高精度，还能在推理时灵活调整轮数换速度（`--valid_iters 4` 就是只跑 4 轮）。

内部的 `SelectiveConvGRU` 是 FFS 的创新点（来自 Selective-IGEV）：

```python
def forward(self, att, h, *x):
    x  = self.conv0(torch.cat(x, dim=1))       # 97 → 100
    hx = self.conv1(torch.cat([x, h], dim=1))  # 160 → 168
    h  = self.small_gru(h, x, hx) * att + self.large_gru(h, x, hx) * (1 - att)
    return h
```

**双路 GRU + 空间注意力选择**：
- `small_gru`：1×1 卷积核 —— 适合**平坦区域**（视差变化平缓，不需要看邻居）
- `large_gru`：3×3 卷积核 —— 适合**边缘/细节区域**（需要空间上下文）

`att`（来自 3.9 的 SAM）是一个 0~1 的空间权重图，网络自己学会在每个像素上决定"这里该用哪种 GRU"。**这就是"选择性（Selective）"的含义**，也是省算力的关键。

`DispHead` 从隐藏态预测 Δd，`mask` 分支同时输出 32 通道特征供上采样用。

## 3.12 `upsample_disp` —— 凸上采样

`core/foundation_stereo.py:182`

视差一直算在 1/4 分辨率上，最后要放大回全分辨率。**直接双线性插值会把物体边缘抹掉。**

FFS 用 RAFT 的**凸上采样（convex upsampling）**：

```python
spx_pred = self.spx_gru(xspx)                    # 输出 9 个通道
spx_pred = F.softmax(spx_pred, 1)                # 在 9 个邻居上归一化
up_disp  = context_upsample(disp*4., spx_pred)   # 加权组合
```

对全分辨率的每个像素，网络预测它应该由 1/4 分辨率下 **3×3 邻域**的 9 个像素**按什么权重**组合而来（权重 softmax 归一化即凸组合）。这样能在上采样的同时**保持边缘锐利**——边缘处的权重会集中在某一侧，而不是平均。

`stem_2x`（1/2 分辨率的浅层特征）在这里提供高分辨率的纹理信息。

## 3.13 `run_hierachical` —— 两阶段由粗到精

`core/foundation_stereo.py:264`

```python
# 先在 0.5 倍缩小的图上跑一遍，得到粗视差
disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters)
disp_small_up = F.interpolate(disp_small, size=(H,W)) * 1/small_ratio

# 再用它作为初始视差，在全分辨率上跑第二遍
init_disp = F.interpolate(disp_small_up, scale_factor=0.25) * 0.25
disp = self.forward(image1, image2, init_disp=init_disp)
```

**用途**：当场景视差范围很大（比如很近的物体超出 `max_disp`）时，先在低分辨率上"看全局"确定大致范围，再在全分辨率上精修。

**代价**：跑了两次完整前向。**实测在本机反而更慢**（1.22s vs 0.94s），因为省下的代价体开销抵不过第二次前向。这个功能主要面向高分辨率 + 大视差的场景。

## 3.14 剪枝残余：`context_zqr_convs`

`core/foundation_stereo.py:148` 定义，但 `forward` 中**从未引用**。占 442,752 参数。

看到它别困惑——这是结构化剪枝没清理干净的遗迹。删掉不影响推理结果。

---

# 第 4 部分 训练 vs 推理的行为差异

`forward()` 有个关键分支（`core/foundation_stereo.py:250`）：

```python
if test_mode and itr < iters-1:
    continue                       # 推理：只在最后一轮做上采样，省时间
disp_up = self.upsample_disp(...)
disp_preds.append(disp_up)

if test_mode:
    return disp_up                 # 推理：只返回最终的
return init_disp, disp_preds       # 训练：返回每一轮的结果
```

| | 返回 | 用途 |
|---|---|---|
| **推理** | 单个 `disp_up` | 只要最终结果，中间轮不上采样省算力 |
| **训练** | `(init_disp, disp_preds)` | **每一轮的预测都要拿去算损失** |

**为什么要每一轮都算损失？** 这是 RAFT 的标准监督方式，让网络在每一轮都学会"再改进一点"，而不是把所有压力放在最后一轮。

标准损失函数（指数加权，越靠后的轮次权重越大）：

```python
loss = sum(gamma ** (N - 1 - i) * |disp_gt - disp_preds[i]|_1  for i in range(N))
# 论文常用 gamma = 0.9，|·|_1 是平滑 L1
```

另外 `disp = disp.detach()` 这一句很重要：**每轮开始时把视差从计算图里断开**，这样反向传播只走当前这一轮，显存从 O(8 轮) 降到 O(1 轮)。这就是为什么"轮数越多越准但不会爆显存"。

还有一个训练/推理的差异：**训练时 `train_iters=22`，推理时 `valid_iters=8`**（TAO 配置）。训练时多迭代是为了让梯度信号更充分。

---

# 第 5 部分 微调方案

## 5.1 先说清楚现实：官方只开源了推理代码

我把仓库翻了一遍：

```
scripts/
├── run_demo.py / run_demo_tensorrt.py / run_demo_single_trt.py ...   # 推理
├── make_onnx.py / make_single_onnx.py / build_plugin_trt.py ...      # 导出
└── profile_speed.py / profile_memory.py                              # 性能分析
# 没有 train.py，没有任何损失函数/数据加载代码
```

**这是 NVIDIA 的商业策略**：
- 推理代码 + 研究权重 → 免费（NVIDIA Source Code License）
- **训练/微调代码 → 走 NVIDIA TAO Toolkit**（商业授权）

readme 第 45 行写得很直白：
> For commercially licensed training and inference code, see NVIDIA TAO Toolkit.

所以你有三条路，我按推荐程度排序。

## 5.2 路线 A：NVIDIA TAO Toolkit（官方支持，最省事）

### 它能做什么

TAO 把 FFS 集成进了 `depth_net` 模块，支持完整链路：

```
train → evaluate → inference → export → gen_trt_engine
```

而且默认开了 **AutoML**（自动搜学习率和衰减）。

### 完整 spec 配置（bp2 商用 checkpoint 专用）

这是微调的核心。下面这份来自 NVIDIA 官方文档，**每个字段都要照抄**，原因见后面的警告：

```yaml
results_dir: /data/result

dataset:
  dataset_name: StereoDataset
  max_disparity: 192
  min_depth: 0.0
  train_dataset:
    data_sources:
      - dataset_name: GenericDataset          # 自己的数据用这个
        data_file: /data/datasets/stereo/train.txt
    batch_size: 1
    workers: 4
    augmentation:
      crop_size: [320, 736]
  val_dataset:
    data_sources:
      - dataset_name: GenericDataset
        data_file: /data/datasets/stereo/val.txt
    batch_size: 1
    workers: 4
    augmentation:
      crop_size: [320, 736]

model:
  model_type: FastFoundationStereo
  encoder: vitl
  hidden_dims: [128]                    # 不是 [128,128,128]
  n_gru_layers: 1                       # 不是 3
  corr_radius: 4
  corr_levels: 2
  n_downsample: 2
  max_disparity: 192                    # 不是 416
  valid_iters: 8
  train_iters: 22                       # 训练时迭代更多
  volume_dim: 28                        # 不是 32
  mixed_precision: false
  gwc_feature_normalize: true           # 必须 true

  # ↓↓↓ 15 个 bp2 剪枝宽度，一个都不能少 ↓↓↓
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
  num_epochs: 10
  precision: fp32
  pretrained_model_path: /data/checkpoints/model_best_bp2_serialize.pth   # ← 用我们下好的权重
  optim:
    optimizer: AdamW
    lr: 1.0e-5
    weight_decay: 0.0001

evaluate:
  num_gpus: 1
  batch_size: 1
  checkpoint: /data/checkpoints/model.pth

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

### ⚠️ 四个必须精确匹配的参数（错了不报错，但结果悄悄变差）

NVIDIA 文档特别警告：bp2 checkpoint 的配置和 TAO schema 的**默认值不一样**。如果你漏写字段，TAO 会静默套用默认值，**不报错但效果崩坏**：

| 参数 | schema 默认 | bp2 必须值 | 写错的后果 |
|---|---|---|---|
| `max_disparity` | 416 | **192** | 代价体建得过大，预测偏离训练视差区间 |
| `gwc_feature_normalize` | — | **true** | 设 false 会在约 **7~8% 的像素**上产生负视差 |
| `volume_dim` | 32 | **28** | 形状不匹配 |
| `hidden_dims` / `n_gru_layers` | [128,128,128] / 3 | **[128] / 1** | 形状不匹配 |

### 数据格式

一个纯文本文件，每行一个样本，空格分隔：

| 列数 | 格式 | 用途 |
|---|---|---|
| 2 | `左图路径 右图路径` | 无真值推理 |
| **3** | `左图路径 右图路径 视差图路径` | **训练 / 评估（用这个）** |
| 4 | `左图路径 右图路径 视差图 遮挡掩码` | 评估（仅 Middlebury / Eth3d 支持） |

训练时用 **3 列**。视差图一般是 `.pfm` 格式（带尺度的浮点图，立体匹配的标准格式）。

支持的 `dataset_name`：

| 值 | 说明 |
|---|---|
| `GenericDataset` | **你自己的数据（训练就用这个）** |
| `Middlebury` / `Kitti` / `Eth3d` | 公开基准 |
| `FSD` | NVIDIA 合成数据集 |
| `IsaacRealDataset` | NVIDIA Isaac 真实机器人数据 |
| `Crestereo` | CREStereo 合成集 |

### 运行方式（Docker）

```bash
# 建可写缓存目录（容器以宿主 UID 运行时需要）
mkdir -p <output_dir>/home <output_dir>/.cache/matplotlib \
         <output_dir>/.cache/torchinductor <output_dir>/.cache/xdg

docker run --gpus 'device=0' --shm-size 16G \
  --user "$(id -u):$(id -g)" -e USER="$(id -un)" \
  -e HOME=<output_dir>/home \
  -e MPLCONFIGDIR=<output_dir>/.cache/matplotlib \
  -e TORCHINDUCTOR_CACHE_DIR=<output_dir>/.cache/torchinductor \
  -e XDG_CACHE_HOME=<output_dir>/.cache/xdg \
  -v <data_root>:<data_root>:ro \
  -v <output_dir>:<output_dir> \
  -v <bp2_ckpt_dir>:<bp2_ckpt_dir>:ro \
  <tao_container> \
  depth_net train -e <spec.yaml>
```

其他 action 把 `train` 换成 `evaluate` / `inference` / `export` / `gen_trt_engine` 即可。

**验证要点**：
- 看每步的 `train_loss`，**别只信 `Execution status: PASS`**（loss 是 NaN 时它也报 PASS）
- 评估指标看 `epe`（端点误差）、`bp1/bp2/bp3`（bad pixel 率）、`d1`
- 忽略 `abs_rel` / `sq_rel` / `rmse_log`（那是单目深度的指标，对立体匹配没意义）

### 这条路的问题

- 需要 NVIDIA NGC 账号 + Docker + nvidia-container-toolkit
- TAO 镜像体积很大（几十 GB）
- **你 4GB 显存跑不动**：`batch_size=1`、`crop_size=[320,736]`、`train_iters=22`、fp32 —— 实测推理全分辨率就要 1.27GB（还没算梯度/优化器状态）。训练至少要 **12GB**（3060/4070 级别），推荐 24GB

## 5.3 路线 B：自己写训练循环（研究用途）

**这条路完全可行**，因为模型就是个标准 `nn.Module`，而且 `forward(test_mode=False)` 已经贴心地返回了每一轮的预测。

### 核心代码骨架

```python
import torch, torch.nn.functional as F
from omegaconf import OmegaConf

# 1) 加载（记得补 normalize，否则报错）
model = torch.load('weights/c-fast/model_best_bp2_serialize.pth', weights_only=False)
OmegaConf.set_struct(model.args, False)
OmegaConf.update(model.args, 'normalize', True, force_add=True)
model.args.mixed_precision = False      # ← 本卡必须 fp32！见部署文档
model.args.train_iters = 22
model.cuda().train()

opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

for left, right, disp_gt in loader:
    left, right, disp_gt = left.cuda(), right.cuda(), disp_gt.cuda()

    # 2) 前向：注意 test_mode=False，返回 (init_disp, disp_preds)
    init_disp, disp_preds = model(left, right, iters=22, test_mode=False)

    # 3) 序列监督损失：越靠后的轮次权重越大
    N = len(disp_preds)
    gamma = 0.9
    loss = sum(gamma ** (N - 1 - i) * F.smooth_l1_loss(disp_preds[i], disp_gt)
               for i in range(N))
    loss += 1.0 * F.smooth_l1_loss(init_disp, disp_gt)   # 初始视差也监督

    # 4) 反向
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)   # TAO 默认 0.1
    opt.step()
```

### 几个必须注意的点

| 事项 | 说明 |
|---|---|
| **必须 fp32** | 本机 cuDNN 的 fp16 卷积会产出全 NaN（详见部署文档第 3 节） |
| **`test_mode=False`** | 否则只返回单个最终结果，拿不到序列损失 |
| **真值视差要下采样** | `disp_preds` 是全分辨率，如果你的 gt 是 1/4 分辨率要对齐 |
| **无效像素要 mask** | gt 里视差为 0/inf 的像素（无匹配、超范围）必须排除，否则学歪 |
| **梯度裁剪** | TAO 默认 `clip_grad_norm=0.1`，比较激进，照抄 |
| **输入必须已极线校正** | 未校正的双目图直接训练会毁掉模型 |
| **显存** | 4GB 绝对不够。梯度 + AdamW 状态（2×17.65M×4 字节）就要 ~140MB，加上 22 轮迭代的激活值，实测量级在 8~12GB |

### 冻结策略（数据少时推荐）

如果只有几百对数据，全量微调容易过拟合。可以只解冻后段：

```python
# 冻结骨干（特征提取已经很强，且是蒸馏来的，动了容易崩）
for p in model.feature.parameters():
    p.requires_grad = False

# 只训练迭代优化部分（0.83M 参数，占 4.7%）
# 或者再加 cost_agg 的末尾几层
```

## 5.4 路线 C：伪标签自蒸馏（最适合你的场景）

**这是 NVIDIA 自己造数据时用的办法**（论文里的 automatic pseudo-labeling pipeline，从 Stereo4D 造了 140 万对）。你也完全可以照做：

```
① 用你自己的双目相机，在实际工作场景拍 N 对 RGB 图（不需要任何真值深度）
        ↓
② 用现有的 FFS 权重跑一遍，得到视差图 d_ffs
        ↓
③ 左右一致性检查（left-right consistency check）：
   用左图视差把右图 warp 过来，比较差异
   差异大的像素判为不可靠 → 丢弃
        ↓
④ 置信度阈值过滤：
   取 softmax 概率的最大值 prob_max，低于阈值（如 0.9）的像素丢弃
        ↓
⑤ 过滤后的视差图当作伪真值 → 路线 A 或 B 微调
```

**为什么有效**：FFS 本身零样本就很强，它的预测在大部分像素上是准的。用一致性检查筛掉它不确定的部分，剩下的就是高质量标签。模型在自己产出的（经过滤的）标签上训练，会把"泛化能力"固化到"你这个具体场景"上，通常能提升边缘质量和稳定性。

**注意**：这一步**不能提升绝对精度上限**——老师不会教出超过自己的学生。它的价值是让模型适配你的**镜头畸变、基线、工作距离、光照**，减少系统性偏差。

### 什么时候值得做

| 情况 | 建议 |
|---|---|
| 你的场景和 FFS 训练数据接近（室内桌面、常见物体） | **别微调**，直接零样本用 |
| 特殊镜头/超短基线/极端光照 | 值得，走路线 C 造数据 |
| 有高精度真值（激光雷达、结构光） | 直接走路线 A/B |
| 只是想要更快的速度 | 别微调，调 `--scale` / `--valid_iters` |

## 5.5 在微调之前，先试这些免费的

FFS 的设计目标就是零样本泛化。在花几周做数据之前，先把这几个旋钮转一遍（都是实测过的）：

```bash
# 1) 分辨率与迭代轮数（最有效）
--scale 0.5 --valid_iters 4      # 0.24s/帧，411MB

# 2) 视差范围
--max_disp 192                   # 默认够用；物体近于 0.1m 才需要加大

# 3) 层次化推理（大视差/高分辨率场景）
--hiera 1

# 4) 输入图必须是 PNG（无损）
#    readme 明确说：lossy 压缩会掉精度
```

readme 里还有几条容易忽略但很关键的：

- **左右图绝对不能互换**（左图必须来自左侧相机，物体在画面里偏右）
- **输入必须已做极线校正**（Zed、RealSense 这类一般已经处理好了）
- **图像宽度最好 < 1000**，超了就 `--scale 0.5` 降下来
- **黑白/红外图也能用**（RealSense D4xx 的 IR 流亲测可用）——这对你的场景很重要

---

# 第 6 部分 在你这台机器上的现实约束

| 项目 | 现状 | 影响 |
|---|---|---|
| 显存 | GTX 1650 4GB | 推理半分辨率仅 411MB，够用；**训练不可行** |
| 精度 | 必须 fp32 | cuDNN 9.1.0 在 Turing 上的 fp16 卷积产出全 NaN |
| 速度 | 半分辨率 0.26s/帧（3.8fps） | 离线测深度够用；实时抓取不行 |
| 微调 | 需 12GB+ 显存 | 建议：在机器上采集数据，训练放到云端/更好的卡 |

**推荐的务实路径**：

1. 先在 4GB 机器上把**深度生成管线跑通**（已完成）
2. 用 RealSense 采集你实际场景的双目图，验证 FFS 零样本效果
3. 如果效果不够，**用路线 C 造伪标签**（本机就能做，只是慢）
4. 训练环节上云（AutoDL / 恒源云，按小时租 24GB 卡）或换卡

---

# 附录 A 配置字段速查（cfg.yaml）

本 checkpoint 权重自带的 args（实测）：

| 字段 | 值 | 含义 |
|---|---|---|
| `vit_size` | `vitl` | DepthAnything 教师规模（影响 `vit_feat_dim`） |
| `hidden_dims` | `[128]` | GRU 隐藏维度列表（剪枝后只有 1 层） |
| `n_gru_layers` | `1` | GRU 层数（原版 3） |
| `corr_levels` | `2` | 相关金字塔层数 |
| `corr_radius` | `4` | 每轮采样偏移 ±4（共 9 个） |
| `n_downsample` | `2` | 视差场分辨率 1/4 |
| `max_disp` | `416` | 训练时的最大视差（推理建议 192） |
| `valid_iters` | `8` | 推理迭代轮数 |
| `mixed_precision` | `True` | 是否开 AMP（**本机必须设 False**） |
| `low_memory` | `0` | 省显存模式 |
| `cv_group`（默认） | `8` | GWC 分组数 |
| `volume_dim`（默认） | `28` | 代价体过滤后的通道数 |
| `normalize`（我们补的） | `True` | GWC 是否做特征归一化 |

运行时可调（不用重新训练）：`valid_iters`、`max_disp`、`scale`、`hiera`。

---

# 附录 B 权重家族对照

| Checkpoint | valid_iters | PyTorch (3090) | TensorRT | 显存 | 获取方式 |
|---|---|---|---|---|---|
| `23-36-37` | 8 | 49.4 ms | 23.4 ms | 653 MB | Google Drive（研究版） |
| `23-36-37` | 4 | 41.1 ms | 18.4 ms | 653 MB | 同上 |
| `20-26-39` | 8 | 43.6 ms | 19.4 ms | 651 MB | 同上 |
| `20-30-48` | 8 | 38.4 ms | 16.6 ms | 646 MB | 同上 |
| `20-30-48` | 4 | 29.3 ms | 14.0 ms | 646 MB | 同上 |
| **c-fast（本机用的）** | 8 | — | — | 68 MB 权重 | **hf-mirror.com/nvidia/c-fast-foundationstereo** |

命名 `23-36-37` 是 NAS 搜出来的结构编号。越往下越快、精度略降。

本机实测（GTX 1650，fp32）：全分辨率 0.94s / 1267MB，半分辨率 0.26s / 411MB。

---

# 附录 C 源码文件地图

| 文件 | 行数 | 内容 |
|---|---:|---|
| `core/foundation_stereo.py` | 444 | 主模型 `FastFoundationStereo`、`hourglass`、TensorRT 包装类 |
| `core/submodule.py` | 675 | 所有基础积木：3D 卷积块、注意力、代价体构建、上采样 |
| `core/extractor.py` | 78 | `Feature`（EdgeNeXt 骨干）、`ContextNetSharedBackbone` |
| `core/update.py` | 108 | `BasicSelectiveMultiUpdateBlock`、`SelectiveConvGRU`、`DispHead` |
| `core/geometry.py` | 80 | `Combined_Geo_Encoding_Volume` |
| `core/distill_block.py` | 50 | ONNX 导出用的序列化辅助类 |
| `core/utils/utils.py` | 122 | `InputPadder`、`bilinear_sampler` |
| `Utils.py` | 88 | 日志、种子、视差可视化、点云转换 |
| `scripts/run_demo.py` | 150 | 官方推理入口（含可视化，无头环境跑不了） |

**我们加的**（部署时用）：

| 文件 | 用途 |
|---|---|
| `batch_infer.py` | 无头批量推理，自动补 `normalize`，NaN 保护 |
| `bench_ffs.py` | 多配置基准测试 |
| `make_vis.py` | 生成三联对比图 |
| `stat_params.py` | 参数量统计 |
| `trace_shapes.py` | 前向形状追踪（本文 2.3 节数据来源） |
| `dump_struct.py` | 打印剪枝后真实结构 |
| `debug_nan*.py` | fp16 NaN 排障过程记录 |

---

## 一句话总结

FFS 用**轻量 EdgeNeXt 骨干**提特征 → **GWC+Concat 双代价体** + **3D 沙漏** + **视差维度 Transformer** 全局搜出一个粗糙视差 → **选择性双路 convGRU 迭代 8 次**每次只预测小残差 → **凸上采样**恢复全分辨率。它的卖点是零样本泛化，所以**先别急着微调**，把 `--scale` / `--valid_iters` / `--max_disp` 调明白，大概率就够了。

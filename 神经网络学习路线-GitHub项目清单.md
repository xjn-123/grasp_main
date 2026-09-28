# 神经网络学习路线 · GitHub 项目清单

> 整理日期：2026-09-27（星数为当日实测值）
> 面向：已在本机跑通 Fast-FoundationStereo、做机器人视觉（立体深度 + 位姿估计）、深度学习基础薄弱的学习者

---

## 写在前面：这份清单为什么这样排

市面上"神经网络学习仓库推荐"的清单很多，但绝大多数是**按热度排的**，不是按**你的目标**排的。

你的处境有三个特殊性，决定了排法不一样：

1. **你已经在用工业级模型了，但不懂原理。** 这是"倒着学"的状态 —— 好处是你能直接看到真实效果，坏处是容易被源码淹没。
2. **你的目标不是发论文，是复现一套机器人抓取系统。** 所以"数学推导的严谨性"要让位于"能不能读懂并改动代码"。
3. **你手上只有一张 4GB 显存的 GTX 1650。** 很多推荐项目你根本跑不动，硬啃会卡死在环境上。

因此这份清单的原则是：**每一个推荐，都必须能直接推进你读懂 FFS（Fast-FoundationStereo）**。不能推进的（比如 LLM 方向），一律放到"暂时别碰"。

---

## 总览：四个阶段

| 阶段 | 目标 | 代表项目 | 建议耗时 | 跳过的后果 |
|---|---|---|---|---|
| ① 拆开黑箱 | 搞懂"反向传播"到底在算什么 | micrograd、nn-zero-to-hero | 1–2 周 | 后面所有公式都只能死记 |
| ② 系统补基础 | 建立知识坐标系（当字典用） | d2l-zh、fastbook | 2–3 周（可长期） | 遇到陌生术语无从下手 |
| ③ 学读工程代码 | 从"能跑 demo"到"能改代码" | ultralytics、timm、Made-With-ML | 2–3 周 | 读不懂任何论文的官方实现 |
| ④ 攻你的方向 | 立体匹配与深度估计 | RAFT-Stereo、monodepth2 | 3–4 周 | FFS 永远是黑箱 |

**这四层是串行的，跳级会卡住。** 但第 ② 层可以和第 ③④ 层并行推进（它就是字典，随查随用）。

---

## 阶段 ① 拆开黑箱

### 1. karpathy/micrograd —— 17.7k ⭐

```
https://github.com/karpathy/micrograd
```

**是什么**：一个只有两个文件、约 150 行的自动微分引擎 + 神经网络库。

**为什么是它**：所有框架里最让你困惑的一行代码一定是 `loss.backward()`。micrograd 把这一行展开成你能逐句执行的代码 —— 它自己实现了一个 `Value` 类，记录每个数的"从哪来、怎么来的"，然后沿图反向走一遍。

**怎么看**（顺序不能变）：

1. `engine.py` 的 `Value` 类 —— 重点看 `__add__`、`__mul__`、`backward` 三个方法
2. 在 notebook 里手写一个 `a = Value(2.0); b = Value(-3.0); ... ` 的表达式
3. 调用 `.backward()`，打印每个中间变量的 `.grad`，**手算验证一遍**
4. `nn.py` —— 看它怎么用 150 行搭出一个 MLP

**过关标准**：不看代码，能手写 `y = (a*b + c).relu()` 对 `a` 的梯度公式。

---

### 2. karpathy/nn-zero-to-hero —— 24.5k ⭐

```
https://github.com/karpathy/nn-zero-to-hero
```

**是什么**：Karpathy 的视频课程代码，从 micrograd 一路讲到 GPT。B 站有带中文字幕的搬运。

**为什么是它**：它是唯一一套"从反向传播讲到 Transformer，全程你能跟着敲"的材料。其他课程要么跳过了底层，要么停留在底层上不去。

**学习顺序**（对应 git 提交顺序）：

| 讲次 | 主题 | 你会得到 |
|---|---|---|
| 1 | micrograd | 反向传播 |
| 2 | makemore：MLP 语言模型 | 第一个"真"的神经网络 |
| 3 | BatchNorm / 残差 | 为什么深网络能训起来 |
| 4 | 手动实现反向传播并对照 PyTorch | 框架到底帮你做了什么 |
| 5 | CNN（ WaveNet 风格） | 卷积的直觉 |
| 6 | Transformer + nanoGPT | 注意力机制 |

**注意**：这个仓库最后更新是 2024 年 8 月，但内容完全不过时 —— 讲的是原理，不是框架 API。

---

## 阶段 ② 系统补基础（当字典，别从头读）

### 3. d2l-ai/d2l-zh（动手学深度学习）—— 81.1k ⭐

```
https://github.com/d2l-ai/d2l-zh
在线阅读：http://zh.d2l.ai/
```

**是什么**：李沐团队的中文深度学习教材，PyTorch 版，每节都是可运行的 Jupyter notebook。被全球 70 多个国家、500 多所大学用于教学。

**⚠️ 最重要的提醒：别按章节顺序从头读到尾。** 这是一本 500 页的教材，从头读的结果一定是第三章弃坑。

**正确用法**：把它当字典。

- 在 FFS 里看到 `nn.BatchNorm2d` → 查"批量规范化"
- 看到残差连接 → 查"残差网络 ResNet"
- 看到 `ConvTranspose2d` → 查"转置卷积"
- 看到学习率调度 → 查"学习率调度器"

**几个对你特别有用的章节**（建议优先看）：

| 章节 | 为什么对你重要 |
|---|---|
| 卷积神经网络（LeNet / AlexNet / VGG） | 看懂 FFS 的 `Feature` 提取器 |
| 批量规范化 | FFS 里到处是 `BasicConv_IN`（IN = InstanceNorm） |
| 注意力机制与 Transformer | FFS 的 `CostVolumeDisparityAttention` 就是它 |
| 优化算法（Adam / 学习率调度） | 后面做微调时必读 |

**安装**：本机 WSL2 里直接 `pip install d2l` 即可，CPU 就能跑完前 60% 的内容。

---

### 4. fastai/fastbook —— 25.3k ⭐

```
https://github.com/fastai/fastbook
```

**是什么**：FastAI 的官方教材，自顶向下 —— 第一章就让你训出一个猫狗分类器，然后才回头讲原理。

**什么时候用它**：如果你啃 d2l 觉得太数学、快劝退了，立刻换成这本。它的哲学是"先有成就感，再补理论"，对自学者的心理韧性更友好。

**取舍**：fastai 封装比 PyTorch 原生厚一层，你会学到一些 fastai 特有的 API。但原理部分（卷积、注意力、损失函数）是通用的，不影响。

---

## 阶段 ③ 学读工程级代码

> 这层是最容易被跳过、但又最关键的一层。**能跑 demo 和能改代码之间，隔着这一整层。**

### 5. ultralytics/ultralytics（YOLO11）—— 62.0k ⭐

```
https://github.com/ultralytics/ultralytics
```

**为什么必须是它**：

1. 你后面做检测环节就要用 YOLO11，**学以致用，不浪费时间**
2. 它的代码组织是教科书级的三层分离，清晰的程度在大型 CV 库里罕见

**看什么**（按这个顺序）：

| 文件 | 看什么 |
|---|---|
| `ultralytics/cfg/default.yaml` | 全部超参一览 —— 先读这个，整个库的配置空间就清楚了 |
| `ultralytics/engine/trainer.py` | 完整训练循环：数据加载 → 前向 → 损失 → 反向 → 调度 |
| `ultralytics/nn/modules/head.py` | 检测头：怎么把特征图变成框和类别 |
| `ultralytics/models/yolo/detect/train.py` | 损失函数的组装过程 |

**一个小练习**：在 `default.yaml` 里改一个参数（比如 `lr0`），重训一遍，观察 loss 曲线的变化。这一步做完，你对"超参"的理解会超过 90% 的自学者。

---

### 6. huggingface/pytorch-image-models（timm）—— 37.2k ⭐

```
https://github.com/huggingface/pytorch-image-models
```

**是什么**：视觉 backbone 大合集（ResNet、ConvNeXt、ViT、Swin…），PyTorch 生态里最常用的模型库。**FFS 就用了 `timm` 里的 ViT。**

**看什么**：

- `timm/models/vision_transformer.py` —— ViT 的规范实现，注释详尽
- `timm/models/resnet.py` —— 对比一下"标准 ResNet"和你从论文里看到的是不是一回事

**用法**：不要通读（几千个模型），**只挑 FFS 用到的那个读**。

---

### 7. GokuMohandas/Made-With-ML —— 49.6k ⭐

```
https://github.com/GokuMohandas/Made-With-ML
```

**是什么**：端到端 MLOps 课程 —— 数据标注、实验跟踪、部署、监控、负责任 AI。

**为什么补它**：所有算法课都不教的东西全在这。当你第一次训完模型，然后发现"忘了记录用的哪组参数、哪个数据集版本"时，你会需要它。

**什么时候看**：等你开始做自己的第一个完整项目（比如用自己相机拍的数据微调）时再看。现在先看前两章建立意识即可。

---

## 阶段 ④ 攻你的方向：立体匹配与深度

### 🎯 关键捷径：RAFT-Stereo 是 FFS 的"精简版"

这是我通读 FFS 源码后最重要的一个发现，也是这份清单里**对你最有价值的一条**：

> FFS 的核心骨架 —— 代价体、GRU 迭代更新、凸上采样 —— **全部继承自 RAFT**。FFS 只在这副骨架前面加了两样东西：ViT 骨干网络 + 视差维度 Transformer。

**所以：直接读 FFS（2000 行 + ViT + 注意力 + 剪枝遗迹）≈ 硬啃；先读 RAFT-Stereo（4 个文件、约 28KB）再看 FFS ≈ 顺水推舟。**

下面是实测的模块对照表（FFS 侧为你本机 `/home/xiao/Depth_Learning/FFS/Fast-FoundationStereo/` 的真实文件）：

| 部件 | RAFT-Stereo | Fast-FoundationStereo | 关系 |
|---|---|---|---|
| 特征提取 | `core/extractor.py` `Feature` | `core/extractor.py` `Feature` | **同名，几乎一样** |
| 匹配代价体 | `core/corr.py` `CorrBlock1D` | `core/submodule.py` `build_gwc_volume_optimized_pytorch1` | FFS 换成 group-wise correlation |
| 迭代更新块 | `core/update.py` `BasicUpdateBlock` | `core/update.py` `BasicSelectiveMultiUpdateBlock` | FFS 改成"选择性双路" |
| GRU 单元 | `core/update.py` `SepConvGRU` | `core/update.py` `SelectiveConvGRU` / `RaftConvGRU` | FFS 让 1×1 与 3×3 两个 GRU 竞争 |
| 运动编码器 | `core/update.py` `BasicMotionEncoder` | `core/update.py` `BasicMotionEncoder` | **同名** |
| 凸上采样 | `core/update.py` `context_upsample` | `core/submodule.py` `context_upsample` | **同名同实现** |
| 视差维度 Transformer | ❌ 无 | `core/submodule.py` `CostVolumeDisparityAttention` | **FFS 新增** |
| 几何编码体 | ❌ 无 | `core/geometry.py` | **FFS 新增** |
| 蒸馏块 | ❌ 无 | `core/distill_block.py` | **FFS 新增** |
| 主模型 | `core/raft_stereo.py` | `core/foundation_stereo.py` | FFS 加了 ViT |

**用法**：先在 RAFT 里搞懂左边那 6 行，然后回到 FFS 看右边 —— 你会发现有一半的代码你已经认识了。

---

### 8. princeton-vl/RAFT-Stereo —— 1.1k ⭐

```
https://github.com/princeton-vl/RAFT-Stereo
```

**别被星数骗了**。它是学术代码库，1.1k 星但**论文引用量数千**，是光流/立体匹配领域的里程碑。仓库最后提交 2026 年 8 月，仍在维护。

**为什么适合学**：核心就 4 个文件，加起来约 28KB：

```
core/
├── raft_stereo.py    6.1 KB   主模型，看 forward 就能懂整体流程
├── extractor.py     10.3 KB   CNN 特征提取
├── corr.py           6.6 KB   相关体（cost volume）
└── update.py         5.4 KB   ★ 最核心：GRU 迭代更新 + 凸上采样
```

**学习步骤**：

1. `python demo.py` 先跑通（Middlebury 的 cones/teddy 图，几百 KB）
2. 读 `core/raft_stereo.py` 的 `forward()` —— 抓住主干：提特征 → 建代价体 → 循环 N 次 GRU → 凸上采样
3. 精读 `core/update.py` —— 这是整个领域的精华
4. **打印张量形状**：每个模块前后加 `print(x.shape)`，看懂数据怎么流动

**过关标准**：能解释"为什么视差要迭代优化，而不是一次预测出来"。

（提示：因为一次预测对"重复纹理/无纹理区"几乎无解；迭代相当于让网络反复查证，每次修正一点。这个思想直接解释了 FFS 里 `valid_iters=8` 这个参数为什么存在。）

---

### 9. nianticlabs/monodepth2 —— 4.5k ⭐

```
https://github.com/nianticlabs/monodepth2
```

**是什么**：自监督单目深度估计的经典实现。只用左图就能出深度，训练时不需要真值深度标签。

**为什么对你重要**：它会彻底解答一个疑问 —— **"没有标注的深度真值，网络靠什么训练？"**

答案是：左右一致性 + 图像重建损失。右图可以视作"免费的监督信号"：网络预测一个视差，用这个视差把左图 warp 到右边，和真实右图对不上就说明预测错了。

**这个思想，正是 NVIDIA 造 FFS 那 140 万对训练数据时用的方法之一。** 理解了 monodepth2，你才能理解 FFS 的"零样本泛化"是怎么来的。

**看什么**：

- `networks/` —— 编码器-解码器结构
- `trainer.py`（25KB）—— 重点看损失函数的组装
- `layers.py` —— 找 `SSIM`、`get_smooth_loss`，就是自监督损失的两个组成

---

### 10. 位姿估计方向（阶段 ④ 后期再看）

等你走完上面九项，再回头看这些（它们是你要复现的机器人抓取系统的另一半）：

| 项目 | 用途 | 备注 |
|---|---|---|
| `NVlabs/FoundationPose` | 免训练 6DoF 位姿 | 换物体只换 CAD，但需 PyTorch3D，8GB 显存起 |
| `lmb-freiburg/FoundPose` | 免训练位姿 | **纯 torch，4GB 卡能跑，适合你** |
| `ZebraPose` | CAD 训练式快速定位 | 对应"需要训练样本"那条路线 |

**提醒**：这三个都依赖"深度图 + 检测框"作为输入，也就是你现在搭的这套 FFS 流水线的下游。**先把深度这条线打通，再接位姿。**

---

## 12 周进度安排（可弹性伸缩）

| 周次 | 任务 | 产出物 |
|---|---|---|
| W1 | micrograd：手推梯度 | 一份手写的梯度推导笔记 |
| W2 | nn-zero-to-hero 讲次 1–3 | 一个 NumPy 版的两层 MLP |
| W3 | nn-zero-to-hero 讲次 4–5（CNN） | 手写卷积前向/反向 |
| W4 | nn-zero-to-hero 讲次 6（Transformer）+ d2l 注意力章节 | 能解释 QKV 在算什么 |
| W5 | ultralytics：读 cfg 与 trainer，跑通一次训练 | 一次成功的 YOLO 训练 |
| W6 | ultralytics：改超参对比实验 | 一份超参对比表 |
| W7 | timm：读 ViT 实现 | 对照 FFS 的 ViT 骨干 |
| W8 | **RAFT-Stereo：跑通 demo + 读 raft_stereo.py** | 能画出数据流图 |
| W9 | **RAFT-Stereo：精读 update.py + 打印形状** | 一完整的 shape 追踪表 |
| W10 | **回头读 FFS，对照模块表逐行验证** | 更新你那份 FFS 原理文档 |
| W11 | monodepth2：理解自监督损失 | 能口述"为什么不需要真值" |
| W12 | 动手实验：用自己的相机数据跑 FFS | 一份实测深度结果 |

**W1–W4 和 W5–W7 可以并行**（前者是原理，后者是应用，互不阻塞）。W8 之后必须串行。

---

## 学习方法：三条比项目本身更重要

### 1. 不要"读"代码，要"改"代码

跑通之后，改一个数字，看输出怎么变。

**你已经用过这个方法了，而且效果很好** —— 排查 FFS 的 fp16 NaN 时，你就是靠"改一个变量 → 看输出变不变"定位到 cuDNN 的。这个方法对**理解原理**同样有效：

- 把 FFS 的 `valid_iters` 从 8 改成 4 → 看深度图哪里变差了（答案：边缘和细节）
- 把 RAFT 的迭代次数改成 1 → 看视差图糊成什么样

**每改一次，你对那个参数的理解就深一层，比读十遍源码都管用。**

### 2. 打印张量形状

每个模块前后 `print(x.shape)`。这招在视觉方向几乎万能。

我在写 FFS 原理文档时，就是靠它确认了一个关键数字：几何编码体的输入通道应为 `2 × 9 × 29 = 522`，而 `BasicMotionEncoder.convc1` 的权重实测正是 `(56, 522, 1, 1)` —— **对上了，说明理解无误**。

形状对得上，理解就对了一大半；对不上，说明你哪里想错了，立刻回头。

### 3. 一篇论文配一份代码看

只看论文 → 空的，全是公式没有实感。
只看代码 → 盲的，知道怎么写不知道为什么。

**正确姿势**：先读论文的图和实验部分（跳过数学推导），再对着代码找论文里提到的模块，最后回头补推导。

---

## 三个坑

### ❌ 别一上来读 `huggingface/transformers`

那是生产库，抽象层厚得离谱 —— 一个 `from_pretrained` 背后套了七八层。新手读它会严重怀疑自己。

**什么时候读**：等你读完 nanoGPT 之后。那时你会发现"结构其实一样，只是他们把它工程化了"。

### ❌ 别同时开两个坑

这四个阶段是串行的。同时看 micrograd 和 RAFT-Stereo 的结果，是两边都半懂不懂。

**唯一允许的例外**：d2l 作为字典可以常开，因为它本来就是查的，不是读的。

### ❌ nanoGPT / llm.c 先别碰

63.4k 和 31k 星，很诱人，但那是**大语言模型方向**，和你做的机器人视觉没有任何交集。等你把 FFS 这条线打通，如果还想了解 LLM，再回头看，届时你会发现很多概念（注意力、位置编码）已经学过了，学起来很快。

---

## 完整清单速查表

> 星数为 2026-09-27 实测值

| # | 项目 | 星数 | 阶段 | 最后更新 | 一句话 |
|---|---|---|---|---|---|
| 1 | [karpathy/micrograd](https://github.com/karpathy/micrograd) | 17.7k | ① | 2026-08 | 150 行搞懂反向传播 |
| 2 | [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | 24.5k | ① | 2024-08 | 从 MLP 讲到 GPT 的视频课 |
| 3 | [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh) | 81.1k | ② | 2024-07 | 中文教材，**当字典用** |
| 4 | [fastai/fastbook](https://github.com/fastai/fastbook) | 25.3k | ② | 2026-09 | 自顶向下，容易坚持 |
| 5 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 62.0k | ③ | 2026-09 | 最好的工程代码入门教材 |
| 6 | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | 37.2k | ③ | 2026-09 | 视觉 backbone 合集 |
| 7 | [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML) | 49.6k | ③ | 2026-03 | 补算法课不教的工程实践 |
| 8 | [princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) | 1.1k | ④ | 2026-08 | ★ **理解 FFS 的最短路径** |
| 9 | [nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2) | 4.5k | ④ | 2024-08 | 搞懂"没有真值怎么训练" |
| 10 | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | — | ④后期 | — | 免训练位姿（需 8GB 显存） |
| 11 | [lmb-freiburg/FoundPose](https://github.com/lmb-freiburg/FoundPose) | — | ④后期 | — | 免训练位姿（4GB 卡可跑） |

---

## 暂时别碰（理由充分后会再推荐）

| 项目 | 星数 | 为什么现在别碰 |
|---|---|---|
| karpathy/nanoGPT | 63.4k | LLM 方向，与你的目标无交集 |
| karpathy/llm.c | 31.0k | 同上，且是 C/CUDA，学习成本极高 |
| huggingface/transformers | 150k+ | 抽象层太厚，读完 nanoGPT 再看 |
| 各种 GAN / 扩散模型仓库 | — | 生成方向，与机器人视觉无关 |
| 强化学习仓库 | — | 除非你之后要做抓取策略学习 |

---

## 下一步

两条路，按你的偏好选：

**A. 立刻动手** —— 我把 RAFT-Stereo 拉到你的 WSL2 里跑通，配上"形状追踪版"的调试脚本，让你直接对着 FFS 的模块表对比着看。这条路最省时间。

**B. 先补原理** —— 从 micrograd 开始，我帮你搭一个本地学习环境（只装 numpy + jupyter，几十 MB），一步步跟着做。这条路最扎实。

如果你想两条并行，建议：**工作日做 A（动手，保持兴趣），周末做 B（补原理，防基础空虚）**。

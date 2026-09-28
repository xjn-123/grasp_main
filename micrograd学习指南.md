# micrograd 学习指南

> 配套脚本：`learn_micrograd.py`（已放在你的项目目录里）
> 项目位置：`/home/xiao/Depth_Learning/karpathy_micrograd`
> 整理日期：2026-09-27

---

## 一、这个项目到底是什么

一句话：**它把 PyTorch 里那句 `loss.backward()` 拆开给你看。**

你平时用 PyTorch 训练模型时，只写三行：

```python
loss.backward()      # 算梯度
optimizer.step()     # 更新参数
optimizer.zero_grad()# 清空梯度
```

这三行背后发生了什么，绝大多数人不知道。micrograd 用 **95 行 engine + 61 行 nn** 把这套机制完整实现了一遍，让你能逐句执行、逐步打印。

**它的价值不在于"能用"，而在于"能看懂"。** 你永远不会用 micrograd 训练真正的模型（它是纯 Python + 标量，慢得离谱），但看完它之后，PyTorch 对你来说就不再是黑箱了。

---

## 二、文件地图

```
karpathy_micrograd/
├── micrograd/
│   ├── engine.py     95 行  ★★★ 核心：自动微分引擎
│   └── nn.py         61 行  ★★  在上面搭的神经网络库
├── demo.ipynb              Karpathy 的完整演示（需要 numpy + matplotlib）
├── trace_graph.ipynb      把计算图画出来（需要 graphviz）
├── test/test_engine.py     单元测试（用 PyTorch 当标准答案做对比）
├── moon_mlp.png / gout.svg / puppy.jpg   只是图片
└── learn_micrograd.py      ← 我给你写的分步练习脚本
```

**你只需要读两个文件**：`engine.py` 和 `nn.py`，加起来 156 行。别的都可以先不管。

---

## 三、阅读顺序（按这个来，别跳）

### 第 1 遍：搞懂 Value（engine.py 第 2–11 行）

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data      # 这个数的值
        self.grad = 0         # 它的梯度（对最终结果的贡献度）
        self._backward = lambda: None   # 每个节点自带的"梯度回传函数"
        self._prev = set(_children)     # 它是从哪些节点算出来的
```

**只有两个字段是你要关心的**：`data`（值）和 `grad`（梯度）。后面三个带下划线的都是内部记账用的。

对应脚本：`python3 learn_micrograd.py 1`

---

### 第 2 遍：搞懂计算图是怎么"自己长出来"的（engine.py 第 13–33 行）

看 `__add__`：

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')   # ① 算结果，同时记下"我来自谁"
    def _backward():                                          # ② 顺便定义好"梯度怎么往回传"
        self.grad += out.grad
        other.grad += out.grad
    out._backward = _backward                                 # ③ 把回传函数挂在新节点上
    return out
```

**关键领悟**：你只是写了 `a + b`，但 Python 实际调用的是 `Value.__add__`。它在算出结果的同时，把**两件事**一起记下来了：

1. 我是从 `a` 和 `b` 来的（存进 `_prev`）
2. 将来梯度要怎么往回传（存进 `_backward`）

所以计算图不是谁画出来的，**是你在做前向计算的时候顺手长出来的**。这就是 PyTorch 说的"动态图"。

再看 `__mul__`（第 24 行），对比一下乘法的回传为什么是 `other.data * out.grad`：

```python
def _backward():
    self.grad  += other.data * out.grad    # d(ab)/da = b
    other.grad += self.data  * out.grad    # d(ab)/db = a
```

对应脚本：`python3 learn_micrograd.py 2`

---

### 第 3 遍：搞懂 backward()（engine.py 第 54–70 行）— 全项目最难的一段

```python
def backward(self):
    topo = []                      # ① 拓扑排序：把图"摊平"成一个顺序
    visited = set()
    def build_topo(v): ...
    build_topo(self)

    self.grad = 1                  # ② 终点对自己的梯度是 1（de/de = 1）
    for v in reversed(topo):       # ③ 倒着走，逐个调用每个节点的回传函数
        v._backward()
```

**为什么要排序？** 因为算 `a` 的梯度之前，必须先把"a 的下游"的梯度算完。倒着走一遍拓扑序，就保证了这个依赖关系。

**为什么要设 `self.grad = 1`？** 因为链式法则需要一个起点：`de/de = 1`。所有其他梯度都是从这里乘下去的。

这段一定要配合脚本看，它会把每一步的梯度变化打印出来：

```bash
python3 learn_micrograd.py 3
```

输出里你能看到：调用 `[3] op='+'` 时谁的 grad 从 0 变成了 1，调用 `[2] op='*'` 时又怎么变成 -2 和 2。**这就是链式法则在代码里的样子。**

---

### 第 4 遍：神经网络部分（nn.py）

三个类，一层套一层：

```
Neuron  → 一个神经元（w 权重 + b 偏置 + 可选 ReLU）
Layer   → 一排神经元
MLP     → 多层 Layer 叠起来
```

重点看 `Neuron.__call__`（第 20–22 行）：

```python
def __call__(self, x):
    act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)   # 加权求和
    return act.relu() if self.nonlin else act                 # 过激活函数
```

这就是你在课本上看到的 `y = ReLU(w·x + b)`，只不过这里每个加法乘法都会被 engine 记录进计算图。

**一个细节值得注意**：micrograd 只处理**标量**，所以一个 8 输入的神经元会被拆成 8 次乘法 + 8 次加法 + 1 次 ReLU，共 17 个计算节点。PyTorch 处理的是张量，一次矩阵乘法搞定。**这就是为什么 micrograd 慢，但也正因为拆得细，你才能看清每一步。**

对应脚本：`python3 learn_micrograd.py 6`

---

## 四、配套脚本怎么用

```bash
cd /home/xiao/Depth_Learning/karpathy_micrograd
python3 learn_micrograd.py        # 从头跑到尾（约 30 秒）
python3 learn_micrograd.py 3      # 只跑第 3 步
```

**不需要装任何东西** —— 脚本是纯标准库写的，你 WSL 里的 `python3` 直接能跑。

| 步骤 | 内容 | 对应源码 |
|---|---|---|
| 1 | Value 是什么 | engine.py 5–11 |
| 2 | 计算图怎么长出来 | engine.py 13–22 |
| 3 | backward 的两步：排序 + 倒着走 | engine.py 54–70 |
| 4 | 复现 README 的例子，验证环境 | — |
| 5 | **数值梯度检验**（证明 grad 是真导数） | — |
| 6 | 一个神经元的完整前向反向 | nn.py 13–28 |
| 7 | 完整训练 MLP，画出决策边界 | nn.py 45–60 |
| 8 | 与 PyTorch 对照（需要在有 torch 的环境跑） | — |

**强烈建议第 5 步认真看**：它用 `(f(x+h) - f(x-h))/2h` 这个导数定义去验证 `backward()` 的结果，误差在 1e-10 量级。看完你就再也不用怀疑"梯度到底是不是导数"了。

第 7 步的最终输出会长这样（实际运行结果）：

```
  step        loss    accuracy
     0    0.896280       56.7%
    50    0.898957       46.7%
   100    0.476722       80.0%
   150    0.130004       98.3%
   199    0.037867      100.0%
```

最后还会用 ASCII 字符画出网络学到的决策边界——一个清晰的圆。

---

## 五、我踩过的坑（你大概率也会踩）

### 坑 1：学习率必须给到 1.0，不是 0.05

这是本次调试最花时间的地方。第一版脚本我用了"看起来稳妥"的 `lr = 0.1 → 0.01`，结果：

| 初始 lr | 200 步后 loss | accuracy |
|---|---|---|
| 0.05 | 0.838 | **57%（完全卡死）** |
| 0.5 | 0.150 | 95% |
| **1.0** | **0.041** | **100%** |

**原因**：loss 做了 `1/N` 归一化，导致单个参数的梯度只有 0.05 量级。再乘 0.05 的学习率，每步只挪 0.0025 —— 200 步总共挪不到 0.5，而模型输出要跨越约 1.5 才能翻转符号，所以**准确率一位小数都不动**。

`demo.ipynb` 里用的就是 `learning_rate = 1.0 - 0.9*k/100`，从 1.0 起步。

**教训**：57% 这个数字很可疑（正好等于正样本比例），说明模型把所有样本判成了一类。遇到"准确率纹丝不动"，先查学习率，别急着改模型。

### 坑 2：ReLU 在 0 处不可导

数值梯度检验时，如果取 `x = -3`，此时 `x*x + 3x` 恰好等于 0，ReLU 正好卡在折点上，数值梯度和 `backward()` 会对不上（-2.0 vs -0.5）。

这不是 bug，是 ReLU 本身的性质。脚本里已经避开这个点了，并在输出里写了说明。

### 坑 3：Python 环境的坑

你的 WSL 里：

- 系统 `python3`（3.10.12）**没有** numpy / matplotlib / torch → 跑不了 `demo.ipynb`
- 但 micrograd 核心**只需要标准库** → `learn_micrograd.py` 可以直接跑
- FFS 的 venv 里有 torch，想做第 8 步的 PyTorch 对照，用：
  ```bash
  source /home/xiao/Depth_Learning/FFS/venv/bin/activate
  python3 learn_micrograd.py 8
  ```

---

## 六、过关标准

不用全记住，能答出下面三个问题就算过了：

1. **`a.grad` 里存的到底是什么？**
   → 是"最终输出对 `a` 的导数"，也就是 `a` 变动一点点时，输出会变多少。

2. **为什么 `backward()` 之前要先拓扑排序？**
   → 因为算某个节点的梯度，必须先知道它下游节点的梯度。倒着走一遍拓扑序就保证了这个顺序。

3. **为什么训练循环里每步都要 `zero_grad()`？**
   → 因为 `_backward()` 里用的是 `+=`（累加）。不清零的话，梯度会一轮一轮累积，越滚越大。

**能手写下面这个梯度推导，就是真的懂了**：

```
e = a*b + a，其中 a=2, b=-3
de/da = ?      de/db = ?
```
答案：`de/da = b + 1 = -2`，`de/db = a = 2`。脚本第 3 步会验证。

---

## 七、下一步

读完这个项目后，按顺序推进：

1. **看 Karpathy 的视频课** `nn-zero-to-hero`（B 站有中文字幕版），第 1 讲就是 micrograd，看完正好接上
2. **补基础**：`d2l-zh` 动手学深度学习，当字典查（详见 `神经网络学习路线-GitHub项目清单.md`）
3. **回到你的主线**：读完 micrograd 之后，去看 RAFT-Stereo —— 你在 FFS 里看到的迭代更新块、GRU、凸上采样，源头都在那里

---

## 附：想跑官方 demo.ipynb 的话

```bash
# 用国内镜像装依赖（WSL 里直连 pypi 很慢）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy matplotlib jupyter

# 然后
cd /home/xiao/Depth_Learning/karpathy_micrograd
jupyter notebook
```

打开 `demo.ipynb`，它就是 micrograd 的"官方教材"，内容和我的脚本第 7 步类似，但用的是 moon 数据集 + 16-16-1 的网络，图更好看。

**不过建议先看我的脚本** —— demo.ipynb 一次性把整个训练循环甩给你，而脚本是拆成 8 步、每步都能单独跑和改的。

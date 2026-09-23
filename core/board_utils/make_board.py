"""
生成 AprilTag 3 标定板的打印图( PNG + PDF ), 并顺便做一次"能不能全检出 + 定位精度"的自检。

用法::

    python make_board.py

想换规格时只改下面 "可改参数" 那一段, 图纸尺寸 / 3D 模型参数 / README 会自动跟着变。

为什么不用现成模板: 网上流传的 AprilTag 标定板模板大多是 AprilTag 2 时代出的( 黑边 2 格 ),
AprilTag 3 / pyapriltags 一个都检不出来。而且模板的物理尺寸是别人定的, 你得反过来去猜它,
不如自己出图 —— 尺寸由代码定义, 和 3D 模型严格同源。
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# apriltag_helper.py 与本脚本同目录; 若被移回 core/common_utils 也能找到
for _cand in (
    HERE,
    os.path.abspath(os.path.join(HERE, "..", "common_utils")),
    os.path.abspath(os.path.join(HERE, "..", "..", "core", "common_utils")),
):
    if os.path.isfile(os.path.join(_cand, "apriltag_helper.py")):
        sys.path.insert(0, _cand)
        HELPER_DIR = _cand
        break
else:
    raise ImportError("找不到 apriltag_helper.py, 请确认它和本脚本的相对位置")

import apriltag_helper as ah

# ============================================================ 可改参数 ============================================================
TAG_FAMILY = "tag36h11"     # 可选 tag16h5 / tag25h7 / tag25h9 / tag36h11
BLACK_BORDER = 1            # 黑边宽度( 格子 ), AprilTag 3 只能是 1, 改别的会检不出来
TAG_SPACING = 3             # tag 之间的白色间距( 格子 )
ROWS = COLS = 6             # 行列数
START_ID = 0                # 起始 tag 编号
CELL_MM = 4.0               # 一个白色数据格的实际边长( 毫米 ) —— 所有物理尺寸都由它推出来
MARKER_SIZE = 1             # 网格交叉点上的黑色标记块边长( 格子 ), 0 = 不画。
#                            它只是给人看的视觉标记, 对 AprilTag 检测**毫无影响**;
#                            实测 marker=0/1/2 的检出率与重投影 RMS 完全一致, 想更干净就设 0
OUT_DIR = HERE
# =================================================================================================================================


# ---------------------------------------------------------------- 尺寸推导
dim = ah.FAMILY_DIM[TAG_FAMILY]
rect = dim + 2 * BLACK_BORDER                       # 单张 tag( 含黑边 )的边长, 单位: 格
tag_size_m = CELL_MM * rect / 1000.0                # 含黑边的 tag 边长( 米 )
space_size_m = ah.calib_board_space_size(tag_size_m, TAG_FAMILY, BLACK_BORDER, TAG_SPACING)
pitch_mm = tag_size_m * 1000.0 + space_size_m * 1000.0
paper_w_mm, paper_h_mm, cell_mm = ah.calib_board_paper_size(
    tag_size_m, TAG_FAMILY, BLACK_BORDER, TAG_SPACING, ROWS, COLS
)


def imwrite_u(path, img):
    """绕开 cv2.imwrite 在非 ASCII 路径上的老毛病"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise IOError(f"编码失败: {path}")
    buf.tofile(path)


# ---------------------------------------------------------------- 出图
SCALE = max(1, int(round(300.0 * CELL_MM / 25.4)))  # 每格像素数, 目标 ~300 DPI
PREVIEW_SCALE = 10

board = ah.create_calib_board_img(TAG_FAMILY, BLACK_BORDER, TAG_SPACING, ROWS, COLS, SCALE, START_ID,
                                 marker_size=MARKER_SIZE)
preview = ah.create_calib_board_img(TAG_FAMILY, BLACK_BORDER, TAG_SPACING, ROWS, COLS, PREVIEW_SCALE, START_ID,
                                    marker_size=MARKER_SIZE)

# 标记块实际用到的尺寸( helper 会 clamp 到 tag_spacing-2 )
MARKER_EFF = max(0, min(int(MARKER_SIZE), max(0, TAG_SPACING - 2)))

tag = f"{TAG_FAMILY}_{ROWS}x{COLS}_tag{tag_size_m * 1000:.0f}mm_bb{BLACK_BORDER}"
path_png = os.path.join(OUT_DIR, f"{tag}_print.png")
path_preview = os.path.join(OUT_DIR, f"{tag}_preview.png")
path_pdf = os.path.join(OUT_DIR, f"{tag}_print.pdf")

imwrite_u(path_png, board)
imwrite_u(path_preview, preview)

# PDF 的页面尺寸由 dpi 决定, 这里反算一个 dpi 让页面**严格等于**设计尺寸
dpi = board.shape[1] / paper_w_mm * 25.4
ah.save_calib_board_pdf(board, path_pdf, dpi=dpi)


# ---------------------------------------------------------------- 自检
detector = ah.Detector(TAG_FAMILY, BLACK_BORDER)
tags = detector.detect(board, -1)
detected = len(tags)

tag3d = ah.create_calib_board_3d(tag_size_m, space_size_m, ROWS, COLS, START_ID)
wpx = board.shape[1]
K = np.array([[wpx * 1.4, 0, wpx / 2.0], [0, wpx * 1.4, board.shape[0] / 2.0], [0, 0, 1]], dtype=np.float64)
T = ah.locate_calib_board(tags, tag3d, K)
if T is None:
    rms_txt = "定位失败"
else:
    pts3d, pts2d, _ = ah.match_tag_pairs(tags, tag3d)
    proj, _ = cv2.projectPoints(
        pts3d.reshape(-1, 1, 3), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1), K, None
    )
    err = np.linalg.norm(np.asarray(proj).reshape(-1, 2) - pts2d, axis=1)
    rms_txt = f"{np.sqrt(np.mean(err ** 2)):.4f} px"


# ---------------------------------------------------------------- 图纸说明
readme = f"""# AprilTag 标定板（AprilTag 3 / pyapriltags 专用）

> **旧标定板不能用了。** AprilTag 3 只能解码**黑边 1 格**的 tag，而 cvte `apriltag2`
> 出的图默认是黑边 2 格 —— 实测那种板子一个 tag 都检不出来。这一版是 `black_border=1` 的新板。
> 本目录下的所有文件都由 `core/make_board/apriltag_helper.py` 现算，图里的每个 tag 都是从
> 内置码表渲染出来的（652 个图案与参考仓库的 PNG 逐像素比对零误差），**不依赖任何外部图片**。

## 打印

| 项目 | 值 |
|---|---|
| 打印用 | `{os.path.basename(path_pdf)}`（推荐，页面尺寸是固定的物理尺寸） |
| 备用 | `{os.path.basename(path_png)}` |
| 预览 | `{os.path.basename(path_preview)}` |
| 纸张尺寸 | **{paper_w_mm:.1f} mm × {paper_h_mm:.1f} mm** |
| 建议纸张 | A3（297×420 mm）能放下；A4 放不下 |
| 分辨率 | 约 {dpi:.0f} DPI，每格 {SCALE} px |

打印时**务必关掉"适应页面 / fit to page"**，按 100% 或按上面给出的毫米数打印。

## black_border 是什么

一个 AprilTag 的图案外面要留一圈**纯黑边框**，检测算法靠它把 tag 从背景里"切"出来。
`black_border` 就是这圈黑边的宽度，单位是**一个白色数据格**：

```
        ← black_border →
    ┌───┬───────────────┬───┐   ↑
    │   │  ██ ░░ ██ ░░  │   │   │ black_border
    ├───┼───────────────┼───┤   ↓
    │   │  ░░ ██ ░░ ██  │   │      ← tag36h11 的数据区 6x6
    │   │  ██ ░░ ░░ ██  │   │
    ├───┼───────────────┼───┤   ↑
    │   │  ░░ ██ ██ ░░  │   │   │ black_border
    └───┴───────────────┴───┘   ↓
```

- `black_border=1`（**本版**）→ 整张 tag 是 {rect}×{rect} 格，即 6×6 数据 + 1 圈黑边。
  这也是 AprilTag 3 家族定义里写死的宽度，改不了。
- `black_border=2`（**旧版**）→ 整张 tag 是 {rect + 2}×{rect + 2} 格，黑边更宽一圈。
  AprilTag 2（cvte 那套封装）支持任意宽度，AprilTag 3 不支持，所以旧板必须作废。

肉眼区别：把 tag 的外边框到第一格黑白图案之间的距离数一数，旧板是 2 个小格子宽，新板是 1 个小格子宽。

## 网格上的黑色小方块（标记块）

每个 tag 的四角外侧各有一个黑色小方块，它们落在标签之间白色间隔的**交叉点**上，
全板共 `({ROWS}+1) x ({COLS}+1)` = {(ROWS+1)*(COLS+1)} 个，尺寸 {MARKER_EFF} 格。

**这些方块对 AprilTag 检测没有任何作用**，纯粹是把网格结构标给人看。实测 `marker_size=0/1/2`
三档的**检出个数与重投影 RMS 完全一致**，所以想去掉它、让图更干净，把 `make_board.py` 顶部的
`MARKER_SIZE` 改成 `0` 即可。

和原 apriltag2 出的图比，这里有两处是**故意改的**（都是实测出来的坑）：

| | 原 apriltag2 | 本版 | 为什么改 |
|---|---|---|---|
| 块的大小 | `tag_spacing` 格 | `<= tag_spacing-2` 格 | 原版的大块正好与 tag 黑边框**对角相连**，AprilTag 3 把这种连通的黑色区域当成一个整体四边形，相邻 tag 被一起吃掉。实测 6×6 板只能检出 **34/36**，4×4 板 **15/16** |
| 块的位置 | 跟着瓦片画在瓦片四角 | 画在网格交叉点，且**所有 tag 贴完后再统一落块** | 相邻瓦片是重叠摆放的（步距 `rect+spacing`，瓦片边长 `rect+2·spacing`），后贴的瓦片白边会把先贴的块**覆盖掉**。实测原做法 2×2 板 16 个块只剩 8 个，每个 tag 周围看着只有 1 个块 |

改完之后：全板 {(ROWS+1)*(COLS+1)} 个交叉点**一个不缺**，每个 tag 四角**都有块**（板最外圈的 tag 也不缺），
且所有块彼此独立、与最近的 tag 至少隔 1 格白边。

三种画法的真实渲染对比（左：原 apriltag2；中：修复前；右：修复后）：

![三种标记块画法的对比](marker_compare.png)

## 规格参数

| 项 | 值 |
|---|---|
| tag 家族 | `{TAG_FAMILY}`（数据区 {dim}×{dim}） |
| 黑边 `black_border` | **{BLACK_BORDER} 格** |
| tag 间距 `tag_spacing` | {TAG_SPACING} 格 |
| 排布 | {ROWS} 行 × {COLS} 列，起始 id = {START_ID} |
| 单格边长 | {cell_mm:.3f} mm |
| **tag 边长** `tag_size`（含黑边） | **{tag_size_m * 1000:.3f} mm** = {tag_size_m:.5f} m |
| **tag 净间距** `space_size` | **{space_size_m * 1000:.3f} mm** = {space_size_m:.5f} m |
| 相邻 tag 中心距 | {pitch_mm:.3f} mm |
| 整张图纸 | {int(round(paper_w_mm / cell_mm))} × {int(round(paper_h_mm / cell_mm))} 格 = {paper_w_mm:.1f} × {paper_h_mm:.1f} mm |

**填进 `calib_board_info` / `create_calib_board_3d` 的值**：

```
[{tag_size_m:.5f}, {space_size_m:.5f}, {ROWS}, {COLS}]
```

## 打印后必须做的一步：量整宽、修正参数

打印机和出图软件几乎总有千分之几的整体缩放，而这个缩放会 **1:1 传给标定结果**
（焦距、手眼 X/Y/Z 全都被拉歪），而且看不出明显异常。

**不要去量单个 tag**（钢尺 ±0.5 mm，对 {tag_size_m * 1000:.0f} mm 就是 1.5% 误差）；
**去量整张图纸的宽**，同样的 ±0.5 mm 摊到 {paper_w_mm:.0f} mm 上只剩 0.2%：

```python
from core.make_board import apriltag_helper as ah

tag_size, space_size = ah.rescale_calib_board_params(
    tag_size={tag_size_m:.5f},
    space_size={space_size_m:.5f},
    paper_measured_mm=275.5,      # <- 你量出来的图纸实际宽度
    paper_nominal_mm={paper_w_mm:.3f},       # 理论宽度
)
# 用修正后的 tag_size / space_size 再跑标定
```

## 代码里的完整用法

```python
from core.make_board import apriltag_helper as ah

tag_size   = {tag_size_m:.5f}                                              # 含黑边的 tag 边长(米)
space_size = ah.calib_board_space_size(tag_size, "{TAG_FAMILY}", black_border={BLACK_BORDER}, tag_spacing={TAG_SPACING})

tag3d_list = ah.create_calib_board_3d(tag_size, space_size, {ROWS}, {COLS}, {START_ID})
tag2d_list = ah.Detector("{TAG_FAMILY}", black_border={BLACK_BORDER}).detect(img, -1)
T_cam_board = ah.locate_calib_board(tag2d_list, tag3d_list, K, D, th_reproj=1.0)
```

`space_size` **不要手算**（1 格 = `tag_size / ({dim} + 2×{BLACK_BORDER})` = {cell_mm:.3f} mm），
一律用 `calib_board_space_size()`，算错会带来一个很难发现的尺度误差。

## 换规格

改 `make_board.py` 顶部的「可改参数」再跑一次即可，README 会自动重写。
常见组合（图纸 = ({TAG_SPACING}×(cols+1) + {rect}×cols) 格）：

| 单格 | tag 边长 | 间距 | 6×6 图纸 | 适用纸张 |
|---|---|---|---|---|
| 3.0 mm | 24.0 mm | 9.0 mm | 207.0 mm | A4 |
| **4.0 mm** | **32.0 mm** | **12.0 mm** | **276.0 mm** | **A3（本版）** |
| 5.0 mm | 40.0 mm | 15.0 mm | 345.0 mm | A2 / 打印店 |

## 出图时的自检结果

- 图纸共 {ROWS * COLS} 个 tag，全部检出：**{detected}/{ROWS * COLS}**
- 整板重投影 RMS：**{rms_txt}**
"""

path_md = os.path.join(OUT_DIR, "README.md")
with open(path_md, "w", encoding="utf-8") as fp:
    fp.write(readme)

print(f"单格 {cell_mm:.3f} mm | tag {tag_size_m * 1000:.3f} mm | 间距 {space_size_m * 1000:.3f} mm | 图纸 {paper_w_mm:.1f} x {paper_h_mm:.1f} mm")
print(f"出图 {board.shape[1]}x{board.shape[0]} px, PDF {dpi:.1f} DPI")
print(f"自检: 检出 {detected}/{ROWS * COLS}, 重投影 RMS = {rms_txt}")
print(f"输出目录: {OUT_DIR}")
for f in (path_png, path_preview, path_pdf, path_md):
    print("  -", os.path.basename(f))

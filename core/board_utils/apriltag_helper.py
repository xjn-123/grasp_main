# -*- coding: utf-8 -*-
"""
apriltag_helper.py
==================

基于 **pyapriltags**(AprilTag 3) 的封装层, 对外提供与 GitHub 仓库
`cvte-robotics/apriltag2 <https://github.com/cvte-robotics/apriltag2>`_ 同名、同语义的高级接口,
工程里原本用 ``apriltag2`` 写的代码基本可以只改一行 import 就能跑。

原 apriltag2 的对外接口( 本模块一一对应 )::

    detector = apriltag2.Detector(tag_family="tag36h11", black_border=2)
    tag2d_list = detector.detect(img, -1)          # Tag2D 列表; 传负数=自动多尺度搜索
    apriltag2.Detector.draw(bgr_img, tag2d_list)   # 可视化( 静态方法 )
    board_img = apriltag2.create_calib_board_img(tag_family, black_border, tag_spacing,
                                                 rows, cols, scale, start_tag_id)
    tag3d_list = apriltag2.create_calib_board_3d(tag_size, space_size, rows, cols, start_tag_id)
    T_cam_board = apriltag2.locate_calib_board(tag2d_list, tag3d_list, K, D, th_reproj=1.0)

⚠️ 从 apriltag2 迁过来唯一的**硬不兼容点**: **黑边框宽度**

    apriltag2(AprilTag 2) 的黑边宽度是可配的, 你们原来常用 black_border=2;
    pyapriltags(AprilTag 3) 把黑边写死成 **1 个格子**。

    实测 bb ∈ {0,1,2,3} x wb ∈ {0,1,2} 共 12 种组合、每种 6 个 id:
    **只有 black_border == 1 的图案能被解码**, 其余全部 MISS( 连 ID 都读不出来 )。

    所以 ``black_border=2`` 的老标定板必须重新按 ``black_border=1`` 出图打印,
    Detector(black_border=2) 只会打一条 warning 但照样检不出东西。
    create_calib_board_img 干脆直接报错拦住, 免得印完才发现。

本模块在 pyapriltags 之上补齐的高级能力::

    1) Tag2D / Tag3D 数据结构( corners 形状与顺序统一, 自带面积/周长/外扩等便捷方法 )
    2) 输入容错: 彩色图自动转灰度, 非连续数组自动整理, 非 uint8 自动量化
    3) 结果过滤: 边界截断过滤、按 hamming / decision_margin / id 白名单过滤、按质量取 top-k
    4) 亚像素角点细化( cv2.cornerSubPix ), 窗口大小随 tag 尺寸自适应, 标定时能明显提升角点精度
    5) **多尺度检测**: detect(img, 负数) 时自动在 1.0/0.75/0.5/0.25 里挑标签最多的那个尺度
    6) **标定板图案生成**: 码表内置, 不依赖任何外部图片资源, 直接画出可打印的标定板
    7) 单 tag 位姿估计与整板 PnP 定位( 带重投影误差迭代剔除离群点 )
    8) 由 3D 角点构建 tag 坐标系( 与旧版 core/vision_utils.compute_tag_pose 的"邻边建法"一致 )

角点顺序约定( 全局统一, 不要改 )::

    corners[0] = tag 自身视角的 左下    corners[1] = 右下
    corners[2] = 右上                   corners[3] = 左上

即"**从 tag 自己的左下角出发逆时针一圈**"。无论 tag 在图像里怎么旋转, 检测器返回的都是这个
自转不变的规范顺序( 已实测验证 ), 因此可以直接和标定板模型的 corners[i] 一一对应。
注意这不是"图像视角从左上开始", 两者正好差半圈, 混用会让位姿整体翻转。

单位约定::

    像素坐标 -> (u, v);  3D 坐标与 tag_size 一律用 **米**;  位姿一律用 4x4 齐次矩阵 T_dst_src
"""

from __future__ import annotations

import base64
import logging
import zlib
import numpy as np
import cv2
import pyapriltags
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# 角点顺序的文字说明, 画图和 debug 时打日志用
CORNER_NAMES: Tuple[str, str, str, str] = ("左下", "右下", "右上", "左上")

ArrayLike = Union[np.ndarray, Sequence, List]


######################################################### 通用工具 #########################################################


def to_gray_u8(img: ArrayLike) -> np.ndarray:
    """
    把任意常见图像输入整理成 pyapriltags 能吃的格式: 单通道 / uint8 / 内存连续
    Args:
        img (ArrayLike): 输入图像, 支持 CV_8UC1 灰度图、CV_8UC3(BGR) 彩色图、CV_8UC4(BGRA), 也支持浮点图
            ( 浮点峰值 <= 1 时按 0~1 归一化图处理, 自动乘 255 )
    Returns:
        (np.ndarray): (H, W) / uint8 / C-contiguous 的灰度图
    """

    arr = np.asarray(img)

    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"unsupported image channels: {channels}")
    elif arr.ndim != 2:
        raise ValueError(f"unsupported image shape: {arr.shape}, expect (H, W) or (H, W, C)")

    if arr.dtype != np.uint8:
        # 浮点图自动归一化: 峰值 <= 1 的( 常见的 img/255.0 )先拉回 0~255 再量化,
        # 否则会被直接截断成 0/1, 变成一张近乎全黑的图, 检测结果全空
        if np.issubdtype(arr.dtype, np.floating) and arr.size > 0:
            peak = float(np.nanmax(arr))
            if 0.0 < peak <= 1.0 + 1e-6:
                arr = arr * 255.0
            # end if
        # end if
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # end if

    # pyapriltags 内部按 stride 直接拷 buffer, 非连续数组必须拷贝一份, 否则结果是错的
    return np.ascontiguousarray(arr)


# end def to_gray_u8


def expand_corners(corners: ArrayLike, ex_ratio: float) -> np.ndarray:
    """
    把四个角点沿各自的对角线方向向外扩展( 与旧版 vision_utils.compute_tag_mask 的算法一致 )
    Args:
        corners (ArrayLike): (4, 2) 角点
        ex_ratio (float): 外延比例, 0.1 表示向外扩展 10%, 负数表示内缩
    Returns:
        (np.ndarray): (4, 2) 扩展后的角点
    """

    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    out = np.empty_like(pts)
    for i in range(4):
        out[i] = pts[i] + (pts[i] - pts[(i + 2) % 4]) * ex_ratio
    return out


# end def expand_corners


def T_from_Rt(R: ArrayLike, t: ArrayLike) -> np.ndarray:
    """
    用旋转矩阵和平移向量组装 4x4 齐次变换矩阵
    Args:
        R (ArrayLike): (3, 3) 旋转矩阵
        t (ArrayLike): (3,) 或 (3, 1) 平移向量
    Returns:
        (np.ndarray): (4, 4) 齐次变换矩阵
    """

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(-1)[:3]
    return T


# end def T_from_Rt


def inv_tf(T: ArrayLike) -> np.ndarray:
    """
    求 4x4 齐次变换矩阵的逆
    Args:
        T (ArrayLike): (4, 4) 齐次变换矩阵
    Returns:
        (np.ndarray): (4, 4) 逆矩阵
    """

    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# end def inv_tf


def as_camera_matrix(intrinsic: ArrayLike) -> np.ndarray:
    """
    把相机内参整理成 3x3 内参矩阵
    Args:
        intrinsic (ArrayLike): [fx, fy, cx, cy] 或已经展开的 3x3 矩阵
    Returns:
        (np.ndarray): (3, 3) 相机内参矩阵
    """

    arr = np.asarray(intrinsic, dtype=np.float64)
    if arr.size == 4:
        fx, fy, cx, cy = arr.reshape(-1)
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return arr.reshape(3, 3)


# end def as_camera_matrix


def is_pose_front_facing(R: ArrayLike, t: ArrayLike) -> bool:
    """
    判断一个位姿是不是"tag 正面朝着相机"的那组解

    Args:
        R (ArrayLike): (3, 3) 旋转矩阵
        t (ArrayLike): (3,) 平移向量

    Returns:
        (bool): True 表示 tag 正面朝向相机( 正常解 ), False 表示是镜像出来的那个解

    Note:
        1) PnP 对平面目标天然存在二义性: 关于目标平面镜像的那组位姿能产生完全相同的投影。
           对于本模块的 tag 坐标系约定( x 轴 左上->右上, y 轴 左上->左下 ), 正常解满足
           dot(R[:, 2], normalize(t)) > 0, 镜像解则为负
        2) cv2 的 SOLVEPNP_IPPE_SQUARE 在部分 OpenCV 版本里( 实测 5.0.0 )会返回镜像解, 必须靠这个判据挑出来
    """

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    norm = float(np.linalg.norm(t))
    if norm < 1e-12:
        return False

    return float(np.dot(R[:, 2], t / norm)) > 0.0


# end def is_pose_front_facing


def build_tag_pose_from_corners3d(corners3d: ArrayLike) -> np.ndarray:
    """
    由 tag 的 3D 角点构建 tag 坐标系到相机坐标系的变换矩阵( "邻边建法" )
    Args:
        corners3d (ArrayLike): (4, 3) tag 的 3D 角点, 顺序必须是 左下/右下/右上/左上
    Returns:
        (np.ndarray): T_cam_tag, (4, 4)

    Note:
        1) 建系方式: x 轴 = 左下->右下, y 轴 = 左下->左上, z 轴 = x × y( 由 tag 指向观察者 )
           这与旧版 core/vision_utils.compute_tag_pose 完全一致, 但与 arm_utils.GripperBody
           的"对角线建法"相差 45°, 两个模块混用时务必注意( 那个坑在工程文档里写得很清楚 )
    """

    pts = np.asarray(corners3d, dtype=np.float64).reshape(4, 3)

    axis_x = pts[1] - pts[0]
    axis_x = axis_x / np.linalg.norm(axis_x)

    axis_y = pts[3] - pts[0]
    axis_y = axis_y / np.linalg.norm(axis_y)

    axis_z = np.cross(axis_x, axis_y)
    axis_z = axis_z / np.linalg.norm(axis_z)

    axis_y = np.cross(axis_z, axis_x)  # 重新正交化, 保证 R 是严格的正交矩阵
    axis_y = axis_y / np.linalg.norm(axis_y)

    return T_from_Rt(np.stack([axis_x, axis_y, axis_z], axis=1), np.mean(pts, axis=0))


# end def build_tag_pose_from_corners3d


def find_tag(tag_list: Sequence["Tag2D"], tag_id: int) -> Optional["Tag2D"]:
    """
    按 id 在检测结果里查找 tag
    Args:
        tag_list (Sequence[Tag2D]): Tag2D 列表
        tag_id (int): 要找的 tag id
    Returns:
        (Tag2D | None): 找到则返回该 tag, 否则返回 None
    """

    for tag in tag_list:
        if tag.id == tag_id:
            return tag
    return None


# end def find_tag


def sort_tags(tag_list: Sequence["Tag2D"], by: str = "id") -> List["Tag2D"]:
    """
    对检测结果排序, 返回新的列表
    Args:
        tag_list (Sequence[Tag2D]): Tag2D 列表
        by (str): "id" 按编号升序; "margin" 按 decision_margin 降序( 质量好的在前 ); "area" 按面积降序
    Returns:
        (List[Tag2D]): 排好序的新列表
    """

    if by == "id":
        return sorted(tag_list, key=lambda t: t.id)
    if by == "margin":
        return sorted(tag_list, key=lambda t: t.decision_margin, reverse=True)
    if by == "area":
        return sorted(tag_list, key=lambda t: t.area, reverse=True)
    raise ValueError(f"unknown sort key: {by}, expect 'id' / 'margin' / 'area'")


# end def sort_tags


############################################### tag 码表与图案生成 ###############################################

FAMILY_DIM: Dict[str, int] = {
    "tag16h5": 4,
    "tag25h7": 5,
    "tag25h9": 5,
    "tag36h9": 6,
    "tag36h11": 6,
}
""" tag 家族 -> 图案边长( 格子数 ), 例如 tag36h11 是 6x6 """

FAMILY_MIN_HAMMING: Dict[str, int] = {
    "tag16h5": 5,
    "tag25h7": 7,
    "tag25h9": 9,
    "tag36h9": 9,
    "tag36h11": 11,
}
""" tag 家族 -> 最小汉明距离( 能纠正 (min_hamming-1)/2 个比特错误 ) """

_TAG_CODES_B64: Dict[str, str] = {
    "tag16h5": "GyOlLmo0uUWmeWt/WLNF51n+bRULOKvwhA02R3KMEK88CbSTA6WPRjfhlVdC3x0c3Omtc1+tMNXKBy6v",
    "tag25h7": "DXdLAOaTFgGrmaUBNaXDAPqqUgGYzawAItnKAa0vLAByNbsANztKAUtShgFMnckA6r8jAHTLQQE50dAB6wpnAXUWhQBOM1AB2D5uAJ1E/QDsVaoAdmHIASibXgGypnwAi8NHAVDJ1gGMDosAURQaAWUrVgHIUz8Beo3VAMmeggDxzPoABeQ2AQYvegDLNAkBVotqAWqipgFFVfgA5MKVAanIJACW/CsBqhNoAb4qpAEkNFcBc0UEAcJWsQARaF4A/ptlAWNa1QFl8FsAZ4biAFS66QFafH0Bgqr1AdG7ogH56RoAUZ4lASsGNAF6F+EAqAftACS+YgGLElkAjz5mAcuDGgBZu0UAWgaJAXCzSwARt28Bd8AiAXqh7AD0wdsAQ9OIAF2sWADoAroAnR0aAOwuxwHFS5IAs8rcABVtiABlyXgBmsZbAGFicQHM4nQB9BDtAahqFQCKKj4A7VInAFHGUwFwFnQBBVt2ALvAGQGDp3IBoaxPAFcS8wD8QSQBSDcNABUfwgA3UKwAkuWAARAyfQCHcaIAr+4rAFf/JgCCDmkAXHZ3ANfhqQEavkABOh6qAVxPlAEyUJsBl5gWALmOBgG8DfMAUaEGAZU+1QHujDQByk/PALWLcgHsHtwA2+hpACMVbgEl+gUBDLuKAV0nxAB2jm0A29boANdv4QCCJqwBW0N3AN1ZowBOnDoAmpEjARdY4gE2qAIApEUVAI2cIAFpX7sAAh/cAX5fXQCBBS0BwoY3AQlU4QCZNaoB2Ko5ASqdsACPSFQAHDU8AXlglwASW7IANNutAa4jywE4VxcBuDswARZ31ADqzogBZ/m6ADltIgGb6TUBxa00AE04LgD605AAEycjALFJfQHWhKoA+N3CAEZWZgFfNE8AsXYiANddJQHMTG8B/K9KAKZtxACzx4UAyx8xAU9snABH2YcB5HiFAAu/4gBMG6AAO0mhAGbXegCC/swAWxuYAYXMygHbLFYAeA5bAcVmjwC/MjMAVOcsAXZqCQC649UBQeonAN8SRAG0uWcAGqXaABfLHQD9Sk0A1TVjADQj7gBVTn0B8LC4AeOZSQH6PVEB8lx2AJCvVgCsFi4BbNjTAZsn/wDdIogBeNSZANLAjQBmtjQAJpXPAD1EhgEpjnoApWqcAX2i8gE2ISsBDc3QACCzLAEL230BOzUFAK8sWwEHpeUB5fEgAVpgFAFM/k4BNIFWAJKfGwGn0nQBHStpAP7kOQA9/6oATCKWAHefPAGP7hAB6nvxAF37mQBBcTMATbUCAHA6IwE=",
    "tag25h9": "8ctVAbbR5AFoC3sBzcnqAc4ULgG7SDUA5ld3AKtdBgHnoroBiKbeACfZgQBBslEArsjbABkO5QHSGVgBgoJtATXgYwGBm50AxO5zAQk6rgBRfF8A/DehAWKV3ABFLoABLFTDAaQPhwAJR5EA8IRmAaXyyAC7PoMAf3FZAFDQPAHRCvoAsGO3Ac6RuQA=",
    "tag36h11": "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",
}
"""
内置码表( 与 ethz_apriltag2 的 TagXXhYY.h 完全一致 )

编码方式: 每个 code 是一个无符号整数, 按 **小端** 打包成 ``ceil(dim*dim/8)`` 个字节后串起来,
          再做 base64。之所以用 base64 而不是直接写字面量数组, 是为了让码表紧凑且不影响阅读。

注意: tag36h9 家族有 5329 个码( 约 26KB ), 体积太大且几乎用不上, 故未内置;
      需要的话可以到 ethz_apriltag2 里取 Tag36h9.h 自行补进来。
"""


def _family_dim(family: str) -> int:
    """
    查某个家族的图案边长, 查不到就抛 ValueError( 而不是 KeyError, 报错信息更友好 )
    Args:
        family (str): 家族名
    Returns:
        (int): 图案边长( 格子数 )
    """

    try:
        return FAMILY_DIM[str(family).strip()]
    except KeyError:
        raise ValueError(f"未知的 tag 家族: {family}, 可选: {sorted(FAMILY_DIM)}")
    # end try


# end def _family_dim


@lru_cache(maxsize=None)
def load_tag_codes(family: str) -> Tuple[int, ...]:
    """
    加载某个 tag 家族的全部码字( 带缓存 )
    Args:
        family (str): 家族名, 如 "tag36h11"
    Returns:
        (Tuple[int, ...]): 码字列表, 下标即 tag id
    Raises:
        ValueError: 家族名不认识或没有内置码表
    """

    family = str(family).strip()
    dim = _family_dim(family)
    if family not in _TAG_CODES_B64:
        raise ValueError(f"家族 {family} 没有内置码表( 体积原因未打包 ), 可选: {sorted(_TAG_CODES_B64)}")

    dim = FAMILY_DIM[family]
    nbytes = (dim * dim + 7) // 8
    raw = base64.b64decode(_TAG_CODES_B64[family])
    return tuple(int.from_bytes(raw[i * nbytes:(i + 1) * nbytes], "little") for i in range(len(raw) // nbytes))


# end def load_tag_codes


def list_tag_families(with_codes_only: bool = False) -> List[str]:
    """
    列出支持的 tag 家族
    Args:
        with_codes_only (bool): True 时只返回"本模块内置了码表、能直接画图"的家族
    Returns:
        (List[str]): 家族名列表
    """

    source = _TAG_CODES_B64 if with_codes_only else FAMILY_DIM
    return sorted(source)


# end def list_tag_families


def tag_family_count(family: str) -> int:
    """
    某个家族一共有多少个可用 tag id
    Args:
        family (str): 家族名
    Returns:
        (int): tag 数量
    """

    return len(load_tag_codes(family))


# end def tag_family_count


def get_tag_code(family: str, tag_id: int) -> int:
    """
    取某个 tag 的码字
    Args:
        family (str): 家族名
        tag_id (int): tag 编号, 必须落在 [0, tag_family_count(family)) 内
    Returns:
        (int): 码字
    Raises:
        IndexError: tag_id 越界
    """

    codes = load_tag_codes(family)
    tag_id = int(tag_id)
    if tag_id < 0 or tag_id >= len(codes):
        raise IndexError(f"tag id {tag_id} 越界, 家族 {family} 只有 {len(codes)} 个 tag")
    return codes[tag_id]


# end def get_tag_code


def render_tag_pattern(family: str, tag_id: int) -> np.ndarray:
    """
    渲染 tag 的核心图案区( 不含黑边和白色静区 )
    Args:
        family (str): 家族名
        tag_id (int): tag 编号
    Returns:
        (np.ndarray): (dim, dim) uint8 灰度图, 0=黑 255=白

    Note:
        1) 比特排布( 已对 ethz_apriltag2 自带的全部 652 张 PNG 做过逐像素校验, 零误差 ):
               bit_idx(row, col) = dim*dim - 1 - (dim*row + col)
               code 的第 bit_idx 位: **1 -> 白像素, 0 -> 黑像素**
           其中 row 从上往下数, col 从左往右数
    """

    dim = _family_dim(family)
    code = get_tag_code(family, tag_id)

    rows = np.arange(dim).reshape(dim, 1)
    cols = np.arange(dim).reshape(1, dim)
    bit_idx = dim * dim - 1 - (dim * rows + cols)
    bits = (np.uint64(code) >> np.asarray(bit_idx, dtype=np.uint64)) & np.uint64(1)

    black_mask = (bits == 0)
    pattern = np.full((dim, dim), 255, dtype=np.uint8)
    pattern[black_mask] = 0
    return pattern


# end def render_tag_pattern


def render_tag_image(
    family: str,
    tag_id: int,
    black_border: int = 1,
    white_border: int = 1,
    scale: int = 1,
) -> np.ndarray:
    """
    渲染单张 tag 图片( 白静区 + 黑边框 + 核心图案 )
    Args:
        family (str): 家族名
        tag_id (int): tag 编号
        black_border (int): 核心图案外面黑色边框的宽度( 格子数 ), 默认 1
        white_border (int): 黑色边框外面再留一圈白色静区的宽度( 格子数 ), 默认 1
        scale (int): 放大倍数, 用最近邻插值保证边缘锐利, 默认 1( 不放大 )
    Returns:
        (np.ndarray): uint8 灰度图, 边长 = dim + 2*black_border + 2*white_border ( 再乘 scale )

    Note:
        1) black_border=1 / white_border=1 出来的图与 apriltag2 仓库 python/apriltag2/tag_images/
           下的 PNG **逐像素一致**, 可以直接替换那批图片资源
        2) scale 用 INTER_NEAREST, 放大后每个格子是 scale x scale 的正方形块
    """

    if black_border < 0 or white_border < 0:
        raise ValueError("black_border / white_border 不能为负数")

    pattern = render_tag_pattern(family, tag_id)
    dim = pattern.shape[0]

    full = dim + 2 * black_border + 2 * white_border
    img = np.full((full, full), 255, dtype=np.uint8)          # 底: 纸是白的

    if black_border > 0:
        inner = white_border + black_border
        img[white_border:full - white_border, white_border:full - white_border] = 0   # 黑边框
        img[inner:full - inner, inner:full - inner] = pattern                          # 核心图案
    else:
        img[white_border:full - white_border, white_border:full - white_border] = pattern
    # end if

    if scale != 1:
        img = cv2.resize(img, (full * scale, full * scale), interpolation=cv2.INTER_NEAREST)

    return img


# end def render_tag_image


######################################################### 数据结构 #########################################################


@dataclass
class Tag2D:
    """
    图像里检测到的一个 AprilTag( 对应 apriltag2.Tag2D )
    """

    id: int
    """ 标签编号 """

    corners: ArrayLike
    """ 四个角点的像素坐标, (4, 2), 顺序: 左下/右下/右上/左上( tag 自身视角 ) """

    center: Optional[ArrayLike] = None
    """ 标签中心的像素坐标, (2,), 不填时自动取四个角点的均值 """

    family: str = "tag36h11"
    """ tag family 名称字符串( pyapriltags 原始返回是 bytes, 这里已解码成 str ) """

    hamming: int = 0
    """ 解码时纠正掉的比特数, 越大越不可靠 """

    decision_margin: float = 0.0
    """ 二值解码质量, 越大越好 """

    homography: Optional[ArrayLike] = None
    """ (3, 3) 单应矩阵, 从理想 tag 坐标映射到像素坐标 """

    pose_R: Optional[ArrayLike] = None
    """ 位姿旋转矩阵 (3, 3), 仅当估计了位姿时非空 """

    pose_t: Optional[ArrayLike] = None
    """ 位姿平移向量 (3,), 单位米, 仅当估计了位姿时非空 """

    pose_err: Optional[float] = None
    """ 位姿估计的对象空间误差, 仅当估计了位姿时非空 """

    tag_size: Optional[float] = None
    """ 估计位姿时使用的 tag 边长( 米 ) """

    good_hamming: int = 2
    """ 判定为高置信( good )所允许的最大纠错位数 """

    good_margin: float = 20.0
    """ 判定为高置信( good )所需的最小 decision_margin """

    def __post_init__(self):
        self.id = int(self.id)
        self.corners = np.asarray(self.corners, dtype=np.float64).reshape(4, 2)
        if self.center is None:
            self.center = np.mean(self.corners, axis=0)
        else:
            self.center = np.asarray(self.center, dtype=np.float64).reshape(2)
        if self.homography is not None:
            self.homography = np.asarray(self.homography, dtype=np.float64).reshape(3, 3)
        if self.pose_R is not None:
            self.pose_R = np.asarray(self.pose_R, dtype=np.float64).reshape(3, 3)
        if self.pose_t is not None:
            self.pose_t = np.asarray(self.pose_t, dtype=np.float64).reshape(3)

    # end def __post_init__

    @property
    def polygon(self) -> np.ndarray:
        """ 可直接喂给 cv2.polylines / cv2.fillConvexPoly 的 int32 顶点, (4, 1, 2) """

        return np.ascontiguousarray(self.corners, dtype=np.int32).reshape(4, 1, 2)

    # end def polygon

    @property
    def area(self) -> float:
        """ 标签在图像上的面积( 像素^2 ) """

        return float(cv2.contourArea(np.ascontiguousarray(self.corners, dtype=np.float32)))

    # end def area

    @property
    def perimeter(self) -> float:
        """ 标签在图像上的周长( 像素 ) """

        pts = self.corners
        return float(sum(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)))

    # end def perimeter

    @property
    def good(self) -> bool:
        """ 是否高置信检测( 对应 apriltags2_ethz 里 AprilTagDetection.good 的语义 ) """

        return bool(self.hamming <= self.good_hamming and self.decision_margin >= self.good_margin)

    # end def good

    @property
    def pose(self) -> Optional[np.ndarray]:
        """ T_cam_tag, (4, 4), 只有在检测时估计过位姿才非空 """

        if self.pose_R is None or self.pose_t is None:
            return None
        return T_from_Rt(self.pose_R, self.pose_t)

    # end def pose

    def expand(self, ex_ratio: float) -> np.ndarray:
        """ 返回沿对角线外扩后的角点, (4, 2), 不修改自身 """

        return expand_corners(self.corners, ex_ratio)

    # end def expand

    def __repr__(self) -> str:
        return (f"Tag2D(id={self.id}, family={self.family}, center={np.round(self.center, 2).tolist()}, "
                f"hamming={self.hamming}, margin={self.decision_margin:.1f})")

    # end def __repr__

    def __str__(self) -> str:
        return self.__repr__()

    # end def __str__


# end class Tag2D


@dataclass
class Tag3D:
    """
    标定板上的一个 AprilTag( 对应 apriltag2.Tag3D ), 坐标在标定板坐标系下
    """

    id: int
    """ 标签编号 """

    corners: ArrayLike
    """ 四个角点的 3D 坐标, (4, 3), 单位米, 顺序: 左下/右下/右上/左上( tag 自身视角 ), 与 Tag2D.corners 一一对应 """

    def __post_init__(self):
        self.id = int(self.id)
        self.corners = np.asarray(self.corners, dtype=np.float64).reshape(4, 3)

    # end def __post_init__

    @property
    def center3d(self) -> np.ndarray:
        """ tag 中心点的 3D 坐标, (3,) """

        return np.mean(self.corners, axis=0)

    # end def center3d

    @property
    def size(self) -> float:
        """ tag 的边长( 米 ), 取四条边长度的均值 """

        pts = self.corners
        return float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))

    # end def size

    def __repr__(self) -> str:
        return f"Tag3D(id={self.id}, center={np.round(self.center3d, 4).tolist()}, size={self.size:.4f})"

    # end def __repr__

    def __str__(self) -> str:
        return self.__repr__()

    # end def __str__


# end class Tag3D


######################################################### 检测器 #########################################################


class Detector:
    """
    AprilTag 检测器( 对应 apriltag2.Detector ), 内部包了一层 pyapriltags.Detector

    Args:
        tag_family (str): tag 家族, 默认 "tag36h11"
            AprilTag 3 额外支持 "tagStandard41h12" / "tagStandard52h13" / "tagCircle21h7" 等新家族,
            也可以一次传多个, 用空格分隔: "tag36h11 tagStandard41h12"
        black_border (int): **仅作接口兼容, 实际不参与计算, 而且必须和你的 tag 图案一致**。
            AprilTag 3 把黑边框宽度写死在各家族的定义里( 永远是 **1 个格子** ), C 库不再暴露这个参数。
            实测( bb=0/1/2/3 x wb=0/1/2 共 12 种组合, 每个组合 6 个 id )发现:
            **只有 black_border == 1 的图案能被解码**, 0 / 2 / 3 全部 MISS。
            所以这里只记录你的 tag 是用几格黑边印的: 传 1 表示图案正确( 不会报警 );
            传 2 会打一条 warning 提醒你"这块板子在 AprilTag 3 下检不出来"( 见 create_calib_board_img )。
            pyapriltags 返回的 corners 是"含黑边在内的整张 tag 的外四角", 与 apriltag2 的语义一致
        nthreads (int): 线程数, 默认 4
        quad_decimate (float): 四边形检测时的降采样倍率, 1.0 = 不降采样( 精度最高 )。
            pyapriltags 库默认 2.0, 但**做标定时必须设 1.0**, 否则角点精度会被牺牲掉
        quad_sigma (float): 检测前的高斯模糊标准差( 像素 ), 噪声大的图可以给 0.8 左右
        refine_edges (int): 是否把四边形边"吸附"到强梯度上, 默认开
        decode_sharpening (float): 解码时的锐化强度, 小 tag 可以调大( 默认 0.25 )
        debug (int): 传 1 会导出中间调试图, 非常慢, 平时保持 0
        max_hamming (int | None): 过滤阈值, 纠错位数大于它的检测结果直接丢弃, None 表示不过滤
        min_decision_margin (float): 过滤阈值, decision_margin 小于它的检测结果直接丢弃
        tag_ids (Sequence[int] | None): id 白名单, 只保留这些编号的检测结果, None 表示不过滤
        refine_corners (bool): 是否用 cv2.cornerSubPix 做亚像素角点细化。
            **默认开**, 与原 apriltag2 的 C++ 实现一致( 它总是做细化 ); 只想要最快速度时才关掉
        min_border (int): 边界过滤, 四个角点里只要有一个离图像边缘小于这么多像素就丢弃该检测结果,
            默认 5( 与原 apriltag2 一致 ), 传 0 关闭。被画幅边缘截断的 tag 角点是不可信的
    """

    def __init__(
        self,
        tag_family: str = "tag36h11",
        black_border: int = 1,
        nthreads: int = 4,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
        debug: int = 0,
        max_hamming: Optional[int] = None,
        min_decision_margin: float = 0.0,
        tag_ids: Optional[Sequence[int]] = None,
        refine_corners: bool = True,
        min_border: int = 5,
        searchpath: Optional[List[str]] = None,
        good_hamming: int = 2,
        good_margin: float = 20.0,
    ):

        self.tag_family = tag_family
        self.black_border = int(black_border)
        self.max_hamming = max_hamming

        if self.black_border != 1:
            logger.warning(
                f"black_border={self.black_border}: AprilTag 3 只能解码**黑边宽度 = 1 个格子**的 tag, "
                f"其它宽度的图案会一个都检不出来。若你的标定板是用 apriltag2 的 black_border=2 印的, "
                f"请用本模块的 create_calib_board_img(black_border=1) 重新出图打印"
            )
        # end if
        self.min_decision_margin = min_decision_margin
        self.tag_ids = list(tag_ids) if tag_ids is not None else None
        self.refine_corners = refine_corners
        self.min_border = int(min_border)
        self.good_hamming = good_hamming
        self.good_margin = good_margin

        detector_kwargs: Dict[str, object] = dict(
            families=tag_family,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            debug=debug,
        )
        if searchpath is not None:
            detector_kwargs["searchpath"] = searchpath

        self.detector = pyapriltags.Detector(**detector_kwargs)

    # end def __init__

    # 多尺度搜索时依次尝试的缩放比例( 与原 apriltag2 一致, 从大到小且用严格 > 比较,
    # 因此比例 1.0 只要能检测到就会胜出 —— 不会为了"多检几个"牺牲角点精度 )
    AUTO_SCALE_LIST: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

    def _detect_once(
        self,
        gray_img: np.ndarray,
        scale_factor: float,
        camera_params: Optional[ArrayLike],
        tag_size: Optional[float],
        estimate_tag_pose: bool,
    ) -> List[Tag2D]:
        """
        在单一尺度上跑一次检测, 返回 Tag2D 列表( 已还原到原图坐标系, 但还没做过滤 )
        Args:
            gray_img (np.ndarray): (H, W) uint8 灰度图
            scale_factor (float): 缩放比例, 1.0 表示不缩放
            camera_params (ArrayLike | None): 相机内参 [fx, fy, cx, cy]
            tag_size (float | None): tag 边长( 米 )
            estimate_tag_pose (bool): 是否顺带解算位姿
        Returns:
            (List[Tag2D]): 检测结果列表
        """

        height, width = gray_img.shape[:2]

        if abs(scale_factor - 1.0) > 1e-6:
            if scale_factor < 0.2 - 1e-2 or scale_factor > 1.0 + 1e-2:
                raise ValueError(f"scale_factor 必须在 [0.2, 1.0] 之间, 收到 {scale_factor}")
            work = cv2.resize(gray_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            inv_scale = 1.0 / scale_factor
        else:
            work = gray_img
            inv_scale = 1.0
        # end if

        raw_tags = self.detector.detect(
            work,
            estimate_tag_pose=estimate_tag_pose,
            camera_params=camera_params,
            tag_size=tag_size,
        )

        tag_list: List[Tag2D] = []
        for raw in raw_tags:
            family = raw.tag_family
            if isinstance(family, (bytes, bytearray)):
                family = family.decode("utf-8")

            corners = np.asarray(raw.corners, dtype=np.float64).reshape(4, 2)
            center = np.asarray(raw.center, dtype=np.float64).reshape(2)
            if inv_scale != 1.0:
                corners = corners * inv_scale
                center = center * inv_scale

            # 边界过滤: 被画幅截断的 tag, 角点是被拉偏的, 直接丢掉
            if self.min_border > 0:
                hit_edge = (
                    np.any(corners[:, 0] < self.min_border) or np.any(corners[:, 0] > width - self.min_border)
                    or np.any(corners[:, 1] < self.min_border) or np.any(corners[:, 1] > height - self.min_border)
                )
                if hit_edge:
                    logger.debug(f"tag {raw.tag_id} 的角点贴到了图像边界, 已丢弃")
                    continue
            # end if

            tag_list.append(
                Tag2D(
                    id=raw.tag_id,
                    corners=corners,
                    center=center,
                    family=family,
                    hamming=raw.hamming,
                    decision_margin=float(raw.decision_margin),
                    homography=raw.homography,
                    pose_R=raw.pose_R,
                    pose_t=raw.pose_t,
                    pose_err=raw.pose_err,
                    tag_size=raw.tag_size,
                    good_hamming=self.good_hamming,
                    good_margin=self.good_margin,
                )
            )
        # end for

        return tag_list

    # end def _detect_once

    def _refine_corners_adaptive(self, gray_img: np.ndarray, tag_list: List[Tag2D]) -> None:
        """
        亚像素角点细化, 窗口大小随 tag 在图像上的尺寸自适应( 与原 apriltag2 的 C++ 实现同一套策略 )
        Args:
            gray_img (np.ndarray): (H, W) uint8 灰度图
            tag_list (List[Tag2D]): 检测结果, **原地**修改其 corners
        """

        if len(tag_list) == 0:
            return

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        for tag in tag_list:
            pts = np.ascontiguousarray(tag.corners.astype(np.float32))

            # 取最短边作为尺度参考, 窗口 = 0.25 * 最短边长, 再截断到 [3, 5]
            min_dist2 = float("inf")
            for i in range(4):
                edge = pts[i] - pts[(i + 1) % 4]
                min_dist2 = min(min_dist2, float(np.dot(edge, edge)))
            win = int(0.25 * np.sqrt(min_dist2))
            win = 3 if win < 3 else (5 if win > 5 else win)
            if win % 2 == 0:            # cv2.cornerSubPix 的窗口必须是奇数
                win += 1

            cv2.cornerSubPix(gray_img, pts, (win, win), (-1, -1), criteria)
            tag.corners = pts.astype(np.float64)
            tag.center = np.mean(tag.corners, axis=0)
        # end for

    # end def _refine_corners_adaptive

    def detect(
        self,
        img: ArrayLike,
        max_tags: Optional[int] = None,
        camera_params: Optional[ArrayLike] = None,
        tag_size: Optional[float] = None,
        estimate_tag_pose: bool = False,
        tag_ids: Optional[Sequence[int]] = None,
        scale_factor: Optional[float] = None,
    ) -> List[Tag2D]:
        """
        在图像中检测 AprilTag
        Args:
            img (ArrayLike): 输入图像, 灰度图优先; 传 BGR/BGRA 彩色图会自动转灰度
            max_tags (int | None): 最多返回多少个检测结果, 超过时优先保留质量好的。
                **None 或负数**: 不限数量; 其中显式传负数还会开启多尺度自动搜索( 见下方 Note )
            camera_params (ArrayLike | None): 相机内参 [fx, fy, cx, cy], 估计位姿时必填
            tag_size (float | None): tag 边长( 米 ), 估计位姿时必填, 也可以传 {id: size} 字典
            estimate_tag_pose (bool): 是否顺带解算位姿, 解开后 Tag2D.pose_R / pose_t / pose_err / pose 可用
            tag_ids (Sequence[int] | None): 临时 id 白名单, 覆盖构造时传入的白名单
            scale_factor (float | None): 图像缩放比例, 范围 [0.2, 1.0];
                传 **负数**会在 self.AUTO_SCALE_LIST 里自动挑检测数量最多的那个尺度
        Returns:
            (List[Tag2D]): 检测结果列表, 按 id 升序排列, 没有检测到时返回空列表

        Note:
            第二个位置参数按 **apriltag2 的 scale_factor 语义**来兼容, 判定规则::

                detector.detect(img)         # 单尺度 1.0, 不限制数量
                detector.detect(img, -1)     # 多尺度自动搜索        <- 项目里最常用的写法
                detector.detect(img, 0.5)    # scale_factor = 0.5
                detector.detect(img, 1)      # 整数也按 scale 处理( 原 apriltag2 没有\"数量限制\"这个概念 )
                detector.detect(img, 10)     # > 1 的整数才当 max_tags( 本模块自己扩展的能力 )

            建议新代码用关键字传 ``max_tags=`` / ``scale_factor=``, 语义最清楚。
            多尺度只在比较时用 **严格大于** 更新最优, 所以比例 1.0 能检测到就永远选 1.0,
            不会因为"缩小后碰巧多检出几个"而牺牲角点精度。
        """

        # 兼容 apriltag2 的 detect(img, scale_factor) 写法: 见上面 Note 的判定规则
        if max_tags is not None:
            value = float(max_tags)
            if value < 0:
                auto_scale_requested = True
            elif value <= 1.0:
                auto_scale_requested = False
                scale_factor = min(max(value, 0.2), 1.0)
            else:
                auto_scale_requested = False
                max_tags = int(value)
                scale_factor = scale_factor if scale_factor is not None else 1.0
            max_tags = None if max_tags is None or max_tags <= 1 else max_tags
        else:
            auto_scale_requested = None

        gray_img = to_gray_u8(img)

        # 是否走多尺度搜索
        auto_scale = False
        if scale_factor is None:
            if auto_scale_requested is True:
                auto_scale = True
            scale_factor = 1.0
        elif scale_factor < 0 or auto_scale_requested is True:
            auto_scale = True
        # end if

        if auto_scale:
            best_tags: List[Tag2D] = []
            best_scale = 1.0
            for scale in self.AUTO_SCALE_LIST:
                cur = self._detect_once(gray_img, scale, camera_params, tag_size, estimate_tag_pose)
                logger.debug(f"scale={scale}, 检测到 {len(cur)} 个 tag")
                if len(cur) > len(best_tags):
                    best_tags, best_scale = cur, scale
            if best_scale != 1.0:
                logger.warning(f"原图没能检出全部标签, 已自动降到 scale={best_scale}, "
                               f"检出 {len(best_tags)} 个( 此时角点精度会略差, 建议复核 )")
            tag_list = best_tags
        else:
            tag_list = self._detect_once(gray_img, float(scale_factor), camera_params, tag_size, estimate_tag_pose)
        # end if

        # 按质量过滤
        if self.max_hamming is not None:
            tag_list = [t for t in tag_list if t.hamming <= self.max_hamming]
        if self.min_decision_margin > 0.0:
            tag_list = [t for t in tag_list if t.decision_margin >= self.min_decision_margin]

        id_filter = tag_ids if tag_ids is not None else self.tag_ids
        if id_filter is not None:
            id_set = set(int(i) for i in id_filter)
            tag_list = [t for t in tag_list if t.id in id_set]

        # 亚像素细化, 相机标定时能明显降低重投影误差
        if self.refine_corners and len(tag_list) > 0:
            self._refine_corners_adaptive(gray_img, tag_list)
        # end if

        # 数量截断: 优先保留解码质量高的
        if max_tags is not None and max_tags > 0 and len(tag_list) > max_tags:
            tag_list = sort_tags(tag_list, by="margin")[:max_tags]

        return sort_tags(tag_list, by="id")

    # end def detect

    def extract_tags(self, img: ArrayLike, max_tags: Optional[int] = None, **kwargs) -> List[Tag2D]:
        """
        detect() 的别名, 为了对齐 apriltags2_ethz( kalibr 那套封装 )里 ``extract_tags`` 的调用习惯
        Args:
            img (ArrayLike): 输入图像
            max_tags (int): 最多返回多少个检测结果, -1 表示不限
            kwargs: 透传给 detect()
        Returns:
            (List[Tag2D]): 检测结果列表
        """

        return self.detect(img, max_tags, **kwargs)

    # end def extract_tags

    @staticmethod
    def draw(
        img: ArrayLike,
        tag_list: Sequence[Tag2D],
        draw_center: bool = True,
        draw_id: bool = True,
        draw_orientation: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        在原图上画出检测结果( 对应 apriltag2 里 Detector.draw 的用法 )

        与原仓库一样是**静态方法**, 两种调用方式都成立::

            apriltag2.Detector.draw(bgr_img, tag_list)     # 类上直接调用
            detector.draw(bgr_img, tag_list)               # 实例调用, 效果相同

        Args:
            img (ArrayLike): 输入图像, 灰度图或 BGR 彩色图均可( 灰度图会自动转成 BGR 再画 )
            tag_list (Sequence[Tag2D]): detect() 的返回结果
            draw_center (bool): 是否画中心点
            draw_id (bool): 是否标注 tag 编号
            draw_orientation (bool): 是否画朝向箭头( 左上 -> 右上 的方向 )
            color (Tuple[int, int, int]): 绘制颜色 (B, G, R)
            thickness (int): 线宽
        Returns:
            (np.ndarray): 画好结果的 BGR 图像
        """

        arr = np.asarray(img)
        if arr.ndim == 2:
            out = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            out = arr.copy()

        for tag in tag_list:
            cv2.polylines(out, [tag.polygon], isClosed=True, color=color, thickness=thickness)

            # __post_init__ 已经保证 center 是 (2,) 的浮点数组
            cu, cvr = int(np.round(tag.center[0])), int(np.round(tag.center[1]))

            if draw_center:
                cv2.circle(out, (cu, cvr), 3, (0, 0, 255), -1)

            if draw_orientation:
                direction = tag.corners[1] - tag.corners[0]
                tip = tag.center + direction
                cv2.arrowedLine(
                    out,
                    (cu, cvr),
                    (int(np.round(tip[0])), int(np.round(tip[1]))),
                    color=(0, 255, 255),
                    thickness=1,
                )
            # end if

            if draw_id:
                left_top = tag.corners[0]
                cv2.putText(
                    out,
                    str(tag.id),
                    (int(left_top[0]), int(left_top[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            # end if
        # end for

        return out

    # end def draw

    def __repr__(self) -> str:
        return f"Detector(family={self.tag_family}, params={self.detector.params})"

    # end def __repr__


# end class Detector


######################################################### 标定板 #########################################################


def make_default_detector(tag_family: str = "tag36h11", **kwargs) -> Detector:
    """
    用默认参数创建检测器( 对应 apriltags2_ethz 里的 make_default_detector )
    Args:
        tag_family (str): tag 家族
        kwargs: 透传给 Detector 的其他参数
    Returns:
        (Detector): 检测器实例
    """

    return Detector(tag_family=tag_family, **kwargs)


# end def make_default_detector


def create_calib_board_img(
    tag_family: str = "tag36h11",
    black_border: int = 1,
    tag_spacing: int = 3,
    rows: int = 6,
    cols: int = 6,
    scale: int = 1,
    start_tag_id: int = 0,
    marker_size: int = 1,
    interp: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """
    生成可直接打印的 AprilTag 标定板图案( 对应 apriltag2.create_calib_board_img )
    Args:
        tag_family (str): 标签族, 支持 "tag16h5" / "tag25h7" / "tag25h9" / "tag36h11"
        black_border (int): tag 核心图案外面黑色边框的宽度( 单位: 一个白色格子 ), **只能传 1**。
            这是与原 apriltag2 最大的**不兼容点**: apriltag2 支持任意黑边宽度( 你们原来常用 2 ),
            而 AprilTag 3 把黑边写死成 1 格, 实测 bb=0/2/3 的图案在 pyapriltags 下 **一个都检不出来**。
            传别的值会报错挡住, 免得印完才发现检测不到
        tag_spacing (int): tag 之间的白色间距( 单位: 一个白色格子 ), **>= 2**
        rows (int): 行数
        cols (int): 列数
        scale (int): 整块板子的放大倍数( 最近邻 ), 用来出高分辨率图
        start_tag_id (int): 起始 tag 编号, 编号按行优先递增
        marker_size (int): 黑色标记块的边长( 格子 ), 0 表示不画。**这些块对 AprilTag 检测毫无作用**,
            纯粹是把板子的网格结构标出来给人看, 可以放心关掉。
            它们画在板子网格的所有交叉点上( 共 ``(rows+1) x (cols+1)`` 个 ), 每个交叉点被周围
            4 个 tag 共享, 所以每个 tag 的四角都恰好各有一个块( 与原 apriltag2 的观感一致 )。
            与原版有两个实测出来的差别:

            1) 大小: 原版取 tag_spacing 格, 块正好与 tag 的黑边框**对角相连**, AprilTag 3 会把这些
               连通的黑色区域当成一个整体四边形, 相邻 tag 被一起吃掉 —— 实测 6x6 板 34/36、
               4x4 板 15/16。这里强制 ``<= tag_spacing - 2``, 块与最近的 tag 至少隔 1 格白边。
               副作用: ``tag_spacing=3`` 时块最大只能 1 格, spacing 更小时干脆不画
            2) 位置: 原版按"每个瓦片四角"画, 而相邻瓦片是重叠摆放的, 后贴的瓦片白边会把先贴的块
               **覆盖掉**( 实测 2x2 板 16 个块只剩 8 个, 每个 tag 看着只有 1 个块 )。
               这里改为贴完全部 tag 之后再按交叉点统一落块, 不会被覆盖
        interp (int): 放大时用的插值方式, 默认 INTER_NEAREST( 保证边缘锐利 )
    Returns:
        (np.ndarray): uint8 灰度图

    Note:
        1) 排布与 ``create_calib_board_3d`` **严格配套**::

               | (rows-1)*cols ... |        <- 图像上方, Y 大
               | cols  cols+1 ...  |
               | 0     1     ...   |        <- 图像下方, Y 小( 含编号为 start_tag_id 的 tag )

           即: 编号沿 +X 方向按列递增, 沿 +Y 方向按行递增, **第 0 行在图像下方**( 板子 Y 轴朝上 )。
           打印时必须用本函数出图 + ``create_calib_board_3d`` 建模型, 两者混搭其它工具容易天地颠倒
        2) 板上有黑色标记块, 画在网格的交叉点上, 详见 marker_size 的说明。
           它们只是视觉标记, 对检测没有任何影响, 不要为了"对齐原版"去放大它
        3) 长度单位是"格子"( 一个白色正方形小格 ), 打印时按
           ``1 格子 = tag_size / (dim + 2*black_border)`` 把物理量换算到毫米/英寸。
           其中 tag_size 是**含黑边**的整张 tag 边长, 也就是 create_calib_board_3d 里那个 tag_size
        4) 标记块是在**所有 tag 都贴完之后**才按网格交叉点统一画的, 不能跟着瓦片一起贴:
           相邻瓦片的步距是 ``rect+spacing``, 而瓦片边长是 ``rect+2*spacing``, 两者重叠了
           ``spacing`` 格, 后贴的瓦片白边会把先贴的块覆盖掉
    """

    if tag_family not in _TAG_CODES_B64:
        raise ValueError(f"不支持的 tag 家族: {tag_family}, 可选: {sorted(_TAG_CODES_B64)}")
    if black_border != 1:
        raise ValueError(
            f"black_border={black_border}: AprilTag 3 只能解码黑边宽度为 1 个格子的 tag "
            f"( apriltag2 支持任意宽度, 这点是硬不兼容 )。请改成 1 重新出图打印"
        )
    if tag_spacing < 2:
        raise ValueError("tag_spacing 必须 >= 2( 留至少 1 格白边把相邻 tag 和标记块隔开 )")
    if rows < 1 or cols < 1:
        raise ValueError("rows / cols 必须 >= 1")
    if marker_size < 0:
        raise ValueError("marker_size 不能为负数")

    n_tags = tag_family_count(tag_family)
    last_id = start_tag_id + rows * cols - 1
    if start_tag_id < 0 or last_id >= n_tags:
        raise ValueError(f"tag 编号越界: 需要 [{start_tag_id}, {last_id}], 但家族 {tag_family} 只有 {n_tags} 个 tag")
    # end if

    dim = FAMILY_DIM[tag_family]
    rect_size = dim + 2 * black_border          # 单个 tag( 含黑边 )的边长
    final_size = rect_size + 2 * tag_spacing    # 瓦片边长( 再套一圈白色间距 )
    pitch = final_size - tag_spacing            # 相邻瓦片的摆放步距 => 相邻 tag 净间距 = tag_spacing

    board_w = tag_spacing * (cols + 1) + rect_size * cols
    board_h = tag_spacing * (rows + 1) + rect_size * rows
    board_img = np.full((board_h, board_w), 255, dtype=np.uint8)
    logger.info(f"标定板原始尺寸: {board_w} x {board_h} 像素( 单 tag {rect_size} 像素 )")

    for r in range(rows):
        for c in range(cols):
            tag_id = start_tag_id + r * cols + c

            tile = np.full((final_size, final_size), 255, dtype=np.uint8)
            tile[tag_spacing:tag_spacing + rect_size, tag_spacing:tag_spacing + rect_size] = render_tag_image(
                tag_family, tag_id, black_border=black_border, white_border=0
            )

            col_start = c * pitch
            row_start = (rows - 1 - r) * pitch          # 第 0 行放在图像下方
            board_img[row_start:row_start + final_size, col_start:col_start + final_size] = tile
        # end for
    # end for

    # 标记块: 画在板子上所有"网格交叉点"( 共 (rows+1) x (cols+1) 个 )。
    # 每个交叉点被周围 4 个 tag 共享, 所以每个 tag 的四角恰好各有一个块, 板子最外圈的 tag 也不会缺。
    # ms 必须 <= tag_spacing - 2: 块与 tag 之间要留 >= 1 格白边, 否则块会与 tag 的黑边框
    # 8-邻接连通, AprilTag 3 把连通的黑色区域当成一个整体四边形, 相邻 tag 会被一起吃掉
    # ( 实测原版那种"块 = tag_spacing 格"的画法, 6x6 板只能检出 34/36, 4x4 板 15/16 )
    ms = max(0, min(int(marker_size), max(0, tag_spacing - 2)))
    if ms > 0:
        off = max(1, (tag_spacing - ms) // 2)
        for k in range(rows + 1):
            for j in range(cols + 1):
                r0 = k * pitch + off
                c0 = j * pitch + off
                board_img[r0:r0 + ms, c0:c0 + ms] = 0
            # end for
        # end for
    # end if

    if scale != 1:
        board_img = cv2.resize(board_img, (board_w * scale, board_h * scale), interpolation=interp)
        logger.info(f"放大 {scale} 倍后尺寸: {board_img.shape[1]} x {board_img.shape[0]} 像素")

    return board_img


# end def create_calib_board_img


def calib_board_paper_size(
    tag_size: float,
    tag_family: str = "tag36h11",
    black_border: int = 1,
    tag_spacing: int = 3,
    rows: int = 6,
    cols: int = 6,
) -> Tuple[float, float, int]:
    """
    算出 ``create_calib_board_img`` 出图后应该打印成多大, 用来安排纸张和做打印后的**尺寸验收**
    Args:
        tag_size (float): 含黑边的整张 tag 边长( 米 )
        tag_family (str): tag 家族
        black_border (int): 黑边宽度( 格子 )
        tag_spacing (int): tag 间距( 格子 )
        rows (int): 行数
        cols (int): 列数
    Returns:
        (Tuple[float, float, int]): (图纸宽, 图纸高, 单格边长), 单位都是**毫米**

    Note:
        拿这个值去验收打印结果: 打印后量整张图纸的宽, 实测值除以这里返回的宽,
        就是打印机的实际缩放系数, 再用 ``rescale_calib_board_params`` 修正 ``tag_size`` / ``space_size``。
        **不要去量单个 tag 或多格**, 量整张对角线/整宽误差最小( 分摊到 69 格上 )
    """

    dim = _family_dim(tag_family)
    rect = dim + 2 * int(black_border)
    cell_mm = tag_size * 1000.0 / rect
    cells_w = tag_spacing * (cols + 1) + rect * cols
    cells_h = tag_spacing * (rows + 1) + rect * rows
    return cells_w * cell_mm, cells_h * cell_mm, cell_mm


# end def calib_board_paper_size


def rescale_calib_board_params(
    tag_size: float,
    space_size: float,
    paper_measured_mm: float,
    paper_nominal_mm: float,
) -> Tuple[float, float]:
    """
    按"打印后实测的纸张宽度"修正标定板参数( 打印机/出图几乎总有千分之几的整体缩放 )
    Args:
        tag_size (float): 标定板模型里用的 tag 边长( 米 ), 即 nominal 值
        space_size (float): 标定板模型里用的 tag 间距( 米 )
        paper_measured_mm (float): **打印好后用尺子量出来的**图纸宽度( 毫米 )
        paper_nominal_mm (float): 图纸宽度的理论值( 毫米 ), 即 ``calib_board_paper_size`` 的返回值
    Returns:
        (Tuple[float, float]): (修正后的 tag_size, 修正后的 space_size), 单位米

    Note:
        为什么要这一步: 印刷整体缩放会 1:1 传染给标定出的所有尺度( 相机焦距、手眼 X/Y/Z )。
        钢尺量单个 tag 误差能到 ±0.5mm( 对 35mm 就是 1.5% ), 而量整块 276mm 的板,
        同样的 ±0.5mm 只摊薄到 0.18% —— **量大的、再按比例换算, 永远是更准的办法**
    """

    k = float(paper_measured_mm) / float(paper_nominal_mm)
    if abs(k - 1.0) > 0.005:
        logger.warning(f"打印缩放 {(k - 1.0) * 100:+.2f}%, 超出常见打印机误差范围, 确认下是不是被'适应页面'缩放了")
    logger.info(f"打印缩放系数 {k:.6f}, tag_size: {tag_size:.6f} -> {tag_size * k:.6f} m")

    return tag_size * k, space_size * k


# end def rescale_calib_board_params


def save_calib_board_pdf(
    img: ArrayLike,
    path: str,
    dpi: float = 300.0,
) -> str:
    """
    把标定板灰度图导出成 PDF( **零第三方依赖**: 手写的最小 PDF writer, 只用标准库 zlib )
    Args:
        img (ArrayLike): ``create_calib_board_img`` 的返回值, (H, W) uint8 灰度图
        path (str): 输出文件路径
        dpi (float): 打印分辨率, 决定 PDF 页面的物理尺寸::

                页面宽( mm ) = 图像宽( px ) / dpi * 25.4

            改 dpi 就等于改打印出来的物理大小, **要保证标定板实际尺寸正确就别乱动它**
    Returns:
        (str): 实际写出的文件路径

    Note:
        1) 为什么要有 PDF: PNG 在很多看图/打印软件里默认是"适应页面( fit to page )",
           一旦被缩放, tag 的物理尺寸就不再是 ``tag_size``, 标定出来的尺度会**整体失真**,
           而且看不出明显异常。PDF 的页面有固定物理尺寸, 打印时只要不勾"缩放"就是严格 1:1
        2) 图像以 FlateDecode + DeviceGray 内嵌, 不做重采样, 边缘保持锐利
        3) PNG 的原点在左上、PDF 用户空间原点在左下, 写之前会把行序倒过来,
           否则打印出来整块板会上下颠倒( id 排布跟着翻 )
    """

    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(arr, dtype=np.uint8)

    height, width = gray.shape[:2]
    raw = zlib.compress(np.ascontiguousarray(gray[::-1]).tobytes(), 9)  # 行序翻转, 见 Note 3

    page_w = width / float(dpi) * 72.0
    page_h = height / float(dpi) * 72.0
    content = f"q {page_w:.4f} 0 0 {page_h:.4f} 0 0 cm /Im0 Do Q\n".encode("ascii")

    objects: List[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.4f} {page_h:.4f}] "
            f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray "
            f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(raw)} >>"
        ).encode("ascii"),
        f"<< /Length {len(content)} >>".encode("ascii"),
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: List[int] = []
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode("ascii")
        out += body + b"\n"
        if idx == 4:
            out += b"stream\n" + raw + b"\nendstream\n"
        elif idx == 5:
            out += b"stream\n" + content + b"endstream\n"
        out += b"endobj\n"
    # end for

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii")

    with open(path, "wb") as fp:
        fp.write(bytes(out))

    logger.info(f"标定板 PDF 已导出: {path} ( 页面 {width / dpi * 25.4:.2f} x {height / dpi * 25.4:.2f} mm, {dpi:.0f} DPI )")
    return path


# end def save_calib_board_pdf


def calib_board_space_size(
    tag_size: float,
    tag_family: str = "tag36h11",
    black_border: int = 1,
    tag_spacing: int = 3,
) -> float:
    """
    由 tag 的物理边长反算出 ``create_calib_board_3d`` 需要的 ``space_size``,
    保证 3D 模型和 ``create_calib_board_img`` 出的图**严格配套**
    Args:
        tag_size (float): **含黑边**的整张 tag 边长( 米 )
        tag_family (str): tag 家族
        black_border (int): 出图时用的黑边宽度( 格子 ), 必须与 create_calib_board_img 传的一致
        tag_spacing (int): 出图时用的 tag 间距( 格子 ), 必须与 create_calib_board_img 传的一致
    Returns:
        (float): 相邻 tag 之间的净间距( 米 ), 直接喂给 create_calib_board_3d 的 space_size

    Note:
        手算很容易错( 1 格 = tag_size / (dim + 2*black_border) ), 一旦算错, 整块板解出来的位姿
        会带一个固定比例的尺度误差, 而且**看不出明显异常** —— 建议一律用这个函数换算
    """

    dim = _family_dim(tag_family)
    cell = tag_size / float(dim + 2 * int(black_border))
    return cell * float(tag_spacing)


# end def calib_board_space_size


def create_calib_board_3d(
    tag_size: float,
    space_size: float,
    rows: int,
    cols: int,
    start_tag_id: int = 0,
    origin: str = "corner",
) -> List[Tag3D]:
    """
    生成 AprilTag 标定板的 3D 模型( 对应 apriltag2.create_calib_board_3d )
    Args:
        tag_size (float): 单个 tag 的边长( 米 ), 注意是**含黑边的整张 tag** 的边长
        space_size (float): 相邻 tag 之间的净间距( 米 )
        rows (int): 行数
        cols (int): 列数
        start_tag_id (int): 左下角第一个 tag 的编号, 编号按行优先递增( 先从下往上, 再从左往右 )
        origin (str): 坐标系原点的位置, 可选:
            - ``"corner"``   (**默认, 与原 apriltag2 完全一致**): 原点 = start_tag_id 这个 tag 的左下角
            - ``"tag0_center"``: 原点 = start_tag_id 这个 tag 的中心
            - ``"center"``   : 原点 = 整块板的几何中心

    Note:
        ``space_size`` 建议用 ``calib_board_space_size(tag_size, ...)`` 换算, 手算容易和出图参数对不上
    Returns:
        (List[Tag3D]): 标定板上所有 tag 的 3D 角点信息

    Note:
        1) 标定板坐标系(**Y 轴朝上**, 右手系): x 轴沿列增大方向( 向右 ), y 轴沿行增大方向( 向上 ),
           z = 0 为板面, z 轴垂直板面指向观察者; 原点在编号为 start_tag_id 的 tag 的左下角
        2) 每个 tag 的角点顺序是 tag 自身视角的 左下/右下/右上/左上, 与 Tag2D.corners 一一对应
        3) 必须和 ``create_calib_board_img`` 出的图配套使用, 否则整套坐标会天地颠倒
    """

    if origin not in ("corner", "tag0_center", "center"):
        raise ValueError(f"未知的 origin: {origin}, 可选 'corner' / 'tag0_center' / 'center'")

    if origin == "tag0_center":
        offset = np.array([tag_size / 2.0, tag_size / 2.0, 0.0], dtype=np.float64)
    elif origin == "center":
        offset = np.array([
            ((cols - 1) * (tag_size + space_size) + tag_size) / 2.0,
            ((rows - 1) * (tag_size + space_size) + tag_size) / 2.0,
            0.0,
        ], dtype=np.float64)
    else:
        offset = np.zeros(3, dtype=np.float64)
    # end if

    step = tag_size + space_size

    tag3d_list: List[Tag3D] = []
    for r in range(rows):
        for c in range(cols):
            tag_id = start_tag_id + r * cols + c

            x0, y0 = c * step, r * step
            x1, y1 = x0 + tag_size, y0 + tag_size
            tag3d_list.append(
                Tag3D(
                    id=tag_id,
                    corners=[
                        [x0, y0, 0.0],     # 左下
                        [x1, y0, 0.0],     # 右下
                        [x1, y1, 0.0],     # 右上
                        [x0, y1, 0.0],     # 左上
                    ],
                )
            )
        # end for
    # end for

    if np.any(offset != 0.0):
        for tag3d in tag3d_list:
            tag3d.corners = tag3d.corners - offset
    # end if

    return tag3d_list


# end def create_calib_board_3d


def match_tag_pairs(
    tag2d_list: Sequence[Tag2D],
    tag3d_list: Sequence[Tag3D],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    按 id 把图像里的检测和标定板模型配对, 摊平成 solvePnP 需要的点对
    Args:
        tag2d_list (Sequence[Tag2D]): detect() 的检测结果
        tag3d_list (Sequence[Tag3D]): create_calib_board_3d() 生成的标定板模型
    Returns:
        (Tuple[np.ndarray, np.ndarray, List[int]]): (pts3d, pts2d, id_list)
            pts3d: (N*4, 3) 标定板坐标系下的角点; pts2d: (N*4, 2) 对应的像素坐标; id_list: 用到的 tag 编号
    """

    pts3d: List[np.ndarray] = []
    pts2d: List[np.ndarray] = []
    id_list: List[int] = []

    for tag2d in tag2d_list:
        tag3d = next((t for t in tag3d_list if t.id == tag2d.id), None)
        if tag3d is None:
            logger.warning(f"tag id {tag2d.id} 在图像里检测到了, 但标定板模型里没有, 已跳过")
            continue
        pts3d.append(np.asarray(tag3d.corners, dtype=np.float64))
        pts2d.append(np.asarray(tag2d.corners, dtype=np.float64))
        id_list.append(tag2d.id)
    # end for

    if len(pts3d) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2)), []

    return np.vstack(pts3d), np.vstack(pts2d), id_list


# end def match_tag_pairs


def locate_calib_board(
    tag2d_list: Sequence[Tag2D],
    tag3d_list: Sequence[Tag3D],
    K: ArrayLike,
    D: Optional[ArrayLike] = None,
    min_tags: int = 4,
    th_reproj: float = 1.0,
    iters: int = 3,
    refine: bool = True,
    flags: Optional[int] = None,
    return_stats: bool = False,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Dict[str, float]]]:
    """
    用 PnP 求标定板相对相机的位姿( 对应 apriltag2.locate_calib_board )
    Args:
        tag2d_list (Sequence[Tag2D]): detect() 的检测结果
        tag3d_list (Sequence[Tag3D]): create_calib_board_3d() 生成的标定板模型
        K (ArrayLike): 相机内参 [fx, fy, cx, cy] 或 3x3 矩阵
        D (ArrayLike | None): 畸变系数 [k1, k2, p1, p2, k3], 传了就会带着畸变模型一起解, 不传视为无畸变
        min_tags (int): 至少要匹配到多少个 tag 才认为定位成功, 否则返回 None
        th_reproj (float): 重投影误差阈值( 像素 ), 超过的点对会被当作离群点剔除, 与原 apriltag2 一致
        iters (int): 剔除-重解 的迭代轮数, 与原 apriltag2 一致( 3 轮 )
        refine (bool): 是否用 LM 迭代再精修一次
        flags (int | None): cv2.solvePnP 的方法 flag, 默认用 SOLVEPNP_ITERATIVE
        return_stats (bool): True 时返回 (T, info) 二元组, info 里带重投影误差统计
    Returns:
        (np.ndarray | None | Tuple): T_cam_board, (4, 4); 定位失败返回 None。
            return_stats=True 时返回 (T_cam_board, info_dict)

    Note:
        1) **没有照搬原 apriltag2 的 SOLVEPNP_IPPE**: IPPE 要求**恰好 4 个点**, 而原实现把 N 个 tag
           的 4N 个点一起喂进去, 一旦检测到 2 个及以上的 tag 就会在 OpenCV 里直接抛异常。
           这里用 SOLVEPNP_ITERATIVE + solvePnPRefineLM, 点数不限且精度更好
        2) 离群点剔除是"先解再剔"的思路: 用当前位姿算所有点的重投影误差, 保留误差小于阈值的点重解,
           循环 iters 轮。若某一轮剩下的点少于 4 个, 会保留上一轮的解并告警, 不会直接返回 None
    """

    pts3d, pts2d, id_list = match_tag_pairs(tag2d_list, tag3d_list)

    if len(id_list) < min_tags:
        logger.warning(f"标定板定位失败: 只匹配到 {len(id_list)} 个 tag, 至少需要 {min_tags} 个")
        return (None, {}) if return_stats else None

    n_points = len(pts3d)
    if n_points < 4:
        logger.warning("标定板定位失败: 可用点对不足 4 个")
        return (None, {}) if return_stats else None

    camera_matrix = as_camera_matrix(K)
    dist_coeffs = np.zeros(5, dtype=np.float64) if D is None else np.asarray(D, dtype=np.float64).reshape(-1)

    status = np.ones(n_points, dtype=bool)
    best: Optional[Tuple[np.ndarray, np.ndarray]] = None
    info: Dict[str, float] = {}

    for it in range(max(1, iters)):
        good3d = np.ascontiguousarray(pts3d[status].reshape(-1, 1, 3))
        good2d = np.ascontiguousarray(pts2d[status].reshape(-1, 1, 2))
        if len(good3d) < 4:
            logger.warning("滤掉离群点后有效点不足 4 个, 保留上一轮的解")
            break

        ok, rvec, tvec = solve_pnp(good3d, good2d, camera_matrix, D=dist_coeffs, refine=refine, flags=flags)
        if not ok:
            logger.warning("标定板定位失败: solvePnP 没有收敛")
            break

        # 用当前解算全部点的重投影误差, 据此更新有效点集合
        proj, _ = cv2.projectPoints(
            np.ascontiguousarray(pts3d.reshape(-1, 1, 3)), rvec, tvec, camera_matrix, dist_coeffs
        )
        errors = np.linalg.norm(np.asarray(proj, dtype=np.float64).reshape(-1, 2) - pts2d, axis=1)
        next_status = errors < th_reproj

        info = {
            "iteration": it + 1,
            "min_reproj_err": float(np.min(errors)),
            "max_reproj_err": float(np.max(errors)),
            "mean_reproj_err": float(np.mean(errors)),
            "rms_reproj_err": float(np.sqrt(np.mean(errors ** 2))),
            "inliers": int(np.sum(next_status)),
            "points": int(n_points),
        }
        logger.info(
            f"Iteration {it + 1}: min={info['min_reproj_err']:.3f} max={info['max_reproj_err']:.3f} "
            f"mean={info['mean_reproj_err']:.3f} inliers={info['inliers']}/{n_points}"
        )

        best = (rvec, tvec)
        if np.array_equal(next_status, status) or int(np.sum(next_status)) < 4:
            status = status if int(np.sum(next_status)) < 4 else next_status
            break
        status = next_status
    # end for

    if best is None:
        return (None, info) if return_stats else None

    rvec, tvec = best
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = T_from_Rt(R, tvec)
    return (T, info) if return_stats else T


# end def locate_calib_board


######################################################### 位姿估计 #########################################################


def solve_pnp(
    pts3d: ArrayLike,
    pts2d: ArrayLike,
    K: ArrayLike,
    D: Optional[ArrayLike] = None,
    refine: bool = True,
    flags: Optional[int] = None,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    统一的 solvePnP 封装, 支持带畸变、支持 LM 精修
    Args:
        pts3d (ArrayLike): (N, 3) 目标坐标系下的 3D 点( 单位米 )
        pts2d (ArrayLike): (N, 2) 对应的像素坐标
        K (ArrayLike): 相机内参 [fx, fy, cx, cy] 或 3x3 矩阵
        D (ArrayLike | None): 畸变系数, 不传视为无畸变
        refine (bool): 是否用 solvePnPRefineLM 精修
        flags (int | None): cv2.solvePnP 的 flag, 默认 SOLVEPNP_ITERATIVE
    Returns:
        (Tuple[bool, np.ndarray, np.ndarray]): (ok, rvec(3,), tvec(3,))
    """

    object_points = np.ascontiguousarray(np.asarray(pts3d, dtype=np.float64).reshape(-1, 1, 3))
    image_points = np.ascontiguousarray(np.asarray(pts2d, dtype=np.float64).reshape(-1, 1, 2))
    camera_matrix = as_camera_matrix(K)
    dist_coeffs = np.zeros(5, dtype=np.float64) if D is None else np.ascontiguousarray(np.asarray(D, dtype=np.float64).reshape(-1))

    if flags is None:
        flags = getattr(cv2, "SOLVEPNP_ITERATIVE", 0)

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=flags,
    )

    if ok and refine and hasattr(cv2, "solvePnPRefineLM"):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, criteria)
        except cv2.error as exc:
            logger.warning(f"solvePnPRefineLM 精修失败, 使用初始解: {exc}")
    # end if

    return ok, np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)


# end def solve_pnp


def make_tag_object_points(tag_size: float) -> np.ndarray:
    """
    生成单个 tag 在其自身坐标系下的四个 3D 角点
    Args:
        tag_size (float): tag 边长( 米 )
    Returns:
        (np.ndarray): (4, 3), 顺序 左下/右下/右上/左上

    Note:
        1) tag 自身坐标系: 原点在 tag 中心, x 轴沿 左下->右下, y 轴沿 左下->左上, z 轴 = x × y
           与 build_tag_pose_from_corners3d 的"邻边建法"是同一套约定, 两者结果可直接对比
    """

    half_tag = tag_size / 2.0
    return np.array(
        [
            [-half_tag, -half_tag, 0.0],
            [+half_tag, -half_tag, 0.0],
            [+half_tag, +half_tag, 0.0],
            [-half_tag, +half_tag, 0.0],
        ],
        dtype=np.float64,
    )


# end def make_tag_object_points


def estimate_tag_pose(
    tag2d: Union[Tag2D, ArrayLike],
    tag_size: float,
    K: ArrayLike,
    D: Optional[ArrayLike] = None,
    use_ippe: bool = False,
    refine: bool = True,
    front_only: bool = True,
) -> Optional[np.ndarray]:
    """
    估计单个 tag 的位姿
    Args:
        tag2d (Tag2D | ArrayLike): Tag2D 实例, 或直接传 (4, 2) 的角点
        tag_size (float): tag 边长( 米 )
        K (ArrayLike): 相机内参 [fx, fy, cx, cy] 或 3x3 矩阵
        D (ArrayLike | None): 畸变系数, 不传视为无畸变
        use_ippe (bool): 是否尝试 SOLVEPNP_IPPE_SQUARE( 专门为正方形标记设计 ), 默认 False
        refine (bool): 是否用 LM 精修, 只在走迭代法时生效
        front_only (bool): 是否要求解出来的 tag 正面朝相机, 用于剔除 PnP 的镜像二义解
    Returns:
        (np.ndarray | None): T_cam_tag, (4, 4); 解算失败返回 None

    Note:
        1) 默认走 SOLVEPNP_ITERATIVE + solvePnPRefineLM, 实测在无噪数据上能稳定到 1e-11 量级
        2) use_ippe=True 时会先用 IPPE 解一版, 再用朝向判据验证, 拿到镜像解/退化解就自动回退迭代法。
           **为什么默认关**: 实测 OpenCV 5.0.0 的 SOLVEPNP_IPPE_SQUARE 对 6x6 标定板这类数据会返回
           t≈0 的退化解( 与角点顺序无关 ), 换 cv2 版本后如果表现正常可以打开试试
    """

    corners = np.asarray(tag2d.corners if isinstance(tag2d, Tag2D) else tag2d, dtype=np.float64).reshape(4, 2)
    object_points = make_tag_object_points(tag_size)

    camera_matrix = as_camera_matrix(K)
    dist_array = np.zeros(5, dtype=np.float64) if D is None else np.asarray(D, dtype=np.float64).reshape(-1)
    has_distortion = bool(np.any(np.abs(dist_array) > 1e-12))

    ippe_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", None) if use_ippe else None
    if ippe_flag is not None:
        # IPPE 不吃畸变, 有畸变就先把像素点去畸变( 等价性由 undistortPoints 的 P=K 保证 )
        if has_distortion:
            undistorted = cv2.undistortPoints(
                corners.reshape(-1, 1, 2), camera_matrix, dist_array, P=camera_matrix
            )
            ippe_points = np.asarray(undistorted, dtype=np.float64).reshape(4, 2)
        else:
            ippe_points = corners

        ok, rvec, tvec = solve_pnp(object_points, ippe_points, camera_matrix, None, refine=False, flags=ippe_flag)
        if ok:
            R_ippe, _ = cv2.Rodrigues(rvec)
            if (not front_only) or is_pose_front_facing(R_ippe, tvec):
                return T_from_Rt(R_ippe, tvec)
            logger.debug("IPPE 返回了镜像二义解, 回退到迭代法重解")
        # end if
    # end if

    iterative_flag = getattr(cv2, "SOLVEPNP_ITERATIVE", 0)
    ok, rvec, tvec = solve_pnp(object_points, corners, K, D=D, refine=refine, flags=iterative_flag)
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    return T_from_Rt(R, tvec)


# end def estimate_tag_pose


def estimate_tag_poses(
    tag2d_list: Sequence[Tag2D],
    tag_size: Union[float, Dict[int, float]],
    K: ArrayLike,
    D: Optional[ArrayLike] = None,
    use_ippe: bool = True,
) -> Dict[int, np.ndarray]:
    """
    批量估计多个 tag 的位姿
    Args:
        tag2d_list (Sequence[Tag2D]): detect() 的检测结果
        tag_size (float | Dict[int, float]): tag 边长( 米 ); 不同 tag 尺寸不同时可以传 {tag_id: size}
        K (ArrayLike): 相机内参
        D (ArrayLike | None): 畸变系数
        use_ippe (bool): 是否使用 IPPE_SQUARE
    Returns:
        (Dict[int, np.ndarray]): {tag_id: T_cam_tag}, 解算失败的 tag 不会出现在字典里
    """

    pose_dict: Dict[int, np.ndarray] = {}
    for tag2d in tag2d_list:
        size = tag_size[tag2d.id] if isinstance(tag_size, dict) else tag_size
        T = estimate_tag_pose(tag2d, size, K, D=D, use_ippe=use_ippe)
        if T is not None:
            pose_dict[tag2d.id] = T
    return pose_dict


# end def estimate_tag_poses


######################################################### 兼容层 #########################################################


class apriltag_helper:
    """
    门面类( facade ), 把本模块的能力挂在一个类名下面, 方便写 ``from ... import apriltag_helper`` 后
    用 ``apriltag_helper.Detector`` / ``apriltag_helper.create_calib_board_3d`` 这种方式调用,
    习惯直接 import 函数的话用上面那些顶层函数完全等价。
    """

    Tag2D = Tag2D
    Tag3D = Tag3D
    Detector = Detector

    create_calib_board_3d = staticmethod(create_calib_board_3d)
    create_calib_board_img = staticmethod(create_calib_board_img)
    calib_board_space_size = staticmethod(calib_board_space_size)
    calib_board_paper_size = staticmethod(calib_board_paper_size)
    rescale_calib_board_params = staticmethod(rescale_calib_board_params)
    save_calib_board_pdf = staticmethod(save_calib_board_pdf)
    make_default_detector = staticmethod(make_default_detector)
    locate_calib_board = staticmethod(locate_calib_board)
    match_tag_pairs = staticmethod(match_tag_pairs)
    estimate_tag_pose = staticmethod(estimate_tag_pose)
    estimate_tag_poses = staticmethod(estimate_tag_poses)
    solve_pnp = staticmethod(solve_pnp)
    build_tag_pose_from_corners3d = staticmethod(build_tag_pose_from_corners3d)
    expand_corners = staticmethod(expand_corners)
    inv_tf = staticmethod(inv_tf)
    T_from_Rt = staticmethod(T_from_Rt)
    to_gray_u8 = staticmethod(to_gray_u8)
    find_tag = staticmethod(find_tag)
    sort_tags = staticmethod(sort_tags)

    render_tag_pattern = staticmethod(render_tag_pattern)
    render_tag_image = staticmethod(render_tag_image)
    load_tag_codes = staticmethod(load_tag_codes)
    get_tag_code = staticmethod(get_tag_code)
    list_tag_families = staticmethod(list_tag_families)
    tag_family_count = staticmethod(tag_family_count)


# end class apriltag_helper

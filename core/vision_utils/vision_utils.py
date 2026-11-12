"""
视觉定位相关的工具函数和类
"""
from  dataclasses import dataclass
import logging
import math
import os
import sys

import cv2
import numpy as np
import open3d

#import apriltag2

root_dir = os.path.normpath(f"{os.path.dirname(os.path.realpath(__file__))}/../../../")# 导入本工程的模块
sys.path.append(root_dir)
from core.common_utils .utils import Common_data

from core.board_utils.apriltag_helper import apriltag_helper

Logger = logging.getLogger(__name__)

########################点云生成函数########################

_DEPTH_DTYPES = (np.uint16, np.float32)
"""支持的深度图类型: uint16( 原始深度, 单位由 depth_scale 决定 ) 或 float32( 已换算成米 )"""


def _prepare_depth(depth_img: np.ndarray,
                   depth_scale: float,
                   depth_range: tuple[float, float] = None) -> np.ndarray:
    '''校验深度图参数, 并按深度范围( 米 )过滤

    Args:
        depth_img (np.ndarray): 深度图像, uint16 或 float32
        depth_scale (float): 深度图单位换算成米的系数, z( 米 ) = 像素值 * depth_scale
        depth_range (tuple[float, float], optional): 保留的深度范围 (min, max), 单位米

    Returns:
        (np.ndarray): 过滤后的深度图, 保持原始 dtype, 可直接交给 Open3D
    '''

    assert isinstance(depth_img, np.ndarray) and depth_img.ndim == 2, "depth_img must be a 2D array"
    assert depth_img.dtype in _DEPTH_DTYPES, \
        f"depth_img dtype must be uint16 or float32, got {depth_img.dtype}"
    assert isinstance(depth_scale, (int, float)) and depth_scale > 0, \
        "depth_scale must be a positive number, e.g. 0.001 for a uint16 depth map in millimeters"

    if depth_range is None:
        return depth_img
    # end if

    min_depth, max_depth = depth_range
    assert min_depth >= 0 and max_depth > min_depth, \
        "depth_range must be (min_depth, max_depth) in meters with min_depth >= 0"

    # 把以米为单位的范围换算回深度图的原始单位, 避免整幅图先做一次浮点乘法
    raw_min = min_depth / depth_scale
    raw_max = max_depth / depth_scale
    filter_depth_img = depth_img.copy()
    filter_depth_img[(filter_depth_img < raw_min) | (filter_depth_img > raw_max)] = 0
    return filter_depth_img
# end def _prepare_depth


def rgbd_to_point_cloud(rgb_img: np.ndarray,
                        depth_img: np.ndarray,
                        intrinsic: list[float],
                        depth_range: tuple[float, float] = None,
                        depth_trunc: float = 3.0,
                        depth_scale: float = 1.0) -> open3d.geometry.PointCloud:

    '''将 RGB-D 图像转换为点云
    Args:
        rgb_img (np.ndarray): RGB 图像 CV_8UC3
        depth_img (np.ndarray): 深度图像, uint16 或 float32
        intrinsic (list[float]): 相机内参 [fx, fy, cx, cy]
        depth_scale (float): 深度图单位换算成米的系数, z( 米 ) = 像素值 * depth_scale
            - uint16 毫米深度图传 0.001; 已经用米表示的 float32 图传 1.0( 默认 )
        depth_range (tuple[float, float], optional): 保留的深度范围 (min, max), 单位米
        depth_trunc (float): 超过该深度( 米 )的点会被丢弃, 默认 3.0; 传 float('inf') 表示不截断
    Returns:
        open3d.geometry.PointCloud: 生成的点云
    '''

    #判断图像类型
    assert rgb_img.dtype == np.uint8 and rgb_img.ndim == 3 and rgb_img.shape[2] == 3, "rgb_img must be CV_8UC3"
    assert intrinsic is not None and len(intrinsic) == 4, "intrinsic must be a list of 4 elements [fx, fy, cx, cy]"
    assert rgb_img.shape[:2] == depth_img.shape[:2], "rgb_img and depth_img must have the same resolution"

    #将超出深度范围的深度置0
    filter_depth_img = _prepare_depth(depth_img, depth_scale, depth_range)

    #将RGB-D图像转换为点云
    o3d_depth = open3d.geometry.Image(np.ascontiguousarray(filter_depth_img))  #将深度图像转换为Open3D格式
    o3d_rgb = open3d.geometry.Image(np.ascontiguousarray(rgb_img))  #将RGB图像转换为Open3D格式

    #将彩色图像和深度图像对应点对匹配( Open3D 的 depth_scale 定义是"像素值/米", 与本函数的定义互为倒数 )
    o3d_rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb,
        o3d_depth,
        depth_scale=1.0 / depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    #生成点云
    rgbd_pc = open3d.geometry.PointCloud.create_from_rgbd_image(
        o3d_rgbd,
        #针孔相机模型对象，用于像素到空间点的反投影，为点云提供xy方向的物理信息
        open3d.camera.PinholeCameraIntrinsic(rgb_img.shape[1], rgb_img.shape[0], intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]),
    )

    return rgbd_pc
# end def rgbd_to_point_cloud


def depth_to_point_cloud(depth_img: np.ndarray,
                         intrinsic: list[float],
                         depth_range: tuple[float, float] = None,
                         depth_trunc: float = 3.0,
                         depth_scale: float = 1.0) -> open3d.geometry.PointCloud:
    '''将深度图像转换为点云
    Args:
        depth_img (np.ndarray): 深度图像, uint16 或 float32
        intrinsic (list[float]): 相机内参 [fx, fy, cx, cy]
        depth_scale (float): 深度图单位换算成米的系数, z( 米 ) = 像素值 * depth_scale
            - uint16 毫米深度图传 0.001; 已经用米表示的 float32 图传 1.0( 默认 )
        depth_range (tuple[float, float], optional): 保留的深度范围 (min, max), 单位米
        depth_trunc (float): 超过该深度( 米 )的点会被丢弃, 默认 3.0; 传 float('inf') 表示不截断
    Returns:
        open3d.geometry.PointCloud: 生成的点云
    '''

    assert intrinsic is not None and len(intrinsic) == 4, "intrinsic must be a list of 4 elements [fx, fy, cx, cy]"

    filter_depth_img = _prepare_depth(depth_img, depth_scale, depth_range)

    depth_pc = open3d.geometry.PointCloud.create_from_depth_image(
        open3d.geometry.Image(np.ascontiguousarray(filter_depth_img)),
        open3d.camera.PinholeCameraIntrinsic(depth_img.shape[1], depth_img.shape[0], intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]),
        depth_scale=1.0 / depth_scale,
        depth_trunc=depth_trunc,
    )

    return depth_pc
# end def depth_to_point_cloud

def depth_mean_filter(depth_img_list: list[np.ndarray],
                    obs_ratio: float = 0.5) -> np.ndarray:
    """
    对深度图像列表进行均值滤波:
    1) 每个像素仅统计深度 >0 的观测次数
    2) 平均深度 = 该像素所有深度和 / 观测次数
    3) 观测次数 < obs_ratio * 列表长度时,该像素深度置 0
    Args:
        depth_img_list (List[np.ndarray]): 深度图像列表, CV_16UC1 格式
        obs_ratio (float): 观测次数阈值比例
    Returns:
        (np.ndarray): 经过均值滤波后的深度图像, CV_16UC1 格式
    """

    if len(depth_img_list) == 0:
        raise ValueError(f"{Common_data.RED}depth_img_list is empty.{Common_data.RESET}")
    if obs_ratio < 0.1 or obs_ratio > 1:
        raise ValueError(f"{Common_data.RED}obs_ratio must be in the range [0.1, 1].{Common_data.RESET}")

    #将深度图像列表转换为(N,H,W)Numpy数组,并转为uint32类型避免加和溢出
    depth_img=np.vstack(depth_img_list,axis=0).astype(np.uint32)
    n=depth_img.shape[0]

    #计算每像素的观测次数，即每像素深度非0的总帧数
    obs_count=np.count_nonzero(depth_img,axis=0)

    #计算每像素的深度和，即每像素深度的总和
    depth_sum=np.sum(depth_img,axis=0,dtype=np.uint32)

    #有效条件: 观测次数 >= obs_ratio * 帧数,且观测次数 >0
    th_ob_cnt=max(1,math.ceil(obs_ratio*n))
    #生成掩码，有效像素置1,无效像素置0
    valid_mask=(obs_count>=th_ob_cnt)

    #对有效像素做除法, depth_sum / obs_count, 无效像素结果置 0
    depth_mean=np.zeros_like(depth_sum,dtype=np.float32)
    np.divide(depth_sum,obs_count,out=depth_mean,where=valid_mask)

    #转回 CV_16UC1
    return np.rint(depth_mean).astype(np.uint16)



###########图像处理及定位辅助函数########################

def compute_tag_pose_2d(tag:apriltag_helper.Tag2D, intrinsic:list[float])->tuple[float,float,float]:
    """
    计算2D标签在图像中的位姿
    Args:
        tag (apriltag_helper.Tag2D): apriltag2检测到的标签对象
        intrinsic (list[float]): 相机内参 [fx, fy, cx, cy]
    Returns:
        tuple[float,float,float]: 标签在图像中的位姿 (tx, ty, theta)
    """

    center=tag.center
    u,v=center[0],center[1]
    nx=(u-intrinsic[2])/intrinsic[0]#标签在图像中的归一化坐标
    ny=(v-intrinsic[3])/intrinsic[1]#标签在图像中的归一化坐标
    theta=np.arctan2(tag.corners[1]-tag.corners[0])

    return (nx,ny,theta)

def compute_discrete_locate_error(expected_T_cam_model: np.ndarray,
                                actual_T_cam_model: np.ndarray,
                                sym_tfs: np.ndarray = None) -> tuple[float, float]:
    """
        计算定位的期望值与实际值的误差, 支持对称工件（比如方形、正多边形，转 90° 也是同一个工件），
        遍历所有对称姿态，选出旋转误差最小的那一组，输出位置误差 (mm)、角度误差 (°)
        Args:
            expected_T_cam_model (np.ndarray): 期望的物体到相机的变换矩阵, 4*4,也就是抓取模板的物体到相机位姿
            actual_T_cam_model (np.ndarray): 实际的物体到相机的变换矩阵, 4*4
            sym_tfs (np.ndarray): 对称变换矩阵数组, N * 4*4, 如果物体具有离散对称性, 则在计算误差时应该考虑所有的对称变换, 取误差最小的那个作为最终误差. 
                如果没有对称变换, 则传入 None 或者空数组即可
        Returns:
            (Tuple[float, float]): 位置误差( mm )和角度误差( deg )
    """

    if sym_tfs is None:
        # 没有对称变换,即构造旋转单位阵表示无旋转对称性
        sym_tfs = np.array([np.eye(4)], dtype=np.float32)

    N = sym_tfs.shape[0]
    min_delta_deg = 1000.0
    for i in range(N):
        sym_tf = sym_tfs[i]
        #对于每个对称变换，计算其在相机下的位姿，
        #且变换矩阵是定义在物体坐标系下，故为右乘变换矩阵
        T_cam_model = actual_T_cam_model @ sym_tf
        #计算T_cam_model到expected_T_cam_model之间的旋转矩阵
        delta_rot = expected_T_cam_model[:3, :3] @ T_cam_model[:3, :3].T
        cosine=(np.trace(delta_rot)-1.0)/2.0
        #确保数值的稳定性
        cosine=np.clip(cosine,-1.0,1.0)
        delta_deg=abs(np.arccos(cosine))
        if delta_deg<min_delta_deg:
            min_delta_deg=delta_deg
            delta_pos=expected_T_cam_model[:3,3]-T_cam_model[:3,3]

    pos_err=np.linalg.norm(delta_pos)*1000.0
    rot_err=min_delta_deg*180.0/np.pi

    return pos_err,rot_err

def compute_projective_transformation(src_K: np.ndarray,
                                    src_img_size: tuple[int, int],
                                    R_dst_src: np.ndarray) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """
    已知 src 相机内参和 src 图像尺寸, 以及从 src 相机坐标系到 dst 相机坐标系的旋转矩阵,
    计算从 src 图像到 dst 图像的射影变换矩阵 H_dst_src, 以及 dst 相机内参和 dst 图像尺寸, 使得 src 图像内容尽量完整地投影到 dst 图像上
    Args:
        src_K (np.ndarray): src 相机的内参矩阵, 3*3 float64
        src_img_size (Tuple[int, int]): src 图像尺寸 (w,h)
        R_dst_src (np.ndarray): 从 src 相机坐标系到 dst 相机坐标系的旋转矩阵
    Returns:
        ( Tuple[np.ndarray, Tuple[int, int], np.ndarray] ): 包含以下内容的元组:
            - dst_K (np.ndarray): dst 相机的内参矩阵( 焦距与 src 相机保持一致 ), 3*3 float64
            - dst_img_size (Tuple[int, int]): dst 图像尺寸 (w,h)
            - H_dst_src (np.ndarray): 从 src 图像到 dst 图像的射影变换矩阵, 3*3 float64
    Note:
        该函数一般用于将机械臂末端相机拍摄的图像变换到与机械臂末端坐标系方向对齐的视角, 以简化后续的视觉处理
    """

    # 计算 src 图像四角的像素的齐次坐标
    src_w, src_h = src_img_size
    corners = np.array([
        [0, 0, 1.0],                  # 左上
        [src_w - 1, 0, 1.0],          # 右上
        [src_w - 1, src_h - 1, 1.0],  # 右下
        [0, src_h - 1, 1.0],          # 左下
    ], dtype=np.float64).T  # (3,4)

    src_rays = np.linalg.inv(src_K) @ corners    # (3,4)  角点在 src 相机坐标系中的射线（方向向量,未归一化）
    dst_rays = R_dst_src @ src_rays              # (3,4)  角点在 dst 相机坐标系中的射线（方向向量,未归一化）

    z = dst_rays[2, :]
    if np.any(z <= 1e-6):
        Logger.error(f"{Common_data.RED}有 src 的图像角点在旋转后落在 dst 相机的后方(z<=0), 无法投影到 dst 图像平面z{Common_data.RESET}")
        return None, None, None
    # end if

    # 不含主点偏移的投影的像素坐标 u_hat = fx * (x/z), v_hat = fy * (y/z), 其中 (x,y,z) 是 dst_rays 中的坐标
    fx, fy = src_K[0, 0], src_K[1, 1]
    u_hat = fx * (dst_rays[0, :] / z)
    v_hat = fy * (dst_rays[1, :] / z)

    u_min, u_max = float(np.min(u_hat)), float(np.max(u_hat))
    v_min, v_max = float(np.min(v_hat)), float(np.max(v_hat))

    # 通过主点平移把内容移入正像素区域
    dst_cx = -u_min
    dst_cy = -v_min

    # 最小不裁剪输出尺寸
    dst_w = int(np.ceil(u_max - u_min)) + 1
    dst_h = int(np.ceil(v_max - v_min)) + 1

    dst_K = np.array([
        [fx, 0.0, dst_cx],
        [0.0, fy, dst_cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 射影变换矩阵 H_dst_src: pixel_src --> pixel_dst
    H_dst_src = dst_K @ R_dst_src @ np.linalg.inv(src_K)

    return dst_K, (dst_w, dst_h), H_dst_src
# end def compute_projective_transformation

def compute_tag_mask(tag: apriltag_helper.Tag2D,
                    mask_img: np.ndarray,
                    ex_ratio: float = 0.3) -> None:
    """
    根据标签顶点计算掩码
    Args:
        tag (apriltag2.Tag2D): AprilTag 实例
        mask_img (np.ndarray): 掩码图像
        ex_ratio (float): 外延比例

    Note:
        1) 掩码图像应该在函数外部创建好, 并且与输入的深度图像尺寸相同, 格式为 CV_8UC1
    """

    corners2d = tag.corners

    # 计算外延点
    ex_corners2d = np.zeros((4, 2), dtype=np.float32)
    ex_corners2d[0] = corners2d[0] + (corners2d[0] - corners2d[2]) * ex_ratio
    ex_corners2d[1] = corners2d[1] + (corners2d[1] - corners2d[3]) * ex_ratio
    ex_corners2d[2] = corners2d[2] + (corners2d[2] - corners2d[0]) * ex_ratio
    ex_corners2d[3] = corners2d[3] + (corners2d[3] - corners2d[1]) * ex_ratio

    # 制作掩码
    cv2.fillConvexPoly(mask_img, ex_corners2d.astype(np.int32), 255)
# end def compute_tag_mask


def compute_tag_corners_3d(tag: apriltag_helper.Tag2D,
                        depth_img: np.ndarray,
                        intrinsic: np.ndarray,
                        depth_scale: float,
                        ex_ratio: float = 0.1
                        ) -> np.ndarray:
    """
    根据标签的 2D 角点和深度图像计算标签的 3D 角点坐标
    Args:
        tag (apriltag2.Tag2D): AprilTag 实例
        depth_img (np.ndarray): 深度图像
        intrinsic (np.ndarray): 相机内参矩阵
        depth_scale (float): 深度缩放因子
        ex_ratio (float): 外延比例, 用于扩展标签的边界以获取更多的深度信息, 例如 ex_ratio=0.1 表示将标签边界向外扩展 10% 的距离
    Returns:
        (np.ndarray): 标签的 3D 角点坐标, (4,3) 格式
    """

    # 绘制掩码
    mask_img = np.zeros(depth_img.shape, dtype=np.uint8)
    compute_tag_mask(tag, mask_img, ex_ratio=ex_ratio)

    # 将非掩码区域设为0
    flt_depth_img = depth_img.astype(np.float32) * depth_scale
    flt_depth_img[mask_img == 0] = 0.0

    # 提取掩码区域的点云
    pc = open3d.geometry.PointCloud.create_from_depth_image(
        open3d.geometry.Image(flt_depth_img),
        open3d.camera.PinholeCameraIntrinsic(flt_depth_img.shape[1], flt_depth_img.shape[0],
                                            intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]),
        np.eye(4),
        depth_scale=1.0,
        depth_trunc=0.5
    )

    # 平面拟合
    plane, inliers = pc.segment_plane(distance_threshold=0.002, ransac_n=6, num_iterations=1000)
    Logger.info(f"{Common_data.GREEN}plane equation: {plane}, inliers count: {len(inliers)}, inliers ratio: {len(inliers)/len(pc.points)}{Common_data.RESET}")

    # 计算角点的空间坐标( 根据平面方程计算 )
    def compute_pt3d(corner: np.ndarray, intrinsic: np.ndarray, plane: np.ndarray) -> np.ndarray:
        nx = (corner[0] - intrinsic[2]) / intrinsic[0]
        ny = (corner[1] - intrinsic[3]) / intrinsic[1]
        A, B, C, D = plane
        z = -D / (A * nx + B * ny + C)
        x = nx * z
        y = ny * z
        return np.array([x, y, z])
    # end def compute_pt3d

    corners2d = tag.corners
    corners3d = np.zeros((4, 3), dtype=np.float32)
    for i in range(4):
        corners3d[i] = compute_pt3d(corners2d[i], intrinsic, plane)
    # end for

    return corners3d
# end def compute_tag_corners3d


def compute_tag_pose_3d(corners3d: np.ndarray) -> np.ndarray:
    """
    根据标签的 3D 角点坐标计算标签的位姿( 从标签坐标系到相机坐标系的变换矩阵 )
    Args:
        corners3d (np.ndarray): 标签的 3D 角点坐标, (4,3) 格式
    Returns:
        (np.ndarray): 从标签坐标系到相机坐标系的变换矩阵, 4*4 格式
    """

    center3d = np.mean(corners3d, axis=0)

    axis_x = corners3d[1] - corners3d[0]
    axis_x = axis_x / np.linalg.norm(axis_x)

    axis_y = corners3d[3] - corners3d[0]
    axis_y = axis_y / np.linalg.norm(axis_y)

    axis_z = np.cross(axis_x, axis_y)
    axis_z = axis_z / np.linalg.norm(axis_z)

    #确保xyz轴在数学上是垂直的，避免xy之间计算是不垂直的
    axis_y = np.cross(axis_z, axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)

    R = np.stack([axis_x, axis_y, axis_z], axis=1)  # (3,3)

    T_cam_tag = np.eye(4, dtype=np.float32)
    T_cam_tag[:3, :3] = R
    T_cam_tag[:3, 3] = center3d

    return T_cam_tag





#########################畸变矫正器类########################
class ImageUndistorter:

    '''图像畸变矫正器'''

    def __init__(self, intrinsic:list[float], distortion:list[float] | None=None) -> None:

        """
        构造函数
        Args:
            intrinsic (List[float]): 相机内参 [fx, fy, cx, cy]
            distortion (List[float]): 相机畸变参数, 格式与 OpenCV 的畸变参数格式相同, 可选, 若不提供或提供空列表则认为无畸变
        """
        self.D = None
        if distortion is None and len(distortion) >0:
            self.D = np.array(distortion, dtype=np.float64)

        self._undistort_maps = []# 畸变矫正映射映射表

        self.K = np.array([
            [intrinsic[0], 0.0, intrinsic[2]],
            [0.0, intrinsic[1], intrinsic[3]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def undistort_img(self, src_img:np.ndarray) -> np.ndarray:
        """
            对输入图像进行畸变矫正,如果构造函数中未提供畸变参数,则直接返回输入图像的副本
            Args:
                src_img (np.ndarray): 输入图像, CV_8UC1/CV_8UC3 格式
            Returns:
                (np.ndarray): 畸变矫正后的图像, CV_8UC1/CV_8UC3 格式
        """
        if self.D is None:
            return src_img.copy()

        if len(self._undistort_maps) == 0:
            map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (src_img.shape[1], src_img.shape[0]), cv2.CV_32FC1)
            self._undistort_maps = [map1, map2]
            Logger.info(f"{Common_data.GREEN}图像畸变映射表初始化成功{Common_data.RESET}")

        undistorted_img = cv2.remap(src_img, self._undistort_maps[0], self._undistort_maps[1], interpolation=cv2.INTER_LINEAR)
        return undistorted_img

    def undistort_points(self, src_pts: np.ndarray, do_normalize: bool = False) -> np.ndarray:
        """
        对输入的点进行畸变矫正,如果构造函数中未提供畸变参数,则直接返回输入点的副本
        Args:
            src_pts (np.ndarray): 输入点, shape=(N, 2), dtype=np.float32/np.float64
            do_normalize (bool): 是否对输出点进行归一化, 若为 True 则输出点为归一化坐标, 若为 False 则输出点为像素坐标
        Returns:
            (np.ndarray): 畸变矫正后的点, shape=(N, 2), dtype=np.float32/np.float64
        """
        if self.D is None:
            return src_pts.copy()

        src_pts = src_pts.reshape(-1, 1, 2)
        undistorted_pts = cv2.undistortPoints(src_pts, self.K, self.D, P=self.K if not do_normalize else None)
        undistorted_pts = undistorted_pts.reshape(-1, 2)
        return undistorted_pts


    
    
#########################2D标签匹配器类########################
class TagMatcher2D:
    '''2D标签匹配器类，用于在图像标签识别及平面位姿估计'''

    @dataclass
    class Config:
        '''2D标签匹配器存储配置类'''
        intrinsic: list[float]
        """ 相机内参 [fx, fy, cx, cy] """
                    
        distortion: list[float] = None
        """ 相机畸变参数, 可选, 格式与 OpenCV 相同 """

        depth_scale: float 
        """ 深度图像的缩放比例 """

        tag_family: str = "tag36h11"
        """ AprilTag 的 tag family, 可选, 默认为 "tag36h11" """

        black_border: int = 1
        """ AprilTag 的黑色边框宽度, 可选, 默认为 1 """

        debug_dir: str = None
        """ 调试结果保存目录, 可选, 如果为 None 则不保存, 如果执行其他成员函数时设置了 debug_level > 0, 则会将中间结果保存到该目录下 """

    @dataclass
    class Result:
        """
        匹配结果类, 用于存储匹配结果和相关信息
        """

        id = int(-1)
        """标签 ID"""

        center = np.zeros(2, dtype=np.float32)
        """标签中心点 [x, y]"""

        corners: np.ndarray = np.zeros((4, 2), dtype=np.float32)
        """标签四个角点 [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]"""

        pose_2d: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """标签的平面位姿 [tx, ty, theta], 其中 tx 和 ty 是标签中心点相对于相机归一化坐标系原点的平面坐标, 
        theta 是标签坐标系相对于相机归一化坐标系的旋转角( 单位: rad )"""

    def __init__(self, config: Config):
        """
        构造函数
        Args:
            Config (TagMatcher2D.Config): 2D标签匹配器配置对象
        """
        self.intrinsic = np.array(config.intrinsic, dtype=np.float32)
        self.distortion = np.array(config.distortion, dtype=np.float32) if config.distortion is not None else None
        self.depth_scale = config.depth_scale
        self.tag_family = config.tag_family
        self.black_border = config.black_border
        self.debug_dir = config.debug_dir

        # 初始化 AprilTag 检测器
        self.detector = apriltag_helper.AprilTagDetector(family=self.tag_family, black_border=self.black_border)
        # 初始化图像去畸变工具
        self.undistorter = ImageUndistorter(intrinsic=self.intrinsic, distortion=self.distortion)

        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)
            Logger.info(f"{Common_data.GREEN}2D标签匹配器调试目录已创建: {self.debug_dir}{Common_data.RESET}")

    def match(self, bgr_img: np.ndarray, top_k: int = 0, debug_level: int = 0) -> list[Result]: 

        """
        匹配: 识别图像中的 AprilTag 实例, 并估计其位姿
        Args:
            bgr_img (np.ndarray): 彩色图像, 可能存在畸变
            top_k (int): 仅匹配分数最高的 k 个实例, 0 表示匹配所有实例
            debug_level (int): 调试等级( 0: 不保存任何结果; 1: 保存输入信息; 2: 保存分割结果和位姿估计结果; )
        Returns:
            (Tuple[List[Result], str]): 匹配结果列表和返回的消息
        """

        result_dir=f"{self.debug_dir}/match"

        if self.debug_dir is not None and not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
            Logger.info(f"{Common_data.BLUE}创建结果目录: {result_dir}{Common_data.RESET}")

        # 去畸变
        undistort_img=self.undistorter.undistort_img(bgr_img)
        # 匹配apriltag
        tag_list = self.detector(undistort_img, -1)

        if (len(tag_list)==0 or debug_level>0) and os.path.exists(result_dir):
            cv2.imwrite(f"{result_dir}/color.png",undistort_img)

        result_list=[]
        for tag in tag_list:
            result=self.Result()
            result.id=tag.id
            result.center=tag.center
            result.corners=tag.corners
            result.pose_2d=compute_tag_pose_2d(result.corners)
            result_list.append(result)

        if len(result_list)>0 and top_k<=len(result_list):
            result_list.sort(key=lambda r:r.pose_2d[0]**2+r.pose_2d[1]**2)
            result_list=result_list[:top_k]

        return result_list

    def draw(self, bgr_img:np.ndarray, result_list:list[Result]):
        '''绘制匹配结果'''
        for result in result_list:

            # 绘制轮廓
            corners = result.corners.astype(np.int32)
            cv2.polylines(bgr_img, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

            # 绘制朝向
            dir = result.corners[1] - result.corners[0]
            cv2.arrowedLine(bgr_img, tuple(result.center.astype(np.int32)), tuple((result.center + dir).astype(np.int32)),
                            color=(0, 255, 255), thickness=1)


class TagMatcher3D:
    '''3D 标签匹配器，利用彩色图和深度图输出标签位姿'''

    @dataclass
    class Config:
        """
        配置类, 用于存储 TagMatcher3D 的配置参数
        """

        intrinsic: list[float]
        """ RGB-D 相机内参 [fx, fy, cx, cy] """

        depth_scale: float
        """ 深度图像缩放比例 """

        distortion: list[float] = None
        """ 相机畸变参数, 可选, 格式与 OpenCV 相同 """

        tag_family: str = "tag36h11"
        """ AprilTag 的 tag family, 可选, 默认为 "tag36h11" """

        black_border: int = 1
        """ AprilTag 的黑色边框宽度, 可选, 默认为 1 """

        debug_dir: str = None
        """ 调试结果保存目录, 可选, 如果为 None 则不保存, 如果执行其他成员函数时设置了 debug_level > 0, 则会将中间结果保存到该目录下 """

    @dataclass
    class Result:
        '''
        匹配结果类
        '''
        
        id = int(-1)
        """标签 ID"""

        center = np.zeros(2, dtype=np.float32)
        """标签中心点 [x, y]"""

        corners: np.ndarray = np.zeros((4, 2), dtype=np.float32)
        """标签四个角点 [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]"""

        T_cam_tag: np.ndarray = np.eye(4, dtype=np.float32)
        """ 从标签坐标系到相机坐标系的变换矩阵, 4*4 """

    def __init__(self, config: Config):
        """
        Args:
            config ( Config ): TagMatcher3D 的配置对象, 包含了物体检测和位姿估计所需的各种参数
        """

        self.intrinsic = config.intrinsic
        self.depth_scale = config.depth_scale
        self.undistorter = ImageUndistorter(intrinsic=self.intrinsic, distortion=config.distortion)

        self.detector = apriltag_helper.Detector(tag_family=config.tag_family, black_border=config.black_border)

        self.tag_size = None

        self.debug_dir = config.debug_dir
        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)
            Logger.info(f"{Common_data.GREEN}已创建调试文件夹{self.debug_dir}{Common_data.RESET}")

        Logger.info(f"{Common_data.GREEN}Apriltag3D匹配器对象已创建{Common_data.RESET}")


    def macth(self,
            bgr_img: np.ndarray,
            depth_img: np.ndarray,
            top_k: int = 0,
            debug_level: int = 0) -> tuple[list[Result], str]:

        """
            匹配: 识别图像中的 AprilTag 实例, 并估计其位姿
            Args:
                bgr_img (np.ndarray): 彩色图像, 可能存在畸变
                depth_img (np.ndarray): 深度图像
                top_k (int): 仅匹配分数最高的 k 个实例, 0 表示匹配所有实例
                debug_level (int): 调试等级( 0: 不保存任何结果; 1: 保存输入信息; 2: 保存分割结果和位姿估计结果; )
            Returns:
                (Tuple[List[Result], str]): 匹配结果列表和返回的消息
        """

        result_dir=os.path.join(self.debug_dir, "match")

        if self.debug_diris is not None and not os.path.exists(result_dir):
            os.makedirs(result_dir,exist_ok=True)
            Logger.info(f"{Common_data.BLUE}保存结果到{result_dir}{Common_data.RESET}")

        #畸变矫正
        bgr_img = self.undistorter.undistort_img(bgr_img)

        # apriltag检测
        tag_list = self.detector.detect(bgr_img, -1)

        #定位每个标签
        result_list =[]
        for tag in tag_list:
            result=self.Result()
            result.tag_id = tag.id
            result.center = tag.center
            result.corners = tag.corners
            corners3d=compute_tag_corners_3d(tag,depth_img,self.intrinsic,self.depth_scale)
            self.T_cam_tag=compute_tag_pose_3d(corners3d)
            result_list.append(result)

            if self.tag_size is  None:
                self.tag_size= self.tag_size = np.linalg.norm(corners3d[0] - corners3d[2]) / math.sqrt(2)
                Logger.info(f"{Common_data.BLUE}tag size estimated to be: {self.tag_size}{Common_data.RESET}")

        # 选取分数最高的 top_k 个实例( 以 T_cam_tag[2,3] 为深度分数, 即深度越小越小越好 )
        if top_k > 0 and len(result_list) > top_k:
            result_list.sort(key=lambda r: r.T_cam_tag[2, 3])  # 按深度分数排序
            result_list = result_list[:top_k]  # 选取前 top_k 个实例
        # end if

        return result_list

    def track(self,
            bgr_img: np.ndarray,
            depth_img: np.ndarray,
            init_T_cam_tag: np.ndarray,
            debug_level: int = 0) -> tuple[np.ndarray, str]:
        """
            跟踪: 跟踪上一帧的 AprilTag 实例, 并估计其位姿,
            依据上一帧的标签裁剪下一帧的图像，保留标签部分图像，提高计算速度
            Args:
                bgr_img (np.ndarray): 彩色图像, 存在畸变
                depth_img (np.ndarray): 深度图像
                init_T_cam_tag (np.ndarray): 上一帧的标签位姿, 4*4
                debug_level (int): 调试等级( 0: 不保存任何结果; 1: 保存输入信息; 2: 保存分割结果和位姿估计结果; )
            Returns:
                    (Tuple[np.ndarray, str]): 跟踪结果, 4*4 的变换矩阵, 和返回的消息
        """ 
        result_dir = f"{self.debug_dir}/track"
        
        # 创建保存跟踪结果的目录
        if self.debug_dir is not None and not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
            Logger.info(f"{Common_data.GREEN}track results will be saved to: {result_dir}{Common_data.RESET}")
        # end if

        corners3d = np.zeros((4, 3), dtype=np.float32)
        half_size = self.tag_size / 2
        corners3d[0] = np.array([-half_size, -half_size, 0], dtype=np.float32)
        corners3d[1] = np.array([half_size, -half_size, 0], dtype=np.float32)
        corners3d[2] = np.array([half_size, half_size, 0], dtype=np.float32)
        corners3d[3] = np.array([-half_size, half_size, 0], dtype=np.float32)

        corners3d = corners3d @ init_T_cam_tag[:3, :3].T + init_T_cam_tag[:3, 3]  # 将标签坐标系下的角点转换到相机坐标系

        # 计算角点在图像中的投影位置
        proj_corners2d = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            x, y, z = corners3d[i]
            u = self.intrinsic[0] * (x / z) + self.intrinsic[2]
            v = self.intrinsic[1] * (y / z) + self.intrinsic[3]
            proj_corners2d[i] = [u, v]
        # end for

        # 计算投影的最小外接矩形( 用于后续位姿细化的初始掩码 )
        min_x, min_y = np.min(proj_corners2d, axis=0)
        max_x, max_y = np.max(proj_corners2d, axis=0)

        # 放大矩形以包含更多的像素
        rect_w = (max_x - min_x) * 2.0
        rect_h = (max_y - min_y) * 2.0
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        min_x = max(0, center_x - rect_w / 2)
        max_x = min(bgr_img.shape[1] - 1, center_x + rect_w / 2)
        min_y = max(0, center_y - rect_h / 2)
        max_y = min(bgr_img.shape[0] - 1, center_y + rect_h / 2)

        # 畸变矫正
        un_img = self.undistorter.undistort_img(bgr_img)

        # 根据投影的最小外接矩形裁剪图像
        crop_bgr_img = un_img[int(min_y):int(max_y), int(min_x):int(max_x)]
        tag_list = self.detector.detect(crop_bgr_img, -1)

        if (len(tag_list) == 0 or debug_level >= 1) and os.path.exists(result_dir):
            vis_img = un_img.copy()
            cv2.rectangle(vis_img, (int(min_x), int(min_y)), (int(max_x), int(max_y)),
                            color=(0, 0, 255), thickness=1)
            cv2.imwrite(f"{result_dir}/proj_area.png", vis_img)
        # end if

        if len(tag_list) == 0:
            msg = "no tag detected in cropped image"
            return None, msg
        # end if

        # 恢复标签坐标到原图坐标系
        tag = tag_list[0]
        tag.center += np.array([min_x, min_y])
        tag.corners += np.array([min_x, min_y])

        # 定位
        corners3d = compute_tag_corners_3d(tag, depth_img, self.intrinsic, self.depth_scale)
        T_cam_tag = compute_tag_pose_3d(corners3d)

        msg = "track successful"
        return T_cam_tag
    # end def track
# end class TagMatcher3D
    





        

    

        
        





    










        



    

    

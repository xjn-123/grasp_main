"""
视觉定位相关的工具函数和类
"""

import dataclasses
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

    @dataclasses
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

    @dataclasses
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






        



    

    

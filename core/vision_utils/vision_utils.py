"""
视觉定位相关的工具函数和类
"""

import os
import sys
import logging
import dataclasses

import math

import numpy as np
import cv2
import open3d

#import apriltag2

root_dir = os.path.normpath(f"{os.path.dirname(os.path.realpath(__file__))}/../../../")# 导入本工程的模块
sys.path.append(root_dir)
from core.common_utils .utils import Common_data

from core.board_utils.apriltag_helper import apriltag_helper

Logger = logging.getLogger(__name__)

########################点云生成函数########################

def rgb_to_point_cloud():
    pass

def depth_to_point_cloud():
    pass
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

    def match(self, bgr_img: np.ndarray, top_k: int = 0, debug_level: int = 0) -> tuple[list[Result], str]: 

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






        



    

    

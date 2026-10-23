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

        T_cam_tag: np.ndarray = np.eye(4, dtype=np.float32)
        """ 从标签坐标系到相机坐标系的变换矩阵, 4*4 """

    

    

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


########################点云生成函数########################

def rgb_to_point_cloud():
    pass

def depth_to_point_cloud():
    pass

class TagMatcher2D:
    '''2D标签匹配器类，用于在图像标签识别及平面位姿估计'''

    @dataclasses
    class Config:
        '''2D标签匹配器存储配置类'''
        intrinsic: list[float]
        """ 相机内参 [fx, fy, cx, cy] """
                    
        distortion: list[float] = None
        """ 相机畸变参数, 可选, 格式与 OpenCV 相同 """

        tag_family: str = "tag36h11"
        """ AprilTag 的 tag family, 可选, 默认为 "tag36h11" """

        black_border: int = 2
        """ AprilTag 的黑色边框宽度, 可选, 默认为 2 """

        debug_dir: str = None
        """ 调试结果保存目录, 可选, 如果为 None 则不保存, 如果执行其他成员函数时设置了 debug_level > 0, 则会将中间结果保存到该目录下 """

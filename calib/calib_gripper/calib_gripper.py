''' calib_gripper.py
    用于标定夹爪，需满足针孔相机模型和apriltag标定条件，目前仅兼容二指夹爪
'''


import os
import sys 
import json
import dataclasses
import logging  

import numpy as np
import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(code_dir, "/../../")
sys.path.append(root_dir)

from core.board_utils import apriltag_helper
from core.common_utils import utils
from core.common_utils.utils import Common_data

logger = logging.getLogger(__name__)

''' 计算apritag在相机坐标系下的坐标，辅助标定夹爪'''
def campute_corners3d(  
    gray_img: np.ndarray,
    depth_img: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    depth_scale: float,
    expand_scale: float = 3.0,     # apriltag的扩展比例,默认3.0
) -> tuple[np.ndarray, np.ndarray]:
    
    ''' 畸变矫正'''
    K=np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]],dtype=np.float32)
    un_img = cv2.undistort(gray_img, K, distortion)


    ''' 检测apriltag的角点'''
    detector = apriltag_helper.Detector(tag_family="tag36h11")
    tag = detector.detect(un_img)
    tag_corners = tag.corners       #四个角点的图像坐标

    #构建点云掩膜,
    tag_center = tag.coenter
    expand_corners = []

    for corner in tag_corners:
        delta=corner-tag_center
        expand_corner = tag_center+delta*expand_scale
        expand_corners.append(expand_corner)

    expand_corners = np.array(expand_corners)

    mask = np.zeros_like(gray_img, dtype=np.uint8)
    pts = expand_corners.astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)

    #将非掩码区域设为0
    masked_depth = depth_img.copy()
    masked_depth[mask == 0] = 0

    #提取掩码区域的点云f

    #拟合apriltag平面

    #计算角点空间内坐标







    

    
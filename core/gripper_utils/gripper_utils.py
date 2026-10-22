'''检测夹爪状态的工具函数，包含夹爪与环境的碰撞以及姿态校验等'''

import logging
import os
import sys
import time

import cv2
import numpy as np
import open3d
import transforms3d

root_dir = os.path.normpath(f"{os.path.dirname(os.path.realpath(__file__))}/../../")
sys.path.append(root_dir)
from core.common_utils.utils import Common_data,inv_tf
from core.vision_utils.vision_utils import rgb_to_point_cloud,depth_to_point_cloud

logger = logging.getLogger(__name__)

class Gripper:
    pass

#####################夹爪位姿调整与检查函数#######################
def check_gripper_pose():
    pass

def com_axis_aligned_pose():
    pass



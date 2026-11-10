"""
功能说明: 创建3D抓取模板的 ROS 节点,也可以用于发布机械臂状态
模板需要保存以下数据:
- 1.末端刚好抓取到物体时的状态, 包含机械臂末端位姿和夹爪距离等信息
- 2.处于抓取准备阶段时的状态, 包含机械臂末端位姿、夹爪距离、以及物体在相机坐标系中的位姿
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import rclpy

root_dir = os.path.normpath(os.path.dirname(os.path.realpath(__file__)), "/../../")
sys.path.append(root_dir)

from core.camera_ros_utils.camera_ros_utils import CameraNode
from core.common_utils.utils import Common_data, read_camera_params, two_finger_gripper
from core.gripper_utils.gripper_utils import com_axis_aligned_pose
from core.robot_helper.robot_helper import ArmHelper
from core.vision_utils.vision_utils import TagMatcher3D, depth_mean_filter

Logger = logging.getLogger(__name__)

"""
功能: 创建基于 **pyapriltags**(AprilTag 3) 的 2D 抓取模板数据, 包含物体在图像上的位置和朝向信息, 以及机械臂末端位姿和夹爪距离等状态信息
模板需要保存以下数据:
- 1.末端刚好抓取到物体时的状态, 包含机械臂末端位姿和夹爪距离等信息
- 2.相机距离物体较近时的状态, 包含机械臂末端位姿、夹爪距离、以及物体在图像上的位置和朝向等信息
- 3.相机距离物体较近时的状态( 在 2 的基础上, 末端在 xy 平面平移一小段距离得到 ), 包含机械臂末端位姿、夹爪距离、以及物体在图像上的位置和朝向等信息
- 4.相机距离物体较远时的状态, 包含机械臂末端位姿、夹爪距离、以及物体在图像上的位置和朝向等信息
- 5.相机距离物体较远时的状态( 在 4 的基础上, 末端在 xy 平面平移一小段距离得到 ), 包含机械臂末端位姿、夹爪距离、以及物体在图像上的位置和朝向等信息

"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import rclpy

root_dir = os.path.normpath(f"{os.path.dirname(os.path.realpath(__file__))}/../../../")
sys.path.append(root_dir)

# 导入本工程的模块
from core.robot_helper.robot_helper import ArmHelper
from core.common_utils.utils import Common_data
from core.camera_ros_utils.cam_ros_utils import CamNode
from core.vision_utils.vision_utils import TagMatcher2D





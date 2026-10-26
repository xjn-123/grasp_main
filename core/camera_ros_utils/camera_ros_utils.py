'''相机ROS工具类和函数'''
import logging
import os
import sys
import time

import rclpy
import sensor_msgs.msg
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from .common_utils.utils import Common_data

logger = logging.getLogger(__name__)


class CamNode(Node):
    def __init__(self, ):
        pass

    def get_frames(self, 
                frames_num: int = 1,
                do_spin_once: bool = True,
                timeout: float = 5.0) -> list[list[float]]:
        pass
        


'''相机ROS工具类和函数'''
import os
import sys  
import logging
import time

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs.msg

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
        


'''机械臂控制封装类
以A3机械臂为参考，按需修改机械臂控制接口'''

import logging
import os
import sys
import time

import carm
import numpy as np
import transforms3d

root_dir= os.path.normpath(os.path.dirname(os.path.realpath(__file__)), "/../../")
sys.path.append(root_dir)
from core.common_utils import Common_data

logger = logging.getLogger(__name__)

class ArmHelpper:

    class ControlMode:
        IDLE = 0#空闲
        POSITION = 1#位置控制
        MIT=2#力矩
        TORQUE = 3#拖动
        PIT=4#力位混合模式
    def __init__(self, ip: str = "10.42.0.101", control_mode: int = carm.ControlMode.POSITION, speed_level: int = 50) -> None:
        self.init_speed_level = speed_level
        self.arm = carm.CarmCol(ip)
        self.init_joints = [0, 0, 0, 0, 0, 0]

    def array_to_matrix(pose: list) -> np.ndarray:
        '''将6维数组转换为4x4矩阵'''
        return np.array(pose).reshape(4, 4)

    def get_external_force(self) -> list:






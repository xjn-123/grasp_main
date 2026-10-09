
import logging
import os
import sys

import carm
import numpy as np
import transforms3d

# 导入本地工具类
core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{core_dir}/../../")
sys.path.append(root_dir)
from core.common_utils.utils import Common_data

logger = logging.getLogger(__name__)

class ArmHelper:
    '''机械臂封装类'''

    class ControlMode:
        """机械臂控制模式"""
        IDLE = 0
        """空闲模式"""

        POSITION = 1
        """位置控制模式"""

        MIT = 2
        """力矩-阻抗-轨迹模式"""

        TEACH = 3
        """拖动模式"""

        PF = 4
        """力位混合模式"""
    def __init__(self, ip: str = "10.42.0.101", control_mode: int = carm.ControlMode.POSITION, speed_level: int = 50) -> None:
        """
                构造函数    
                Args:
                    ip (str): 机械臂的 IP 地址
                    control_mode (int): 机械臂的控制模式, 0: 空闲模式, 1: 位置控制模式, 2: MIT 模式, 3: 拖动模式, 4: PF 模式
                    speed_level (int): 机械臂的速度等级, 范围为 1~100, 数值越大速度越快
        """
        self.arm = carm.Carm(ip)
        self.init_joints = [0, 0, 0, 0, 0, 0]
        self.init_speed_level = speed_level

        self.arm.connect()#连接机械臂
        self.arm.set_ready()#使能机械臂
        self.arm.set_control_mode(control_mode)#设置机械臂控制模式
        self.arm.set_speed_level(speed_level) #设置机械臂速度等级

        logger.info(f"{Common_data.GREEN}机械臂已连接，IP: {ip}, 控制模式: {control_mode}, 速度等级: {speed_level}{Common_data.RESET}")

    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, 'arm') and self.arm is not None:
                try:   
                    self.arm.set_speed_level(self.init_speed_level)  # 将机械臂速度等级设置为 0
                except Exception:
                    pass
                try:
                    self.arm.disconnect()
                except Exception:
                    pass
                logger.info(f'{Common_data.GREEN}机械臂断开连接{Common_data.RESET}')
        except Exception:  # noqa: BLE001, S110
            pass
            



    def array_to_matrix(pose: list[float]) -> np.ndarray:
        '''将数组转换成矩阵'''
        return np.array(pose).reshape(4, 4)

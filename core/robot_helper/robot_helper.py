
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

    def is_connected(self) -> bool:
        """检查机械臂是否连接"""
        return self.arm.is_connected()  

    def set_control_mode(self, control_mode: int) -> None:
        """设置机械臂控制模式"""
        logger.info(f"{Common_data.GREEN}设置机械臂控制模式为: {control_mode}{Common_data.RESET}")
        return self.arm.set_control_mode(control_mode)

    def set_speed_level(self, speed_level: int) -> None:
        """设置机械臂速度等级
        Args:
            speed_level (int): 机械臂的速度等级, 范围为 1~100, 数值越大速度越快
        """
        logger.info(f"{Common_data.GREEN}设置机械臂速度等级为: {speed_level}{Common_data.RESET}")
        return self.arm.set_speed_level(float(speed_level)/10.0,20)

    def get_joint(self) -> list[float]:
        """获取机械臂关节角度"""
        return [float(rad) for rad in self.arm.joint_positions()]
    
    def array_to_matrix(pose: list[float]) -> np.ndarray:
        '''将位姿列表转化为位姿矩阵
        
        将位姿列表转换为位姿矩阵    
                Args:
                    pose (List[float]): 位姿列表 [tx, ty, tz, qx, qy, qz, qw]
                Returns:
                    (np.ndarray): 位姿矩阵, 4*4 的齐次变换矩阵
        '''
        T=np.eye(4)
        T[0:3,3]=np.array(pose[0:3])
        q=pose[-1:-5]#将位姿列表中的四元数部分提取出来并转换为[qw, qx, qy, qz]的顺序
        T[0:3,0:3]=transforms3d.quaternions.qmult(q)

        return T

    def matrix_to_array(T: np.ndarray) -> list[float]:
        '''将位姿矩阵转化为位姿列表
        
        将位姿矩阵转换为位姿列表    
                Args:
                    T (np.ndarray): 位姿矩阵, 4*4 的齐次变换矩阵
                Returns:
                    (List[float]): 位姿列表 [tx, ty, tz, qx, qy, qz, qw]
        '''
        p=T[0:3,3].tolist
        q=transforms3d.quaternions.mat2quat(T[0:3,0:3])
        return [p[0],p[1],p[2],q[1],q[2],q[3],q[0]]

    def get_cart_matrix(self) -> np.ndarray:
        """获取机械臂末端位姿矩阵"""
        pose = self.arm.cart_pose
        T_base_end = self.array_to_matrix(pose)    
        return T_base_end

    def get_gripper_dist(self) -> float:
        """获取机械臂夹爪距离"""
        return self.arm.gripper_dist

    def get_external_force(self) -> list[float]:
        """
        获取机械臂末端的外部力信息    
        Returns:
            (List[float]): 机械臂末端的外部力列表 [fx, fy, fz, tx, ty, tz], 前三个元素为力的分量( 单位: N ), 后三个元素为力矩的分量( 单位: N*m )
        """
        return self.arm.external_force

    def set_joints(self,
                target_joints: list[float],
                desire_time: float = -1,
                move_line: bool = False) -> bool:
        """
        设置机械臂的关节角度     
        Args:
            target_joints (List[float]): 目标机械臂的关节角度列表,长度为 6
            desire_time (float): 期望的运动时间, 单位( 秒 ), 默认值为-1表示使用机械臂默认速度, 当 move_line=True 时, 该参数会被忽略
            move_line (bool): 是否进行直线运动, 默认值为 False
        Returns:
            (bool): 设置是否成功
        """

        assert len(target_joints) == 6, "Joint angles must be a list of 6 elements."

        if move_line:
            is_ok = self.arm.move_line_joint(target_joints)
        else:
            is_ok = self.arm.move_joint(target_joints, desire_time)

        if not is_ok:
            logger.error(f"{Common_data.RED}设置机械臂关节角度失败: {target_joints}{Common_data.RESET}")
            return False

        return True

    def set_pose(self,
                T_base_end: np.ndarray,
                desire_time: float = -1,
                move_line: bool = False,) -> bool:
        """ 
        设置机械臂的位姿        
        Args:
            T_base_end (np.ndarray): 目标机械臂的位姿矩阵, 4*4 的齐次变换矩阵
            move_line (bool): 是否进行直线运动, 默认值为 False
        Returns:
            (bool): 设置是否成功
        """

        target_pose = self.matrix_to_array(T_base_end)

        if move_line:
            is_ok = self.arm.move_line_pose(target_pose)
        else:
            is_ok = self.arm.move_pose(target_pose, desire_time)
        # end if

        if not is_ok:
            logger.error(f"{Common_data.RED}设置机械臂位姿失败: {target_pose}{Common_data.RESET}")
            return False
        # end if

        return True

    def track_pose(self, target_pose: np.ndarray, gripper_dist: float = -1) -> bool:
        """
        机械臂末端位姿跟踪
        Args:
            target_pose (np.ndarray): 目标机械臂的位姿矩阵, 4*4 的齐次变换矩阵
            gripper_dist (float): 目标机械臂夹爪距离, 单位( 米 ),范围为0-0.8m
        Returns:
            (bool): 设置是否成功
        """
        target_pose_array = self.matrix_to_array(target_pose)

        if gripper_dist >= 0:
            is_ok = self.arm.track_pose(target_pose_array, gripper_dist)
        else:
            is_ok = self.arm.track_pose(target_pose_array)

        if not is_ok:
            logger.error(f"{Common_data.RED}机械臂末端位姿跟踪失败: {target_pose_array}, gripper_dist: {gripper_dist}{Common_data.RESET}")
            return False

        return True

    def set_gripper_dist(self, 
                        gripper_dist: float,
                        grip_force: float = 25.0,
                        th_dist_err: float=-1.0) -> bool:
        """
        设置机械臂夹爪距离
        Args:
            gripper_dist (float): 目标机械臂夹爪距离, 单位( 米 ),范围为0-0.8m
            grip_force (float): 夹爪抓取力, 单位( N ), 默认值为25N,范围为0-100N
            th_dist_err (float): 夹爪距离误差阈值, 单位( 米 ), 默认值为-1.0表示不进行检查

        Returns:
            (bool): 设置是否成功
        """
        is_ok = self.arm.set_gripper_dist(gripper_dist,grip_force)

        if not is_ok:
            logger.error(f"{Common_data.RED}设置机械臂夹爪距离失败: {gripper_dist}{Common_data.RESET}")
            return False

        if th_dist_err > 0.0 and abs(self.arm.get_gripper_dist() - gripper_dist) > th_dist_err:
            logger.error(f"{Common_data.RED}机械臂夹爪距离设置误差大于阈值: 误差值: {abs(self.arm.get_gripper_dist() - gripper_dist)}{Common_data.RESET}")
            return False

        return True

    def inverse_kinematics(self, 
                        T_base_end_list: np.ndarray,
                        ref_joints_list: list[list[float]]) -> list[list[float]]:
        """
        逆运动学计算( 机械臂的末端位姿 --> 机械臂的各关节的角度 )    
        Args:
            T_base_end_list (List[np.ndarray]): 机械臂的位姿矩阵列表 N*4*4
            ref_joints_list (List[List[float]]): 机械臂的参考关节角度列表, N*6, 若为 None 则使用机械臂当前的关节角度
        Returns:
            (List[List[float]]): 计算得到的机械臂关节角度列表,长度为 N, 对于逆解失败的位姿, 列表中对应位置为 []
        """

        if ref_joints_list is None:
            ref_joints = self.arm.get_joint()
            ref_joints_list = [ref_joints] * len(T_base_end_list)
        else:
            assert len(T_base_end_list) == len(ref_joints_list), "T_base_end_list 和 ref_joints_list 长度不一致"

        pose_list = [self.matrix_to_array(T) for T in T_base_end_list]
        joints_list = self.arm.inverse_kinematics(pose_list, ref_joints_list)

        return joints_list

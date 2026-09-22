"""
自研机械臂 CARM 控制器封装类
"""

import logging

import transforms3d
import numpy as np

from typing_extensions import List

# 自研机械臂 CARM 的 Python API
import carm

# 导入本工程的模块
from .utils import GREEN, YELLOW, BLUE, RED, RESET


######################################################### 类定义 #########################################################


class ArmWrapper:
    """ 
    机械臂控制封装类( 后续适配其他机械臂时, 只需要修改该类的实现即可, 不需要修改其他代码 )
    """

    class ControlMode:
        """机械臂控制模式枚举类"""
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
    # end class ControlMode

    def __init__(self,
                 ip: str = "10.42.0.101",
                 control_mode: int = ControlMode.POSITION,
                 speed_level: int = 50) -> None:
        """
        构造函数    
        Args:
            ip (str): 机械臂的 IP 地址
            control_mode (int): 机械臂的控制模式, 0: 空闲模式, 1: 位置控制模式, 2: MIT 模式, 3: 拖动模式, 4: PF 模式
            speed_level (int): 机械臂的速度等级, 范围为 1~100, 数值越大速度越快
        """

        self.init_speed_level = speed_level
        """机械臂的初始速度等级"""

        # 如果是 arm64 平台, 则使用本地 IP 地址连接机械臂, 127.0.0.1
        import platform
        if platform.machine() == 'aarch64':
            ip = '127.0.0.1'
            logging.info(f'Running on arm64 platform, using local IP address: {GREEN}{ip}{RESET}')
        # end if

        # 初始化 CARM
        self.arm = carm.Carm(ip)

        self.arm.set_ready()  # 使能机械臂
        self.arm.set_control_mode(control_mode)  # 设置机械臂控制模式
        self.set_speed_level(speed_level)

        self.init_joints = [0, 0, 0, 0, 0, 0]
        """机械臂的初始关节角度"""

        logging.info(f'{GREEN}Arm initialized.{RESET}')
    # end def __init__

    def __del__(self):
        # Python 解释器退出阶段可能导致模块被卸载,析构里只做尽力清理并忽略异常
        try:
            if hasattr(self, 'arm') and self.arm is not None:
                try:
                    self.set_speed_level(self.init_speed_level)  # 恢复机械臂初始速度等级
                except Exception:
                    pass

                try:
                    self.arm.disconnect()  # 断开机械臂连接
                except Exception:
                    pass

                logging.info(f'{GREEN}Arm disconnected.{RESET}')
        except Exception:
            pass
    # end def __del__

    def is_connected(self) -> bool:
        """
        检查机械臂是否连接  
        Returns:
            (bool): 机械臂是否连接
        """

        return self.arm.is_connected()
    # end def is_connected

    def set_control_mode(self, control_mode: int) -> bool:
        """
        设置机械臂的控制模式    
        Args:
            control_mode (int): 机械臂的控制模式, 0: 空闲模式, 1: 位置控制模式, 2: MIT 模式, 3: 拖动模式, 4: PF 模式
        Returns:
            (bool): 是否设置成功
        """

        logging.info(f'Setting control mode to: {GREEN}[{control_mode}]{RESET}...')

        return self.arm.set_control_mode(control_mode)
    # end def set_control_mode

    def set_speed_level(self, speed_level: int) -> bool:
        """
        设置机械臂的速度等级    
        Args:
            speed_level (int): 速度等级, 范围为 1~100, 数值越大速度越快
        Returns:
            (bool): 是否设置成功
        """

        logging.info(f'Setting speed level to: {GREEN}[{speed_level}]{RESET}...')

        return self.arm.set_speed_level(float(speed_level) / 10.0, 20)
    # end def set_speed_level

    def get_joints(self) -> List[float]:
        """
        获取机械臂的关节角度
        Returns:
            (List[float]): 机械臂的关节角度列表,长度为6
        """

        joints = [float(v) for v in self.arm.joint_pos]  # 返回值为 List[float], 长度为6

        return joints
    # end def get_joints

    def get_pose(self) -> np.ndarray:
        """
        获取机械臂的位姿    
        Returns:
            (np.ndarray): 从机械臂末端到机械臂基座的欧式变换矩阵 T_base_end, 4*4 
        """

        pose = self.arm.cart_pose  # 返回值为 List[float], 长度为7, [tx, ty, tz, qx, qy, qz, qw]
        T_base_end = self.array_to_matrix(pose)

        return T_base_end
    # end def get_pose

    def get_gripper_dist(self) -> float:
        """
        获取机械爪的夹持距离
        Returns:
            (float): 机械爪的夹持距离( 单位: 米 )
        """

        dist = self.arm.gripper_pos

        return dist
    # end def get_gripper_dist

    def get_external_force(self) -> List[float]:
        """
        获取机械臂末端的外部力信息    
        Returns:
            (List[float]): 机械臂末端的外部力列表 [fx, fy, fz, tx, ty, tz], 前三个元素为力的分量( 单位: N ), 后三个元素为力矩的分量( 单位: N*m )
        """

        force_info = self.arm.cart_external_force  # 返回值为 List[float], [fx, fy, fz, tx, ty, tz]

        return force_info
    # end def get_external_force

    def set_joints(self,
                   target_joints: List[float],
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
        # end if

        if not is_ok:
            logging.error(f'Failed to reach target joint angles: {target_joints}')
            return False

        return True
    # end def set_joints

    def set_pose(self,
                 T_base_end: np.ndarray,
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
            is_ok = self.arm.move_pose(target_pose)
        # end if

        if not is_ok:
            logging.error(f'Failed to send pose to arm sdk: {target_pose}')
            return False
        # end if

        return True
    # end def set_pose

    def track_pose(self,
                   T_base_end: np.ndarray,
                   gripper_dist: float = -1) -> bool:
        """
        发布位姿和夹爪距离, 作为机械臂的目标状态, 用于实时跟踪控制        
        Args:
            T_base_end (np.ndarray): 目标机械臂的位姿矩阵, 4*4 的齐次变换矩阵
            gripper_dist (float): 目标机械爪的夹持距离, 范围为 0.0~0.08 米
        Returns:
            (bool): 指令发送是否成功 
        """

        target_pose = self.matrix_to_array(T_base_end)

        is_ok = self.arm.track_pose(target_pose, gripper_dist)
        if not is_ok:
            logging.error(f'Failed to send track pose command to arm sdk: {target_pose}, gripper_dist: {gripper_dist}')
            return False
        # end if

        return True
    # end def track_pose

    def set_gripper_dist(self,
                         target_dist: float,
                         grip_force: float = 25,
                         th_dist_err: float = -1.0) -> bool:
        """
        设置机械爪的夹持距离
        Args:
            target_dist (float): 两个爪指之间的目标夹持距离, 范围为 0.0~0.08 米
            grip_force (float): 机械爪的夹持力, 范围为 0.0~100.0 N
            th_dist_err (float): 夹持距离误差阈值( 单位: 米 ), 默认值为 -1.0, 如果该值小于等于0, 则不进行结果检查, 直接返回成功
        Returns:
            (bool): 是否设置成功
        """

        is_ok = self.arm.set_gripper(target_dist, grip_force)

        if not is_ok:
            logging.error(f'Failed to send set gripper command to arm sdk: target_dist: {target_dist}, grip_force: {grip_force}')
            return False
        # end if

        current_dist = self.get_gripper_dist()
        dist_err = abs(current_dist - target_dist)
        if th_dist_err > 0 and dist_err > th_dist_err:
            logging.error(f'Gripper distance( m ) error is too large: {dist_err:.4f}, target: {target_dist:.4f}, current: {current_dist:.4f}')
            return False
        # end if

        return True
    # end def set_gripper_dist

    def inverse_kinematics(self,
                           T_base_end_list: List[np.ndarray],
                           ref_joints_list: List[List[float]] = None) -> List[List[float]]:
        """
        逆运动学计算( 机械臂的末端位姿 --> 机械臂的各关节的角度 )    
        Args:
            T_base_end_list (List[np.ndarray]): 机械臂的位姿矩阵列表 N*4*4
            ref_joints_list (List[List[float]]): 机械臂的参考关节角度列表, N*6, 若为 None 则使用机械臂当前的关节角度
        Returns:
            (List[List[float]]): 计算得到的机械臂关节角度列表,长度为 N, 对于逆解失败的位姿, 列表中对应位置为 []
        """

        if ref_joints_list is None:
            ref_joints = self.get_joints()
            ref_joints_list = [ref_joints] * len(T_base_end_list)
        else:
            assert len(ref_joints_list) == len(T_base_end_list), "length of ref_joints_list must match length of T_base_end_list."
        # end if

        pose_list = []
        for T in T_base_end_list:
            pose = self.matrix_to_array(T)
            pose_list.append(pose)
        # end for

        joints_list = self.arm.inverse_kine(pose_list, ref_joints_list)

        return joints_list
    # end def inverse_kinematics

    @staticmethod
    def array_to_matrix(pose: List[float]) -> np.ndarray:
        """
        将位姿列表转换为位姿矩阵    
        Args:
            pose (List[float]): 位姿列表 [tx, ty, tz, qx, qy, qz, qw]
        Returns:
            (np.ndarray): 位姿矩阵, 4*4 的齐次变换矩阵
        """

        T = np.eye(4)
        T[:3, 3] = np.array(pose[:3])
        q = [pose[6], pose[3], pose[4], pose[5]]  # 转换为 [qw, qx, qy, qz]
        T[:3, :3] = transforms3d.quaternions.quat2mat(q)

        return T
    # end def array_to_matrix

    @staticmethod
    def matrix_to_array(T: np.ndarray) -> List[float]:
        """
        将位姿矩阵转换为位姿列表    
        Args:
            T (np.ndarray): 位姿矩阵, 4*4 的齐次变换矩阵
        Returns:
            (List[float]): 位姿列表 [tx, ty, tz, qx, qy, qz, qw]
        """

        p = T[:3, 3].tolist()
        q = transforms3d.quaternions.mat2quat(T[:3, :3]).tolist()  # 将旋转矩阵转换为四元数 [qw, qx, qy, qz]
        pose = [p[0], p[1], p[2], q[1], q[2], q[3], q[0]]  # 四元数和位置 tx, ty, tz, qx, qy, qz, qw

        return pose
    # end def matrix_to_array
# end class ArmWrapper

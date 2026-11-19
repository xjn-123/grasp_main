'''机械臂示例抓取ros节点'''

import argparse
import json
import logging
import os
import sys
import time

import mmengine
import numpy as np
import rclpy
import transforms3d
from rclpy.node import Node

root_dir = os.path.normpath(os.path.dirname(os.path.realpath(__file__)), "/../../")
sys.path.append(root_dir)

from core.common_utils.utils import (
    Common_data,
    two_finger_gripper,read_camera_params,
    wait_key,reset_empty_str,inv_tf)


from core.robot_helper.robot_helper import ArmHelper

from core.gripper_utils.gripper_utils import (
    TH_ANGLE_Z,
    TH_GRIPPER_HEIGHT,
    CollisionDetector,
    Gripper,
    check_gripper_pose,
    com_axis_aligned_pose,
)

from core.robot_ros_utils.robot_ros_utils import (
    ArmNode,
)

from core.camera_ros_utils.camera_ros_utils import (
    CameraNode,
)


from core.common_utils.grasp_utils import (
    Grasp_computer,
)

Logger = logging.getLogger(__name__)


######################################################### 全局常量( 仅本文件使用 ) #########################################################

CHECK_HEIGHT = 0.18
"""检查位姿时物体到相机的垂直距离, 单位: 米"""

COLLISION_DIR = os.path.normpath(f"{root_dir}/results/app/collision")
"""保存碰撞检测结果的目录"""

class Matcher:

    '''匹配器占位类'''
###########################函数定义########################

def compute_check_state(T_end_cam: np.ndarray,
                        arm: ArmHelper,
                        cur_joints: list[float],
                        cur_T_base_end: np.ndarray,
                        cur_T_cam_model: np.ndarray,
                        height: float,) -> tuple[np.ndarray, list[float]]:
    """
        计算当前位姿是否满足检查位姿要求,
        Args:
            T_end_cam (np.ndarray): 从相机到机械臂末端的变换矩阵, 4*4
            cur_T_base_end (np.ndarray): 当前从机械臂末端到基座的变换矩阵, 4*4
            cur_T_cam_model (np.ndarray): 当前从物体到相机的变换矩阵, 4*4
            height (float): 物体到相机光心的垂直距离, 单位: 米
        Returns:
            (Tuple[np.ndarray, List[float]]): 计算得到的机械臂位姿 T_base_end , 机械臂关节角度列表
    """

    cur_T_base_cam = cur_T_base_end @ T_end_cam
    cur_T_base_model = cur_T_base_cam @ cur_T_cam_model

    target_dir = np.array([0.0, 0.0, -1.0])
    cur_dir = cur_T_base_cam[:3, 2]
    cosine = np.dot(target_dir, cur_dir)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.arccos(cosine)
    Logger.info(f'cur_dir: {Common_data.GREEN}{cur_dir}{Common_data.RESET}, '
                f'target_dir: {Common_data.GREEN}{target_dir}{Common_data.RESET}, '
                 f'angle: {Common_data.GREEN}{angle * 180.0 / np.pi:.2f}{Common_data.RESET} deg')

    #计算调整后的几台
    axis=np.cross(cur_dir, target_dir)
    delta_R = transforms3d.axangles.axangle2mat(axis, angle)

    target_T_base_cam = np.eye(4)
    target_T_base_cam[:3, :3] = delta_R @ cur_T_base_cam[:3, :3]
    target_T_base_cam[:3, 3] = cur_T_base_model[:3, 3] + np.array([0.0, 0.0, height])

    #相机光轴竖直向下时，可以同时存在多个绕相机光轴旋转的位姿，末端位姿需要进行过滤
    
    T_base_end_list = []
    angle_step=15.0*np.pi/180.0
    for angle_yaw in np.arange(0.0, np.pi, angle_step):
        R_z=transforms3d.axangles.axangle2mat(np.array([0.0, 0.0, 1.0]), angle_yaw)
        T_base_cam=np.eye(4,dtype=np.float32)
        T_base_cam[:3,:3]=R_z @ target_T_base_cam[:3,:3]
        T_base_cam[:3,3]=target_T_base_cam[:3,3]

        T_base_end = T_base_cam @ inv_tf(T_end_cam)
        T_base_end_list.append(T_base_end)

    #计算机械臂逆解
    st=time.time()
    joints_list = arm.inverse_kinematics(T_base_end_list,[cur_joints]*len(T_base_end_list))

    if joints_list is None:
        Logger.error(f'inverse kinematics failed!')
        return None, None

    #计算当前最接近的关节解（第一个关节不参与计算）
    min_delta_joints = 1000.0
    best_idx = -1
    for idx, joints in enumerate(joints_list):
        if joints is None: continue

        if len(joints)!=len(cur_joints): continue

        delta=0.0

        for i in range(1,len(joints)):
            delta+=(joints[i]-cur_joints[i])

        if delta<min_delta_joints:
            min_delta_joints=delta
            best_idx=idx

    if best_idx==-1:
        Logger.error(f'no valid joints found!')
        return None, None

    best_joints=joints_list[best_idx]
    best_T_base_end=T_base_end_list[best_idx]

    Logger.info(f'compute_check_state, time: {time.time()-st:.2f}s')

    return best_T_base_end, best_joints


def do_grasp(T_end_cam: np.ndarray,
            sym_tfs: np.ndarray,
            tmpl_state_list: list[dict],
            gripper_body: Gripper,
            collision_detector: CollisionDetector,
            arm: ArmHelper,
            cam_node: CameraNode,
            arm_node: ArmNode,
            client_node: Node,
            use_cache: bool,
            cache_T_base_end: np.ndarray,
            cache_joints: list[float],
            debug_level: int,
            debug: bool = False) -> int: 
    """
        执行抓取动作
        Args:
            T_end_cam (np.ndarray): 从相机到机械臂末端的变换矩阵, 4*4
            sym_tfs (np.ndarray): 模板变换矩阵列表, 4*4*N
            tmpl_state_list (list[dict]): 模板状态列表
            gripper_body (Gripper): 抓取体
            collision_detector (CollisionDetector): 碰撞检测器
            arm (ArmHelper):机械臂
            cam_node (CameraNode): 相机节点
            arm_node (ArmNode):机械臂节点
            client_node (Node): 客户端节点(位姿匹配器)
            use_cache (bool): 是否使用缓存位姿
            cache_T_base_end (np.ndarray): 缓存位姿 T_base_end
            cache_joints (list[float]): 缓存关节角度列表
            debug_level (int): 调试等级
            debug (bool): 是否调试
            Returns:
                (int): 抓取结果, -1: 退出程序, 0: 任务失败,复用缓存信息作下一次任务; 1: 任务成功, 不需要复用缓存信息
        
    """

    










# -*- coding: utf-8 -*-
"""
功能说明: 基于自研机械臂 CARM 的 3D 抓取( 6 个自由度 )示例 ROS2 节点    
"""

import rclpy

import logging
import argparse
import os
import sys
import time
import json
import mmengine
from typing_extensions import List, Tuple, Dict

import numpy as np
import transforms3d


# 导入本工程的模块

code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f'{code_dir}/../../../')
sys.path.append(root_dir)

from core.utils import (
    GREEN, YELLOW, BLUE, RED, RESET,
    wait_key, reset_empty_str,
    read_rgbd_params, read_handeye_calib, inv_tf
)

from core.arm_wrapper import ArmWrapper

from core.arm_utils import (
    TH_ANGLE_Z,
    TH_GRIPPER_HEIGHT,
    GripperBody,
    CollisionDetector,
    check_arm_pose,
)

from core.arm_ros_utils import TargetArmNode

from core.cam_ros_utils import (
    CamNode,
)

from core.vision_utils import (
    compute_locate_error
)

from examples.app.src.client_matching3d import (
    ClientNode
)  # 同目录下的模块


######################################################### 全局常量( 仅本文件使用 ) #########################################################

CHECK_HEIGHT = 0.18
"""检查位姿时物体到相机的垂直距离, 单位: 米"""

COLLISION_DIR = os.path.normpath(f"{root_dir}/results/app/collision")
"""保存碰撞检测结果的目录"""


######################################################### 函数定义 #########################################################


def read_tmpl_grasp(tmpl_dir: str) -> Dict:
    """
    读取抓取模板数据
    Args:
        tmpl_dir (str): 抓取模板文件夹路径
    Returns:
        (Dict): 抓取模板数据字典
    """

    # 1. 读取相机处于检测位置时的机械臂状态
    detect_path = os.path.join(tmpl_dir, 'detect.json')
    if not os.path.exists(detect_path):
        logging.error(f'file not found: {detect_path}')
        return None
    # end if

    with open(detect_path, 'r') as f:
        detect_dict = json.load(f)
    # end with
    detect_T_base_end = np.array(detect_dict['T_base_end'], dtype=np.float32)
    detect_joints = detect_dict["joints"]
    detect_gripper_dist = detect_dict["gripper_dist"]
    logging.info(f'arm detect_joints: {detect_joints}')
    logging.info(f'arm detect_gripper_dist: {detect_gripper_dist}')

    # 2. 读取相机处于放置位置时的机械臂状态
    place_path = os.path.join(tmpl_dir, 'place.json')
    if not os.path.exists(place_path):
        logging.error(f'file not found: {place_path}')
        return None
    # end if

    with open(place_path, 'r') as f:
        place_dict = json.load(f)
    # end with
    place_T_base_end = np.array(place_dict['T_base_end'], dtype=np.float32)
    place_joints = place_dict["joints"]
    place_gripper_dist = place_dict["gripper_dist"]
    logging.info(f'arm place_joints: {place_joints}')
    logging.info(f'arm place_gripper_dist: {place_gripper_dist}')

    # 3. 读取抓取模板位姿
    tmpl_state_list = []
    tmpl_cnt = 0
    while True:
        tmpl_dir_i = os.path.join(tmpl_dir, f'{tmpl_cnt}')
        if not os.path.exists(tmpl_dir_i):
            break
        # end if
        grasp_path = os.path.join(tmpl_dir_i, 'grasp.json')
        ready_path = os.path.join(tmpl_dir_i, 'ready.json')
        if not os.path.exists(grasp_path) or not os.path.exists(ready_path):
            logging.warning(f'file not found: {grasp_path} or {ready_path}')
            break
        # end if

        state_dict = {}

        with open(grasp_path, 'r') as f:
            grasp_data = json.load(f)
        # end with
        state_dict['grasp_T_base_end'] = np.array(grasp_data['T_base_end'], dtype=np.float32)
        state_dict['grasp_gripper_dist'] = grasp_data['gripper_dist']

        with open(ready_path, 'r') as f:
            ready_data = json.load(f)
        # end with
        state_dict['ready_T_base_end'] = np.array(ready_data['T_base_end'], dtype=np.float32)
        state_dict['ready_T_cam_model'] = np.array(ready_data['T_cam_model'], dtype=np.float32)
        state_dict['ready_gripper_dist'] = ready_data['gripper_dist']

        tmpl_state_list.append(state_dict)
        tmpl_cnt += 1
    # end while

    if len(tmpl_state_list) == 0:
        logging.error(f'no valid grasp tmpl found in dir: {tmpl_dir}')
        return None
    # end if

    tmpl_dict = {
        'detect_T_base_end': detect_T_base_end,
        'detect_joints': detect_joints,
        'detect_gripper_dist': detect_gripper_dist,

        'place_T_base_end': place_T_base_end,
        'place_joints': place_joints,
        'place_gripper_dist': place_gripper_dist,

        'tmpl_state_list': tmpl_state_list
    }

    logging.info(f'grasp tmpl num: {len(tmpl_state_list)}')
    print()

    return tmpl_dict
# end def read_tmpl_grasp


def compute_tmpl_ready_pose(sym_tfs: np.ndarray,
                            T_end_cam: np.ndarray,
                            ready_T_base_end: np.ndarray,
                            ready_T_cam_model: np.ndarray,
                            cur_T_base_end: np.ndarray,
                            cur_T_cam_model: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    对于特定的抓取模板,计算机械臂位姿,使得机械臂从当前位姿移动到设定位姿时, cur_T_cam_model = ready_T_cam_model    
    Args:
        sym_tfs (np.ndarray): 对称变换矩阵数组, N*4*4
        T_end_cam (np.ndarray): 从相机到机械臂末端的变换矩阵, 4*4
        ready_T_base_end (np.ndarray): 准备阶段从机械臂末端到基座的变换矩阵, 4*4
        ready_T_cam_model (np.ndarray): 准备阶段从物体到相机的变换矩阵, 4*4
        cur_T_base_end (np.ndarray): 当前从机械臂末端到基座的变换矩阵, 4*4
        cur_T_cam_model (np.ndarray): 当前从物体到相机的变换矩阵, 4*4
    Returns:
        (Tuple[np.ndarray, float]): 计算得到的机械臂位姿 T_base_end , 机械臂 ready 和 target 位姿之间的旋转角度差
    """

    N = sym_tfs.shape[0]  # 对称变换数量

    ready_T_model_cam = inv_tf(ready_T_cam_model)
    ready_T_end_base = inv_tf(ready_T_base_end)
    T_cam_end = inv_tf(T_end_cam)

    target_poses = np.zeros((N, 4, 4), dtype=np.float32)  # N*4*4 T_base_end
    delta_angles = [np.pi * 2] * N  # N 旋转矩阵的轴角表示的旋转部分

    for i in range(N):
        sym_tf = sym_tfs[i]
        target_T_base_end = cur_T_base_end @ T_end_cam @ cur_T_cam_model @ sym_tf @ ready_T_model_cam @ T_cam_end
        target_poses[i] = target_T_base_end

        delta_T_base_end = ready_T_end_base @ target_T_base_end

        # 由旋转矩阵计算轴角表示的旋转部分
        delta_R = delta_T_base_end[:3, :3]
        cosine = (np.trace(delta_R) - 1) / 2
        cosine = np.clip(cosine, -1.0, 1.0)  # 数值稳定性处理
        angle = abs(np.arccos(cosine))

        # 计算 ready_T_base_end 和 target_T_base_end 之间的 Z 轴夹角
        ready_z_dir = ready_T_base_end[:3, 2]  # 机械臂末端 Z 轴方向
        target_z_dir = target_T_base_end[:3, 2]  # 机械臂末端 Z 轴方向
        cosine = np.dot(ready_z_dir, target_z_dir) / (np.linalg.norm(ready_z_dir) * np.linalg.norm(target_z_dir))
        cosine = np.clip(cosine, -1.0, 1.0)  # 数值稳定性处理
        angle_z = np.arccos(cosine)

        # logging.info(f'sym tf idx: {i}, between ready and set, Z axis angle(deg): {angle_z * 180.0 / np.pi: .2f}, '
        #              f'rotation angle(deg): {angle * 180.0 / np.pi: .2f}')

        if angle_z > TH_ANGLE_Z:
            angle += np.pi * 2  # 增加一个惩罚值
        # end if

        delta_angles[i] = angle
    # end for

    min_idx = int(np.argmin(delta_angles))
    target_T_base_end = target_poses[min_idx]
    delta_angle = delta_angles[min_idx]

    return target_T_base_end, delta_angle
# end def compute_tmpl_ready_pose


def compute_check_state(T_end_cam: np.ndarray,
                        arm: ArmWrapper,
                        cur_joints: List[float],
                        cur_T_base_end: np.ndarray,
                        cur_T_cam_model: np.ndarray,
                        height: float,) -> Tuple[np.ndarray, List[float]]:
    """
    计算检查位姿对应的机械臂状态,使得相机光心竖直向下看向物体,并且物体位于相机视野中心正下方的 height 米处  
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

    target_z_dir = np.array([0, 0, -1])   # 期望的 Z 轴方向
    cur_z_dir = cur_T_base_cam[:3, 2]  # 当前的 Z 轴方向
    cosine = np.dot(target_z_dir, cur_z_dir)
    cosine = np.clip(cosine, -1.0, 1.0)  # 数值稳定性处理
    angle = np.arccos(cosine)
    logging.warning(f'current cam Z dir: {cur_z_dir}, need adjust to {target_z_dir}, angle(deg): {angle*np.rad2deg(1)}')

    # 计算调整后的姿态
    axis = np.cross(cur_z_dir, target_z_dir)
    delta_R = transforms3d.axangles.axangle2mat(axis, angle)

    target_T_base_cam = np.eye(4, dtype=np.float32)
    target_T_base_cam[:3, :3] = delta_R @ cur_T_base_cam[:3, :3]
    target_T_base_cam[:3, 3] = cur_T_base_model[:3, 3] + np.array([0, 0, height])

    # 计算多个机械臂位姿
    T_base_end_list = []
    angle_step = 15.0 * np.pi / 180.0  # 步长
    T_cam_end = inv_tf(T_end_cam)
    for angle_yaw in np.arange(0, 2 * np.pi, angle_step):
        R_z = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]), angle_yaw)
        T_base_cam = np.eye(4, dtype=np.float32)
        T_base_cam[:3, :3] = R_z @ target_T_base_cam[:3, :3]
        T_base_cam[:3, 3] = target_T_base_cam[:3, 3]

        T_base_end = T_base_cam @ T_cam_end
        T_base_end_list.append(T_base_end)
    # end for

    # 计算机械臂逆解
    st = time.time()
    joints_list = arm.inverse_kinematics(T_base_end_list, [cur_joints] * len(T_base_end_list))
    logging.info(f'compute inverse kinematics cost time( ms ): {(time.time() - st)*1000:.2f}')

    if joints_list is None:
        return None, None
    # end if

    # print(joints_list)

    # 计算最接近当前关节角( 第一个关节除外 )的解
    min_delta = 1000.0
    best_idx = -1
    for idx, joints in enumerate(joints_list):
        if joints is None:
            continue
        # end if

        if len(joints) != len(cur_joints):
            continue
        # end if

        delta = 0.0
        for i in range(1, len(joints)):
            delta += abs(joints[i] - cur_joints[i])
        # end for
        if delta < min_delta:
            min_delta = delta
            best_idx = idx
        # end if
    # end for

    if best_idx == -1:
        return None, None
    # end if

    best_T_base_end = T_base_end_list[best_idx]
    best_joints = joints_list[best_idx]

    logging.info(f'compute check state, best joints idx: {best_idx}, delta joints angle sum( deg ): {min_delta*180.0/np.pi:.2f}')

    return best_T_base_end, best_joints
# end def compute_check_state


def compute_ready_pose(tmpl_state_list: List[Dict],
                       sym_tfs: np.ndarray,
                       T_end_cam: np.ndarray,
                       cur_T_base_end: np.ndarray,
                       cur_T_cam_model: np.ndarray,
                       tmpl_idx: int = -1) -> Tuple[np.ndarray, int]:
    """
    选择并计算最佳抓取模板的预备位姿    
    Args:
        tmpl_state_list (List[Dict]): 抓取模板状态列表
        sym_tfs (np.ndarray): 对称变换矩阵数组, N*4*4
        T_end_cam (np.ndarray): 从相机到机械臂末端的变换矩阵, 4*4
        cur_T_base_end (np.ndarray): 当前从机械臂末端到基座的变换矩阵, 4*4
        cur_T_cam_model (np.ndarray): 当前从物体到相机的变换矩阵, 4*4
        tmpl_idx (int): 指定抓取模板索引, 默认为 -1, 表示不指定,需要计算最佳模板
    Returns:
        (Tuple[np.ndarray, int]): 计算得到的机械臂位姿 T_base_end , 抓取模板索引
    """

    best_target_T_base_end = None
    best_tmpl_idx = -1

    # 计算最佳模板
    if tmpl_idx >= 0:  # 已经指定了模板索引
        grasp_tmpl = tmpl_state_list[tmpl_idx]

        ready_T_base_end = grasp_tmpl['ready_T_base_end']
        ready_T_cam_model = grasp_tmpl['ready_T_cam_model']
        target_T_base_end, _ = compute_tmpl_ready_pose(sym_tfs,
                                                       T_end_cam,
                                                       ready_T_base_end,
                                                       ready_T_cam_model,
                                                       cur_T_base_end,
                                                       cur_T_cam_model)
        best_target_T_base_end = target_T_base_end
        best_tmpl_idx = tmpl_idx
    else:  # 未指定模板索引, 计算最佳模板
        tmpl_num = len(tmpl_state_list)
        min_delta_angle = 1000.0
        for i in range(tmpl_num):
            grasp_tmpl = tmpl_state_list[i]

            ready_T_base_end = grasp_tmpl['ready_T_base_end']
            ready_T_cam_model = grasp_tmpl['ready_T_cam_model']
            target_T_base_end, delta_angle = compute_tmpl_ready_pose(sym_tfs,
                                                                     T_end_cam,
                                                                     ready_T_base_end,
                                                                     ready_T_cam_model,
                                                                     cur_T_base_end,
                                                                     cur_T_cam_model)
            if delta_angle < min_delta_angle:
                min_delta_angle = delta_angle
                best_tmpl_idx = i
                best_target_T_base_end = target_T_base_end
            # end if
        # end for
    # end if

    return best_target_T_base_end, best_tmpl_idx
# end def compute_ready_pose


def do_grasp(T_end_cam: np.ndarray,
             sym_tfs: np.ndarray,
             tmpl_state_list: List[Dict],
             gripper_body: GripperBody,
             collision_detector: CollisionDetector,
             arm: ArmWrapper,
             cam_node: CamNode,
             arm_node: TargetArmNode,
             client_node: ClientNode,
             use_cache: bool,
             cache_T_base_end: np.ndarray,
             cache_joints: List[float],
             debug_level: int,
             debug: bool = False) -> int:
    """
    执行一次 3D 抓取任务
    Args:
        T_end_cam (np.ndarray): 从相机到机械臂末端的变换矩阵, 4*4
        sym_tfs (np.ndarray): 对称变换矩阵数组, N*4*4
        tmpl_state_list (List[Dict]): 抓取模板状态列表
        gripper_body (GripperBody): 夹爪几何体
        collision_detector (CollisionDetector): 碰撞检测器
        arm (ArmWrapper): 机械臂
        cam_node (CamNode): 相机节点
        arm_node (TargetArmNode): 机械臂目标节点
        client_node (ClientNode): 客户端节点
        use_cache (bool): 是否使用缓存的信息来定位物体
        cache_T_base_end (np.ndarray): 缓存的机械臂末端到基座的变换矩阵, 4*4
        cache_joints (List[float]): 缓存的关节角度列表
        debug_level (int): 调试级别
        debug (bool): 是否启用调试模式
    Returns:
        (int): 执行结果, -1: 退出程序; 0: 任务失败,复用缓存信息作下一次任务; 1: 任务成功, 不需要复用缓存信息
    """

    T_cam_end = inv_tf(T_end_cam)

    max_refine_cnt = 2  # 最大细化次数

    show_locate_err = False  # 是否显示定位误差

    # 临时变量
    prev_T_cam_model = None
    prev_T_base_end = None
    target_T_base_end = None
    tmpl_idx = -1

    ######## 1. 定位物体( 通过匹配 ) ########
    print()
    logging.info(f'grasp-step [1] , {BLUE}locate model by matching{RESET}')
    if not wait_key(debug):
        return -1
    # end if

    # 匹配物体
    cur_T_cam_model, _ = client_node.locate_model(timeout_sec=20.0,
                                                  use_cache=use_cache,
                                                  debug_level=debug_level)
    if cur_T_cam_model is None:
        logging.error(f'{RED}match model failed, exit run.{RESET}')
        return -1  # 已经是使用缓存了,但仍然失败,说明没有物体,可以退出程序
    # end if

    logging.info(f'match T_cam_model: \n{GREEN}{cur_T_cam_model}{RESET}')

    ######## 2. 计算检查位姿( 使物体处于相机视野中心正前方 ) ########
    print()
    logging.info(f'grasp-step [2] , {BLUE}compute check pose{RESET}')
    # if not wait_key(debug):
    #     return -1
    # # end if

    # 计算检查位姿
    check_height = CHECK_HEIGHT
    cur_T_base_end = cache_T_base_end  # 这里必须使用处于检测位置时候的末端位姿,而不是当前位姿,因为当复用缓存来定位物体时当前位姿可能不是检测位置
    cur_joints = cache_joints
    target_T_base_end, target_joints = compute_check_state(T_end_cam=T_end_cam,
                                                           arm=arm,
                                                           cur_joints=cur_joints,
                                                           cur_T_base_end=cur_T_base_end,
                                                           cur_T_cam_model=cur_T_cam_model,
                                                           height=check_height)
    if target_T_base_end is None:
        logging.warning(f"{RED}compute check state failed.{RESET} try check next label.")
        return 0
    # end if

    logging.info(f'check T_base_end: \n{GREEN}{target_T_base_end}{RESET}')
    logging.info(f'check joints: {target_joints}')

    # 发布将要到达的位姿( 用于可视化 )
    arm_node.publish_pose(target_T_base_end)

    # 检查机械臂位姿合理性
    if not check_arm_pose(T_base_end=target_T_base_end,
                          T_end_cam=T_end_cam,
                          gripper_body=gripper_body,
                          gripper_dist=arm.get_gripper_dist(),
                          th_angle_z=TH_ANGLE_Z,
                          th_gripper_height=TH_GRIPPER_HEIGHT):
        logging.warning(f"{RED}arm pose check failed at ready pose.{RESET} try check next label.")
        return 0
    # end if

    ######## 3. 移动机械臂到检查位姿 ########
    print()
    logging.info(f'grasp-step [3] , {BLUE}move arm to check pose{RESET}')
    if not wait_key(debug):
        return -1
    # end if

    # 移动之前更新临时变量
    prev_T_cam_model = cur_T_cam_model
    prev_T_base_end = cur_T_base_end

    # 移动到检查位姿
    is_ok = arm.set_joints(target_joints)
    if not is_ok:
        logging.error(f"{RED}move arm to check joints failed. try next label.{RESET}")
        return 0
    # end if

    ######## 4. 重新定位物体( 通过跟踪 ) ########
    print()
    logging.info(f'grasp-step [4] , {BLUE}relocate model by tracking{RESET}')
    if not wait_key(debug):
        return -1
    # end if

    cur_T_base_end = arm.get_pose()
    cur_T_end_base = inv_tf(cur_T_base_end)
    init_T_cam_model = T_cam_end @ cur_T_end_base @ prev_T_base_end @ T_end_cam @ prev_T_cam_model
    cur_T_cam_model, _ = client_node.locate_model(init_T_cam_model=init_T_cam_model,
                                                  timeout_sec=20.0,
                                                  debug_level=debug_level)

    if cur_T_cam_model is None:
        logging.warning(f"{RED}track model failed.{RESET}")
        return -1  # 相机出问题了,退出任务
    # end if
    logging.info(f'track T_cam_model: \n{GREEN}{cur_T_cam_model}{RESET}')

    ######## 5. 计算预备位姿,选择最佳模板 ########
    print()
    logging.info(f'grasp-step [5] , {BLUE}compute ready pose and select best grasp template{RESET}')

    target_T_base_end, tmpl_idx = compute_ready_pose(tmpl_state_list=tmpl_state_list,
                                                     sym_tfs=sym_tfs,
                                                     T_end_cam=T_end_cam,
                                                     cur_T_base_end=cur_T_base_end,
                                                     cur_T_cam_model=cur_T_cam_model)
    logging.info(f'selected tmpl idx: {GREEN}{tmpl_idx}{RESET}')
    logging.info(f'ready T_base_end: \n{GREEN}{target_T_base_end}{RESET}')

    # 根据模板索引获取,相关的状态
    ready_T_cam_model = tmpl_state_list[tmpl_idx]['ready_T_cam_model']
    ready_T_base_end = tmpl_state_list[tmpl_idx]['ready_T_base_end']
    grasp_T_base_end = tmpl_state_list[tmpl_idx]['grasp_T_base_end']
    grasp_gripper_dist = tmpl_state_list[tmpl_idx]['grasp_gripper_dist']

    # 计算从 ready 位姿到 grasp 位姿的增量
    delta_T_end = inv_tf(ready_T_base_end) @ grasp_T_base_end  # 末端坐标系下的位姿增量
    logging.info(f'delta_T_end: \n{GREEN}{delta_T_end}{RESET}')

    target_gripper_dist = grasp_gripper_dist + 0.015  # 预备位姿时先稍微放开一点夹爪,方便检查和移动

    # 发布将要到达的位姿( 用于可视化 )
    arm_node.publish_pose(target_T_base_end)

    # 检查机械臂位姿合理性
    if not check_arm_pose(T_base_end=target_T_base_end,
                          T_end_cam=T_end_cam,
                          gripper_body=gripper_body,
                          gripper_dist=target_gripper_dist,
                          th_angle_z=TH_ANGLE_Z,
                          th_gripper_height=TH_GRIPPER_HEIGHT):
        logging.warning(f"{RED}arm pose check failed at ready pose.{RESET} try next label.")
        return 0
    # end if

    ######## 6. 检查夹爪是否会碰撞到场景 ########
    print()
    logging.info(f'grasp-step [6] , {BLUE}check if gripper is obstacled{RESET}')
    if not wait_key(debug):
        return -1
    # end if

    # 获取图像
    frames = cam_node.get_frames(do_spin_once=True)
    if frames is None:
        logging.error(f"{RED}get frames failed.{RESET}")
        return -1
    # end if

    is_obstacled = collision_detector.check(gripper_dist=target_gripper_dist,
                                            ref_T_base_end=cur_T_base_end,
                                            target_T_base_end=target_T_base_end,
                                            ref_bgr_img=frames[0][0],
                                            ref_depth_img=frames[0][1],
                                            max_depth_diff=0.01,
                                            debug_level=debug_level)

    logging.info(f'is gripper obstacled: {GREEN}{is_obstacled}{RESET}')

    if is_obstacled:
        logging.warning(f"{RED}gripper is obstacled at grasp pose, cannot proceed to grasp.{RESET} try next label.")
        return 0
    # end if

    ######## 7. 移动到预备位置 ########
    print()
    logging.info(f'grasp-step [7] , {BLUE}move to ready pose{RESET}')
    if not wait_key(debug):
        return -1
    # end if

    # 先收紧夹爪,缩小碰撞范围,防止碰撞到堆叠的物体
    is_ok = arm.set_gripper_dist(target_gripper_dist)
    if not is_ok:
        logging.error(f"{RED}close gripper failed.{RESET}")
        return -1
    # end if

    # 移动之前更新临时变量
    prev_T_cam_model = cur_T_cam_model
    prev_T_base_end = cur_T_base_end

    is_ok = arm.set_pose(target_T_base_end, th_pos_err=0.0005)
    if not is_ok:
        logging.error(f"{RED}move arm to ready pose failed. try next label.{RESET}")
        return 0
    # end if

    # # 计算定位偏差
    # if show_locate_err:
    #     cur_T_base_end = arm.get_pose()
    #     cur_T_end_base = inv_tf(cur_T_base_end)
    #     init_T_cam_model = T_cam_end @ cur_T_end_base @ prev_T_base_end @ T_end_cam @ prev_T_cam_model
    #     cur_T_cam_model, _ = task_node.locate_model(init_T_cam_model=init_T_cam_model,
    #                                                 timeout_sec=20.0,
    #                                                 debug_level=debug_level)
    #     if cur_T_cam_model is None:
    #         logging.error(f"{RED}locate model failed at ready pose.{RESET}")
    #         return -1
    #     # end if

    #     pos_err, rot_err = compute_locate_error(ready_T_cam_model, cur_T_cam_model, sym_tfs)
    #     logging.info(f'locate error at ready pose, pos_err(mm): {pos_err:.2f}, rot_err(deg): {rot_err:.2f}')
    # # end if

    ######## 迭代细化预备位姿 ########
    refine_cnt = 0
    while refine_cnt < max_refine_cnt:
        refine_cnt += 1

        ######## 8. 跟踪物体并计算预备位姿 ########
        print()
        logging.info(f'grasp-step [8-{refine_cnt}] , {BLUE}track model{RESET}')
        if not wait_key(debug):
            return -1
        # end if

        # 跟踪物体
        cur_T_base_end = arm.get_pose()
        cur_T_end_base = inv_tf(cur_T_base_end)
        init_T_cam_model = T_cam_end @ cur_T_end_base @ prev_T_base_end @ T_end_cam @ prev_T_cam_model
        cur_T_cam_model, _ = client_node.locate_model(init_T_cam_model=init_T_cam_model,
                                                      timeout_sec=20.0,
                                                      debug_level=debug_level)
        if cur_T_cam_model is None:
            logging.error(f"{RED}track model failed.{RESET}")
            return -1
        # end if
        logging.info(f'track T_cam_model: \n{GREEN}{cur_T_cam_model}{RESET}')

        # 计算定位偏差
        if show_locate_err:
            pos_err, rot_err = compute_locate_error(ready_T_cam_model, cur_T_cam_model, sym_tfs)
            logging.info(f'locate error at ready pose, pos_err(mm): {pos_err:.2f}, rot_err(deg): {rot_err:.2f}')
        # end if

        # 计算预备位姿
        target_T_base_end, _ = compute_ready_pose(tmpl_state_list=tmpl_state_list,
                                                  sym_tfs=sym_tfs,
                                                  T_end_cam=T_end_cam,
                                                  cur_T_base_end=cur_T_base_end,
                                                  cur_T_cam_model=cur_T_cam_model,
                                                  tmpl_idx=tmpl_idx)
        logging.info(f'ready T_base_end: \n{GREEN}{target_T_base_end}{RESET}')

        # 发布将要到达的位姿( 用于可视化 )
        arm_node.publish_pose(target_T_base_end)

        # 检查机械臂位姿合理性
        if not check_arm_pose(T_base_end=target_T_base_end,
                              T_end_cam=T_end_cam,
                              gripper_body=gripper_body,
                              gripper_dist=arm.get_gripper_dist(),
                              th_angle_z=TH_ANGLE_Z,
                              th_gripper_height=TH_GRIPPER_HEIGHT):
            logging.error(f"{RED}arm pose check failed at ready pose. try next label{RESET}")
            return 0
        # end if

        ######## 9. 再次移动到预备位置并计算抓取位姿 ########
        print()
        logging.info(f'grasp-step [9-{refine_cnt}] , {BLUE}move to ready pose again{RESET}')
        if not wait_key(debug):
            return -1
        # end if

        # 移动之前更新临时变量
        prev_T_cam_model = cur_T_cam_model
        prev_T_base_end = cur_T_base_end

        # 移动到预备位置
        is_ok = arm.set_pose(target_T_base_end, th_pos_err=0.0005)
        if not is_ok:
            logging.error(f"{RED}move arm to ready pose failed. try next label.{RESET}")
            return 0
        # end if
    # end while

    # 达到最大细化次数时,计算抓取位姿
    logging.info('reached max refine count.')

    cur_T_base_end = arm.get_pose()

    # 计算定位偏差
    if show_locate_err and debug:
        # 跟踪物体
        cur_T_end_base = inv_tf(cur_T_base_end)
        init_T_cam_model = T_cam_end @ cur_T_end_base @ prev_T_base_end @ T_end_cam @ prev_T_cam_model
        cur_T_cam_model, _ = client_node.locate_model(init_T_cam_model=init_T_cam_model,
                                                      timeout_sec=20.0,
                                                      debug_level=debug_level)
        if cur_T_cam_model is None:
            logging.error('track model failed.')
            return -1
        # end if
        logging.info(f'track T_cam_model: \n{GREEN}{cur_T_cam_model}{RESET}')

        # 计算定位偏差
        pos_err, rot_err = compute_locate_error(ready_T_cam_model, cur_T_cam_model, sym_tfs)
        logging.info(f'locate error at ready pose, pos_err(mm): {pos_err:.2f}, rot_err(deg): {rot_err:.2f}')
    # end if

    # 计算抓取位姿
    target_T_base_end = cur_T_base_end @ delta_T_end
    logging.info(f'grasp T_base_end: \n{GREEN}{target_T_base_end}{RESET}')

    target_gripper_dist = grasp_gripper_dist + 0.008  # 刚好比物体宽一点,不会碰到其他物体

    # 设置夹爪位置
    is_ok = arm.set_gripper_dist(target_gripper_dist)
    if not is_ok:
        logging.error(f"{RED}set gripper to grasp position failed.{RESET}")
        return -1
    # end if

    # 发布将要到达的位姿( 用于可视化 )
    arm_node.publish_pose(target_T_base_end)

    # 检查机械臂位姿合理性
    if not check_arm_pose(T_base_end=target_T_base_end,
                          T_end_cam=T_end_cam,
                          gripper_body=gripper_body,
                          gripper_dist=target_gripper_dist,
                          th_angle_z=TH_ANGLE_Z,
                          th_gripper_height=TH_GRIPPER_HEIGHT):
        logging.warning(f"{RED}arm pose check failed at ready pose.{RESET} try next label.")
        return 0
    # end if

    ######## 10. 再次检查夹爪是否会碰撞到场景 ########
    print()
    logging.info(f'grasp-step [10] , {BLUE}check if gripper is obstacled at grasp pose{RESET}')
    if not wait_key(debug):
        return -1
    # end if

    # 获取图像
    frames = cam_node.get_frames(do_spin_once=True)
    if frames is None:
        logging.error(f"{RED}get frames failed.{RESET}")
        return -1
    # end if

    is_obstacled = collision_detector.check(gripper_dist=grasp_gripper_dist + 0.004,
                                            ref_T_base_end=cur_T_base_end,
                                            target_T_base_end=target_T_base_end,
                                            ref_bgr_img=frames[0][0],
                                            ref_depth_img=frames[0][1],
                                            max_depth_diff=0.015,
                                            debug_level=debug_level)

    logging.info(f'is gripper obstacled: {GREEN}{is_obstacled}{RESET}')

    if is_obstacled:
        logging.warning(f"{YELLOW}gripper is obstacled at grasp pose, cannot proceed to grasp.{RESET} try next label.")
        return 0
    # end if

    ######## 11. 抓取 ########
    print()
    logging.info(f'grasp-step [11] , {BLUE}move to grasp pose{RESET}')
    if not wait_key(debug):
        return -1
    # end if

    use_cache = False  # 抓取之后就不能再使用缓存了, 因为机械臂可能会碰到物体导致位姿发生变化

    # 移动到抓取位姿
    is_ok = arm.set_pose(target_T_base_end, move_line=True)
    if not is_ok:
        logging.error(f"{RED}move arm to grasp pose failed. try next label.{RESET}")
        return 0
    # end if

    # 闭合夹爪
    is_ok = arm.set_gripper_dist(grasp_gripper_dist - 0.008)
    if not is_ok:
        logging.error(f"{RED}close gripper failed.{RESET}")
        return -1
    # end if

    time.sleep(0.3)  # 等待夹爪闭合完成

    # 提高 Z 轴高度, 避免碰撞
    target_T_base_end[2, 3] += 0.1
    logging.info(f"{GREEN} try move arm to higher pose...{RESET}")
    is_ok = arm.set_pose(target_T_base_end, move_line=True)
    if not is_ok:
        logging.warning(f"{YELLOW}move arm to higher pose failed. release gripper and try next label.{RESET}")
        if not arm.set_gripper_dist(arm.get_gripper_dist() + 0.01):
            logging.error(f"{RED}release gripper failed.{RESET}")
            return -1
        else:
            return 0
        # end if
    # end if

    return 1
# end def do_grasp


def run(T_end_cam: np.ndarray,
        sym_tfs: np.ndarray,
        grasp_tmpl_dict: Dict,
        gripper_body: GripperBody,
        collision_detector: CollisionDetector,
        arm: ArmWrapper,
        cam_node: CamNode,
        arm_node: TargetArmNode,
        client_node: ClientNode,
        debug_level: int,
        debug: bool = False):
    """
    循环执行 3D 抓取任务    
    """

    detect_joints = grasp_tmpl_dict['detect_joints']               # 机械臂处于检测状态时的位置关节角度
    detect_gripper_dist = grasp_tmpl_dict['detect_gripper_dist']   # 机械臂处于检测状态时的夹爪位置
    place_T_base_end = grasp_tmpl_dict['place_T_base_end']         # 机械臂处于放置状态时的末端位姿
    place_joints = grasp_tmpl_dict['place_joints']                 # 机械臂处于放置状态时的位置关节角度
    place_gripper_dist = grasp_tmpl_dict['place_gripper_dist']     # 机械臂处于放置状态时的夹爪位置
    tmpl_state_list = grasp_tmpl_dict['tmpl_state_list']           # 抓取模板状态列表

    use_cache = False  # 是否使用缓存的位姿
    cache_T_base_end = None
    cache_joints = None
    while rclpy.ok():

        print(f"\n{GREEN}start loop {RESET}")

        ######## 0. 移动到检测位置 ########
        if not use_cache:  # 不使用缓存, 需要重新检测
            logging.info(f'step [0] , {BLUE}move to detect pose{RESET}')
            if not wait_key(debug):
                return -1
            # end if

            logging.info(f"{GREEN}try move arm to detect pose...{RESET}")
            is_ok = arm.set_gripper_dist(detect_gripper_dist)
            if not is_ok:
                logging.error(f"{RED}set gripper to detect pose failed.{RESET}")
                break
            # end if

            detect_T_base_end = grasp_tmpl_dict['detect_T_base_end']
            is_ok = arm.set_pose(detect_T_base_end, th_pos_err=0.005)
            if not is_ok:
                logging.error(f"{RED}move arm to detect pose failed, try again.{RESET}")
                break
            # end if
            # is_ok = arm.set_joints(detect_joints, th_angle_err=0.01)
            # if not is_ok:
            #     logging.error(f"{RED}move arm to detect pose failed, try again.{RESET}")
            #     break
            # # end if

            cache_joints = arm.get_joints()
            cache_T_base_end = arm.get_pose()
        # end if

        status = do_grasp(T_end_cam=T_end_cam,
                          sym_tfs=sym_tfs,
                          tmpl_state_list=tmpl_state_list,
                          gripper_body=gripper_body,
                          collision_detector=collision_detector,
                          arm=arm,
                          cam_node=cam_node,
                          arm_node=arm_node,
                          client_node=client_node,
                          use_cache=use_cache,
                          cache_T_base_end=cache_T_base_end,
                          cache_joints=cache_joints,
                          debug_level=debug_level,
                          debug=debug)

        if status == -1:  # 退出程序
            logging.info(f"{GREEN}exiting program...{RESET}")
            break
        elif status == 0:  # 任务失败, 但可以继续尝试
            use_cache = True  # 任务失败了, 可以尝试复用缓存信息来定位下一个物体, 因为机械臂可能没有碰到物体导致位姿没有发生变化
            logging.info(f"{GREEN}grasp task failed, try next loop...{RESET}")
            continue
        elif status == 1:  # 成功执行抓取任务, 可以进行下一步
            use_cache = False  # 抓取成功了, 就不能再使用缓存了, 因为机械臂可能会碰到物体导致位姿发生变化
            logging.info(f"{GREEN}grasp task success, try next step...{RESET}")
        # end if

        ######## -1. 放置 ########
        print()
        logging.info(f'step [-1] , {BLUE}move to place pose{RESET}')
        if not wait_key(debug):
            return -1
        # end if

        # 移动到放置位置
        logging.info(f"{GREEN} try move arm to place pose...{RESET}")

        target_T_base_end = place_T_base_end.copy()
        target_T_base_end[2, 3] = arm.get_pose()[2, 3]  # 保持当前高度, 只移动 XY 平面位置和姿态
        is_ok = arm.set_pose(target_T_base_end, move_line=True)
        if not is_ok:
            logging.error(f"{RED}move arm to place pose failed.{RESET}")
            break
        # end if

        is_ok = arm.set_joints(place_joints, th_angle_err=0.005)
        if not is_ok:
            logging.error(f"{RED}move arm to place pose failed.{RESET}")
            break
        # end if

        # 打开夹爪
        is_ok = arm.set_gripper_dist(place_gripper_dist)
        if not is_ok:
            logging.error(f"{RED}open gripper to place pose failed.{RESET}")
            break
        # end if

    # end while

    # 机械臂回到零点
    arm.set_joints(arm.init_joints)

    logging.info('run finished.')

# end def run


######################################################### 主函数 #########################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--color_img_topic", type=str, required=True,
                        help="彩色图像的 ROS2 话题名称")

    parser.add_argument("--depth_img_topic", type=str, required=True,
                        help="深度图像的 ROS2 话题名称")

    parser.add_argument("--tmpl_dir", type=str, required=True,
                        help="模板文件的目录")

    parser.add_argument("--model_name", type=str, default=None,
                        help="要匹配的模型名称, 如果提供了该参数, 则服务名称会变成 MATCHING3D_SRV_BASE_NAME/{model_name}, 适用于同时运行多个匹配服务的情况")

    parser.add_argument("--debug", action='store_true',
                        help="是否开启调试模式")
    args = parser.parse_args()

    color_img_topic = args.color_img_topic
    if color_img_topic is None:
        logging.error("color_img_topic is not provided.")
        exit(0)
    # end if

    depth_img_topic = args.depth_img_topic
    if depth_img_topic is None:
        logging.error("depth_img_topic is not provided.")
        exit(0)
    # end if

    tmpl_dir = args.tmpl_dir
    if tmpl_dir is None:
        logging.error('no tmpl_dir specified, exiting')
        sys.exit(-1)
    # end if

    model_name = reset_empty_str(args.model_name)

    debug = args.debug
    debug_level = 0
    if debug is True:
        debug_level = 3
    # end if
    logging.info(f'debug level: {GREEN}{debug_level}{RESET}')

    print()
    print(f"color image topic: {GREEN}{color_img_topic}{RESET}")
    print(f"depth image topic: {GREEN}{depth_img_topic}{RESET}")
    print(f'load grasp template from: {GREEN}{tmpl_dir}{RESET}')
    print(f'model name for matching: {GREEN}{model_name}{RESET}')
    print(f'enable debug mode: {GREEN}{debug}{RESET}')
    print()

    # 读取相机参数
    camera_param_path = os.path.join(root_dir, 'data/calib/cam_params.json')
    camera_param_path = os.path.normpath(camera_param_path)
    intrinsic, distortion, depth_scale = read_rgbd_params(camera_param_path)
    if intrinsic is None or depth_scale is None:
        exit(1)
    # end if

    # 读取手眼标定矩阵
    print()
    handeye_calib_path = os.path.join(root_dir, 'data/calib/calib_handeye.json')
    T_end_cam, _ = read_handeye_calib(handeye_calib_path)
    logging.info(f'T_end_cam: \n{GREEN}{T_end_cam} {RESET}')

    # 读取夹爪模型
    gripper_path = os.path.join(root_dir, 'data/calib/gripper_body.json')
    if not os.path.exists(gripper_path):
        logging.error(f'no gripper body file found at: {gripper_path}, exiting')
        exit(1)
    # end if
    gripper_data_dict = mmengine.load(gripper_path)
    gripper_width = gripper_data_dict['width']
    gripper_thickness = gripper_data_dict['thickness']
    T_cam_gripper = np.array(gripper_data_dict['T_cam_gripper'], dtype=np.float32)
    gripper_body = GripperBody(width=gripper_width,
                               thickness=gripper_thickness,
                               T_cam_gripper=T_cam_gripper)
    logging.info(f"gripper width: {GREEN}{gripper_body.width}{RESET}, thickness: {GREEN}{gripper_body.thickness}{RESET}")
    logging.info(f"T_cam_gripper: \n{GREEN}{gripper_body.T_cam_gripper}{RESET}")
    print()

    # 读取抓取模板
    print()
    tmpl_dir = os.path.normpath(tmpl_dir)  # 规范化路径
    tmpl_dict = read_tmpl_grasp(tmpl_dir)
    if tmpl_dict is None:
        logging.error('read grasp tmpl failed.')
        sys.exit(-1)
    # end if

    # 创建机械臂对象
    arm = ArmWrapper()
    if not arm.is_connected():
        logging.error(f'{RED}failed to connect to arm, exiting{RESET}')   # 红色打印
        exit(1)
    # end if

    # 设置夹爪先闭合再打开,表明程序已经启动
    is_ok = arm.set_gripper_dist(0.02)
    if not is_ok:
        logging.error('set gripper initial position failed, exiting')
        exit(1)
    # end if
    time.sleep(0.5)
    is_ok = arm.set_gripper_dist(0.07)
    if not is_ok:
        logging.error('set gripper initial position failed, exiting')
        exit(1)
    # end if

    # 初始化碰撞检测器
    collision_detector = CollisionDetector(gripper_body=gripper_body,
                                           T_end_cam=T_end_cam,
                                           intrinsic=intrinsic,
                                           depth_scale=depth_scale,
                                           distortion=distortion,
                                           debug_dir=COLLISION_DIR)

    # 初始化 ROS2 节点
    rclpy.init(args=None)

    cam_node = CamNode(img_topic_list=[color_img_topic, depth_img_topic])
    arm_node = TargetArmNode()
    client_node = ClientNode(model_name=model_name)

    # 等待匹配服务可用
    wait_cnt = 0
    max_wait_cnt = 10
    while not client_node.client.service_is_ready():
        wait_cnt += 1
        logging.warning(f'{YELLOW}waiting for service [{client_node.client.srv_name}] ready, wait cnt: {wait_cnt}/{max_wait_cnt} {RESET}')  # 黄色打印
        time.sleep(1.0)

        if wait_cnt >= max_wait_cnt:
            logging.error(f'{RED}service not available, exiting{RESET}')
            exit(1)
        # end if
    # end while

    # 获取物体对称性信息
    sym_tfs, _ = client_node.get_symmetric_info(timeout_sec=5.0)
    if sym_tfs is None:
        logging.error('get symmetric info failed. exiting')
        exit(1)
    # end if
    print()

    # 运行
    run(T_end_cam=T_end_cam,
        sym_tfs=sym_tfs,
        grasp_tmpl_dict=tmpl_dict,
        gripper_body=gripper_body,
        collision_detector=collision_detector,
        arm=arm,
        cam_node=cam_node,
        arm_node=arm_node,
        client_node=client_node,
        debug_level=debug_level,
        debug=debug)

    cam_node.destroy_node()
    arm_node.destroy_node()
    client_node.destroy_node()
    rclpy.shutdown()
    logging.info('shutdown')

# end if __name__ == '__main__'

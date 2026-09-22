"""
功能说明: 创建3D抓取模板的 ROS 节点,也可以用于发布机械臂状态
模板需要保存以下数据:
- 1.末端刚好抓取到物体时的状态, 包含机械臂末端位姿和夹爪距离等信息
- 2.处于抓取准备阶段时的状态, 包含机械臂末端位姿、夹爪距离、以及物体在相机坐标系中的位姿
"""

import logging

import rclpy

import argparse
import os
import sys
import time
import json

import cv2


# 导入本工程的模块
# 导入本工程的模块
code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f'{code_dir}/../../../')
sys.path.append(root_dir)

from core.utils import (
    GREEN, YELLOW, BLUE, RED, RESET,
    KeyboardReader, read_calib_handeye, read_rgbd_params
)
from core.arm_wrapper import ArmWrapper
from core.arm_utils import compute_axis_aligned_pose
from core.cam_ros_utils import CamNode
from core.vision_utils import TagMatcher3D, depth_mean_filter





######################################################### 全局变量 #########################################################


######################################################### 类定义 #########################################################


######################################################### 主函数 #########################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--cam_params_path", type=str, required=True,
                        help="相机参数文件的路径, 包含内参和畸变参数")

    parser.add_argument("--calib_handeye_path", type=str, required=True,
                        help="手眼标定文件的路径, 包含相机与机械臂的位姿关系")

    parser.add_argument("--color_img_topic", type=str, required=True,
                        help="RGB 图像的 ROS2 话题名称")

    parser.add_argument("--depth_img_topic", type=str, required=True,
                        help="深度图像的 ROS2 话题名称")

    parser.add_argument("--tmpl_dir", type=str, required=True,
                        help="模板文件的目录")

    args = parser.parse_args()

    cam_params_path = args.cam_params_path
    calib_handeye_path = args.calib_handeye_path

    color_img_topic = args.color_img_topic
    if color_img_topic is None:
        logging.error("Error: color_img_topic is not provided.")
        exit(0)
    # end if

    depth_img_topic = args.depth_img_topic
    if depth_img_topic is None:
        logging.error("Error: depth_img_topic is not provided.")
        exit(0)
    # end if

    tmpl_dir = args.tmpl_dir
    if tmpl_dir is None:
        logging.error('no tmpl_dir specified, exiting')
        exit(1)
    # end if

    print()
    print(f"RGB-D camera parameters file: {BLUE}{cam_params_path}{RESET}")
    print(f"handeye calib file: {BLUE}{calib_handeye_path}{RESET}")
    print(f"color image topic: {BLUE}{color_img_topic}{RESET}")
    print(f"depth image topic: {BLUE}{depth_img_topic}{RESET}")
    print(f'grasp template will be saved to: {BLUE}{tmpl_dir}{RESET}')
    print()

    # 读取相机参数
    intrinsic, distortion, depth_scale = read_rgbd_params(cam_params_path)
    if intrinsic is None or distortion is None or depth_scale is None:
        logging.error('read camera parameters failed, exiting')
        exit(1)
    # end if

    config = TagMatcher3D.Config(
        intrinsic=intrinsic,
        distortion=distortion,
        depth_scale=depth_scale
    )
    matcher = TagMatcher3D(config)

    # 读取手眼标定矩阵
    print()
    T_end_cam, _ = read_calib_handeye(calib_handeye_path)
    if T_end_cam is None:
        logging.error('read handeye calib failed, exiting')
        exit(1)
    # end if

    # 创建机械臂对象
    arm = ArmWrapper()
    if not arm.is_connected():
        logging.error(f'{RED}failed to connect to arm, exiting{RESET}')   # 红色打印
        exit(1)
    # end if

    # 初始化 ROS2 节点
    rclpy.init(args=None)
    cam_node = CamNode(img_topic_list=[color_img_topic, depth_img_topic])

    # 创建键盘读取对象
    keyboard_reader = KeyboardReader()

    # 创建文件夹用于保存结果
    tmpl_dir = os.path.normpath(tmpl_dir)  # 规范化路径
    os.makedirs(tmpl_dir, exist_ok=True)

    print()
    print(f'use keyboard to control: \n{BLUE}'
          f'  q: 退出程序\n'
          f'  <: 缩小夹爪距离\n'
          f'  >: 放大夹爪距离\n'
          f'  a: 使末端的 z 轴方向与基座的 -z 轴平行\n'
          f'  c: 使相机的 z 轴方向与基座的 -z 轴平行\n'
          f'  g: 保存抓取时的状态\n'
          f'  r: 保存准备阶段的状态\n'
          f'{RESET}')

    while rclpy.ok():

        time.sleep(0.05)  # 避免占用过多 CPU

        key = keyboard_reader.read_key()
        if key is None:
            continue
        # end if
        # logging.info(f'pressed: {key}')

        if key == 'q':  # 退出
            print()
            logging.info('Quit.')
            break
        # end if

        T_base_end = arm.get_pose()  # 获取机械臂末端位姿
        gripper_dist = arm.get_gripper_dist()  # 获取夹爪距离
        joints = arm.get_joints()  # 获取机械臂关节角度

        # 夹爪控制
        gripper_step = 0.001  # 夹爪每次移动的步长, 单位: m
        if key == ',':  # 缩小夹爪
            set_dist = gripper_dist - gripper_step
            arm.set_gripper_dist(set_dist)
            logging.info(f'set gripper dist {gripper_dist:.3f} -->> {set_dist:.3f}, actual: {arm.get_gripper_dist():.3f}')
        elif key == '.':  # 放大夹爪
            set_dist = gripper_dist + gripper_step
            arm.set_gripper_dist(set_dist)
            logging.info(f'set gripper dist {gripper_dist:.3f} -->> {set_dist:.3f}, actual: {arm.get_gripper_dist():.3f}')
        # end if

        # 调整末端,使末端坐标系的 Z 轴指向下方
        if key == 'a':
            target_T_base_end = compute_axis_aligned_pose(T_base_end, base_axis_idx=-3, obj_axis_idx=3)
            if target_T_base_end is None:
                continue
            # end if

            logging.info(f'try move to new T_base_end:\n{target_T_base_end}')
            is_ok = arm.set_pose(target_T_base_end)
            if not is_ok:
                logging.warning('failed to move to new T_base_end')
            # end if
            print()
        # end if

        # 调整末端,使相机坐标系的 Z 轴指向下方
        if key == 'c':
            target_T_base_end = compute_axis_aligned_pose(T_base_end, base_axis_idx=-3, obj_axis_idx=3, T_end_obj=T_end_cam)
            if target_T_base_end is None:
                continue
            # end if

            logging.info(f'try move to new T_base_end:\n{target_T_base_end}')
            is_ok = arm.set_pose(target_T_base_end)
            if not is_ok:
                logging.warning('failed to move to new T_base_end')
            # end if
            print()
        # end if

        # 保存抓取时的数据
        if key == 'g':

            print()
            logging.info(f"{GREEN}task: save grasp data {RESET}")

            # 获取 RGB-D 图像
            frames = cam_node.get_frames(do_spin_once=True)
            if frames is None:
                logging.warning('No RGB-D frame available yet.')
                continue
            # end if

            color_img, depth_img = frames[0]
            logging.info(f"rgb_img shape: {color_img.shape}, depth_img shape: {depth_img.shape}")

            logging.info(f"T_base_end:\n{T_base_end}")
            logging.info(f"gripper_dist(m): {gripper_dist:.3f}")

            # 保存结果
            rgb_path = os.path.join(tmpl_dir, f'grasp-color.png')
            depth_path = os.path.join(tmpl_dir, f'grasp-depth.png')

            cv2.imwrite(rgb_path, color_img)
            cv2.imwrite(depth_path, depth_img)
            logging.info(f'Saved color images to: {rgb_path}')

            # 保存机械臂状态
            data_dict = {
                "T_base_end": T_base_end.tolist(),
                "gripper_dist": gripper_dist
            }
            file_path = os.path.join(tmpl_dir, f'grasp.json')
            with open(file_path, 'w') as f:
                json.dump(data_dict, f, indent=4)
            logging.info(f'Saved grasp state to: {file_path}')
        # end if

        # 保存准备阶段的数据
        if key == 'r':

            print()
            logging.info(f"{GREEN}task: save ready data {RESET}")

            # 获取 RGB-D 图像
            frames = cam_node.get_frames(do_spin_once=True, frames_num=5)
            if frames is None:
                logging.warning('No RGB-D frame available yet.')
                continue
            # end if

            color_img, depth_img = frames[0]
            logging.info(f"rgb_img shape: {color_img.shape}, depth_img shape: {depth_img.shape}")

            logging.info(f"T_base_end:\n{T_base_end}")
            logging.info(f"gripper_dist(m): {gripper_dist:.3f}")

            # 获取匹配结果
            depth_img_list = [frame[1] for frame in frames]  # 获取所有帧的深度图像列表
            depth_img = depth_mean_filter(depth_img_list, obs_ratio=0.5)  # 对深度图像列表进行均值滤波
            result_list, msg = matcher.match(color_img, depth_img, top_k=1)
            if len(result_list) == 0:
                logging.warning(f'3D match failed, {msg}')
                continue
            # end if
            T_cam_model = result_list[0].T_cam_tag
            if T_cam_model is None:
                logging.warning('3D match failed.')
                continue
            # end if
            logging.info(f"T_cam_model:\n{T_cam_model}")

            # 保存结果
            rgb_path = os.path.join(tmpl_dir, f'ready-color.png')
            depth_path = os.path.join(tmpl_dir, f'ready-depth.png')

            cv2.imwrite(rgb_path, color_img)
            cv2.imwrite(depth_path, depth_img)
            logging.info(f'Saved color images to: {rgb_path}')

            # 保存机械臂状态
            data_dict = {
                "T_base_end": T_base_end.tolist(),
                "T_cam_model": T_cam_model.tolist(),
                "gripper_dist": gripper_dist
            }
            file_path = os.path.join(tmpl_dir, f'ready.json')
            with open(file_path, 'w') as f:
                json.dump(data_dict, f, indent=4)
            logging.info(f'Saved ready state to: {file_path}')
        # end if

    # end while

    cam_node.destroy_node()
    rclpy.shutdown()
    logging.info('shutdown')

# end if __name__ == '__main__

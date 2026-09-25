# -*- coding: utf-8 -*-
"""
功能说明: 标定从夹爪坐标系到相机坐标系的位姿变换
"""

import argparse
import json
import logging
import os
import sys

# 导入其他库
import apriltag2
import cv2
import mmengine
import numpy as np
import open3d
import rclpy


# 导入本工程的模块
# 导入本工程的模块
code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{code_dir}/../../../")
sys.path.append(root_dir)

from core.arm_ros_utils import ArmNode, pose_to_transform_stamped
from core.arm_utils import GripperBody, compute_axis_aligned_pose
from core.arm_wrapper import ArmWrapper
from core.cam_ros_utils import CamNode
from core.utils import (
    BLUE,
    GREEN,
    RED,
    RESET,
    KeyboardReader,
    read_calib_handeye,
    read_rgbd_params,
)
from core.vision_utils import depth_mean_filter
from tf2_ros import TransformBroadcaster
from typing_extensions import Tuple

######################################################### 全局变量 #########################################################


######################################################### 函数定义 #########################################################


def compute_corners3d(
    gray_img: np.ndarray,
    depth_img: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    depth_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:

    # 畸变矫正
    K = np.array(
        [[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]],
        dtype=np.float32,
    )
    D = np.array(distortion, dtype=np.float32) if distortion is not None else None
    un_img = cv2.undistort(gray_img, K, D)

    # cv2.imshow('undistorted', un_img)
    # cv2.waitKey(0)

    # 检测 AprilTag, ID: 0
    detector = apriltag2.Detector(tag_family="tag36h11", black_border=2)
    tags = detector.detect(un_img)
    if len(tags) == 0:
        logging.error("No AprilTag detected.")
        return None
    # end if

    tag = None
    for t in tags:
        if t.id == 0:
            tag = t
            break
        # end if
    # end for

    if tag is None:
        logging.error("AprilTag ID 0 not detected.")
        return None
    # end if

    # 获取 tag 的四个角点
    corners = tag.corners  # (4,2)

    # 绘制检测结果
    vis_img = cv2.cvtColor(un_img, cv2.COLOR_GRAY2BGR)
    pt0 = (int(corners[0][0]), int(corners[0][1]))
    pt1 = (int(corners[1][0]), int(corners[1][1]))
    pt2 = (int(corners[2][0]), int(corners[2][1]))
    pt3 = (int(corners[3][0]), int(corners[3][1]))
    cv2.line(vis_img, pt0, pt1, (0, 0, 255), 1)
    cv2.line(vis_img, pt1, pt2, (255, 0, 0), 1)
    cv2.line(vis_img, pt2, pt3, (255, 0, 0), 1)
    cv2.line(vis_img, pt3, pt0, (0, 255, 0), 1)
    cv2.rectangle(
        vis_img,
        (int(pt0[0] - 5), int(pt0[1] - 5)),
        (int(pt0[0] + 5), int(pt0[1] + 5)),
        (0, 255, 255),
        1,
    )
    cv2.rectangle(
        vis_img,
        (int(pt2[0] - 5), int(pt2[1] - 5)),
        (int(pt2[0] + 5), int(pt2[1] + 5)),
        (0, 255, 255),
        1,
    )

    # 只在 x64 架构下可视化
    import platform

    if platform.machine() == "x86_64":
        cv2.imshow("tag detection", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # end if

    # 将角点往四周扩展到原来的3倍
    center = tag.center
    expanded_corners = []
    for corner in corners:
        vec = corner - center
        expanded_corner = center + vec * 3.0
        expanded_corners.append(expanded_corner)
    # end for
    expanded_corners = np.array(expanded_corners)  # (4,2)

    # 制作掩码
    mask = np.zeros_like(gray_img, dtype=np.uint8)
    pts = expanded_corners.astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)

    # 将非掩码区域设为0
    masked_depth = depth_img.copy()
    masked_depth[mask == 0] = 0

    # 提取掩码区域的点云f
    pc = open3d.geometry.PointCloud.create_from_depth_image(
        open3d.geometry.Image(masked_depth),
        open3d.camera.PinholeCameraIntrinsic( 
            gray_img.shape[1],
            gray_img.shape[0],
            intrinsic[0],
            intrinsic[1],
            intrinsic[2],
            intrinsic[3],
        ),
        np.eye(4),
        depth_scale=1.0 / depth_scale,
        depth_trunc=0.5,
    )

    # 平面拟合
    plane, inliers = pc.segment_plane(
        distance_threshold=0.002, ransac_n=6, num_iterations=1000
    )
    logging.info(
        f"拟合的平面方程系数: {plane}\n 内点数: {len(inliers)}\n 内点比例: {len(inliers) / len(pc.points)}"
    )

    # 计算角点的空间坐标( 根据平面方程计算 )
    def compute_pt3d(
        corner: np.ndarray, intrinsic: np.ndarray, plane: np.ndarray
    ) -> np.ndarray:
        nx = (corner[0] - intrinsic[2]) / intrinsic[0]
        ny = (corner[1] - intrinsic[3]) / intrinsic[1]
        A, B, C, D = plane
        z = -D / (A * nx + B * ny + C)
        x = nx * z
        y = ny * z
        return np.array([x, y, z])

    # end def compute_pt3d

    corners3d = np.array(
        [compute_pt3d(corner, intrinsic, plane) for corner in corners]
    )  # (4,3)

    return corners3d


# end def compute_corners3d


######################################################### 类定义 #########################################################


######################################################### 主函数 #########################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cam_params_path",
        type=str,
        default="/home/i4/桌面/test_location/calib/collect_image/cam_params.json",
        help="相机参数文件的路径, 包含内参和畸变参数",
    )

    parser.add_argument(
        "--calib_handeye_path",
        type=str,
        default="/home/i4/桌面/test_location/calib/collect_image_handeye/calib_handeye.json",
        help="手眼标定文件的路径, 包含相机与机械臂的位姿关系",
    )

    parser.add_argument(
        "--gripper_path",
        type=str,
        default="/home/i4/桌面/test_location/calib/gripper/gripper_params.json",
        help="夹爪标定文件的路径, 包含夹爪的尺寸和位姿信息",
    )

    parser.add_argument(
        "--color_img_topic",
        type=str,
        default="/gemini305/color/image_raw",
        help="RGB 图像的 ROS2 话题名称",
    )

    parser.add_argument(
        "--depth_img_topic",
        type=str,
        default="/gemini305/depth/image_raw",
        help="深度图像的 ROS2 话题名称",
    )

    parser.add_argument(
        "--pc_frame_id", type=str, default="camera_link", help="点云所在的坐标系名称"
    )

    parser.add_argument(
        "--gripper_size",
        type=str,
        help="夹爪的宽度和厚度(单位: m), 格式: [width,thickness]",
        default="[0.0015,0.002]",
    )
    args = parser.parse_args()

    cam_params_path = args.cam_params_path
    calib_handeye_path = args.calib_handeye_path
    gripper_path = args.gripper_path

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

    pc_frame_id = args.pc_frame_id
    if pc_frame_id is None:
        logging.error("Error: pc_frame_id is not provided.")
        exit(0)
    # end if

    gripper_size = json.loads(args.gripper_size)  # 夹爪的宽度和厚度(单位: m)
    gripper_width = gripper_size[0]
    gripper_thickness = gripper_size[1]

    print()
    print(f"RGB-D camera parameters file: {GREEN}{cam_params_path}{RESET}")
    print(f"handeye calib file: {GREEN}{calib_handeye_path}{RESET}")
    print(f"gripper calib file save path: {GREEN}{gripper_path}{RESET}")
    print(f"color image topic: {GREEN}{color_img_topic}{RESET}")
    print(f"depth image topic: {GREEN}{depth_img_topic}{RESET}")
    print(f"point cloud frame id: {GREEN}{pc_frame_id}{RESET}")
    print(
        f"gripper_width: {GREEN}{gripper_width}{RESET}, gripper_thickness: {GREEN}{gripper_thickness}{RESET}"
    )

    calib_dir = os.path.join(root_dir, "data/calib")
    calib_dir = os.path.normpath(calib_dir)  # 规范化路径

    # 读取相机参数
    intrinsic, distortion, depth_scale = read_rgbd_params(cam_params_path)
    print()

    # 读取手眼标定矩阵
    T_end_cam, _ = read_calib_handeye(calib_handeye_path)
    print()

    # 创建机械臂对象
    arm = ArmWrapper()
    if not arm.is_connected():
        logging.error(f"{RED}failed to connect to arm, exiting {RESET}")  # 红色打印
        exit(1)
    # end if

    # 初始化 ROS2 节点
    rclpy.init(args=None)
    arm_node = ArmNode(pub_gripper_msg=True)
    cam_node = CamNode(img_topic_list=[color_img_topic, depth_img_topic])
    tf_broadcaster = TransformBroadcaster(arm_node)

    # 创建键盘读取对象
    keyboard_reader = KeyboardReader()

    # 初始化夹爪对象
    gb = GripperBody(width=gripper_width, thickness=gripper_thickness)

    # 标定数据字典
    calib_data = {}
    if os.path.exists(gripper_path):
        calib_data = mmengine.load(gripper_path)
        logging.info(
            f"Loaded existing gripper calib data from: {GREEN}{gripper_path}{RESET}"
        )
    # end if

    if "width" in calib_data:
        saved_width = calib_data["width"]
        saved_thickness = calib_data["thickness"]
        if (
            abs(saved_width - gripper_width) > 1e-6
            or abs(saved_thickness - gripper_thickness) > 1e-6
        ):
            logging.warning(
                f"Loaded gripper_size [{saved_width}, {saved_thickness}] is different from ",
                f"current setting [{gripper_width}, {gripper_thickness}], use loaded values.",
            )
        # end if
    else:
        calib_data["width"] = gripper_width
        calib_data["thickness"] = gripper_thickness
    # end if

    if "T_cam_gripper" in calib_data:
        gb.T_cam_gripper = np.array(calib_data["T_cam_gripper"], dtype=np.float32)
        logging.info(f"Loaded T_cam_gripper: \n{GREEN}{gb.T_cam_gripper}{RESET}")
    else:
        calib_data["T_cam_gripper"] = np.eye(4).tolist()
        logging.warning(
            "No T_cam_gripper found in loaded calib_data, use identity matrix as default."
        )
    # end if

    print()
    print(
        f"use keyboard to control: \n{BLUE}"
        f"  q: 退出程序\n"
        f"  a: 调整末端姿态,使末端坐标系的 Z 轴指向下方\n"
        f"  t: 从当前 RGB-D 图像计算夹爪在相机坐标系下的位姿\n"
        f"  <: 缩小夹爪\n"
        f"  >: 放大夹爪\n"
        f"  s: 保存标定结果到文件\n"
        f"{RESET}"
    )

    while rclpy.ok():
        rclpy.spin_once(arm_node, timeout_sec=0.005)

        T_base_end = arm.get_pose()
        gripper_dist = arm.get_gripper_dist()

        T_base_cam = T_base_end @ T_end_cam
        ts = pose_to_transform_stamped(arm_node.frame_id, pc_frame_id, T_base_cam)
        ts.header.stamp = arm_node.get_clock().now().to_msg()
        tf_broadcaster.sendTransform(ts)

        arm_node.publish_pose(T_base_end)  # 发布机械臂状态

        arm_node.publish_grippers(
            gripper_body=gb,
            gripper_dist=gripper_dist,
            T_base_end=T_base_end,
            T_end_cam=T_end_cam,
        )  # 发布夹爪 Marker

        key = keyboard_reader.read_key()
        if key is None:
            continue
        # end if
        # logging.info(f'pressed: {key}')

        if key == "q":
            logging.info("exit by user command")
            break
        # end if

        # 调整末端,使末端坐标系的 Z 轴指向下方
        if key == "a":
            target_T_base_end = compute_axis_aligned_pose(
                T_base_end, base_axis_idx=-3, obj_axis_idx=3
            )
            if target_T_base_end is None:
                continue
            # end if

            logging.info(f"try move to new T_base_end:\n{target_T_base_end}")
            is_ok = arm.set_pose(target_T_base_end)
            if not is_ok:
                logging.warning("failed to move to new T_base_end")
            # end if
            print()
        # end if

        # 夹爪控制
        gripper_step = 0.001  # 夹爪电机每次旋转的步长
        if key == ",":  # 缩小夹爪
            set_dist = gripper_dist - gripper_step
            arm.set_gripper_dist(set_dist)
            logging.info(
                f"set gripper {gripper_dist:.3f} -->> {set_dist:.3f}, actual: {arm.get_gripper_dist():.3f}"
            )
        elif key == ".":  # 放大夹爪
            set_dist = gripper_dist + gripper_step
            arm.set_gripper_dist(set_dist)
            logging.info(
                f"set gripper {gripper_dist:.3f} -->> {set_dist:.3f}, actual: {arm.get_gripper_dist():.3f}"
            )
        # end if

        # 计算夹爪在相机坐标系下的位姿
        if key == "t":
            # 获取 RGB-D 图像
            frames = cam_node.get_frames(do_spin_once=True, frames_num=5)
            if frames is None:
                logging.warning("No RGB-D frame available yet.")
                continue
            # end if

            color_img = frames[0][0]
            depth_img_list = [frame[1] for frame in frames]
            depth_img = depth_mean_filter(depth_img_list)
            logging.info(
                f"rgb_img shape: {color_img.shape}, depth_img shape: {depth_img.shape}"
            )

            gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            corners3d = compute_corners3d(
                gray_img, depth_img, intrinsic, distortion, depth_scale
            )
            if corners3d is None:
                logging.error("failed to compute tag plane, skip this round")
                continue
            # end if

            gb.initialize(corners3d)

            calib_data["T_cam_gripper"] = gb.T_cam_gripper.tolist()
        # end if

        # 保存标定结果
        if key == "s":
            if "T_cam_gripper" not in calib_data:
                logging.error("no T_cam_gripper in calib_data, cannot save")
                continue
            # end if

            mmengine.dump(calib_data, gripper_path, indent=4)
            logging.info(
                f"Saved gripper calibration data to: {GREEN}{gripper_path}{RESET}"
            )
        # end if

    # end while

    arm_node.destroy_node()
    cam_node.destroy_node()
    rclpy.shutdown()
    logging.info("shutdown")

# end if __name__ == '__main__

"""
文件说明:
    读取图片以及对应的机械臂末端位姿数据,定位每张图像中标定板的位姿,再利用这些数据进行手眼标定,最终得到机械臂末端与相机之间的变换关系.
"""

import argparse
import glob
import json
import logging
import os
import sys

import apriltag2
import cv2
import numpy as np
import transforms3d


# 导入本工程的模块
# 导入本工程的模块
code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{code_dir}/../../../")
sys.path.append(root_dir)

from core.arm_wrapper import ArmWrapper
from core.utils import BLUE, GREEN, RED, RESET, YELLOW, inv_tf, read_cam_params
from typing import Dict, List, Tuple


######################################################### 函数定义 #########################################################


def calib_handeye(
    arm_pose_dict: Dict[int, np.ndarray],
    cam_pose_dict: Dict[int, np.ndarray],
    eye_in_hand: bool = True,
) -> np.ndarray:
    """
    手眼标定, 计算机械臂末端与相机之间的变换关系
    Args:
        arm_pose_dict (Dict[int,np.ndarray]): 机械臂末端位姿数据, key为图像id, value为机械臂末端位姿 T_base_end
        cam_pose_dict (Dict[int,np.ndarray]): 相机位姿数据, key为图像id, value为相机位姿 T_cam_board
        eye_in_hand (bool): 是否为眼在手标定, 默认值为 True

    Returns:
        np.ndarray: 机械臂末端与相机之间的变换关系(4x4矩阵), 如果标定失败则返回 None
    """

    def solve_axxb(T01s: np.ndarray, T23s: np.ndarray) -> np.ndarray:
        """
        求解 AX=XB 问题, 形式如下: 给定两组相同数量位姿 { T01 } 和 { T23 },
        对于任意两对位姿 <T01_i , T23_i> 和 <T01_j , T23_j> 都满足等式: T01_i * T12 * T23_i = T01_j * T12 * T23_j ,
        即: (T01_j^-1 * T01_i) * T12 = T12 * (T23_i * T23_j^-1), 目标是求解常量 T12
        Args:
            T01s (np.ndarray): 从坐标系 1-->0 的位姿变化矩阵, shape为(N, 4, 4)
            T23s (np.ndarray): 从坐标系 3-->2 的位姿变化矩阵, shape为(N, 4, 4)

        Returns:
            T12 (np.ndarray): 从坐标系 2-->1 的变换矩阵, shape为(4, 4), 如果标定失败则返回 None
        """

        R, t = cv2.calibrateHandEye(
            T01s[:, :3, :3],
            T01s[:, :3, 3],
            T23s[:, :3, :3],
            T23s[:, :3, 3],
            method=cv2.CALIB_HAND_EYE_PARK,  # 这个方法最稳定
        )

        T12 = np.eye(4, dtype=np.float64)
        T12[:3, :3] = R
        T12[:3, 3] = t.ravel()

        return T12

    # end def solve_axxb

    def compute_error(
        T01s: np.ndarray, T23s: np.ndarray, T12: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算标定结果的误差:
        计算所有位姿对的平均旋转误差(度)和平均平移误差(毫米).
        对每对 (i, j), 先计算 T03_i = T01_i * T12 * T23_i 和 T03_j = T01_j * T12 * T23_j,
        再计算 dT = T03_j * inv(T03_i), 其中旋转误差 = angle(dT.R), 平移误差 = ||dT.t||.

        Args:
            T01s (np.ndarray): 从坐标系 1-->0 的位姿变化矩阵, shape为(N, 4, 4)
            T23s (np.ndarray): 从坐标系 3-->2 的位姿变化矩阵, shape为(N, 4, 4)
            T12 (np.ndarray): 从坐标系 2-->1 的变换矩阵, shape为(4, 4)

        Returns:
            Tuple[float, float]: (平均旋转误差(度), 平均平移误差(毫米))
        """

        assert T01s.shape[0] == T23s.shape[0], (
            f"T01s and T23s must have the same number of poses, got {T01s.shape[0]} vs {T23s.shape[0]}"
        )

        N = T01s.shape[0]
        if N < 2:
            return 0.0, 0.0
        # end if

        # T03_i = T01_i * T12 * T23_i,  shape: (N, 4, 4)
        T03s = T01s @ T12 @ T23s

        sum_delta_angle = 0.0
        sum_delta_position = 0.0
        count = 0

        for i in range(N):
            for j in range(i + 1, N):
                # dT = T03_j * inv(T03_i)
                dT = T03s[j] @ np.linalg.inv(T03s[i])

                # 旋转误差: 从旋转矩阵提取旋转角
                R = dT[:3, :3]
                cos_angle = (np.trace(R) - 1.0) / 2.0
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                delta_angle = np.arccos(cos_angle)

                # 平移误差: 平移向量的模
                delta_position = np.linalg.norm(dT[:3, 3])

                sum_delta_angle += delta_angle
                sum_delta_position += delta_position
                count += 1
            # end for
        # end for

        error_r = sum_delta_angle / count * 180.0 / np.pi  # 弧度 -> 度
        error_p = sum_delta_position / count * 1000.0  # 米 -> 毫米

        return error_r, error_p

    # end def compute_error

    arm_pose_list = []  # T_base_end
    cam_pose_list = []  # T_cam_board
    for img_id in arm_pose_dict:
        if img_id in cam_pose_dict:
            arm_pose_list.append(arm_pose_dict[img_id])
            cam_pose_list.append(cam_pose_dict[img_id])
        else:
            logging.warning(
                f"{YELLOW}No camera pose for image {img_id}, skipping.{RESET}"
            )
        # end if
    # end for

    if len(arm_pose_list) < 4:
        logging.error(
            f"{RED}Not enough valid data for hand-eye calibration. Need at least 4 pairs of poses, but got {len(arm_pose_list)}. Exiting.{RESET}"
        )
        return None
    # end if

    if eye_in_hand:
        logging.info(
            f"Using {GREEN}{len(arm_pose_list)}{RESET} valid pairs of arm and camera poses for calibration."
        )
        T01s = np.array(arm_pose_list)  # T_base_end
        T23s = np.array(cam_pose_list)  # T_cam_board
        T_end_cam = solve_axxb(T01s, T23s)
        err_r, err_p = compute_error(T01s, T23s, T_end_cam)

        logging.info(f"Calibration result (T_end_cam):\n{GREEN}{T_end_cam}{RESET}")
        logging.info(
            f"Calibration error: rotation(deg) = {GREEN}{err_r:.4f}{RESET}, translation(mm) = {GREEN}{err_p:.4f}{RESET}"
        )

        return T_end_cam
    else:
        T01s = np.zeros((len(arm_pose_list), 4, 4), dtype=np.float64)
        T23s = np.zeros((len(arm_pose_list), 4, 4), dtype=np.float64)
        for i in range(len(arm_pose_list)):
            T01s[i] = inv_tf(arm_pose_list[i])  # T_end_base
            T23s[i] = inv_tf(cam_pose_list[i])  # T_board_cam
        # end for
        T_base_cam = solve_axxb(T01s, T23s)
        err_r, err_p = compute_error(T01s, T23s, T_base_cam)

        logging.info(f"Calibration result (T_base_cam):\n{GREEN}{T_base_cam}{RESET}")
        logging.info(
            f"Calibration error: rotation(deg) = {GREEN}{err_r:.4f}{RESET}, translation(mm) = {GREEN}{err_p:.4f}{RESET}"
        )

        return T_base_cam
    # end if


# end def calib_handeye


######################################################### 主函数 #########################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手眼标定")

    parser.add_argument(
        "--cam_param_path",
        type=str,
        default="/home/i4/桌面/test_location/calib/collect_image/cam_params.json",
        help="相机参数文件的路径",
    )

    parser.add_argument(
        "--calib_board_info",
        type=str,
        default="[0.0245,0.0075 , 6, 6]",
        help="标定板信息 [tag_size, space_size, tag_rows, tag_cols]",
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        default="/home/i4/桌面/test_location/calib/collect_image_handeye/cam0",
        help="保存图像的目录",
    )

    parser.add_argument(
        "--arm_pose_path",
        type=str,
        default="/home/i4/桌面/test_location/calib/collect_image_handeye/arm_pose.json",
        help="机械臂末端位姿文件的路径",
    )

    args = parser.parse_args()

    cam_param_path = args.cam_param_path
    calib_board_info = json.loads(args.calib_board_info)
    img_dir = args.img_dir
    arm_pose_path = args.arm_pose_path

    print()
    print(f"相机参数文件路径: {BLUE}{cam_param_path}{RESET}")
    print(f"标定板信息: {BLUE}{calib_board_info}{RESET}")
    print(f"图像目录: {BLUE}{img_dir}{RESET}")
    print(f"机械臂末端位姿文件路径: {BLUE}{arm_pose_path}{RESET}")
    print()

    # 读取相机参数
    intrinsic, distortion = read_cam_params(cam_param_path)
    if intrinsic is None:
        logging.error("Failed to read camera parameters. Exiting.")
        sys.exit(1)
    # end if
    K = np.array(
        [[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]]
    )
    D = np.array(distortion) if distortion is not None else None

    # 读取机械臂末端位姿数据
    pose_dict = json.load(open(arm_pose_path, "r"))
    if len(pose_dict) == 0:
        logging.error(
            f"{RED}No arm pose data found in {arm_pose_path}. Exiting.{RESET}"
        )
        sys.exit(1)
    # end if
    eye_in_hand = pose_dict.get("eye_in_hand", True)  # 默认值为 True

    # 创建标定板
    tag3d_list = apriltag2.create_calib_board_3d(
        tag_size=calib_board_info[0],
        space_size=calib_board_info[1],
        rows=calib_board_info[2],
        cols=calib_board_info[3],
        start_tag_id=0,
    )

    # 创建检测器
    detector = apriltag2.Detector(
        tag_family="tag36h11",
        black_border=2,
    )

    # 遍历每张图像, 定位标定板, 计算手眼标定所需的数据
    arm_pose_dict = {}
    cam_pose_dict = {}

    img_path_list = glob.glob(f"{img_dir}/*.png")
    img_path_list.sort()
    if len(img_path_list) == 0:
        logging.error(f"{RED}No images found in {img_dir}. Exiting.{RESET}")
        sys.exit(1)
    # end if

    for img_path in img_path_list:
        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]

        logging.info(f"Processing image: {BLUE}{img_id}{RESET}")

        if f"{img_id}" not in pose_dict:
            logging.warning(
                f"{YELLOW}No arm pose data for image {img_name}, skipping.{RESET}"
            )
            continue
        # end if
        arm_pose = pose_dict[f"{img_id}"]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(
                f"{YELLOW}Failed to read image {img_name}, skipping.{RESET}"
            )
            continue
        # end if

        tag2d_list = detector.detect(img, -1)

        # if True:
        #     bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     detector.draw(bgr_img, tag2d_list)
        #     cv2.imshow("tag detection", bgr_img)
        #     cv2.waitKey(500)
        # # end if

        cam_pose = apriltag2.locate_calib_board(
            tag2d_list=tag2d_list, tag3d_list=tag3d_list, K=K, D=D
        )
        if cam_pose is None:
            logging.warning(
                f"{YELLOW}Failed to locate calibration board in image {img_name}, skipping.{RESET}"
            )
            continue
        # end if

        cam_pose_dict[img_id] = cam_pose
        arm_pose_dict[img_id] = ArmWrapper.array_to_matrix(arm_pose)

        print()
    # end for

    # 执行手眼标定
    T_arm_cam = calib_handeye(arm_pose_dict, cam_pose_dict, eye_in_hand=eye_in_hand)
    if T_arm_cam is None:
        logging.error(f"{RED}Hand-eye calibration failed. Exiting.{RESET}")
        sys.exit(1)
    # end if

    # 保存标定结果
    save_path = os.path.join(os.path.dirname(arm_pose_path), "calib_handeye.json")
    result_dict = {"QuaternionFormat": "qw,qx,qy,qz"}
    R = T_arm_cam[:3, :3]
    t = T_arm_cam[:3, 3]
    q = transforms3d.quaternions.mat2quat(R).tolist()  # qw,qx,qy,qz
    if eye_in_hand:
        result_dict["T_armend_cam"] = {"t": t.tolist(), "q": q, "R": R.tolist()}
    else:
        result_dict["T_armbase_cam"] = {"t": t.tolist(), "q": q, "R": R.tolist()}
    # end if

    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    # end if

    logging.info(f"Result saved to: {GREEN}{save_path}{RESET}")

# end if __name__ == '__main__':

"""
功能: 创建基于 AprilTag2 的 2D 抓取模板数据, 包含物体在图像上的位置和朝向信息, 以及机械臂末端位姿和夹爪距离等状态信息
模板需要保存以下数据:
- 1.末端刚好抓取到物体时的状态, 包含机械臂末端位姿和夹爪距离等信息
- 2.相机距离物体较近时的状态, 包含机械臂末端位姿、夹爪距离、以及物体在图像上的位置和朝向等信息
- 3.相机距离物体较近时的状态( 在 2 的基础上, 末端在 xy 平面平移一小段距离得到 ), 包含机械臂末端位姿、夹爪距离、以及物体在图像上的位置和朝向等信息
- 4.相机距离物体较远时的状态, 包含机械臂末端位姿、夹爪距离、以及物体在图像上的位置和朝向等信息
- 5.相机距离物体较远时的状态( 在 4 的基础上, 末端在 xy 平面平移一小段距离得到 ), 包含机械臂末端位姿、夹爪距离、以及物体在图像上的位置和朝向等信息

"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import rclpy



# 导入本工程的模块
# 导入本工程的模块
code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{code_dir}/../../../")
sys.path.append(root_dir)

from core.arm_utils import compute_axis_aligned_pose
from core.arm_wrapper import ArmWrapper
from core.cam_ros_utils import CamNode
from core.utils import BLUE, GREEN, RED, RESET, YELLOW, KeyboardReader, read_cam_params
from core.vision_utils import TagMatcher2D

######################################################### 全局变量 #########################################################


######################################################### 函数定义 #########################################################


def save_state(
    cam_node: CamNode, arm: ArmWrapper, save_dir: str, matcher: TagMatcher2D = None
):
    """
    保存当前状态为模板文件
    Args:
        cam_node (CamNode): 相机 ROS2 节点
        arm (ArmWrapper): 机械臂对象
        save_dir (str): 保存路径
        matcher (TagMatcher2D): 可选的 2D 匹配器对象, 若提供则会检测物体位置并保存到模板中
    """

    # 获取图像
    imgs = cam_node.get_frames(do_spin_once=True)
    if imgs is None:
        logging.error(f"{RED}failed to get images, skip saving tmpl{RESET}")  # noqa: LOG015
        return
    # end if
    rgb_img = imgs[0][0]  # 取第一帧第一摄像头图像

    # 获取机械臂末端位姿
    T_base_end = arm.get_pose()

    # 获取夹爪距离
    gripper_dist = arm.get_gripper_dist()

    data_dict = {"T_base_end": T_base_end.tolist(), "gripper_dist": gripper_dist}

    if matcher is not None:  # 获取物体位置
        result_list, msg = matcher.match(rgb_img, top_k=1)
        if len(result_list) == 0:
            logging.warning(f"{YELLOW}no match found, skip saving tmpl, {msg}{RESET}")  # noqa: LOG015
            return
        # end if

        pose_2d = result_list[0].pose_2d
        data_dict["obj_pose_2d"] = [float(x) for x in pose_2d]

        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        matcher.draw(bgr_img, result_list)  # 绘制匹配结果
        img_path = os.path.join(save_dir, "tag.png")
        cv2.imwrite(img_path, bgr_img)
    # end if

    # 保存模板数据
    img_path = os.path.join(save_dir, "color.png")
    cv2.imwrite(img_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    logging.info(f"saved image to: {GREEN}{img_path}{RESET}")  # noqa: LOG015

    file_path = os.path.join(save_dir, f"state.json")
    with open(file_path, "w") as f:
        json.dump(data_dict, f, indent=4)
    logging.info(f"saved state to: {GREEN}{file_path}{RESET}")  # noqa: LOG015


# end def save_state


######################################################### 主函数 #########################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cam_params_path",
        type=str,
        required=True,
        help="相机参数文件的路径, 包含内参和畸变参数",
    )

    parser.add_argument(
        "--color_img_topic", type=str, required=True, help="RGB 图像的 ROS2 话题名称"
    )

    parser.add_argument("--tmpl_dir", type=str, required=True, help="模板文件的目录")

    args = parser.parse_args()

    cam_params_path = args.cam_params_path

    color_img_topic = args.color_img_topic
    if color_img_topic is None:
        logging.error("Error: color_img_topic is not provided.")  # noqa: LOG015
        exit(0)
    # end if

    tmpl_dir = args.tmpl_dir
    if tmpl_dir is None:
        logging.error("no tmpl_dir specified, exiting")
        exit(1)
    # end if

    print()
    print(f"camera parameters file: {BLUE}{cam_params_path}{RESET}")
    print(f"color image topic: {BLUE}{color_img_topic}{RESET}")
    print(f"grasp_2d template will be saved to: {GREEN}{tmpl_dir}{RESET}")
    print()

    # 读取相机参数
    intrinsic, distortion = read_cam_params(cam_params_path)

    config = TagMatcher2D.Config(
        intrinsic=intrinsic,
        distortion=distortion,
    )
    matcher = TagMatcher2D(config)

    tmpl_dir = os.path.normpath(tmpl_dir)  # 规范化路径
    os.makedirs(tmpl_dir, exist_ok=True)

    # 创建机械臂对象
    arm = ArmWrapper()
    if not arm.is_connected():
        logging.error(f"{RED}failed to connect to arm, exiting {RESET}")
        exit(1)
    # end if

    # 初始化 ROS2 节点
    rclpy.init(args=None)
    cam_node = CamNode([color_img_topic])

    # 创建键盘读取对象
    keyboard_reader = KeyboardReader()

    print()
    print(
        f"use keyboard to control: \n{BLUE}"
        f"  q: 退出程序\n"
        f"  a: 使末端的 z 轴方向与基座的 -z 轴平行\n"
        f"  <: 缩小夹爪之间的距离\n"
        f"  >: 放大夹爪之间的距离\n"
        f"  g: 保存抓取时的状态\n"
        f"  n: 保存相机距离物体较近时的状态\n"
        f"  b: 保存下一个相机距离物体较近时的状态\n"
        f"  f: 保存相机距离物体较远时的状态\n"
        f"  d: 保存下一个相机距离物体较远时的状态\n"
        f"{RESET}"
    )

    while rclpy.ok():
        key = keyboard_reader.read_key()
        if key is None:
            time.sleep(0.03)
            continue
        # end if
        # logging.info(f'pressed: {key}')
        print()

        if key == "q":  # 退出
            logging.info("exiting...")
            break
        # end if

        if key == "a":  # 调整末端的 z 轴方向, 使它与基座的 z 轴平行
            logging.info(f"{BLUE}调整末端的 z 轴方向 ...{RESET}")

            T_base_end = arm.get_pose()
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
        gripper_step = 0.001  # 夹爪每次移动的步长
        gripper_dist = arm.get_gripper_dist()
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

        if key == "g":  # 保存抓取时的状态
            logging.info(f"{BLUE}开始保存抓取时的状态 ...{RESET}")

            save_dir = os.path.join(tmpl_dir, "grasp")
            os.makedirs(save_dir, exist_ok=True)

            save_state(cam_node, arm, save_dir)
        # end if

        if key == "n":  # 保存相机距离物体较近时的状态
            logging.info(f"{BLUE}开始保存相机距离物体较近时的状态 ...{RESET}")

            save_dir = os.path.join(tmpl_dir, "near")
            os.makedirs(save_dir, exist_ok=True)

            save_state(cam_node, arm, save_dir, matcher=matcher)
        # end if

        if (
            key == "b"
        ):  # 保存相机距离物体较近时的状态, 即在当前位置的基础上在XY平面上移动一小段距离
            logging.info(f"{BLUE}开始保存下一个相机距离物体较近时的状态 ...{RESET}")

            save_dir = os.path.join(tmpl_dir, "next_near")
            os.makedirs(save_dir, exist_ok=True)

            save_state(cam_node, arm, save_dir, matcher=matcher)
        # end if

        if key == "f":  # 保存相机距离物体较远时的状态
            logging.info(f"{BLUE}开始保存相机距离物体较远时的状态 ...{RESET}")

            save_dir = os.path.join(tmpl_dir, "far")
            os.makedirs(save_dir, exist_ok=True)

            save_state(cam_node, arm, save_dir, matcher=matcher)
        # end if

        if (
            key == "d"
        ):  # 保存相机距离物体较远时的状态, 即在当前位置的基础上在XY平面上移动一小段距离
            logging.info(f"{BLUE}开始保存下一个相机距离物体较远时的状态 ...{RESET}")

            save_dir = os.path.join(tmpl_dir, "next_far")
            os.makedirs(save_dir, exist_ok=True)

            save_state(cam_node, arm, save_dir, matcher=matcher)
        # end if
    # end while
# end if __name__ == '__main__'

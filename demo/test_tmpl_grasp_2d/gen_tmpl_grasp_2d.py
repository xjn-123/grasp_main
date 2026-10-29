"""
功能: 创建基于 **pyapriltags**(AprilTag 3) 的 2D 抓取模板数据, 包含物体在图像上的位置和朝向信息, 以及机械臂末端位姿和夹爪距离等状态信息
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

root_dir = os.path.normpath(f"{os.path.dirname(os.path.realpath(__file__))}/../../../")
sys.path.append(root_dir)

# 导入本工程的模块
from core.camera_ros_utils.camera_ros_utils import CameraNode
from core.common_utils.utils import Common_data, KeyboardReader, read_camera_params
from core.gripper_utils.gripper_utils import com_axis_aligned_pose
from core.robot_helper.robot_helper import ArmHelper
from core.vision_utils.vision_utils import TagMatcher2D

Logger = logging.getLogger(__name__)


def save_state(
    cam_node: CameraNode, arm: ArmHelper, save_dir: str, matcher: TagMatcher2D = None
):
    """
    保存当前状态为模板文件
    Args:
        cam_node (CamNode): 相机 ROS2 节点
        arm (ArmHelper): 机械臂对象
        save_dir (str): 保存路径
        matcher (TagMatcher2D): 可选的 2D 匹配器对象, 若提供则会检测物体位置并保存到模板中
    保存的数据:
        data_dict = {"T_base_end": T_base_end.tolist(), 
        "gripper_dist": gripper_dist,
        可选"T_cam_model": T_cam_model.tolist()}

        color.png: 相机拍摄的彩色图像

    """

    # 获取图像
    imgs = cam_node.get_frames(do_spin_once=True)
    if imgs is None:
        Logger.error(f"{Common_data.RED}获取图像失败{Common_data.RESET}")  # noqa: LOG015
        return

    img=imgs[0][0]  # 取第一帧第一摄像头图像

    # 获取机械臂末端位姿矩阵
    T_base_end = arm.get_cart_matrix()

    # 获取夹爪距离
    gripper_dist = arm.get_gripper_dist()

    data_dict = {"T_base_end": T_base_end.tolist(), "gripper_dist": gripper_dist}

    if matcher is not None: 
        result_list = matcher.match(img, top_k=1)
        if len(result_list) == 0:
            Logger.warning(f"{Common_data.YELLOW}未匹配到apriltag{Common_data.RESET}")
            return

        pose_2d = result_list[0].pose_2d
        data_dict["obj_pose_2d"] = [float(x) for x in pose_2d]

        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_bgr_img = bgr_img.copy()#用于保存的bgr图像

        matcher.draw(bgr_img, result_list)#绘制匹配结果
        save_tag_dir = os.path.join(save_dir, "tag.png")
        save_color_dir = os.path.join(save_dir, "color.png")
        save_data_dir = os.path.join(save_dir, "state.json")

        cv2.imwrite(save_tag_dir, bgr_img)
        cv2.imwrite(save_color_dir, save_bgr_img)

        with open(save_data_dir, "w") as f:
            json.dump(data_dict, f, indent=4)

        Logger.info(f"{Common_data.GREEN}保存模板到{save_dir}{Common_data.RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cam_param_path", require=True,type=str,help="相机参数文件路径")
    parser.add_argument("--color_img_topic", require=True,type=str,help="彩色图像话题名称")
    parser.add_argument("--save_dir", require=True,type=str,help="保存模板文件的目录")

    args = parser.parse_args()

    if args.cam_param_path is None or args.color_img_topic is None or args.save_dir is None:  # noqa: PLR5501
        print(f"{Common_data.RED}参数错误！{Common_data.RESET}")
        sys.exit(1)

    #加载参数
    cam_params_path = args.cam_param_path
    intrinsic, distortion, depth_scale = read_camera_params(cam_params_path)
    color_img_topic = args.color_img_topic
    save_dir = args.save_dir
    #规范化路径
    save_dir = os.path.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"{Common_data.GREEN}相机参数:\nintrinsic:{intrinsic}\ndistortion{distortion}\ndepth_scale{depth_scale}{Common_data.RESET}")
    print(f"{Common_data.GREEN}保存模板文件目录:{save_dir}{Common_data.RESET}")
    print(f"{Common_data.GREEN}图像话题名{color_img_topic}{Common_data.RESET}")

    #创建匹配器对象
    config=TagMatcher2D.Config(intrinsic=intrinsic, distortion=distortion)
    matcher = TagMatcher2D(config)

    #创建机械臂对象
    arm = ArmHelper()
    if not arm.is_connected():
        Logger.error(f"{Common_data.RED}机械臂连接失败！{Common_data.RESET}")
        sys.exit(1)

    #创建相机节点
    rclpy.init(args=None)
    cam_node = CameraNode([color_img_topic])

    #创建键盘读取器对象
    key_reader = KeyboardReader()

    print()
    print(
        f"use keyboard to control: \n{Common_data.BLUE}"
        f"  q: 退出程序\n"
        f"  a: 使末端的 z 轴方向与基座的 -z 轴平行\n"
        f"  <: 缩小夹爪之间的距离\n"
        f"  >: 放大夹爪之间的距离\n"
        f"  g: 保存抓取时的状态\n"
        f"  n: 保存相机距离物体较近时的状态\n"
        f"  b: 保存下一个相机距离物体较近时的状态\n"
        f"  f: 保存相机距离物体较远时的状态\n"
        f"  d: 保存下一个相机距离物体较远时的状态\n"
        f"{Common_data.RESET}"
    )

    while rclpy.ok():
        key=key_reader.read_key()

        if key == "q": 
            Logger.info(f"{Common_data.YELLOW}退出程序{Common_data.RESET}") # 退出程序
            break

        if key == "a":
            '使末端的 z 轴方向与基座的 -z 轴平行'
            current_T_base_end = arm.get_cart_matrix()
            target_T_base_end = com_axis_aligned_pose(current_T_base_end, base_axis_idx=-3, obj_axis_idx=3)
            if target_T_base_end is None:
                Logger.error(f"{Common_data.RED}目标位姿计算失败{Common_data.RESET}")
                continue
            is_ok=arm.set_pose(target_T_base_end)
            if not is_ok:
                Logger.error(f"{Common_data.RED}位姿调整失败{Common_data.RESET}")

        # 夹爪控制
        gripper_step = 0.001  # 夹爪每次移动的步长
        gripper_dist = arm.get_gripper_dist()

        if key == ",":  # 缩小夹爪
            set_dist = gripper_dist - gripper_step
            arm.set_gripper_dist(set_dist)
            Logger.info(
                f"{Common_data.BLUE}调整夹爪 {gripper_dist:.3f} -->> {set_dist:.3f}, 已调整夹爪至: {arm.get_gripper_dist():.3f}{Common_data.RESET}"
            )
        elif key == ".":  # 放大夹爪
            set_dist = gripper_dist + gripper_step
            arm.set_gripper_dist(set_dist)
            Logger.info(
                f"{Common_data.BLUE}调整夹爪 {gripper_dist:.3f} -->> {set_dist:.3f}, 已调整夹爪至: {arm.get_gripper_dist():.3f}{Common_data.RESET}"
            )

        if key == "g":  # 保存抓取时的状态
            Logger.info(f"{Common_data.BLUE}开始保存抓取时的状态{Common_data.RESET}")

            save_dir=os.path.join(save_dir, "grasp")
            os.makedirs(save_dir, exist_ok=True)

            save_state(cam_node, arm, save_dir, matcher)

        if key == "g":  # 保存近距离时的状态
            Logger.info(f"{Common_data.BLUE}开始近距离时的状态{Common_data.RESET}")

            save_dir=os.path.join(save_dir, "near")
            os.makedirs(save_dir, exist_ok=True)
            
            save_state(cam_node, arm, save_dir, matcher)

        if key == "b":  # 保存下一个近距离时的状态
            Logger.info(f"{Common_data.BLUE}开始保存下一个近距离时的状态{Common_data.RESET}")

            save_dir=os.path.join(save_dir, "next_near")
            os.makedirs(save_dir, exist_ok=True)

            save_state(cam_node, arm, save_dir, matcher)

        if key == "f":  # 保存距离物体较远时的状态
            Logger.info(f"{Common_data.BLUE}开始保存距离物体较远时的状态{Common_data.RESET}")

            save_dir=os.path.join(save_dir, "far")
            os.makedirs(save_dir, exist_ok=True)

            save_state(cam_node, arm, save_dir, matcher)
        #
        if key == "d":  # 保存下一个距离物体较远时的状态
            Logger.info(f"{Common_data.BLUE}开始保存下一个距离物体较远时的状态{Common_data.RESET}")

            save_dir=os.path.join(save_dir, "next_far")
            os.makedirs(save_dir, exist_ok=True)

            save_state(cam_node, arm, save_dir, matcher)


        

        


    







    




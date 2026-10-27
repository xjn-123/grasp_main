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
from core.robot_helper.robot_helper import ArmHelper
from core.common_utils.utils import Common_data
from core.camera_ros_utils.camera_ros_utils import CameraNode
from core.vision_utils.vision_utils import TagMatcher2D

Logger = logging.getLogger(__name__)


def save_sate(
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

    # 获取机械臂末端位姿
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






    


    
    # end if


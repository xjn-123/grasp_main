"""
功能: 测试基于 AprilTag2 的 2D 抓取模板数据的使用
适用条件:
    1. 物体表面贴有 AprilTag, 放置在平面上, 且平面与机械臂基座的XOY平面平行, 物体在平面上的自由度为 3 (x,y,theta), 其中 theta 为物体在平面上的朝向角
    2. 机械臂末端安装有相机, 且相机的 z 轴与末端的 z 轴夹角为小于 45 度
    3. 机械臂执行抓取任务时, 保持末端的 z 轴与平面法向方向平行, 且末端的 z 轴始终指向平面
实现思路:
    1. 多线程架构:
       - 主线程: 运行 rclpy.spin_once 循环( 配合 stop_event ), 持续处理相机订阅回调,
         确保 frame_callback 实时触发, 始终缓存最新图像帧到 cam_node.imgs 中.
       - 子线程: 执行 run() 抓取主循环, 调用 do_grasp() 完成 4 步 2D 视觉伺服抓取流程.
    2. 帧获取机制:
       - CamNode 仅订阅单个 RGB 话题, ApproximateTimeSynchronizer 单输入直通.
       - get_frames(do_spin_once=False) 在主线程 spin 的前提下, 仅轮询 self.stamp 等待新帧,
         不再自行调用 spin_once, 避免与主线程 spin 冲突.
    3. 模板数据结构( read_tmpl_grasp_2d ):
       - grasp: 最终抓取位姿与夹爪开度.
       - near / next_near: 近距离下两个相邻时刻的末端位姿和物体 2D 位姿, 用于计算近距离 Jacobian 比值.
       - far / next_far: 远距离下两个相邻时刻的末端位姿和物体 2D 位姿, 用于计算远距离 Jacobian 比值.
    4. 视觉伺服控制( compute_delta_end_pose ):
       - 引入"虚拟相机"概念: 与真实相机同原点, 但坐标轴与末端对齐, 内参为单位矩阵( 归一化坐标 ).
       - 将 near/far 模板和当前观测的 2D 位姿统一变换到虚拟相机归一化坐标系.
       - 根据当前 Z 高度在 near_z 和 far_z 之间线性插值, 计算目标 Jacobian 比值(target_ratio)和目标归一化坐标(target_uv).
       - 旋转校正: 计算当前观测与 near 模板的角度差 delta_theta, 构建绕 Z 轴的旋转矩阵.
       - 平移校正: 当前归一化坐标与目标归一化坐标的偏差 x target_ratio → 末端平移增量.
       - Z 轴步进限制: 每次最多移动 z_step=0.05m, 避免过冲.
    5. 抓取流程( do_grasp ):
       Step 1: 检测 AprilTag → 获取物体 2D 位姿 [nx, ny, theta].
       Step 2: 计算末端位姿增量( compute_delta_end_pose ) → 移动机械臂.
       Step 3: 重复 Step 1-2, 直到增量足够小( 平移<1mm 且旋转<2° )或达到最大尝试次数(20次).
       Step 4: 末端相对于 near 位姿执行 final_T_end 增量到达抓取位姿 → 闭合夹爪 → 抬高.
    6. 放置流程:
       - 直接移动到放置位姿 → 打开夹爪释放物体.
    7. 退出清理:
       - Ctrl+C 或子线程正常结束 → 主线程退出 spin 循环 → finally 块依次执行:
         机械臂回零位 → 等待子线程 → 恢复速度并断连 → 销毁节点 → 关闭 rclpy.

"""


import json
import logging
import os
import sys
import threading
import time

import numpy as np
import rclpy

root_dir = os.path.normpath(os.path.dirname(os.path.realpath(__file__)), "/../../")
sys.path.append(root_dir)

from core.camera_ros_utils.camera_ros_utils import CameraNode
from core.common_utils.utils import (
    Common_data,
    inv_tf,
    read_calib_params,
    read_camera_params,
    wait_key,
)
from core.gripper_utils.gripper_utils import com_axis_aligned_pose
from core.robot_helper.robot_helper import ArmHelper
from core.vision_utils.vision_utils import TagMatcher2D



def read_tmpl_grasp_2d(tmpl_dir: str) -> Dict:
    




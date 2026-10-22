"""
相机相关的 ROS2 工具函数和类
"""

import logging

from typing_extensions import List, Tuple, Dict
import time
import platform

import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs.msg

# 导入本工程的模块
from .utils import (
    GREEN, YELLOW, BLUE, RED, RESET
)


######################################################## 类定义 ########################################################

class CamNode(Node):
    """
    ROS2 节点, 用于:
        - 1. 同步接收一帧( 如 RGB-D/Stereo )中的多张图像     
        - 2. 获取相机内参
    """

    def __init__(self,
                 img_topic_list: List[str],
                 cam_info_topic_list: List[str] = None,
                 reliability: int = 1):
        """
        初始化   
        Args:
            img_topic_list (List[str]): 图像话题列表
            cam_info_topic_list (List[str], optional): 相机信息话题列表. 默认值为 None
            reliability (int, optional): 消息可靠性. 0: SYSTEM_DEFAULT, 1: RELIABLE, 2: BEST_EFFORT. 默认值为 1
        """

        super().__init__('cam_node')  # 初始化节点名称

        assert len(img_topic_list) > 0, "img_topic_list must contain at least one topic."

        if reliability == 0:
            reliability_policy = ReliabilityPolicy.SYSTEM_DEFAULT
        elif reliability == 1:
            reliability_policy = ReliabilityPolicy.RELIABLE
        elif reliability == 2:
            reliability_policy = ReliabilityPolicy.BEST_EFFORT
        else:
            raise ValueError("Invalid reliability value. Must be 0 (SYSTEM_DEFAULT), 1 (RELIABLE), or 2 (BEST_EFFORT).")
        # end if

        qos = QoSProfile(
            reliability=reliability_policy,  # RELIABLE / BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST,            # 保留最后几条消息
            depth=1                                     # 队列深度
        )

        # 创建订阅者
        self.bridge = CvBridge()  # 创建 CvBridge 实例
        self.img_sub_list = []
        for img_topic in img_topic_list:
            sub = Subscriber(self, sensor_msgs.msg.Image, img_topic, qos_profile=qos)
            self.img_sub_list.append(sub)
        # end for

        # 创建时间同步器( 允许 50 ms 的时间差 )
        self.sync = ApproximateTimeSynchronizer(
            self.img_sub_list,
            queue_size=1,
            slop=0.05  # 50ms 容差
        )
        self.sync.registerCallback(self.frame_callback)

        # 初始化图像缓存
        self.enable_receive_frame = False
        self.imgs = [None] * len(img_topic_list)
        self.stamp = None

        # cam_info_topic_list 的处理
        self.cam_info_sub_list = []
        if cam_info_topic_list is not None:
            for cam_info_topic in cam_info_topic_list:
                sub = Subscriber(self, sensor_msgs.msg.CameraInfo, cam_info_topic)
                self.cam_info_sub_list.append(sub)
            # end for

            # 创建时间同步器
            self.cam_info_sync = ApproximateTimeSynchronizer(
                self.cam_info_sub_list,
                queue_size=1,
                slop=0.1
            )
            self.cam_info_sync.registerCallback(self.cam_info_callback)

            self.cam_infos = [None] * len(self.cam_info_sub_list)  # 初始化相机信息缓存
        # end if

        self.callback_cnt = 0            # 回调计数器, 用于调试
        self.callback_first_time = None  # 首次触发的回调时间, 用于调试
        self.callback_duration = 0.0     # 回调持续时间, 用于调试

        logging.info(f"platform: {GREEN}{platform.machine()}{RESET}")
        logging.info(f'{GREEN}CamNode initialized.{RESET}')
    # end def __init__

    def frame_callback(self, *img_msgs: sensor_msgs.msg.Image):
        """
        同步的图像回调
        """

        # 只在非 x64 架构下打印调试信息
        if platform.machine() != 'x86_64':

            if self.callback_first_time is None:
                self.callback_first_time = time.time()
            # end if

            duration = time.time() - self.callback_first_time
            self.callback_cnt += 1

            if int(duration) % 5 == 0 and int(duration) != int(self.callback_duration):  # 每隔 5 秒打印一次日志
                logging.info(f"frame_callback triggered, count: {self.callback_cnt}, duration: {duration:.2f} s")
            # end if

            self.callback_duration = duration
        # end if

        if not self.enable_receive_frame:
            return
        # end if

        self.imgs = [self.bridge.imgmsg_to_cv2(img_msg, desired_encoding=img_msg.encoding) for img_msg in img_msgs]
        self.stamp = img_msgs[0].header.stamp

        # logging.info(f"Received synchronized images at {self.stamp.sec}.{self.stamp.nanosec}")

    # end def frame_callback

    def get_frames(self,
                   frames_num: int = 1,
                   timeout_sec: float = 5.0,
                   do_spin_once: bool = False) -> List[List[np.ndarray]]:
        """
        获取多帧   
        Args:
            frames_num (int): 需要获取的帧的数量,每一帧可能包含多张图像
            timeout_sec (float): 超时时间( 单位: s )
            do_spin_once (bool): 是否在函数内部执行 rclpy.spin_once. 如果调用该函数前已经在外部执行了 rclpy.spin 或者 rclpy.spin_once, 
                                 则可以将该参数设置为 False 以避免重复调用
        Returns:
            (List[List[np.ndarray]]): 帧列表,如果失败则返回 None. 每一帧是一个包含多张图像的列表, 图像的顺序与 img_topic_list 中话题的顺序一致
        """

        self.enable_receive_frame = True
        self.stamp = None
        imgs_list = []  # 帧列表
        st = time.time()
        while rclpy.ok():
            if do_spin_once:
                rclpy.spin_once(self, timeout_sec=0.1)
            # end if

            if time.time() - st > timeout_sec:
                logging.error(f'{RED}get frame timeout.{RESET}')
                break
            # end if

            if self.stamp is None:
                time.sleep(0.03)  # 等待图像到来
                continue
            # end if

            imgs_list.append(self.imgs.copy())
            self.stamp = None  # 重置时间戳以等待下一帧

            if len(imgs_list) >= frames_num:
                break
            # end if
        # end while

        self.enable_receive_frame = False

        if len(imgs_list) < frames_num:
            logging.error(f'{RED}not enough frames, got {len(imgs_list)} < {frames_num}.{RESET}')
            return None
        # end if

        logging.info(f'get_frames cost time( ms ): {(time.time() - st)*1000:.2f}')

        return imgs_list

    # end def get_frames

    def cam_info_callback(self, *cam_info_msgs: sensor_msgs.msg.CameraInfo):
        """
        同步的相机信息回调
        """
        if self.cam_infos[0] is not None:
            return  # 只需要获取一次相机信息
        # end if

        for i, cam_info_msg in enumerate(cam_info_msgs):
            resolution = [cam_info_msg.width, cam_info_msg.height]
            intrinsic = [cam_info_msg.k[0], cam_info_msg.k[4], cam_info_msg.k[2], cam_info_msg.k[5]]  # fx, fy, cx, cy
            if len(cam_info_msg.d) == 0:
                distortion = []  # 无畸变
            else:
                distortion = list(cam_info_msg.d)
            # end if

            self.cam_infos[i] = {
                'resolution': resolution,
                'intrinsic': intrinsic,
                'distortion': distortion
            }
        # end for
    # end def cam_info_callback

    def get_cam_infos(self,
                      timeout_sec: float = 5.0,
                      do_spin_once: bool = False) -> List[Dict]:
        """
        获取相机信息
        Args:
            timeout_sec (float): 超时时间( 单位: s )
            do_spin_once (bool): 是否在函数内部执行 rclpy.spin_once. 如果调用该函数前已经在外部执行了 rclpy.spin 或者 rclpy.spin_once, 
                                 则可以将该参数设置为 False 以避免重复调用
        Returns:
            (List[Dict]): 相机信息列表,每个字典包含以下字段:
                - 'resolution': [width, height]
                - 'intrinsic': [fx, fy, cx, cy]
                - 'distortion': 畸变系数列表( 可选 ), 格式与 OpenCV 相同
        """

        if len(self.cam_info_sub_list) == 0:    # 未订阅相机信息话题
            logging.warning(f'{YELLOW}cam_info_topic_list is empty, no cam info to wait.{RESET}')
            return None
        # end if

        if self.cam_infos[0] is not None:
            return self.cam_infos
        # end if

        logging.info("waiting for cameras' info via ros topic ...")

        st = time.time()
        while rclpy.ok():
            if do_spin_once:
                rclpy.spin_once(self, timeout_sec=0.1)
            # end if

            if time.time() - st > timeout_sec:
                logging.error(f'{RED}wait cam info timeout.{RESET}')
                return None
            # end if

            if self.cam_infos[0] is not None:
                break
            # end if

            time.sleep(0.1)  # 等待相机信息到来
        # end while

        logging.info(f"received cameras' info num: {len(self.cam_infos)}")

        return self.cam_infos
    # end def get_cam_infos
# end class CamNode

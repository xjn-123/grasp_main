'''相机ROS工具类和函数'''
import logging
import os
import sys
import time

import rclpy
import sensor_msgs.msg
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
import numpy as np

from common_utils.utils import Common_data

Logger = logging.getLogger(__name__)


class CameraNode(Node):

    """
    相机节点
    """

    def __init__(self,img_topic_list: list[str], cam_info_topic_list: list[str] = None, reliability: int = 1 ):

        '''构造函数
        args:img_topic_list:订阅的图像话题列表
        cam_info_topic_list:订阅的相机信息话题列表
        reliability:相机话题的传输质量，0为尽力而为，1为可靠传输，2为尽力而为且可靠传输
        '''

        super().__init__("camera_node")
        if len(img_topic_list) <1:
            raise ValueError("图像话题列表不能为空")

        if reliability ==0:
            reliability_policy = ReliabilityPolicy.SYSTEM_DEFAULT
        elif reliability ==1:
            reliability_policy = ReliabilityPolicy.RELIABLE
        elif reliability ==2:
            reliability_policy = ReliabilityPolicy.BEST_EFFORT
        else:
            raise ValueError("reliability参数错误")

        qos=QoSProfile(reliability=reliability_policy, history=HistoryPolicy.KEEP_Last  , depth=1 )

        # 创建订阅者
        self.bridge = CvBridge()
        self.img_sub_list=[]
        for img_topic in img_topic_list:
            self.img_sub_list.append(Subscriber(
                sensor_msgs.msg.Image,
                img_topic,
                qos
            ))

        #创建时间同步器
        self.sync = ApproximateTimeSynchronizer(self.img_sub_list, queue_size=1, slop=0.05)
        self.sync.registerCallback(self.frame_callback)

        #初始化图像缓存
        self.enable_receive_frame = False #是否开始接收图像
        self.imgs = [None] * len(img_topic_list)#cv2格式的图片缓存
        self.stamp = None#当前图像的时间戳

        #创建相机信息订阅器
        self.cam_info_sub_list = []
        if cam_info_topic_list is not None:
            for cam_info_topic in cam_info_topic_list:
                sub=Subscriber(self, sensor_msgs.msg.CameraInfo, cam_info_topic)
                self.cam_info_sub_list.append(sub)

            self.cam_info_sync = ApproximateTimeSynchronizer(self.cam_info_sub_list, queue_size=1, slop=0.1)
            self.cam_info_sync.registerCallback(self.cam_info_callback)

            self.cam_info_list=[None]*len(self.cam_info_sub_list)#创建一个长度为cam_info_sub_list的长度的列表，用于存储相机信息
        else:
            Logger.warning(f"{Common_data.GREEN}相机信息列表为空{Common_data.RESET}")

        self.callback_cnt = 0            # 回调计数器, 用于调试
        self.callback_first_time = None  # 首次触发的回调时间, 用于调试
        self.callback_duration = 0.0     # 回调持续时间, 用于调试
        
        Logger.info(f"{Common_data.GREEN}相机节点初始化成功{Common_data.RESET}")  


    def frame_callback(self, *img_msgs: sensor_msgs.msg.Image):  

        """
            同步的图像回调
        """

        if not self.enable_receive_frame:
            return
        
        # 将ROS消息转换成OpenCV格式,并保留相机的原始图像编码格式
        self.imgs=[self.bridge.imgmsg_to_cv2(img_msg,desired_endcoding=img_msg.encoding) for img_msg in img_msgs]
        # 保存时间戳
        self.stamp = img_msgs[0].header.stamp

    def get_frames(self, 
                frames_num: int = 1,
                do_spin_once: bool = False,
                timeout: float = 5.0) -> list[list[float]]:
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
        self.stamp=None
        imgs_list=[]
        st=time.time()
        while rclpy.ok():
            if do_spin_once:
                rclpy.spin_once(self, timeout_sec=timeout)

            if  time.time()-st>timeout:
                Logger.error(f"{Common_data.RED}等待图像超时{Common_data.RESET}")

            if self.stamp is not None:
                time.sleep(0.03)
                continue

            imgs_list.append(self.imgs.copy())
            self.stamp=None

            if len(imgs_list)>=frames_num:
                break

        self.enable_receive_frame = False

        if len(imgs_list) <frames_num:
            Logger.error(f"{Common_data.RED}未收集到足够的图像，需收集{frames_num},已收集{len(imgs_list)}{Common_data.RESET}") 
        
        Logger.info(f"{Common_data.RED}收集完成，共耗时{(time.time()-st)*1000:.2f}ms{Common_data.RESET}")
        
        return imgs_list
    
    
    def cam_info_callback(self, *cam_info_msgs: sensor_msgs.msg.CameraInfo) -> None: 
        """
        相机信息回调函数
        """

        if self.cam_info_list[0] is not None:# 已经收集过相机信息不再继续收集
            return

        for i, cam_info_msg in enumerate(cam_info_msgs):

            resolution = (cam_info_msg.width, cam_info_msg.height)
            intrinsic = [cam_info_msg.K[0], cam_info_msg.K[4], cam_info_msg.K[2], cam_info_msg.K[5]]

            if len(cam_info_msg.d)==0:
                distortion = []
            else:  
                distortion=list(cam_info_msg.D)

            self.cam_info_list[i]={
                'resolution': resolution,
                'intrinsic': intrinsic,
                'distortion': distortion
            }

    def get_cam_infos(self, timeout_sec: float = 5.0, do_spin_once: bool = False) -> list[dict]:

        """
        获取相机信息
        :param timeout_sec: 等待相机信息超时时间
        :param do_spin_once: 是否只执行一次spin
        :return: 相机信息列表
        """

        if self.cam_info_list is not None:
            return self.cam_info_list

        if len(self.cam_info_sub_list)==0:
            Logger.warning(f"{Common_data.RED}相机信息订阅器为空{Common_data.RESET}")
            return None
        
        st=time.time()
        while rclpy.ok():
            if do_spin_once:
                rclpy.spin_once(self, timeout_sec=0.1)

            if time.time()-st>timeout_sec:
                Logger.error(f"{Common_data.RED}等待相机信息超时{Common_data.RESET}")
                return None

            if self.cam_info_list is not None:
                break
            time.sleep(0.03)

        Logger.info(f"{Common_data.RED}获取相机信息完成，共耗时{(time.time()-st)*1000:.2f}ms{Common_data.RESET}")
        return self.cam_info_list
    


        



            


        
        


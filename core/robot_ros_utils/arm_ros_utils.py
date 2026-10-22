"""
机械臂相关的 ROS2 工具函数和类
"""

import logging

from typing_extensions import List, Tuple

import numpy as np
import transforms3d

import rclpy
from rclpy.node import Node
import sensor_msgs.msg
import geometry_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformStamped


# 导入本工程的模块
from .utils import (
    GREEN, YELLOW, BLUE, RED, RESET
)
from .arm_utils import GripperBody


######################################################## 函数定义 #########################################################

def pose_to_msg(T: np.ndarray) -> geometry_msgs.msg.PoseStamped:
    """
    将位姿矩阵转换为 ROS PoseStamped 消息,注意这里没有设置 frame_id 以及时间戳    
    Args:
        T (np.ndarray): 位姿矩阵,形状为 (4, 4)
    Returns:
        (geometry_msgs.msg.PoseStamped): ROS PoseStamped 消息
    """

    # 提取四元数和位置
    p = T[:3, 3]  # [tx, ty, tz]
    q = transforms3d.quaternions.mat2quat(T[:3, :3])  # [qw, qx, qy, qz]
    msg = geometry_msgs.msg.PoseStamped()

    msg.pose.position.x = float(p[0])
    msg.pose.position.y = float(p[1])
    msg.pose.position.z = float(p[2])

    msg.pose.orientation.w = float(q[0])
    msg.pose.orientation.x = float(q[1])
    msg.pose.orientation.y = float(q[2])
    msg.pose.orientation.z = float(q[3])

    return msg
# end def pose_to_msg


def grippers_to_msg(gripper_rects_3d: np.ndarray,
                    stamp: rclpy.time.Time,
                    frame_id: str,
                    color: Tuple[float, float, float, float]) -> MarkerArray:
    """
    发布两个夹爪的可视化 Marker( 使用两个长方体表示夹爪 )
    Args:
        gripper_rects_3d (np.ndarray): 夹爪的 3D 顶点坐标,形状为 (8, 3), 前四个点为第一个夹爪的四个顶点,后四个点为第二个夹爪的四个顶点
        stamp (rclpy.time.Time): 时间戳
        frame_id (str): 参考坐标系的 frame_id
        color (Tuple[float, float, float, float]): 夹爪的颜色, RGBA 格式, 每个分量的取值范围为 [0.0, 1.0]
    Returns:
        (MarkerArray): 夹爪的 MarkerArray 消息
    """

    assert gripper_rects_3d is not None and gripper_rects_3d.shape == (8, 3), "Invalid gripper_rects_3d data."

    gripper_length = 0.05  # 夹爪长度 (单位: m, Z 轴方向), 从夹爪顶点中心到夹爪底部的距离, 这个值用于可视化, 与实际夹爪尺寸无关

    def create_marker(pts: np.ndarray) -> Marker:
        edge_x = pts[1] - pts[0]
        edge_y = pts[0] - pts[3]
        size_x = np.linalg.norm(edge_x)
        size_y = np.linalg.norm(edge_y)

        axis_x = edge_x / size_x
        axis_y = edge_y / size_y
        axis_z = np.cross(axis_x, axis_y)

        R = np.column_stack((axis_x, axis_y, axis_z))
        q = transforms3d.quaternions.mat2quat(R)  # (w, x, y, z)
        p = pts.mean(axis=0)
        p -= R @ np.array([0, 0, gripper_length / 2.0])  # 调整中心点到长方体中心

        marker = Marker()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.scale.x = float(size_x)
        marker.scale.y = float(size_y)
        marker.scale.z = float(gripper_length)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        marker.pose.position.x = float(p[0])
        marker.pose.position.y = float(p[1])
        marker.pose.position.z = float(p[2])
        marker.pose.orientation.w = float(q[0])
        marker.pose.orientation.x = float(q[1])
        marker.pose.orientation.y = float(q[2])
        marker.pose.orientation.z = float(q[3])

        return marker
    # end def create_marker

    marker_array = MarkerArray()
    for idx in range(2):
        pts = gripper_rects_3d[idx * 4:(idx + 1) * 4]
        marker = create_marker(pts)

        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'gripper'
        marker.id = idx

        marker_array.markers.append(marker)
    # end for

    return marker_array
# end def grippers_to_msg


def pose_to_transform_stamped(parent_frame: str,
                              child_frame: str,
                              T: np.ndarray) -> TransformStamped:
    """
    将位姿矩阵转换为 TransformStamped 消息,注意这里没有设置时间戳  
    Parameters
    ----------
        parent_frame (str): 父坐标系名称
        child_frame (str): 子坐标系名称
        T (np.ndarray): 4x4 位姿矩阵
    Returns
    -------
        (TransformStamped): ROS 消息
    """

    p = T[:3, 3]
    q = transforms3d.quaternions.mat2quat(T[:3, :3])  # qw,qx,qy,qz

    ts = TransformStamped()
    ts.header.frame_id = parent_frame
    ts.child_frame_id = child_frame
    ts.transform.translation.x = float(p[0])
    ts.transform.translation.y = float(p[1])
    ts.transform.translation.z = float(p[2])
    ts.transform.rotation.x = float(q[1])
    ts.transform.rotation.y = float(q[2])
    ts.transform.rotation.z = float(q[3])
    ts.transform.rotation.w = float(q[0])

    return ts
# end def pose_to_transform_stamped


######################################################## 类定义 ########################################################

class ArmNode(Node):
    """
    ROS2 节点, 用于发布机械臂状态   
    """

    def __init__(self,
                 pub_arm_joints: bool = False,
                 pub_gripper_msg: bool = False,
                 frame_id: str = 'base_link'):
        """
        初始化   
        Args:
            pub_arm_joints (bool): 是否发布机械臂的关节角度. 默认值为 False
            pub_gripper_msg (bool): 是否发布机械爪的 Marker. 默认值为 False
            frame_id (str): 发布的消息的参考坐标系的 frame_id. 默认值为 'base_link'
        """

        super().__init__('arm_node')

        self.frame_id = frame_id

        # 创建发布者
        self.arm_pose_pub = self.create_publisher(geometry_msgs.msg.PoseStamped, '/arm_pose', 1)

        self.arm_joint_pub = None
        if pub_arm_joints:
            self.arm_joint_pub = self.create_publisher(sensor_msgs.msg.JointState, '/arm_joint', 1)

        self.gripper_marker_pub = None
        if pub_gripper_msg:
            self.gripper_marker_pub = self.create_publisher(MarkerArray, '/grippers', 1)
        # end if

        logging.info(f'{GREEN}ArmNode initialized {RESET}')
    # end def __init__

    def publish_pose(self,
                     T_base_end: np.ndarray):
        """
        发布机械臂的状态信息, 包括位姿和关节角度
        Args:
            T_base_end (np.ndarray): 从机械臂末端到基座坐标系下的位姿变换矩阵,形状为 (4, 4)
        """

        # 时间戳
        stamp = rclpy.time.Time().to_msg()

        # 发布位姿
        msg = pose_to_msg(T_base_end)
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        self.arm_pose_pub.publish(msg)
    # end def publish_pose

    def publish_joints(self,
                       joints: List[float],):
        """
        发布机械臂的状态信息, 包括位姿和关节角度
        Args:
            joints (List[float]): 机械臂的关节角度列表,长度为 6
        """

        if self.arm_joint_pub is None:
            logging.warning('arm_joint_pub is None, skip publish joints.')
            return
        # end if

        # 时间戳
        stamp = rclpy.time.Time().to_msg()

        # 发布关节角度
        msg = sensor_msgs.msg.JointState()
        msg.header.stamp = stamp
        msg.position = joints
        self.arm_joint_pub.publish(msg)
    # end def publish_joints

    def publish_grippers(self,
                         gripper_body: GripperBody,
                         gripper_dist: float,
                         T_base_end: np.ndarray,
                         T_end_cam: np.ndarray):
        """
        发布机械臂夹爪的位姿
        Args:
            gripper_body (GripperBody): 机械爪模型对象
            gripper_dist (float): 夹爪开合距离
            T_base_end (np.ndarray): 从基座坐标系到机械臂末端坐标系的位姿矩阵,形状为 (4, 4)
            T_end_cam (np.ndarray): 从相机坐标系到机械臂末端坐标系的位姿矩阵,形状为 (4, 4)
        """

        if self.gripper_marker_pub is None:
            logging.warning('gripper_marker_pub is None, skip publish grippers.')
            return
        # end if

        T_base_cam = T_base_end @ T_end_cam

        gripper_rects_3d = gripper_body.get_rects_3d(dist=gripper_dist,
                                                     T_target_cam=T_base_cam)  # 获取夹爪的 3D 顶点坐标( 基座坐标系下 )

        stamp = rclpy.time.Time().to_msg()

        msg = grippers_to_msg(gripper_rects_3d, stamp, self.frame_id,
                              (0.0, 1.0, 0.0, 0.5))  # RGBA 绿色半透明
        self.gripper_marker_pub.publish(msg)
    # end def publish_grippers
# end class ArmNode


class TargetArmNode(Node):
    """
    ROS2 节点, 发布机械臂的目标状态( 非实际状态 ),用于在 rviz 中可视化, 检查目标状态是否正常( 如: 是否会与环境发生碰撞 )   
    """

    def __init__(self,
                 pub_gripper_msg: bool = False,
                 frame_id: str = 'base_link'):
        """
        初始化   
        Args:
            pub_gripper_msg (bool): 是否发布机械爪的 Marker. 默认值为 False
            frame_id (str): 发布的消息的参考坐标系的 frame_id. 默认值为 'base_link'
        """

        super().__init__('target_arm_node')

        self.frame_id = frame_id

        # 创建发布者
        self.arm_pose_pub = self.create_publisher(geometry_msgs.msg.PoseStamped, '/target_arm_pose', 1)

        self.gripper_marker_pub = None
        if pub_gripper_msg:
            self.gripper_marker_pub = self.create_publisher(MarkerArray, '/target_grippers', 1)
        # end if

        logging.info(f'{GREEN}TargetArmNode initialized {RESET}')
    # end def __init__

    def publish_pose(self,
                     T_base_end: np.ndarray):
        """
        发布机械臂的状态信息, 包括位姿和关节角度
        Args:
            T_base_end (np.ndarray): 从机械臂末端到基座坐标系下的位姿变换矩阵,形状为 (4, 4)
        """

        # 时间戳
        stamp = rclpy.time.Time().to_msg()

        # 发布位姿
        pose_msg = pose_to_msg(T_base_end)
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.frame_id
        self.arm_pose_pub.publish(pose_msg)
    # end def publish_pose

    def publish_grippers(self,
                         gripper_body: GripperBody,
                         gripper_dist: float,
                         T_base_end: np.ndarray,
                         T_end_cam: np.ndarray):
        """
        发布机械臂夹爪的位姿
        Args:
            gripper_body (GripperBody): 机械爪模型对象
            gripper_dist (float): 夹爪开合距离
            T_base_end (np.ndarray): 从基座坐标系到机械臂末端坐标系的位姿矩阵,形状为 (4, 4)
            T_end_cam (np.ndarray): 从相机坐标系到机械臂末端坐标系的位姿矩阵,形状为 (4, 4)
        """

        if self.gripper_marker_pub is None:
            logging.warning('gripper_marker_pub is None, skip publish grippers.')
            return
        # end if

        T_base_cam = T_base_end @ T_end_cam

        gripper_rects_3d = gripper_body.get_rects_3d(dist=gripper_dist,
                                                     T_target_cam=T_base_cam)  # 获取夹爪的 3D 顶点坐标( 基座坐标系下 )

        stamp = rclpy.time.Time().to_msg()

        msg = grippers_to_msg(gripper_rects_3d, stamp, self.frame_id,
                              (0.0, 0.0, 1.0, 0.3))  # RGBA 蓝色半透明
        self.gripper_marker_pub.publish(msg)
    # end def publish_grippers
# end class TargetArmNode

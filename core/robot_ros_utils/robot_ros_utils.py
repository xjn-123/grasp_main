'''机械臂位姿信息和夹爪信息的ros发布节点
ros_geometry消息不支持四元数
工具函数和类，不进行发布和发布进入TF树
'''

import logging
import os
import sys

import geometry_msgs.msg
import numpy as np
import rclpy
import sensor_msgs.msg
import transforms3d
from rclpy.node import Node
from tf2_ros import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

root_dir = os.path.normpath(f"{os.path.dirname(os.path.realpath(__file__))}/../../")
sys.path.append(root_dir)
from core.common_utils.utils import Common_data
from core.gripper_utils.gripper_utils import Gripper

Logger = logging.getLogger(__name__)

####################消息封装函数####################


def pose_to_msg(T: np.ndarray) -> geometry_msgs.msg.PoseStamped:

    """
    将位姿矩阵转换为 ROS PoseStamped 四元数消息,注意这里没有设置 frame_id 以及时间戳    
    Args:
        T (np.ndarray): 位姿矩阵,形状为 (4, 4)
    Returns:
        (geometry_msgs.msg.PoseStamped): ROS PoseStamped 消息
    """
    #将位姿矩阵转换为 ROS PoseStamped 四元数消息
    p=T[:3,3]
    q=transforms3d.quaternions.mat2quat(T[:3,:3])
    msg=geometry_msgs.msg.PoseStamped()

    msg.pose.position.x = float(p[0])
    msg.pose.position.y = float(p[1])
    msg.pose.position.z = float(p[2])

    msg.pose.orientation.w = float(q[0])
    msg.pose.orientation.x = float(q[1])
    msg.pose.orientation.y = float(q[2])
    msg.pose.orientation.z = float(q[3])

    return msg


def pose_to_transformstamped(parent_frame:str,child_frame:str,T:np.ndarray) -> TransformStamped:
    """
    将 PoseStamped 消息转换为 TransformStamped 消息,注意这里没有设置时间戳
    Args:
       
        parent_frame (str): 父坐标系名称
        child_frame (str): 子坐标系名称
        T (np.ndarray): 4x4 位姿矩阵
    Returns:    
        (TransformStamped): ROS TransformStamped 消息
    """

    #将位姿矩阵转换为 ROS TransformStamped 四元数消息

    p = T[:3, 3]
    q = transforms3d.quaternions.mat2quat(T[:3, :3])
    ts = TransformStamped()

    ts.header.frame_id = parent_frame
    ts.child_frame_id = child_frame

    ts.transform.translation.x = float(p[0])
    ts.transform.translation.y = float(p[1])
    ts.transform.translation.z = float(p[2])

    ts.transform.rotation.w = float(q[0])
    ts.transform.rotation.x = float(q[1])
    ts.transform.rotation.y = float(q[2])
    ts.transform.rotation.z = float(q[3])

    return ts
   

def gripper_to_msg(gripper_rects_3d: np.ndarray, stamp: rclpy.time.Time, frame_id: str, color: tuple[float, float, float, float]) -> MarkerArray:

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

    assert gripper_rects_3d is not None and gripper_rects_3d.shape == (8, 3), 'gripper_rects_3d is not valid'

    gripper_length = 0.05
    # 夹爪的长度 (单位: m, Z 轴方向), 从夹爪顶点中心到夹爪底部的距离, 这个值用于可视化, 与实际夹爪尺寸无关

    def create_marker(pts: np.ndarray) -> Marker:

        edge_x = pts[1] - pts[0]
        edge_y = pts[0] - pts[3]

        size_x = np.linalg.norm(edge_x)
        size_y = np.linalg.norm(edge_y)
        size_z = gripper_length

        axis_x = edge_x / size_x
        axis_y = edge_y / size_y
        axis_z = np.cross(axis_x, axis_y)

        R = np.column_stack((axis_x, axis_y, axis_z))
        p= pts.mean(axis=0)#沿第一个维度求均值，即剩余维度在这个维度方向上的均值
        p-= R @ np.array([0, 0, gripper_length / 2.0])
        q = transforms3d.quaternions.mat2quat(R)#(w,x,y,z)

        marker=Marker()
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

    marker_array = MarkerArray()

    for idx in range(2):
        pts = gripper_rects_3d[idx * 4:(idx + 1) * 4]
        marker = create_marker(pts)

        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'gripper'
        marker.id=idx

        marker_array.markers.append(marker)

    return marker_array


#################节点类################
class ArmNode(Node):
    """机械臂状态发布节点( 实际状态与目标状态共用同一份实现 )

    通过 topic_prefix 区分用途, 例如:
        ArmNode()                                                  -> /arm_pose, /arm_joint, /grippers
        ArmNode(topic_prefix='/target_')                           -> /target_arm_pose, /target_grippers
        ArmNode(topic_prefix='/arm1/', frame_id='arm1_base_link')   # 多臂场景
    """

    def __init__(self,
                 node_name: str = 'arm_node',
                 topic_prefix: str = '',
                 frame_id: str = 'base_link',
                 pub_arm_joints: bool = False,
                 pub_gripper_msg: bool = False,
                 gripper_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5),
                 joint_names: list[str] = None):
        """
        Args:
            node_name (str): ROS2 节点名, 同一进程内的多个实例必须不同
            topic_prefix (str): 话题名前缀, 直接拼在话题名前面; '' 表示不加前缀
            frame_id (str): 所有消息的参考坐标系
            pub_arm_joints (bool): 是否发布关节角消息
            pub_gripper_msg (bool): 是否发布夹爪 Marker
            gripper_color (tuple): 夹爪 Marker 的 RGBA
            joint_names (list[str]): 关节名, 需与 URDF 一致, 默认 joint_1 ~ joint_6
        """

        super().__init__(node_name)

        self.frame_id = frame_id
        self.gripper_color = gripper_color
        self.joint_names = (list(joint_names) if joint_names is not None
                            else [f'joint_{i}' for i in range(1, 7)])

        self.arm_pose_pub = self.create_publisher(
            geometry_msgs.msg.PoseStamped, f'{topic_prefix}arm_pose', 1)

        self.arm_joint_pub = None
        if pub_arm_joints:
            self.arm_joint_pub = self.create_publisher(
                sensor_msgs.msg.JointState, f'{topic_prefix}arm_joint', 1)
        # end if

        self.gripper_marker_pub = None
        if pub_gripper_msg:
            self.gripper_marker_pub = self.create_publisher(
                MarkerArray, f'{topic_prefix}grippers', 1)
        # end if

        Logger.info(f"{Common_data.GREEN}{node_name} initialized{Common_data.RESET}")
    # end def __init__

    def publish_pose(self, T_base_end: np.ndarray) -> None:
        """发布机械臂位姿消息
        Args:
            T_base_end (np.ndarray): 末端在基座坐标系下的 4x4 位姿矩阵
        """

        pose = pose_to_msg(T_base_end)
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()

        self.arm_pose_pub.publish(pose)
    # end def publish_pose

    def publish_joints(self, joints: list[float]) -> None:
        """发布机械臂关节角消息
        Args:
            joints (list[float]): 关节角列表, 长度需与 joint_names 一致
        """

        if self.arm_joint_pub is None:
            Logger.warning(f"{Common_data.YELLOW}arm_joint_pub is None{Common_data.RESET}")
            return
        # end if

        assert len(joints) == len(self.joint_names), \
            f'joints length {len(joints)} != joint_names length {len(self.joint_names)}'

        joint_state = sensor_msgs.msg.JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = list(self.joint_names)
        joint_state.position = [float(q) for q in joints]

        self.arm_joint_pub.publish(joint_state)
    # end def publish_joints

    def publish_grippers(self, gripper: Gripper, gripper_dist: float,
                         T_base_end: np.ndarray, T_end_cam: np.ndarray) -> None:
        """发布夹爪的可视化 Marker
        Args:
            gripper (Gripper): 夹爪模型
            gripper_dist (float): 夹爪开合距离 (单位: m)
            T_base_end (np.ndarray): 末端在基座坐标系下的 4x4 位姿矩阵
            T_end_cam (np.ndarray): 相机在末端坐标系下的 4x4 位姿矩阵
        """

        if self.gripper_marker_pub is None:
            Logger.warning(f"{Common_data.YELLOW}gripper_marker_pub is None{Common_data.RESET}")
            return
        # end if

        T_base_cam = T_base_end @ T_end_cam

        # 获取夹爪的 3D 顶点坐标( 基座坐标系下 )
        gripper_rects_3d = gripper.get_rects_3d(dist=gripper_dist,
                                                T_target_cam=T_base_cam)

        msg = gripper_to_msg(gripper_rects_3d, self.get_clock().now().to_msg(),
                             self.frame_id, self.gripper_color)

        self.gripper_marker_pub.publish(msg)
    # end def publish_grippers
# end class ArmNode


class TargetArmNode(ArmNode):
    """目标位姿节点: 只是 ArmNode 的一组预设参数, 保留用于兼容旧调用

    等价于:
        ArmNode(node_name='target_arm_node', topic_prefix='/target_',
                pub_gripper_msg=..., gripper_color=(0.0, 0.0, 1.0, 0.3))
    话题名与旧版一致: /target_arm_pose, /target_grippers
    """

    def __init__(self, pub_gripper_msg: bool = False, frame_id: str = 'base_link'):
        super().__init__(node_name='target_arm_node',
                         topic_prefix='/target_',
                         frame_id=frame_id,
                         pub_arm_joints=False,
                         pub_gripper_msg=pub_gripper_msg,
                         gripper_color=(0.0, 0.0, 1.0, 0.3))
    # end def __init__
# end class TargetArmNode
    



        
        









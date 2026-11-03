'''检测夹爪状态的工具函数，包含夹爪与环境的碰撞以及姿态校验等'''

import logging
import os
import sys
import time

import cv2
import numpy as np
import open3d
import transforms3d

root_dir = os.path.normpath(f"{os.path.dirname(os.path.realpath(__file__))}/../../")
sys.path.append(root_dir)
from core.common_utils.utils import Common_data,inv_tf
from core.vision_utils.vision_utils import rgb_to_point_cloud,depth_to_point_cloud

Logger = logging.getLogger(__name__)

################全局变量#####################
TH_ANGLE_Z = 45.0 * np.pi / 180.0
"""两个坐标系之间的 Z 轴夹角阈值, 单位: 弧度"""

TH_GRIPPER_HEIGHT = -0.01
"""夹爪中心在基座坐标系的高度阈值, 单位: 米"""

class Gripper:
    """
    夹爪对象

    夹爪几何体,使用两个矩形来表示夹爪与环境的接触面.预设条件:     
    1) 仅适用于眼在手的场景
    2) 夹爪坐标系的原点位于两片爪尖的几何中心
    3) 以夹爪张开方向为夹爪坐标系的 X 轴,从左爪指向右爪;    
    4) 夹爪坐标系的 Z 轴垂直于夹爪平面,朝向与末端坐标系 Z 轴的夹角近似平行;

    """

    def __init__(self, width: float, thickness: float, T_cam_gripper: np.ndarray = np.eye(4)):

        '''构造函数'''
        self.width = width#夹爪宽度
        self.thickness = thickness#夹爪厚度
        self.T_cam_gripper = T_cam_gripper#夹爪坐标系到 RGB-D 相机的变换矩阵

    def two_finger_gripper(self, corner3d: np.ndarray):
        """
        计算双指夹爪到相机坐标的旋转矩阵
        """
        assert corner3d.shape == (4, 3)# 检查输入的corner3d的形状
        center=corner3d.mean(axis=0)   # 计算四角形的中心点,沿数组的第0维（行）进行求平均

        #计算xy轴
        edge_x=corner3d[0]-corner3d[2]
        edge_y=corner3d[3]-corner3d[1]

        if np.linalg.norm(edge_x) < 1e-6 or np.linalg.norm(edge_y) < 1e-6:
            Logger.error(f"{Common_data.RED}请检查输入的corner3d,对角点的x或y值相差过小{Common_data.RESET}")
            return None
        
        #np.linalg.norm 计算向量的模长
        x_axis=edge_x/np.linalg.norm(edge_x)
        y_axis_raw=edge_y/np.linalg.norm(edge_y)

        #y方向向量减去其在x轴上的投影，实现xy向量的正交化
        y_axis=y_axis_raw-np.dot(y_axis_raw, x_axis)*x_axis
        y_axis=y_axis/np.linalg.norm(y_axis)

        #Z=X x Y  由xy的叉乘计算Z,Z的朝向与相机坐标系Z轴一致（由aprit tag的角点保证）
        z_axis=np.cross(x_axis, y_axis)
        z_axis=z_axis/np.linalg.norm(z_axis)
        assert z_axis[2]>0 # 确保Z轴朝向相机坐标系Z轴

        #初始化旋转矩阵
        T_cam_gripper = np.eye(4)
        T_cam_gripper[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        T_cam_gripper[:3, 3] = center.reshape(3, 1)

        Logger.info(f"{Common_data.GREEN}相机和gripper之间的变换矩阵: {T_cam_gripper}{Common_data.RESET}")

        return T_cam_gripper

    def get_rects_3d(self, dist: float, T_target_cam: np.ndarray)-> np.ndarray:

        """
        用两个空间矩形表示夹爪的位置,计算夹爪的八个角点的3D坐标( 目标坐标系下 )     
        Args:
            dist (float): 两个夹爪矩形之间的距离 (单位: m)
            T_target_cam (np.ndarray): 从相机坐标系到目标坐标系的变换矩阵, 形状为 (4, 4)
        Returns:
            (np.ndarray): 两个夹爪夹爪共八个角点的3D坐标, 形状为 (8, 3), 每个夹爪四个角点按[左下,右下,右上,左上]顺序排列
        """
        w=self.width
        t=self.thickness
        hd=dist/2.0
        hw=w/2.0

        # 计算夹爪在夹爪坐标系下的八个角点坐标
        left_rect=np.array([[-hd-t,hw,0,1],
                            [-hd,hw,0,1],
                            [-hd,-hw,0,1],
                            [-hd-t,-hw,0,1]],dtype=np.float32).T
        
        right_rect=np.array([[hd,hw,0,1],
                            [hd+t,hw,0,1],
                            [hd+t,-hw,0,1],
                            [hd,-hw,0,1]],dtype=np.float32).T
        # 将夹爪角点从夹爪坐标系转换到目标坐标系
        T_target_gripper=T_target_cam@self.T_cam_gripper
        left_rect_3d=(T_target_gripper@left_rect).T[:,:3]
        right_rect_3d=(T_target_gripper@right_rect).T[:,:3]
        rects_3d=np.vstack([left_rect_3d,right_rect_3d])
        return rects_3d
        
class CollisionDetector:
    """
    夹爪碰撞检测器类,用于检测夹爪与环境的碰撞，
    通过将夹爪的两个矩形与环境点云进行碰撞检测来判断夹爪是否与环境发生碰撞
    仅适用于eye-in-hand场景
    """

    def __init__(self, gripper: Gripper):
        self.gripper = gripper



#####################夹爪位姿调整与检查函数#######################
def check_gripper_pose():
    pass

def com_axis_aligned_pose(T_base_end: np.ndarray, 
                        base_axis_idx: int, 
                        obj_axis_idx: int, 
                        T_end_obj: np.ndarray = np.eye(4), 
                        th_angle: float = 45.0):
    """
        计算一个新的末端位姿,使得物体坐标系的第 obj_axis_idx 个轴与机械臂基座坐标系的第 base_axis_idx 个轴对齐
        Args:
            T_base_end (np.ndarray): 从机械臂末端到基座坐标系下的变换矩阵, 4*4
            base_axis_idx (int): 末端坐标系中需要对齐的轴的索引, 1,2,3 分别表示 X,Y,Z 轴的正方向, -1,-2,-3 分别表示 X,Y,Z 轴的负方向
            obj_axis_idx (int): 物体坐标系中需要对齐的轴的索引, 1,2,3 分别表示 X,Y,Z 轴的正方向, -1,-2,-3 分别表示 X,Y,Z 轴的负方向
            T_end_obj (np.ndarray): 从物体到机械臂末端坐标系下的变换矩阵, 4*4
            th_angle (float): 夹角阈值, 单位: 度. 如果当前夹角大于该阈值, 则认为无法调整, 返回 None
        Returns:
            (np.ndarray): 调整后的末端位姿, 4*4. 如果调整失败( 夹角过大 ), 返回 None
    """

    if abs(base_axis_idx) not in [1, 2, 3]:
        Logger.error(f'{Common_data.RED}base_axis_idx错误: {base_axis_idx}{Common_data.RESET}')
        return 

    if abs(obj_axis_idx) not in [1, 2, 3]:
        Logger.error(f'{Common_data.RED}obj_axis_idx错误: {obj_axis_idx}{Common_data.RESET}')
        return 

    T_base_obj = T_base_end @ T_end_obj

    # 期望的对齐方向, 由 base_axis_idx 确定
    target_dir = np.eye(3)[:, abs(base_axis_idx) - 1] * np.sign(base_axis_idx)
    # 当前的方向, 由 obj_axis_idx 确定
    current_dir = T_base_obj[:3, abs(obj_axis_idx) - 1] * np.sign(obj_axis_idx)

    # 计算夹角
    cosine = np.dot(target_dir, current_dir)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.arccos(cosine)

    Logger.info(f'compute_axis_aligned_pose, 目标方向: {Common_data.GREEN}{target_dir}{Common_data.RESET}, '
                 f'当前方向: {Common_data.GREEN}{current_dir}{Common_data.RESET}, '
                 f'调整角度: {Common_data.GREEN}{angle * 180.0 / np.pi:.2f}{Common_data.RESET} deg')

    if angle > np.deg2rad(th_angle):  # 夹角过大
        Logger.warning(f'{Common_data.RED}调整角度 > {th_angle} deg')
        return 

    #旋转轴
    axis=np.cross(current_dir, target_dir)
    delta_R = transforms3d.axangles.axangle2mat(axis, angle)

    #计算调整后的姿态
    target_T_base_obj = T_base_obj.copy()
    target_T_base_obj[:3, :3] = delta_R @ T_base_obj[:3, :3]
    target_T_base_end = target_T_base_obj @ inv_tf(T_end_obj)

    return target_T_base_end


    


    


    



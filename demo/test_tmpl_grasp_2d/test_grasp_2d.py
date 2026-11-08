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
import transforms3d

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

Logger = logging.getLogger(__name__)


def read_tmpl_grasp_2d(tmpl_dir: str) -> dict:
   '''读取模板数据'''

   #读取抓取位姿的模板数据
   filepath = os.path.join(tmpl_dir, "/grasp/state.json")
   with open(filepath, "r") as f:
      grasp_staste = json.load(f)

   grasp_dist = grasp_staste["gripper_dist"]
   grasp_T_base_end = np.array(grasp_staste["T_base_end"])

   #读取位于近距离的模板数据
   near_filepath = os.path.join(tmpl_dir, "/near/state.json")
   with open(near_filepath, "r") as f:
      near_staste = json.load(f)

   near_dist = near_staste["gripper_dist"]
   near_T_base_end = np.array(near_staste["T_base_end"])
   near_obj_pose_2d  = np.array(near_staste["obj_pose_2d"])

   near_next_filepath = os.path.join(tmpl_dir, "/near_next/state.json")
   with open(near_next_filepath, "r") as f:
      near_next_staste = json.load(f)

   near_next_dist = near_next_staste["gripper_dist"]
   near_next_T_base_end = np.array(near_next_staste["T_base_end"])
   near_next_obj_pose_2d  = np.array(near_next_staste["obj_pose_2d"])

   #读取位于远方的模板数据
   far_filepath = os.path.join(tmpl_dir, "/far/state.json")
   with open(far_filepath, "r") as f:
      far_staste = json.load(f)

   far_dist = far_staste["gripper_dist"]
   far_T_base_end = np.array(far_staste["T_base_end"])
   far_obj_pose_2d  = np.array(far_staste["obj_pose_2d"])

   #读取位于远方下一个模板数据
   far_next_filepath = os.path.join(tmpl_dir, "/far_next/state.json")
   with open(far_next_filepath, "r") as f:
      far_next_staste = json.load(f)

   far_next_dist = far_next_staste["gripper_dist"]
   far_next_T_base_end = np.array(far_next_staste["T_base_end"])
   far_next_obj_pose_2d  = np.array(far_next_staste["obj_pose_2d"])

   tmpl_dict={
      'grasp_grapper_dist':grasp_dist,
      'grasp_T_base_end':grasp_T_base_end,

      'near_grapper_dist':near_dist,
      'near_T_base_end':near_T_base_end,
      'near_obj_pose_2d':near_obj_pose_2d,
      'near_next_grapper_dist':near_next_dist,
      'near_next_T_base_end':near_next_T_base_end,
      'near_next_obj_pose_2d':near_next_obj_pose_2d,

      'far_grapper_dist':far_dist,
      'far_T_base_end':far_T_base_end,
      'far_obj_pose_2d':far_obj_pose_2d,
      'far_next_grapper_dist':far_next_dist,
      'far_next_T_base_end':far_next_T_base_end,
      'far_next_obj_pose_2d':far_next_obj_pose_2d
   }

   return tmpl_dict


def transform_pose_2d(R_dst_src: np.ndarray,
                     src_pose_2d: list[float]) -> np.ndarray:
   """将物体在 src 像素坐标系下的归一位姿转换到 dst 归一化坐标系下的位姿
      Args:
         R_dst_src (np.ndarray): 从 src 坐标系到 dst 坐标系的旋转矩阵 (3,3)
         src_pose_2d (List[float]): 物体在 src 坐标系下的位姿 [nx, ny, theta], 其中 (nx, ny) 是归一化坐标, theta 是物体的朝向角, 单位为弧度
      Returns:
         (np.ndarray): 物体在 dst 坐标系下的位姿 [nx, ny, theta], 其中 (nx, ny) 是归一化坐标, theta 是物体的朝向角, 单位为弧度

         将物体在原坐标系下的方向向量，转换为实际的射线再投影回dst的归一化坐标
   """

   # 为便于区分, 这里将三维空间中的位移用 xyz 表示, 将归一化坐标中的位移用 nx,ny 表示, 图像/归一化坐标中的旋转用 theta 表示

   src_nx, src_ny, src_theta = src_pose_2d

   step=0.01 #用于计算src坐标中物体朝向的向量步长
   delta_src_x = src_nx+step * np.cos(src_theta)
   delta_src_y = src_ny+step * np.sin(src_theta)

   # 计算物体在 src 像素坐标系下的两个点的坐标，
   src_pts=np.array([[src_nx, src_ny,1.0], 
                     [delta_src_x, delta_src_y,1.0]],
                     detype=np.float64).T#(3,2)

   #src_pts在dst像素坐标系下的坐标
   dst_pts=R_dst_src @ src_pts
   #计算src_pts在dst归一化坐标系的坐标
   dst_rays=dst_pts/(dst_pts[2:3, :] + 1e-6) #X/Z,Y/Z,Z/Z

   dst_nx, dst_ny = dst_rays[0, 0], dst_rays[1, 0]
   delta_dst_nx, delta_dst_ny = dst_rays[0, 1], dst_rays[1, 1]
   dst_theta = np.arctan2(delta_dst_ny-dst_ny, delta_dst_nx-dst_nx)

   return np.array([dst_nx, dst_ny, dst_theta],detype=np.float64)


def compute_delta_end_pose(T_end_cam: np.ndarray,
                           tmpl_dict: dict,
                           cur_T_base_end: np.ndarray,
                           cur_obj_pose_2d: list[float],
                           z_step: float = 0.05,
                           ) -> np.ndarray:
   """
   计算机械臂末端的位姿增量

   Args:
      T_end_cam (np.ndarray): 从相机到机械臂末端的射影变换矩阵
      tmpl_dict (Dict): 模板数据字典
      cur_T_base_end (np.ndarray): 当前机械臂末端位姿
      cur_obj_pose_2d (List[float]): 当前物体在图像上的位姿 [nx, ny, theta], 其中 (nx, ny) 是归一化坐标, theta 是物体相对于相机的朝向角, 单位为弧度
      z_step (float): 每次控制机械臂在 z 轴上的移动步长, 单位为米

   Returns:
      (np.ndarray): 机械臂末端的位姿增量
   """

   near_T_base_end = tmpl_dict['near_T_base_end']
   near_obj_pose_2d = tmpl_dict['near_obj_pose_2d']
   next_near_T_base_end = tmpl_dict['near_next_T_base_end']
   next_near_obj_pose_2d = tmpl_dict['near_next_obj_pose_2d']

   far_T_base_end = tmpl_dict['far_T_base_end']
   far_obj_pose_2d = tmpl_dict['far_obj_pose_2d']
   next_far_T_base_end = tmpl_dict['far_next_T_base_end']
   next_far_obj_pose_2d = tmpl_dict['far_next_obj_pose_2d']

   # 设置一个虚拟相机, 满足:
   # 1) 虚拟相机的内参 K 为 3*3 的单位矩阵, 即虚拟相机坐标系下的二维坐标为归一化坐标
   # 2) 虚拟相机与末端的坐标轴同向 T_end_virtual[0:3, 3] = T_end_cam[0:3, 3]
   # 3) 虚拟相机与原始相机的原点重合  R_virtual_cam = T_end_cam[:3, :3]
   

   T_end_virtual = np.eye(4)
   T_end_virtual[0:3, :3] = T_end_cam[0:3, :3]
   R_virtual_cam = T_end_cam[:3, :3]

   #末端模板位姿时，计算虚拟相机在基座坐标系下的位姿
   near_T_base_virtual = near_T_base_end @ T_end_virtual
   near_next_T_base_virtual = next_near_T_base_end @ T_end_virtual
   far_T_base_virtual = far_T_base_end @ T_end_virtual
   far_next_T_base_virtual = next_far_T_base_end @ T_end_virtual
   cur_T_base_virtual = cur_T_base_end @ T_end_virtual

   #将物体位姿转换到虚拟相机归一化坐标系下
   near_obj_pose_virtual = transform_pose_2d(R_virtual_cam , near_obj_pose_2d)
   next_near_obj_pose_virtual = transform_pose_2d(R_virtual_cam , next_near_obj_pose_2d)
   far_obj_pose_virtual = transform_pose_2d(R_virtual_cam , far_obj_pose_2d)
   next_far_obj_pose_virtual = transform_pose_2d(R_virtual_cam , next_far_obj_pose_2d)
   cur_obj_pose_virtual = transform_pose_2d(R_virtual_cam , cur_obj_pose_2d)


   #分别计算near和far距离下物理平移与归一化坐标增量的比值
   near_z=near_T_base_virtual[2,3]
   near_delta_xy=inv_tf(near_T_base_virtual)@near_next_T_base_virtual[:3,2]
   near_delta_uv=next_near_obj_pose_virtual-near_obj_pose_virtual
   near_ratio=np.linalg.norm(near_delta_xy)/(np.linalg.norm(near_delta_uv)+1e-6)

   far_z=far_T_base_virtual[2,3]
   far_delta_xy=inv_tf(far_T_base_virtual)@far_next_T_base_virtual[:3,2]
   far_delta_uv=next_far_obj_pose_virtual-far_obj_pose_virtual
   far_ratio=np.linalg.norm(far_delta_xy)/(np.linalg.norm(far_delta_uv)+1e-6)

   #计算虚拟相机从near模板位姿到cur位姿的旋转增量
   delta_theta=cur_obj_pose_virtual[2]-near_obj_pose_virtual[2]
   if delta_theta>np.pi:
      delta_theta-=2*np.pi
   if delta_theta<-np.pi:
      delta_theta+=2*np.pi

   #计算虚拟相机从near模板位姿到cur位姿的旋转矩阵
   delta_R=transforms3d.axangles.axangle2mat([0,0,1], delta_theta)

   # 先旋转物体位姿，计算仅旋转后的物体位姿在当前的虚拟归一化坐标系下的位姿
   R_2d=delta_R.T[:2,:2]
   rotated_cur_obj_pose_virtual=R_2d @ cur_obj_pose_virtual

   #再平移物体位姿，并计算虚拟位姿的平移增量
   cur_z=cur_T_base_virtual[2,3]
   alpha=(cur_z-near_z)/(far_z-near_z)
   target_ratio=near_ratio+alpha*(far_ratio-near_ratio)
   targrt_uv= near_obj_pose_virtual[:2]+target_ratio*(far_obj_pose_virtual[:2]-cur_obj_pose_virtual[:2])
   #物体位姿旋转到与目标位姿一致后，与目标位姿在归一化坐标系中的平移向量，由目标位姿指向当前位姿
   #由于相机的移动方向与物体在视野中的方向相反，故无需变号
   delta_uv=rotated_cur_obj_pose_virtual[:2]-targrt_uv
   delta_xy=target_ratio @ delta_uv

   diff_z=cur_z-near_z
   if abs(diff_z)>z_step:
      delta_z=np.sign(diff_z)*z_step
   else:
      delta_z=diff_z

   # delta_z = 0   # 调试: 先不控制 z 轴移动, 只控制 x,y 平移和旋转
   delta_xyz = np.array([delta_xy[0], delta_xy[1], delta_z])

   #计算最终的虚拟相机位姿增量
   #旋转矩阵
   delta_T0=np.eye(4)
   delta_T0[:3,:3]=delta_R
   #平移矩阵
   delta_T1=np.eye(4)
   delta_T1[0:3,3]=[delta_xyz[0],delta_xyz[1],diff_z]

   delta_T_virtual=delta_T0 @ delta_T1

   #将虚拟相机位姿增量转换到末端坐标系下
   delta_T_end = T_end_virtual@ delta_T_virtual @ inv_tf(T_end_virtual)

   return delta_T_end










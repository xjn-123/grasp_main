"""
检测夹爪与环境的碰撞情况
"""

import os
import logging
import time

from typing_extensions import List, Tuple, Dict

import transforms3d
import numpy as np
import cv2
import open3d

# 导入本工程的模块
from .utils import (
    GREEN, YELLOW, RESET,
    inv_tf
)
from .vision_utils import rgbd_to_point_cloud, depth_to_point_cloud


######################################################### 全局变量 #########################################################

TH_ANGLE_Z = 45.0 * np.pi / 180.0
"""两个坐标系之间的 Z 轴夹角阈值, 单位: 弧度"""

TH_GRIPPER_HEIGHT = -0.01
"""夹爪中心在基座坐标系的高度阈值, 单位: 米"""


######################################################### 类定义 #########################################################

class GripperBody:
    """
    夹爪几何体,使用两个矩形来表示夹爪与环境的接触面.预设条件:     
    1) 仅适用于眼在手的场景
    2) 夹爪坐标系的原点位于两片爪尖的几何中心
    3) 以夹爪张开方向为夹爪坐标系的 X 轴,从左爪指向右爪;    
    4) 夹爪坐标系的 Z 轴垂直于夹爪平面,朝向与末端坐标系 Z 轴的夹角近似平行;
    """

    def __init__(self,
                 width: float,
                 thickness: float,
                 T_cam_gripper: np.ndarray = np.eye(4)):
        """
        构造函数
        Args:
            width (float): 夹爪宽度 (单位: m)
            thickness (float): 夹爪厚度 (单位: m)
            T_cam_gripper (np.ndarray): 从夹爪坐标系到相机坐标系的位姿变换, 形状为 (4, 4)
        """

        self.width = width
        self.thickness = thickness
        self.T_cam_gripper = T_cam_gripper  # 夹爪在相机坐标系下的位姿
    # end def __init__

    def initialize(self,
                   corners3d: np.ndarray):
        """
        初始化: 计算从夹爪坐标系到相机坐标系的位姿变换
        Args:
            corners3d (np.ndarray): 夹爪平面的 AprilTag 标签的四个角点的3D坐标,形状为 (4, 3)
        """

        assert corners3d.shape == (4, 3), "corners3d must have shape (4, 3)"

        # 坐标系定义, 从 corners3d[0] 指向 corners3d[2] 的方向为 X 轴, 从 corner[3] 指向 corner[1] 的方向为 Y 轴
        # Z 轴为平面的法向量,由右手定则确定
        # 原点为 corners3d[0] 和 corners3d[2] 的中点

        center_3d = corners3d.mean(axis=0)  # 更鲁棒的中心
        edge_x = corners3d[2] - corners3d[0]
        edge_y = corners3d[1] - corners3d[3]
        if np.linalg.norm(edge_x) < 1e-9 or np.linalg.norm(edge_y) < 1e-9:
            raise ValueError("角点退化,无法建立坐标系")
        # end if

        axis_x = edge_x / np.linalg.norm(edge_x)

        # 先正交化 Y
        axis_y_raw = edge_y / np.linalg.norm(edge_y)
        axis_y = axis_y_raw - np.dot(axis_y_raw, axis_x) * axis_x
        axis_y /= np.linalg.norm(axis_y)

        # 右手系 Z
        axis_z = np.cross(axis_x, axis_y)
        axis_z /= np.linalg.norm(axis_z)
        assert axis_z[2] > 0, "gripper Z axis direction error"

        self.T_cam_gripper[:3, :3] = np.column_stack((axis_x, axis_y, axis_z))
        self.T_cam_gripper[:3, 3] = center_3d.reshape(3)

        logging.info(f'Gripper pose set. T_cam_gripper:\n{self.T_cam_gripper}')

    # end def initialize

    def get_rects_3d(self,
                     dist: float,
                     T_target_cam: np.ndarray = np.eye(4)) -> np.ndarray:
        """
        用两个空间矩形表示夹爪的位置,计算夹爪的八个角点的3D坐标( 目标坐标系下 )     
        Args:
            dist (float): 两个夹爪矩形之间的距离 (单位: m)
            T_target_cam (np.ndarray): 从相机坐标系到目标坐标系的变换矩阵, 形状为 (4, 4)
        Returns:
            (np.ndarray): 两个夹爪夹爪共八个角点的3D坐标, 形状为 (8, 3), 每个夹爪四个角点按[左下,右下,右上,左上]顺序排列
        """

        w = self.width
        t = self.thickness
        hd = dist / 2.0
        hw = w / 2.0

        # 计算夹爪四个角点在夹爪坐标系下的坐标
        left_rect = np.array([[-hd - t, hw, 0, 1],
                              [-hd, hw, 0, 1],
                              [-hd, -hw, 0, 1],
                              [-hd - t, -hw, 0, 1]], dtype=np.float32).T  # (4,4)

        right_rect = np.array([[hd, hw, 0, 1],
                               [hd + t, hw, 0, 1],
                               [hd + t, -hw, 0, 1],
                               [hd, -hw, 0, 1]], dtype=np.float32).T  # (4,4)

        T_target_gripper = T_target_cam @ self.T_cam_gripper      # 从夹爪坐标系到目标坐标系的变换矩阵
        left_rect_3d = (T_target_gripper @ left_rect).T[:, :3]    # (4,3)
        right_rect_3d = (T_target_gripper @ right_rect).T[:, :3]  # (4,3)
        rects_3d = np.vstack((left_rect_3d, right_rect_3d))       # (8,3)
        return rects_3d
    # end def get_rects_3d
# end class Gripper


class CollisionDetector:
    """
    碰撞检测器( 仅适用于眼在手的场景 )      
    通过将夹爪在目标位置下的空间矩形投影到参考位置下的深度图像上,计算投影深度图与真实深度图的差异来判断是否发生碰撞
    """

    def __init__(self,
                 gripper_body: GripperBody,
                 T_end_cam: np.ndarray,
                 intrinsic: List[float],
                 depth_scale: float,
                 distortion: List[float] = None,
                 debug_dir: str = None):
        """
        构造函数
        Args:
            gripper_body (GripperBody): 夹爪几何体
            T_end_cam (np.ndarray): 从相机坐标系到机械臂末端的变换矩阵, 形状为 (4, 4)
            intrinsic (List[float]): 相机内参 [fx, fy, cx, cy]
            depth_scale (float): 深度图像的缩放比例
            distortion (List[float], optional): 相机畸变参数,格式与 OpenCV 一致. 默认值为 None
            debug_dir (str, optional): 用于保存调试信息的目录. 默认值为 None
        """
        assert len(intrinsic) == 4, "intrinsic must have 4 elements: [fx, fy, cx, cy]"

        self.gripper_body = gripper_body
        self.T_end_cam = T_end_cam

        self.intrinsic = intrinsic
        self.depth_scale = depth_scale
        self.distortion = distortion
        self.color_undistort_maps = []  # 去畸变映射表, [map1, map2], 由 cv2.initUndistortRectifyMap 计算得到

        self.debug_dir = debug_dir
        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)
        # end if
    # end def __init__

    def _undistort_color_img(self,
                             color_img: np.ndarray) -> np.ndarray:
        """
        对彩色图像去畸变
        Args:
            color_img (np.ndarray): 输入的彩色图像 CV_8UC3
        Returns:
            (np.ndarray): 去畸变后的颜色图像 CV_8UC3
        """

        if self.distortion is None:
            return color_img
        # end if

        # 初始化畸变校正映射表
        if len(self.color_undistort_maps) == 0:
            K = np.array([[self.intrinsic[0], 0, self.intrinsic[2]],
                          [0, self.intrinsic[1], self.intrinsic[3]],
                          [0, 0, 1]], dtype=np.float32)
            D = np.array(self.distortion, dtype=np.float32)
            img_w, img_h = color_img.shape[1], color_img.shape[0]
            map1, map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=K,
                distCoeffs=D,
                R=None,
                newCameraMatrix=K,
                size=(img_w, img_h),
                m1type=cv2.CV_16SC2
            )
            self.color_undistort_maps = [map1, map2]

            logging.info("undistort rectify maps initialized.")
        # end if

        undistorted_img = cv2.remap(color_img, self.color_undistort_maps[0], self.color_undistort_maps[1],
                                    interpolation=cv2.INTER_LINEAR)

        return undistorted_img
    # end def _undistort_color_img

    def _proj_rect_3d(self,
                      intrinsic: List[float],
                      rect_3d: np.ndarray,
                      proj_mask_img: np.ndarray,
                      proj_depth_img: np.ndarray,
                      mask_value: int = 255) -> np.ndarray:
        """
        将3D矩形投影到图像平面上,并计算投影深度图、投影掩码图、投影轮廓
        Args:
            intrinsic (List[float]): 相机内参 [fx, fy, cx, cy]
            rect_3d (np.ndarray): 矩形的四个角点的3D坐标,形状为 (4, 3)
            proj_mask_img (np.ndarray): 投影得到的掩码图 CV_8UC1
            proj_depth_img (np.ndarray): 投影得到的深度图 CV_16UC1
            mask_value (int): 掩码图中表示矩形区域的像素值
        Returns:
            (np.ndarray): 投影轮廓点的像素坐标列表,形状为 (N, 2)
        """

        # 计算平面方程 Ax + By + Cz + D = 0
        pt0, pt1, pt2 = rect_3d[0], rect_3d[1], rect_3d[2]
        v1 = pt1 - pt0
        v2 = pt2 - pt0
        normal = np.cross(v1, v2)
        D = -np.dot(normal, pt0)
        A, B, C = normal[0], normal[1], normal[2]

        # 计算投影
        fx, fy, cx, cy = intrinsic
        contour = []
        for i in range(4):
            pt3d = rect_3d[i]
            u = int(pt3d[0] * fx / pt3d[2] + cx)
            v = int(pt3d[1] * fy / pt3d[2] + cy)
            contour.append([u, v])
        # end for
        contour = np.array(contour, dtype=np.int32)

        # 填充多边形区域
        cv2.fillPoly(proj_mask_img, [contour], color=mask_value)

        # 计算投影深度
        vs, us = np.where(proj_mask_img == mask_value)
        for u, v in zip(us, vs):
            # 计算归一化图像坐标
            nx, ny = (u - cx) / fx, (v - cy) / fy

            # 计算平面方程与视线的交点深度值
            z = -D / (A * nx + B * ny + C)
            if z <= 0.001:
                continue
            # end if

            d = np.uint16(z / self.depth_scale)

            if d > proj_depth_img[v, u]:
                proj_depth_img[v, u] = d
            # end if
        # end for

        return contour
    # end def _proj_rect_3d

    def _compute_depth_diff(self,
                            real_depth_img: np.ndarray,
                            proj_depth_img: np.ndarray,
                            proj_mask_img: np.ndarray,
                            proj_contour: np.ndarray,
                            mask_value: int,
                            max_depth_diff: float,) -> Tuple[float, float]:
        """
        计算投影深度图与真实深度图的深度差异
        Args:
            real_depth_img (np.ndarray): 真实深度图 CV_16UC1
            proj_depth_img (np.ndarray): 投影深度图 CV_16UC1
            proj_mask_img (np.ndarray): 投影掩码图 CV_8UC1
            proj_contour (np.ndarray): 投影轮廓点的像素坐标列表,形状为 (N, 2)
            mask_value (int): 掩码图中表示矩形区域的像素值
            max_depth_diff (float): 最大允许的深度差异 (单位: m)
        Returns:
            (Tuple[float, float]): 有效像素点数量, 深度差异过大( 投影深度大于实际深度 )的像素点数量
        """

        th_diff = max_depth_diff / self.depth_scale
        valid_cnt = 0
        bad_cnt = 0
        bbox = cv2.boundingRect(proj_contour)  # x, y, w, h
        v0 = max(bbox[1] - 1, 0)
        v1 = min(bbox[1] + bbox[3], real_depth_img.shape[0] - 1)
        u0 = max(bbox[0] - 1, 0)
        u1 = min(bbox[0] + bbox[2], real_depth_img.shape[1] - 1)
        for v in range(v0, v1):
            for u in range(u0, u1):
                if proj_mask_img[v, u] != mask_value:
                    continue
                # end if

                d_proj = proj_depth_img[v, u]
                d_real = real_depth_img[v, u]
                if d_real == 0 or d_proj == 0:
                    continue
                # end if

                valid_cnt += 1

                if d_proj > d_real + th_diff:
                    bad_cnt += 1
                # end if
            # end for
        # end for

        return valid_cnt, bad_cnt
    # end def _compute_depth_diff

    def check(self,
              gripper_dist: float,
              ref_T_base_end: np.ndarray,
              target_T_base_end: np.ndarray,
              ref_bgr_img: np.ndarray,
              ref_depth_img: np.ndarray,
              max_depth_diff: float = 0.01,
              debug_level: int = 0) -> bool:
        """
        检测夹爪在目标位置是否与环境碰撞
        Args:
            gripper_dist (float): 夹爪张开距离 (单位: m)
            ref_T_base_end (np.ndarray): 参考位置下机械臂末端的位姿矩阵,形状为 (4, 4)
            target_T_base_end (np.ndarray): 目标位置下机械臂末端的位姿矩阵,形状为 (4, 4)
            ref_bgr_img (np.ndarray): 参考位置下的彩色图像 CV_8UC3
            ref_depth_img (np.ndarray): 参考位置下的深度图像 CV_16UC1
            max_depth_diff (float): 最大允许的深度差异 (单位: m). 默认值为 0.01
            debug_level (int): 调试等级, 0: 不输出调试信息; 1: 输出投影图像; 2: 输出点云. 默认值为 0
        Returns:
            (bool): 是否发生碰撞
        """

        st = time.time()

        # 从 target 位置到 ref 位置,相机位姿的变换
        cam_T_ref_target = inv_tf(ref_T_base_end @ self.T_end_cam) @ (target_T_base_end @ self.T_end_cam)

        # 计算 target 位置下的夹爪在 ref 位置下的相机坐标系的空间矩形的顶点 8*3
        gripper_rects_3d = self.gripper_body.get_rects_3d(gripper_dist, cam_T_ref_target)

        # 计算投影
        proj_mask_img = np.zeros_like(ref_depth_img, dtype=np.uint8)
        proj_depth_img = np.zeros_like(ref_depth_img, dtype=np.uint16)

        left_contour = self._proj_rect_3d(self.intrinsic, gripper_rects_3d[0:4],
                                          proj_mask_img, proj_depth_img, mask_value=200)  # 左夹爪
        right_contour = self._proj_rect_3d(self.intrinsic, gripper_rects_3d[4:8],
                                           proj_mask_img, proj_depth_img, mask_value=255)  # 右夹爪

        left_valid_cnt, left_bad_cnt = self._compute_depth_diff(ref_depth_img, proj_depth_img, proj_mask_img, left_contour,
                                                                mask_value=200, max_depth_diff=max_depth_diff)
        right_valid_cnt, right_bad_cnt = self._compute_depth_diff(ref_depth_img, proj_depth_img, proj_mask_img, right_contour,
                                                                  mask_value=255, max_depth_diff=max_depth_diff)

        logging.info(f'check_collision cost time( ms ): {(time.time() - st) * 1000:.1f}')
        logging.info(f'check_collision (bad_cnt/valid_cnt), left: {left_bad_cnt}/{left_valid_cnt}, right: {right_bad_cnt}/{right_valid_cnt}')

        if debug_level >= 1:
            assert self.debug_dir is not None, "debug_dir must be set for debug output"

            # 彩色图像去畸变
            bgr_img = self._undistort_color_img(ref_bgr_img)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # 绘制夹爪投影轮廓
            cv2.polylines(bgr_img, [left_contour], isClosed=True, color=(255, 0, 0), thickness=1)
            cv2.polylines(bgr_img, [right_contour], isClosed=True, color=(0, 0, 255), thickness=1)
            proj_img_path = f"{self.debug_dir}/proj.png"
            cv2.imwrite(proj_img_path, bgr_img)
            logging.info(f"gripper projection image saved to: {proj_img_path}")

            if debug_level >= 2:
                # 保存点云
                depth_range = (0.07, 0.3)
                rgbd_pc = rgbd_to_point_cloud(rgb_img=rgb_img,
                                              depth_img=ref_depth_img.astype(np.float32) * self.depth_scale,
                                              intrinsic=self.intrinsic,
                                              depth_range=depth_range)
                gripper_pc = depth_to_point_cloud(depth_img=proj_depth_img.astype(np.float32) * self.depth_scale,
                                                  intrinsic=self.intrinsic,
                                                  depth_range=depth_range)
                gripper_pc.paint_uniform_color([0.0, 1.0, 0.0])  # 夹爪点云染色

                rgbd_pc += gripper_pc

                pc_path = f"{self.debug_dir}/scene.pcd"
                open3d.io.write_point_cloud(pc_path, rgbd_pc, write_ascii=True)
                logging.info(f"scene point cloud saved to: {pc_path}")
            # end if
        # end if

        if left_bad_cnt > max(20, 0.4 * left_valid_cnt) or right_bad_cnt > max(20, 0.4 * right_valid_cnt):
            is_obstacled = True
        else:
            is_obstacled = False
        # end if

        return is_obstacled
    # end def check
# end class CollisionDetector


######################################################### 函数定义 #########################################################

def check_arm_pose(T_base_end: np.ndarray,
                   T_end_cam: np.ndarray,
                   gripper_body: GripperBody,
                   gripper_dist: float,
                   th_angle_z: float,
                   th_gripper_height: float) -> bool:
    """
    检查机械臂位姿是否合理:     
        - 末端 Z 轴与基座坐标系的 -Z 轴的夹角小于 th_angle_z
        - 夹爪在基座坐标系下的高度高于 th_gripper_height
    Args:
        T_base_end (np.ndarray): 从机械臂末端到基座坐标系下的变换矩阵, 4*4
        T_end_cam (np.ndarray): 从相机到机械臂末端坐标系下的变换矩阵, 4*4
        gripper_body (GripperBody): 夹爪对象
        gripper_dist (float): 夹爪距离
        th_angle_z (float): 末端与基座 Z 轴夹角阈值, 单位: 弧度
        th_gripper_height (float): 夹爪高度( 基坐标系下 )阈值, 单位: 米
    """

    # 1. 与 -Z 轴的夹角检查
    R_base_end = T_base_end[:3, :3]
    z_dir = R_base_end[:, 2]  # 机械臂末端 Z 轴方向
    z_axis = np.array([0, 0, -1], dtype=np.float32)  # 基座坐标系 Z 轴负方向
    cosine = np.dot(z_dir, z_axis) / (np.linalg.norm(z_dir) * np.linalg.norm(z_axis))
    angle = np.arccos(cosine)
    logging.info(f'Arm end-effector Z axis angle check, angle: {angle * 180.0 / np.pi:.2f} deg')

    if angle > th_angle_z:  # 夹角大于45度
        logging.error(f'Arm end-effector Z axis angle check failed, should be less than {th_angle_z* 180.0 / np.pi:.2f} deg')
        return False
    # end if

    # 2. 夹爪位置检查, z 坐标高于一定高度
    T_base_cam = T_base_end @ T_end_cam
    gripper_rect3_3d = gripper_body.get_rects_3d(gripper_dist, T_base_cam)  # 计算夹爪在基座坐标系下的8个顶点坐标 8*3
    gripper_height = np.min(gripper_rect3_3d[:, 2])  # 夹爪最低点的高度
    logging.info(f'Arm gripper height check, height: {gripper_height:.3f} m')

    if gripper_height < th_gripper_height:  # 夹爪高度低于阈值
        logging.error(f'Arm gripper height check failed, should be higher than {th_gripper_height} m')
        return False
    # end if

    return True
# end def check_arm_pose


def compute_axis_aligned_pose(T_base_end: np.ndarray,
                              base_axis_idx: int,
                              obj_axis_idx: int,
                              T_end_obj: np.ndarray = np.eye(4),
                              th_angle: float = 45.0) -> np.ndarray:
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
        logging.error(f'Invalid base_axis_idx: {base_axis_idx}')
        return None
    # end if

    if abs(obj_axis_idx) not in [1, 2, 3]:
        logging.error(f'Invalid obj_axis_idx: {obj_axis_idx}')
        return None
    # end if

    T_base_obj = T_base_end @ T_end_obj

    # 期望的对齐方向, 由 base_axis_idx 决定
    target_dir = np.eye(3)[:, abs(base_axis_idx) - 1] * np.sign(base_axis_idx)

    # 当前的方向, 由 obj_axis_idx 决定
    current_dir = T_base_obj[:3, abs(obj_axis_idx) - 1] * np.sign(obj_axis_idx)

    # 计算夹角
    cosine = np.dot(target_dir, current_dir)
    cosine = np.clip(cosine, -1.0, 1.0)  # 数值稳定性
    angle = np.arccos(cosine)
    logging.info(f'compute_axis_aligned_pose, target_dir: {GREEN}{target_dir}{RESET}, '
                 f'current_dir: {GREEN}{current_dir}{RESET}, '
                 f'angle: {GREEN}{angle * 180.0 / np.pi:.2f}{RESET} deg')

    if angle > np.deg2rad(th_angle):  # 夹角过大
        logging.warning(f'angle > {th_angle} deg, skip align')
        return None
    # end if

    # 计算调整后的姿态
    axis = np.cross(current_dir, target_dir)
    delta_R = transforms3d.axangles.axangle2mat(axis, angle)

    target_T_base_obj = T_base_obj.copy()
    target_T_base_obj[:3, :3] = delta_R @ T_base_obj[:3, :3]
    target_T_base_end = target_T_base_obj @ inv_tf(T_end_obj)

    return target_T_base_end
# end def compute_axis_aligned_pose

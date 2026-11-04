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
from core.vision_utils.vision_utils import (rgbd_to_point_cloud,
                                            depth_to_point_cloud,
                                            ImageUndistorter)

Logger = logging.getLogger(__name__)

################全局变量#####################
TH_ANGLE_Z = 45.0 * np.pi / 180.0
"""两个坐标系之间的 Z 轴夹角阈值, 单位: 弧度"""

TH_GRIPPER_HEIGHT = -0.01
"""夹爪中心在基座坐标系的高度阈值, 单位: 米"""

DEPTH_RANGE = (0.07, 0.3)
"""CollisionDetector.check_collision方法调试信息的点云深度范围, 单位: 米"""

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

    def __init__(self, gripper: Gripper,T_end_cam: np.ndarray, intrinsic: list[float], depth_scale: float, debug_dir: str = None):

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

        if len(intrinsic) != 4:
            Logger.error(f"{Common_data.RED}intrinsic参数长度错误,应为4,实际为{len(intrinsic)}{Common_data.RESET}")
            return

        self.gripper = gripper
        self.T_end_cam = T_end_cam
        self.intrinsic = intrinsic
        self.depth_scale = depth_scale
        self.debug_dir = debug_dir
        self.color_undistort_maps = [] # 去畸变映射表, [map1, map2], 由 cv2.initUndistortRectifyMap 计算得到

        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)

    def _undistort_color_img(self, color_img: np.ndarray) ->np.ndarray:
        pass

    def proj_rect_3d(self, intrinsic: list[float], rect_3d: np.ndarray, proj_mask_img: np.ndarray, proj_depth_img: np.ndarray, mask_value: int = 255)-> np.ndarray:
        """
        将3D点投影到图像平面
        Args:
            intrinsic (list[float]): 相机内参 [fx, fy, cx, cy]
            rect_3d (np.ndarray): 3D点, 形状为 (N, 3)
            proj_mask_img (np.ndarray): 投影后的图像, 形状为 (H, W)
            proj_depth_img (np.ndarray): 投影后的深度图, 形状为 (H, W)
            mask_value (int, optional): 投影后的图像中用于表示投影点位置的值. 默认值为 255
        Returns:
            (np.ndarray): 投影后的图像, 形状为 (H, W)
        """

        #计算夹爪平面的在相机坐标系的平面方程
        pt0,pt1,pt2=rect_3d[0],rect_3d[1],rect_3d[2]
        v1=pt1-pt0
        v2=pt2-pt0
        normal=np.cross(v1,v2)
        A,B,C=normal[0],normal[1],normal[2]
        D=-np.dot(normal,pt0)

        #计算夹爪平面在图像平面的投影
        fx,fy,cx,cy=intrinsic
        #投影坐标
        contour=[]
        for i in range(4):
            pt3d=rect_3d[i]
            u=int(fx*pt3d[0]/pt3d[2]+cx)
            v=int(fy*pt3d[1]/pt3d[2]+cy)
            contour.append([u,v])

        contour=np.array(contour,dtype=np.int32)

        #填充夹爪四个角点围成的多边形
        cv2.fillPoly(proj_mask_img,[contour],mask_value)

        #计算夹爪平面在图像平面的深度值
        vs,us=np.where(proj_mask_img==mask_value)
        for u,v in zip(us,vs):
            #计算归一化坐标
            nx=(u-cx)/fx
            ny=(v-cy)/fy

            #计算平面方程与坐标点(nx,ny,1)的交点值
            z=(-A*nx-B*ny-D)/C
            if z<0.001:
                continue

            d=np.uint16(z/self.depth_scale)

            if d > proj_depth_img[v, u]:
                proj_depth_img[v, u] = d

        return contour

    def compute_depth_diff(self, real_depth_img: np.ndarray, proj_depth_img: np.ndarray, proj_mask_img: np.ndarray, proj_contour: np.ndarray, mask_value: int, max_depth_diff: float) -> Tuple[float, float]:
        """
        计算真实深度图与投影深度图之间的深度差
        Args:
            real_depth_img (np.ndarray): 真实深度图, 形状为 (H, W)
            proj_depth_img (np.ndarray): 投影后的深度图, 形状为 (H, W)
            proj_mask_img (np.ndarray): 投影掩码图, 形状为 (H, W)
            proj_contour (np.ndarray): 投影后的轮廓点, 形状为 (N, 2)
            mask_value (int): 投影后的图像中用于表示投影点位置的值
            max_depth_diff (float): 最大深度差, 单位: 米    
        Returns:
            (Tuple[float, float]):有效像素数量，投影深度大于实际深度的像素数量
        """

        th_diff = max_depth_diff / self.depth_scale#投影与真实深度图最大像素值差
        valid_cnt = 0#有效像素数量
        bad_cnt = 0#投影深度大于实际深度的像素数量

        bbox = cv2.boundingRect(proj_contour)#投影后的轮廓的边界框,投影后的轮廓不再是矩形
        v0=max(bbox[1], 0)
        v1=min(bbox[1]+bbox[3], real_depth_img.shape[0])
        u0=max(bbox[0], 0) 
        u1=min(bbox[0]+bbox[2], real_depth_img.shape[1])

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

        return valid_cnt, bad_cnt


    def check_collision(self,
                        gripper_dist: float, 
                        ref_T_base_end: np.ndarray,
                        target_T_base_end: np.ndarray,
                        ref_bgr_img: np.ndarray,
                        ref_depth_img: np.ndarray, 
                        maxx_depth_diff: float = 0.01,
                        debug_lvel: int = 0) -> bool:
        '''检查夹爪是否发生碰撞
        Args:
            gripper_dist (float): 夹爪张开距离, 单位: 米
            ref_T_base_end (np.ndarray): 参考位姿下的末端位姿, 形状为 (4, 4)
            target_T_base_end (np.ndarray): 目标位姿下的末端位姿, 形状为 (4, 4)
            ref_bgr_img (np.ndarray): 参考位姿下的 BGR 图像, 形状为 (H, W, 3)
            real_depth_img (np.ndarray): 实际深度图像, 形状为 (H, W)
            maxx_depth_diff (float): 最大深度差, 单位: 米
            debug_lvel (int): 调试等级, 0 表示不保存调试信息, 1 输出投影图像, 2输出点云，默认为0 
            
        Returns:
            (bool): True 表示发生碰撞, False 表示未发生碰撞
        '''
        st=time.time()

        # 计算目标位置的相机到参考位姿下的相机的变换矩阵
        cam_T_ref_target = inv_tf(self.T_end_cam) @ inv_tf(ref_T_base_end) @ target_T_base_end @ self.T_end_cam

        #计算目标位置的夹爪在参考位姿下的相机坐标系下的空间矩形角点坐标
        target_rects_3d = self.gripper.get_rects_3d(gripper_dist, cam_T_ref_target)

        #初始化掩码图和投影深度图
        proj_mask_img = np.zeros_like(ref_depth_img, dtype=np.uint8)
        proj_depth_img = np.zeros_like(ref_depth_img, dtype=np.uint16)

        #将夹爪的两个矩形投影到图像平面
        left_contour = self.proj_rect_3d(self.intrinsic, target_rects_3d[:4],
                                        proj_mask_img, proj_depth_img, mask_value=255)
        right_contour = self.proj_rect_3d(self.intrinsic, target_rects_3d[4:], 
                                        proj_mask_img, proj_depth_img, mask_value=255)

        #计算投影深度图与实际深度图之间的深度差
        left_valid_cnt, left_bad_cnt = self.compute_depth_diff(ref_depth_img, proj_depth_img, proj_mask_img, left_contour, 
                                                                mask_value=255, max_depth_diff=maxx_depth_diff)

        right_valid_cnt, right_bad_cnt = self.compute_depth_diff(ref_depth_img, proj_depth_img, proj_mask_img, right_contour, 
                                                                mask_value=255, max_depth_diff=maxx_depth_diff)

        Logger.info(f"{Common_data.GREEN}check_collision, 投影深度图与实际深度图之间的深度差计算耗时: {(time.time() - st)*1000:.2f} ms{Common_data.RESET}")
        Logger.info(f"{Common_data.GREEN}check_collision, 左夹爪投影有效像素数量: {left_valid_cnt}, 投影深度大于实际深度的像素数量: {left_bad_cnt}{Common_data.RESET}")
        Logger.info(f"{Common_data.GREEN}check_collision, 右夹爪投影有效像素数量: {right_valid_cnt}, 投影深度大于实际深度的像素数量: {right_bad_cnt}{Common_data.RESET}")

        #输出调试信息
        if debug_lvel >= 1 and self.debug_dir is not None:

            #彩色图去畸变
            image_undistorter = ImageUndistorter(self.intrinsic)
            undistorted_bgr_img = image_undistorter.undistort(ref_bgr_img)
            rgb_img = cv2.cvtColor(undistorted_bgr_img, cv2.COLOR_BGR2RGB)

            #绘制夹爪轮廓投影
            cv2.polylines(rgb_img, [left_contour], isClosed=True, color=(0, 0, 255), thickness=1)
            cv2.polylines(rgb_img, [right_contour], isClosed=True, color=(0, 0, 255), thickness=1)

            proj_img_path = os.path.join(self.debug_dir, "proj_img.png")
            cv2.imwrite(proj_img_path, rgb_img)
            Logger.info(f"{Common_data.GREEN}check_collision, 投影图像保存到: {proj_img_path}{Common_data.RESET}")

            if debug_lvel >= 2:

                rgbd_pc=rgbd_to_point_cloud(rgb_img=rgb_img, 
                                            depth_img=ref_depth_img, 
                                            intrinsic=self.intrinsic,
                                            depth_range=DEPTH_RANGE,
                                            depth_trunc=3.0,
                                            depth_scale=self.depth_scale
                                        )

                gripper_pc=depth_to_point_cloud(depth_img=proj_depth_img,
                                                intrinsic=self.intrinsic,
                                                depth_range=DEPTH_RANGE,
                                                depth_trunc=3.0,
                                                depth_scale=self.depth_scale
                                        )

                #夹爪点云颜色
                gripper_pc.paint_uniform_color([0.0, 1.0, 0.0])

                rgbd_pc+=gripper_pc

                pc_save_path = os.path.join(self.debug_dir, "pc.pcd")
                open3d.io.write_point_cloud(pc_save_path, rgbd_pc)
                Logger.info(f"{Common_data.GREEN}check_collision, 点云保存到: {pc_save_path}{Common_data.RESET}")

        #判断是否碰撞
        if left_bad_cnt>max(0,0.4*left_valid_cnt) or right_bad_cnt>max(0,0.4*right_valid_cnt):
            Logger.warning(f"{Common_data.RED}夹爪发生碰撞{Common_data.RESET}")
            return True
        return False

        

#####################夹爪位姿调整与检查函数#######################
def check_gripper_pose(T_base_end: np.ndarray, 
                    T_end_cam: np.ndarray,
                    gripper: Gripper,
                    gripper_dist: float,
                    th_angle_z: float ,
                    th_gripper_height: float ) -> bool:
    """
    检查夹爪位姿是否满足约束条件
    - 末端 Z 轴与基座坐标系的 -Z 轴的夹角小于 th_angle_z
    - 夹爪在基座坐标系下的高度高于 th_gripper_height
    Args:
        T_base_end (np.ndarray): 从机械臂末端到基座坐标系下的变换矩阵, 4*4
        T_end_cam (np.ndarray): 从相机到机械臂末端坐标系下的变换矩阵, 4*4
        gripper (Gripper): 夹爪对象
        gripper_dist (float): 夹爪距离
        th_angle_z (float): 末端与基座 Z 轴夹角阈值, 单位: 弧度
        th_gripper_height (float): 夹爪在基座坐标系下的高度阈值, 单位: 米
    Returns:
        (bool): True 表示满足约束条件, False 表示不满足约束条件
    """
    #1.夹角检查

    # 计算末端 Z 轴与基座 -Z 轴的夹角
    R_end_base = T_base_end[:3, :3]
    # 末端 Z 轴在基座坐标系下的方向
    z_dir= R_end_base[:, 2]
    # 基座 -Z 轴在基座坐标系下的方向
    z_axis = np.array([0, 0, -1], dtype=np.float32)
    # 计算夹角
    cosine = np.dot(z_dir, z_axis) / (np.linalg.norm(z_dir) * np.linalg.norm(z_axis))
    ang=np.arccos(np.clip(cosine, -1.0, 1.0))
    Logger.info(f"{Common_data.GREEN}夹爪 Z 轴与基座 Z 轴夹角: {ang * 180.0 / np.pi:.2f} deg{Common_data.RESET}")

    if ang > th_angle_z:
        Logger.warning(f"{Common_data.RED}夹爪 Z 轴与基座 Z 轴夹角太大, 不满足约束条件{Common_data.RESET}")
        return False

    
    #2.高度检查

    # 计算夹爪在基座坐标系下的高度
    T_base_cam = T_base_end @ T_end_cam
    # 夹爪在基座坐标系下的高度
    gripper_rect_3d = gripper.get_rects_3d(gripper_dist, T_base_cam)
    gripper_height = np.min(gripper_rect_3d[:, 2])# 夹爪最低点高度
    Logger.info(f"{Common_data.GREEN}夹爪高度: {gripper_height:.3f} m{Common_data.RESET}")

    if gripper_height < th_gripper_height:
        Logger.warning(f"{Common_data.RED}夹爪高度太低, 不满足约束条件{Common_data.RESET}")
        return False

    return True



    

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


    


    


    



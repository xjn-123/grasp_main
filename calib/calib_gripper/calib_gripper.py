''' calib_gripper.py
    用于标定夹爪，需满足针孔相机模型和apriltag标定条件，目前仅兼容二指夹爪
'''

import os  # noqa: I001
import sys 
import json
import logging
import argparse
import mmengine

import numpy as np
import cv2
import open3d

code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(code_dir, "/../../")
sys.path.append(root_dir)

from core.board_utils import apriltag_helper  # noqa: I001
import core.common_utils as utils
from core.common_utils.utils import Common_data

logger = logging.getLogger(__name__)

''' 计算apritag在相机坐标系下的坐标，辅助标定夹爪'''
def campute_corners3d(  
    gray_img: np.ndarray,
    depth_img: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    depth_scale: float,
    expand_scale: float = 3.0,     # apriltag的扩展比例,默认3.0
) -> tuple[np.ndarray, np.ndarray]:
    
    ''' 畸变矫正'''
    K=np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]],dtype=np.float32)
    un_img = cv2.undistort(gray_img, K, distortion)


    ''' 检测apriltag的角点'''
    detector = apriltag_helper.Detector(tag_family="tag36h11")
    tag = detector.detect(un_img)
    tag_corners = tag.corners       #四个角点的图像坐标

    #构建点云掩膜,
    tag_center = tag.coenter
    expand_corners = []

    for corner in tag_corners:
        delta=corner-tag_center
        expand_corner = tag_center+delta*expand_scale
        expand_corners.append(expand_corner)

    expand_corners = np.array(expand_corners)

    mask = np.zeros_like(gray_img, dtype=np.uint8)
    pts = expand_corners.astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)

    #将非掩码区域设为0
    masked_depth = depth_img.copy()
    masked_depth[mask == 0] = 0

    #提取掩码区域的点云f
    pc=open3d.geometry.PointCloud.create_from_depth_image(
        open3d.geometry.Image(masked_depth),
        open3d.camera.PinholeCameraIntrinsic(
            gray_img.shape[1],
            gray_img.shape[0],
            intrinsic[0],
            intrinsic[1],
            intrinsic[2],
            intrinsic[3],
        ),
        np.eye(4),
        depth_scale=1.0 / depth_scale,
        depth_trunc=0.5,
        )

    #拟合apriltag平面
    plane,inlines = pc.segment_plane(
        distance_threshold=0.002,
        ransac_n=6,
        num_iterations=1000,
    )
    logger.info(
                f"拟合的平面方程系数: {plane}\n 内点数: {len(inlines)}\n 内点比例: {len(inlines) / len(pc.points)}"
    )

    #计算角点笛卡尔坐标
    def compute_pt3d(
        corner: np.ndarray, intrinsic: np.ndarray, plane: np.ndarray
    ) -> np.ndarray:
        nx = (corner[0] - intrinsic[2]) / intrinsic[0]
        ny = (corner[1] - intrinsic[3]) / intrinsic[1]
        A, B, C, D = plane
        z = -D / (A * nx + B * ny + C)
        x = nx * z
        y = ny * z
        return np.array([x, y, z])

    corners3d = np.array(
        [compute_pt3d(corner, intrinsic, plane) for corner in tag_corners]
    )

    return  corners3d

'''计算T_cam_gripper'''
def camputer_T_cam_gripper(
    corners3d: np.ndarray,
    corners: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    ''' 计算apriltag的旋转矩阵'''


def main():

    parser=argparse.ArgumentParser(description="gripper calibration using apriltag")

    parser.add_argument("--color_image_dir",type=str,required=True,help="彩色图像文件夹路径")
    parser.add_argument("--depth_image_dir",type=str,required=True,help="深度图像文件夹路径")
    parser.add_argument("--camera_param_dir",type=str,required=True,help="相机内参文件路径")
    parser.add_argument("--gripper_param_dir",type=str,required=True,help="夹爪标定结果的写入文件保存路径")
    parser.add_argument("--depth_scale",type=float,required=True,help="深度图depth_scale")
    parser.add_argument("--expand_scale",type=float,default=3.0,help="角点缩放系数")

    args=parser.parse_args()

    color_image_dir=args.color_image_dir
    depth_image_dir=args.depth_image_dir
    camera_param_dir=args.intrinsic_dir
    gripper_param_dir=args.gripper_param_dir
    depth_scale=args.depth_scale
    expand_scale=args.expand_scale

    #读取彩色和深度图
    color_image=cv2.imread(color_image_dir,cv2.IMREAD_COLOR)
    depth_image=cv2.imread(depth_image_dir,cv2.IMREAD_UNCHANGED)

    if color_image is None or depth_image is None:
        logger.error("图像读取失败")
        return

    #读取相机内参
    intrinsic,distortion=utils.read_cam_params(camera_param_dir)

    #计算apriltag的角点3d坐标
    consers_3d=campute_corners3d(color_image,depth_image,intrinsic,distortion,depth_scale,expand_scale)

    '''计算T_cam_gripper'''
    T_cam_gripper=utils.camputer_T_cam_gripper(consers_3d)

    #保存夹爪标定结果
    save_json={}
    save_json["T_cam_gripper"]=T_cam_gripper.tolist()
    mmengine.dump(save_json,gripper_param_dir,indent=4)
    logger.info(f"夹爪标定结果保存到{gripper_param_dir}")

if __name__ == "__main__":
    main()



    

    








    








    

    
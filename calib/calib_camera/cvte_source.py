"""
文件说明:
    读取文件夹中的图像( 拍照了标定板 ), 标定相机( 仅适用于针孔相机模型 )
"""

import argparse
import glob
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import apriltag2
import cv2
import numpy as np


# 导入本工程的模块
# 导入本工程的模块
code_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{code_dir}/../../../")
sys.path.append(root_dir)

from core.utils import (
    BLUE,
    GREEN,
    RED,
    RESET,
    YELLOW,
)

######################################################### 函数定义 #########################################################


def calib_camera(
    tag3d_list: List[apriltag2.Tag3D],
    tag2d_list_list: List[List[apriltag2.Tag2D]],
    img_size: Tuple[int, int],
) -> Tuple[List[float], List[float]]:
    """
    标定针孔相机的内参和畸变参数, 需要至少 10 张图像, 每张图像中至少检测到 4 个 tag2d.
    Args:
        tag3d_list (List[apriltag2.Tag3D]): 3D 标定板上的 tag 信息列表
        tag2d_list_list (List[List[apriltag2.Tag2D]]): 每张图像中检测到的 2D tag 信息列表
        img_size (Tuple[int, int]): 图像的尺寸 (宽, 高)

    Returns:
        Tuple[List[float], List[float]]: 相机的内参和畸变系数, 如果标定失败则返回 (None, None)
    """

    pts3d_list = []
    pts2d_list = []
    for tag2d_list in tag2d_list_list:
        pts3d = []
        pts2d = []

        for tag2d in tag2d_list:
            tag_id = tag2d.id
            tag3d = next((t for t in tag3d_list if t.id == tag_id), None)
            if tag3d is None:
                logging.warning(
                    f"{YELLOW}Tag ID {tag_id} detected in 2D but not found in 3D list. Skipping this tag.{RESET}"
                )
                continue
            # end if

            for i in range(4):
                pts3d.append(tag3d.corners[i])
                pts2d.append(tag2d.corners[i])
            # end for
        # end for

        if len(pts3d) < 4:
            logging.warning(
                f"{YELLOW}Only {len(pts3d)} valid 3D-2D point pairs found in this image. At least 4 are required. Skipping this image.{RESET}"
            )
            continue
        # end if

        pts3d_list.append(np.array(pts3d))
        pts2d_list.append(np.array(pts2d))
    # end for

    if len(pts3d_list) < 10:
        logging.error(
            f"{RED}Only {len(pts3d_list)} valid images with sufficient 3D-2D point pairs found. At least 10 are required for reliable calibration. Exiting.{RESET}"
        )
        return None, None
    # end if

    K = np.eye(3)
    D = np.zeros(5)

    # 执行相机标定 (objectPoints/imagePoints 需为每张图像一个数组的列表)
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=pts3d_list,
        imagePoints=pts2d_list,
        imageSize=img_size,
        cameraMatrix=K,
        distCoeffs=D,
    )

    intrinsic = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]  # fx, fy, cx, cy
    distortion = D.tolist()  # k1, k2, p1, p2, k3

    logging.info(f"Camera calibration RMS error: {ret}")
    logging.info(f"Camera matrix: {GREEN}{intrinsic}{RESET}")
    logging.info(f"Distortion coefficients: {GREEN}{distortion}{RESET}")
    print()

    return intrinsic, distortion


# end def calib_camera


######################################################### 主函数 #########################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="相机标定")

    parser.add_argument(
        "--calib_board_info",
        type=str,
        default="[0.0245,0.0075, 6, 6]",
        help="标定板信息 [tag_size, space_size, tag_rows, tag_cols]",
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        default="/home/i4/桌面/test_location/calib/collect_image/cam0",
        help="保存图像的目录",
    )

    args = parser.parse_args()

    calib_board_info = json.loads(args.calib_board_info)
    img_dir = args.img_dir

    print()
    print(f"标定板信息: {BLUE}{calib_board_info}{RESET}")
    print(f"图像目录: {BLUE}{img_dir}{RESET}")
    print()

    # 创建标定板
    tag3d_list = apriltag2.create_calib_board_3d(
        tag_size=calib_board_info[0],
        space_size=calib_board_info[1],
        rows=calib_board_info[2],
        cols=calib_board_info[3],
        start_tag_id=0,
    )

    # 创建检测器
    detector = apriltag2.Detector(
        tag_family="tag36h11",
        black_border=2,
    )

    # 遍历每张图像, 检测 tag
    img_path_list = glob.glob(f"{img_dir}/*.png")
    img_path_list.sort()
    if len(img_path_list) == 0:
        logging.error(f"{RED}No images found in {img_dir}. Exiting.{RESET}")
        sys.exit(1)
    # end if

    tag2d_list_list = []

    for img_path in img_path_list:
        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(
                f"{YELLOW}Failed to read image {img_name}, skipping.{RESET}"
            )
            continue
        # end if

        tag2d_list = detector.detect(img, -1)
        if len(tag2d_list) == 0:
            logging.warning(
                f"{YELLOW}No tags detected in image {img_name}, skipping.{RESET}"
            )
            continue
        # end if

        # if True:
        #     bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     detector.draw(bgr_img, tag2d_list)
        #     cv2.imshow("tag detection", bgr_img)
        #     cv2.waitKey(500)
        # # end if

        tag2d_list_list.append(tag2d_list)
    # end for

    # 执行相机标定
    img_size = (img.shape[1], img.shape[0])
    intrinsic, distortion = calib_camera(
        tag3d_list=tag3d_list, tag2d_list_list=tag2d_list_list, img_size=img_size
    )
    if intrinsic is None or distortion is None:
        logging.error(f"{RED}Camera calibration failed. Exiting.{RESET}")
        sys.exit(1)
    # end if

    # 保存标定结果
    save_path = os.path.join(os.path.dirname(img_dir), "cam_params.json")
    result_dict = {
        "camera_type": "Pinhole",
        "IntrinsicFormat": "fx,fy,cx,cy",
        "DistortionFormat": "k1,k2,p1,p2,k3",
        "resolution": [img_size[0], img_size[1]],
        "intrinsic": intrinsic,
        "distortion": distortion,
    }

    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    # end if

    logging.info(f"Result saved to: {GREEN}{save_path}.{RESET}")
    

# end if __name__ == '__main__':

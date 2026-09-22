import os
import sys
import logging
import json
import glob
import argparse

import numpy as np
import cv2
import transforms3d

#导入本地工程模块
core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{core_dir}/../../")
sys.path.append(root_dir)

from core.board_utils import apriltag_helper

logger = logging.getLogger(__name__)


def calib_handeye(
    T_base_end__dict: dict[int, np.ndarray],
    T_cam_board_dict: dict[int, np.ndarray],
    eye_in_hand: bool = True,
) -> np.ndarray:

    """
    进行手眼标定, 计算 T_cam_arm

    Args:
        T_base_end__dict: 末端执行器到机械臂基座的变换矩阵字典, 键为采集序号
        T_cam_board_dict: 标定板到相机的变换矩阵字典, 键为采集序号
        eye_in_hand: 是否为手眼标定( True )或固定相机标定( False )

    Returns:
        T_cam_arm: 相机到机械臂末端执行器的变换矩阵
    """
    # 将字典转换为列表, 并按键排序

    def save_AXXB(T01s:np.ndarray, T23s:np.ndarray)->np.ndarray:
        """ 计算AX=XB问题的解 
            输出T12s
        """

        R,t = cv2.calibrateHandEye(
            T01s[:, :3, :3], T01s[:, :3, 3],
            T23s[:, :3, :3], T23s[:, :3, 3],
            method=cv2.CALIB_HAND_EYE_PARK
            )

        T12s = np.eye(4, dtype=np.float64)
        T12s[:3, :3] = R
        T12s[:3, 3] = t.ravel()
        
        return T12s


        

    




    

    


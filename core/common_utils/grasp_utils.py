'''抓取位姿计算辅助函数'''
import numpy as np
import transforms3d
import json
import logging

import os
import sys


root_dir = os.path.normpath(f"{os.path.dirname(os.path.realpath(__file__))}/../../../")# 导入本工程的模块  \
sys.path.append(root_dir)

def gen_symmetric_linalg(self, json_file_path: str, scale: float, rotation_step: float = 1) -> np.ndarray:
    """
    读取 CAD 模型的对称性信息,生成对称变换矩阵列表，支持离散旋转对称和连续旋转对称离散化两种
    json_file_path (str): json 文件路径
        json文件的格式为:
        {
            "symmetries_continuous":
            [
                {
                    "axis":[0,0,1],
                    "offset"[0.1,0,0]
                }
            ]
            "symmetries_discrete":
            [
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]
            ]
            #缩放因子( 将位移单位转换到米制单位的比例 )
            "scale":1.0         
        }
    Returns:
        list[np.ndarray]: 对称变换矩阵列表
    
    Args:
        scale (float): 缩放因子( 将位移单位转换到米制单位的比例 )
        rotation_step (float): 旋转步长( 仅适用于连续旋转 ), 单位: 度
    Returns:
        (np.ndarray): 对称变换矩阵 N*4*4, 如果失败则返回 None
    """


def compute_ready_pose(tmpl_state_list: list[dict],
                       sym_tfs: np.ndarray,
                       T_end_cam: np.ndarray,
                       cur_T_base_end: np.ndarray,
                       cur_T_cam_model: np.ndarray,
                       tmpl_idx: int = -1) -> tuple[np.ndarray, int]:
    """选择并计算最佳抓取模板的预备位姿,适用于输入的是离散的对称变换矩阵

    Args:
        tmpl_state_list (List[Dict]): 模板状态列表
        sym_tfs (np.ndarray): 对称变换矩阵列表
        T_end_cam (np.ndarray): 相机到末端的变换矩阵
        cur_T_base_end (np.ndarray): 当前末端到基坐标系的变换矩阵
        cur_T_cam_model (np.ndarray): 当前相机到模板坐标系的变换矩阵
        tmpl_idx (int, optional): 模板状态列表中的索引. Defaults to -1.
        Returns:
            Tuple[np.ndarray, int]: 抓取准备位姿和索引
    """












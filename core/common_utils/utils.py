"""
通用工具函数和类
"""

import logging
# 如果还没有配置 logging, 则配置一个默认的 logging 输出到控制台
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        force=True)

import os
import sys
import termios
import tty
import select
import atexit
from datetime import datetime

import json

from typing_extensions import List, Tuple, Dict

import cv2
import numpy as np
import transforms3d


######################################################### 全局常量 #########################################################

# 彩色打印定义
RED = '\033[91m'
"""在终端开启红色打印"""

GREEN = '\033[92m'
"""在终端开启绿色打印"""

YELLOW = '\033[93m'
"""在终端开启黄色打印"""

BLUE = '\033[94m'
"""在终端开启蓝色打印"""

RESET = '\033[0m'
"""在终端重置打印的颜色"""


######################################################### 函数定义 #########################################################

def reset_empty_str(param):
    """
    1.如果是空字符串,则变成 None;     
    2.如果是路径参数且存在,则变成规范的路径( 去掉多余的./和../ )
    Args:
        param: 待处理参数
    Returns:
        处理后的参数
    """
    if param is not None:
        if isinstance(param, str):
            if len(param) == 0:
                return None
            elif os.path.exists(param):
                param = os.path.normpath(param)  # 变成规范的路径
            # end if
        # end if
    # end if
    return param
# end def reset_empty_str


def read_cam_params(json_file_path: str) -> Tuple[List[float], List[float]]:
    """
    读取相机参数     
    Args:
        json_file_path (str): 相机参数文件路径
    Returns:
        (Tuple[List[float], List[float]]): 包括以下项    
            - 相机内参矩阵 [fx, fy, cx, cy]    
            - 畸变系数( 可以为空 ) [k1, k2, p1, p2, [k3, [k4 ,k5 ,k6]]]    
    """

    logging.info(f"Try to read camera parameters from: {GREEN}{json_file_path}{RESET}")
    try:
        with open(json_file_path, 'r') as f:
            param = json.load(f)
        # end with
    except Exception as e:
        logging.error(f"Failed to read camera parameters from {json_file_path}: {e}")
        return None, None
    # end try

    # 如果有 serial_number, 打印出来
    if 'serial_number' in param:
        logging.info(f"camera serial number: {GREEN}{param['serial_number']}{RESET}")
    # end if

    if 'intrinsic' not in param:
        logging.error("JSON file must contain 'intrinsic' field.")
        return None, None
    # end if

    intrinsic = param['intrinsic']
    if 'distortion' in param:
        distortion = param['distortion']
    else:
        distortion = None
    # end if
    logging.info(f"intrinsic: {GREEN}{intrinsic}{RESET}")
    logging.info(f"distortion: {GREEN}{distortion}{RESET}")

    return intrinsic, distortion
# end def read_cam_params


def read_rgbd_params(json_file_path: str) -> Tuple[List[float], List[float], float]:
    """
    读取 RGB-D 相机参数     
    Args:
        json_file_path (str): 相机参数文件路径
    Returns:
        (Tuple[List[float], List[float], float]): 包括以下项    
            - 相机内参矩阵 [fx, fy, cx, cy]    
            - 畸变系数( 可以为空 ) [k1, k2, p1, p2, [k3, [k4 ,k5 ,k6]]]    
            - 深度缩放系数
    """

    logging.info(f"Try to read RGB-D camera parameters from: {GREEN}{json_file_path}{RESET}")
    try:
        with open(json_file_path, 'r') as f:
            param = json.load(f)
        # end with
    except Exception as e:
        logging.error(f"Failed to read RGB-D camera parameters from {json_file_path}: {e}")
        return None, None, None
    # end try

    # 如果有 serial_number, 打印出来
    if 'serial_number' in param:
        logging.info(f"camera serial number: {GREEN}{param['serial_number']}{RESET}")
    # end if

    if 'intrinsic' not in param or 'depth_scale' not in param:
        logging.error("JSON file must contain 'intrinsic' and 'depth_scale' fields.")
        return None, None, None
    # end if

    intrinsic = param['intrinsic']
    if 'distortion' in param:
        distortion = param['distortion']
    else:
        distortion = None
    # end if
    depth_scale = float(param['depth_scale'])

    logging.info(f"intrinsic: {GREEN}{intrinsic}{RESET}")
    logging.info(f"distortion: {GREEN}{distortion}{RESET}")
    logging.info(f"depth_scale: {GREEN}{depth_scale}{RESET}")

    return intrinsic, distortion, depth_scale
# end def read_rgbd_params


def read_calib_handeye(json_file_path: str) -> Tuple[np.ndarray, bool]:
    """
    读取手眼标定矩阵    
    Args:
        json_file_path( str ): JSON 文件路径
    Returns:
        (Tuple[np.ndarray, bool]): 手眼标定矩阵,是否为眼在手
    """

    if os.path.exists(json_file_path) is False:
        logging.error(f'Hand-eye calibration file not found: {json_file_path}')
        return None, None
    # end if

    with open(json_file_path, 'r') as f:
        calib_dict = json.load(f)
    # end with

    if 'T_armend_cam' in calib_dict:
        q = np.array(calib_dict['T_armend_cam']["q"], dtype=np.float32)  # [qw,qx,qy,qz]
        t = np.array(calib_dict['T_armend_cam']["t"], dtype=np.float32)
        eye_in_hand = True
    elif 'T_armbase_cam' in calib_dict:
        q = np.array(calib_dict['T_armbase_cam']["q"], dtype=np.float32)  # [qw,qx,qy,qz]
        t = np.array(calib_dict['T_armbase_cam']["t"], dtype=np.float32)
        eye_in_hand = False
    else:
        logging.error("JSON file does not contain valid hand-eye calibration data.")
        return None, None
    # end if

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = transforms3d.quaternions.quat2mat(q)
    T[:3, 3] = t

    if eye_in_hand:
        logging.info(f"T_end_cam: \n{GREEN}{T}{RESET}")
    else:
        logging.info(f"T_base_cam: \n{GREEN}{T}{RESET}")
    # end if

    return T, eye_in_hand
# end def read_calib_handeye


def inv_tf(T: np.ndarray) -> np.ndarray:
    """
    计算变换矩阵的逆
    Args:
        T( np.ndarray ): 变换矩阵 4*4
    Returns:
        (np.ndarray): 逆变换矩阵 4*4
    """
    assert T.shape == (4, 4), "Input transformation matrix must be of shape 4x4"

    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
# end def inv_tf


def transform_delta_pose(T_a0_a1: np.ndarray,
                         T_a_b: np.ndarray) -> np.ndarray:
    """
    已知从 b 坐标系到 a 坐标系的固定变换 T_a_b, 将在 a 坐标系下的位姿增量 T_a0_a1 ( 形如: T_w_a0 * T_a0_a1 = T_w_a1 ), 
    变换到 b 坐标系下, 得到 T_b0_b1

    Args:
        T_a0_a1( np.ndarray ): 从 1 时刻的 a 坐标系到 0 时刻的 a 坐标系的变换矩阵, 4*4
        T_a_b( np.ndarray ): 从 b 坐标系到 a 坐标系的固定变换矩阵, 4*4
    Returns:
        T_b0_b1( np.ndarray ): 从 1 时刻的 b 坐标系到 0 时刻的 b 坐标系的变换矩阵, 4*4

    Examples:
        // 已知的变换关系     
        T_w_a * T_a_b = T_w_b, T_w_a0 * T_a0_a1 = T_w_a1, T_w_b0 * T_b0_b1 = T_w_b1     

        // 已知量: T_a_b, T_w_a0, T_w_a1, T_w_b0,  待求解: T_w_b1   

        // 推导     
        T_w_a0 * T_a_b = T_w_b0 , 
        T_w_a1 * T_a_b = T_w_b1 -->> T_w_a0 * T_a0_a1 * T_a_b = T_w_b0 * T_b0_b1 = T_w_a0 * T_a_b * T_b0_b1     
        T_b0_b1 = inv(T_a_b) * T_a0_a1 * T_a_b  # 这行代码就是函数实现    
        T_w_b1 = T_w_b0 * T_b0_b1  # 结果
    """
    T_b0_b1 = inv_tf(T_a_b) @ T_a0_a1 @ T_a_b
    return T_b0_b1
# end def transform_delta_pose


def compute_aligned_pose(T_a_b0: np.ndarray,
                         T_b_c: np.ndarray,
                         T_c0_d: np.ndarray,
                         T_c1_d: np.ndarray) -> np.ndarray:
    """
    计算对齐位姿    
    已知不变量 T_b_c, 0 时刻的 T_a_b0, T_c0_d, 1 时刻的 T_c1_d,           
    求解 1 时刻的 T_a_b1, 使得 T_a_b1 * T_b_c * T_c1_d = T_a_b0 * T_b_c * T_c0_d  

    Args:
        T_a_b0( np.ndarray ): 0 时刻从 b 坐标系到 a 坐标系的变换矩阵, 4*4
        T_b_c( np.ndarray ): 从 c 坐标系到 b 坐标系的固定变换矩阵, 4*4
        T_c0_d( np.ndarray ): 0 时刻从 d 坐标系到 c 坐标系的变换矩阵, 4*4
        T_c1_d( np.ndarray ): 1 时刻从 d 坐标系到 c 坐标系的变换矩阵, 4*4   

    Returns: 
        T_a_b1( np.ndarray ): 1 时刻从 b 坐标系到 a 坐标系的变换矩阵, 4*4   
    """

    T_a_b1 = T_a_b0 @ T_b_c @ T_c0_d @ inv_tf(T_b_c @ T_c1_d)
    return T_a_b1
# end def compute_aligned_pose


def timestamp_to_str(ns: int) -> str:
    """
    将时间戳转换为字符串格式    
    Args:
        ns (int): 时间戳( 纳秒 )
    Returns:
        str: 格式化后的时间字符串 HH:MM:SS.sss
    """
    dt = datetime.fromtimestamp(ns / 1_000_000_000)
    return dt.strftime("%H:%M:%S.%f")[:-3]
    # end if
# end def timestamp_to_str


def wait_key(debug: bool) -> bool:
    """
    当在 debug 模式下会等待按键, 如果输入 'q' 则返回 False, 否则返回 True, 用于控制是否继续执行程序
    Args:
        debug (bool): 是否启用了调试模式
    Returns:
        (bool): 是否执行下一步
    """
    if debug:
        key = input('press, q: break, other: next step \n')
        logging.info(f'pressed: [{key}]')
        if key == 'q':
            return False
        # end if
    # end if

    return True
# end def wait_key


def get_key():
    """
    获取单个按键输入( 无需再按下回车键 ),该函数会阻塞直到有按键输入    
    注意: 该函数会导致直接 ctrl+c 无法中断程序, 必须要使用判断条件 get_key()=='\x03' 来显式捕获 ctrl+c 的输入
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
# end def get_key


######################################################### 类定义 #########################################################


class KeyboardReader:
    """
    键盘输入读取类, 用于非阻塞读取单个按键
    """

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._tcsetattr = termios.tcsetattr
        self._tcgetattr = termios.tcgetattr
        self.old_settings = self._tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self._closed = False
        atexit.register(self._close)
    # end def __init__

    def _close(self):
        if self._closed:
            return
        try:
            self._tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        finally:
            self._closed = True
    # end def close

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass
    # end def __del__

    def read_key(self):
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if rlist else None
    # end def read_key
# end class KeyboardReader

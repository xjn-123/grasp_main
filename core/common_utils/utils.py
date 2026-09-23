"""通用工具函数和常量"""

import os 
import sys 
import json
import dataclasses
import logging




import numpy as np



logger = logging.getLogger(__name__)
@dataclasses
class Common_data:
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


def inv_tf(T: np.ndarray) -> np.ndarray:

    R=T[:3, :3]
    t=T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T@t

    return T_inv

def read_cam_params(json_file_path: str) -> tuple[list[float], list[float],float]:
    """
    读取相机内参和畸变参数,
    兼容读取RGB—D相机深度缩放系数
    """

    try:
        with open(json_file_path, 'r') as f:
            cam_params = json.load(f)
    except Exception as e:
        logger.error(f"{Common_data.RED}读取相机参数文件失败, 请检查路径是否正确:{e}{Common_data.RESET}")

    #读取相机序列号
    if "serial_number" in cam_params:
        logger.info(f"{Common_data.GREEN}相机序列号: {cam_params['serial_number']}{Common_data.RESET}")

    if "intrinsic" not in cam_params or "distortion" not in cam_params:
        logger.error(f"{Common_data.RED}相机参数文件中缺少'intrinsic'或'distortion'字段{Common_data.RESET}")
        return None, None, None

    intrinsic = cam_params["intrinsic"]
    distortion = cam_params["distortion"]

    if "depth_scale"  in cam_params:
        depth_scale = cam_params["depth_scale"]
        logger.info(f"{Common_data.GREEN}相机深度缩放系数: {depth_scale}{Common_data.RESET}")
        logger.info(f"{Common_data.GREEN}相机内参: {intrinsic}{Common_data.RESET}")
        logger.info(f"{Common_data.GREEN}相机畸变: {distortion}{Common_data.RESET}")
        return intrinsic, distortion, depth_scale
    else:    # 兼容旧版本   
        logger.info(f"{Common_data.GREEN}相机内参: {intrinsic}{Common_data.RESET}")
        logger.info(f"{Common_data.GREEN}相机畸变: {distortion}{Common_data.RESET}")
        return intrinsic, distortion
    




    
    

    

    
   

    

 





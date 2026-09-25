"""通用工具函数和常量"""

import dataclasses
import json
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


#
# ***************************************计算相关方法和类****************************************
def inv_tf(T: np.ndarray) -> np.ndarray:
    """
    计算旋转的逆矩阵
    """

    R=T[:3, :3]
    t=T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T@t

    return T_inv

class T_cam_gripper_camputeror:
    """
    计算相机和 gripper 之间的变换矩阵
    """
    def two_finger_gripper(corner3d:np.ndarray) -> np.ndarray:
        """
        计算双指夹爪到相机坐标的旋转矩阵
        """
        assert corner3d.shape == (4, 3)# 检查输入的corner3d的形状
        center=corner3d.mean(axis=0)   # 计算四角形的中心点,沿数组的第0维（行）进行求平均

        #计算xy轴
        edge_x=corner3d[2]-corner3d[0]
        edge_y=corner3d[3]-corner3d[1]

        if np.linalg.norm(edge_x) < 1e-6 or np.linalg.norm(edge_y) < 1e-6:
            logger.error(f"{Common_data.RED}请检查输入的corner3d,对角点的x或y值相差过小{Common_data.RESET}")
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

        logger.info(f"{Common_data.GREEN}相机和gripper之间的变换矩阵: {T_cam_gripper}{Common_data.RESET}")

        return T_cam_gripper














        


#########################################加载读取文件方法#####################################

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





    




    
    

    

    
   

    

 





'''抓取位姿计算辅助函数'''

import json
import logging
import os
import sys

import numpy as np
import transforms3d

root_dir = os.path.normpath(f"{os.path.dirname(os.path.realpath(__file__))}/../../../")# 导入本工程的模块  \
sys.path.append(root_dir)

from core.common_utils.utils import Common_data,inv_tf

Logger = logging.getLogger(__name__)


class Grasp_computer:

    def __init__(self,
                json_file_path,
                tmpl_dir,
                rotation_step: float = -1.0,
                ro_th_angle_z:float=45.0,
                tmpl_idx:int=-1) -> None:

        self.json_file_path = json_file_path
        self.tmpl_dir = tmpl_dir
        self.rotation_step = rotation_step
        self.ro_th_angle_z = ro_th_angle_z
        self.tmpl_idx = tmpl_idx

        #初始化对称信息
        self.sym_tfs=None
        self.axis=None
        self.offest=None
        self.scale=None
        self.discrete_martix = None

        #读取对称信息
        result = self.gen_discrete_symmetric_linalg(self.json_file_path, self.rotation_step)
        self.sym_tfs=result[0]
        self.axis=result[1]
        self.offest=result[2]
        self.scale=result[3]
        self.offest*=self.scale#化为米制单位
        self.discrete_martix = result[4]


    def gen_discrete_symmetric_linalg(self, 
                                    json_file_path: str,
                                    rotation_step: float = 1) -> tuple[np.ndarray,np.ndarray,np.ndarray,float,np.ndarray]:
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

        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            Logger.error(f"{Common_data.RED}failed to read symmetric info from: '{json_file_path}' \n{e}{Common_data.RESET}")
            return None  # 返回空数组表示读取失败

        axis_list=[]
        offest_list=[]

        #读取缩放因子
        if "scale" in data:
            scale = data.get('scale', 1.0)

        # 读取旋转轴( 仅适用于连续旋转 )
        continuous_tf_list = []
        if "symmetries_continuous" in data:
            symmeries = data.get('symmetries_continuous', [])

            for i, symmetry in enumerate(symmeries):
                axis = symmetry.get('axis', [])
                offset = symmetry.get('offset', [])*scale

                axis_list.append(axis)
                offest_list.append(offset)

                #将轴和偏移量转换为 numpy 数组
                axis = np.array(axis)
                offset = np.array(offset)
                Logger.info(f"{Common_data.RED}symmetries_continuous index: {i}, axis: {axis}, offset: {offset}{Common_data.RESET}")

                #生成对称变换矩阵
                offset = offset.reshape((3, 1))
                for angle in np.arange(0, 360, rotation_step):
                    R = transforms3d.axangles.axangle2mat(axis, np.radians(angle))
                    t = offset - R.dot(offset)

                    # 组合旋转矩阵和平移向量
                    tf = np.eye(4)
                    tf[:3, :3] = R
                    tf[:3, 3] = t.flatten()

                    continuous_tf_list.append(tf)
                # end for
        
        #读取离散对称性
        discrete_tf_list = []
        if "symmetries_discrete" in data:
            symmetries = data.get('symmetries_discrete', [])
            for i, symmetry in enumerate(symmetries):
                tf = np.array(symmetry).reshape((4, 4))
                tf[:3, 3] *= scale
                discrete_tf_list.append(tf)

        discrete_tf_list.append(np.eye(4))#添加单位矩阵作为默认变换

        #组合离散和连续变换
        tf_list = []
        for disc_tf in discrete_tf_list:
            if len(continuous_tf_list) > 0:
                for cont_tf in continuous_tf_list:
                    tf_list.append(disc_tf @ cont_tf)
                # end for
            else:
                tf_list.append(disc_tf)
            # end if
        # end for

        sym_tfs = np.array(tf_list)#将变换矩阵列表转换为 numpy 数组
        axis_list = np.array(axis_list)
        offest_list = np.array(offest_list)

        return sym_tfs,axis_list,offest_list,scale,discrete_tf_list

    def compute_tmpl_ready_pose(self,
                                sym_tfs: np.ndarray,
                                T_end_cam: np.ndarray,
                                ready_T_base_end: np.ndarray,
                                ready_T_cam_model: np.ndarray,
                                cur_T_base_end: np.ndarray,
                                cur_T_cam_model: np.ndarray) -> tuple[np.ndarray, float]:
        """
            对于特定的抓取模板,基于离散的对称旋转矩阵，计算机械臂合理的ready位姿,使得机械臂从当前位姿移动到设定位姿时, cur_T_cam_model = ready_T_cam_model    
            Args:
                sym_tfs (np.ndarray): 对称变换矩阵数组, N*4*4
                T_end_cam (np.ndarray): 从相机到机械臂末端的变换矩阵, 4*4
                ready_T_base_end (np.ndarray): 准备阶段从机械臂末端到基座的变换矩阵, 4*4
                ready_T_cam_model (np.ndarray): 准备阶段从物体到相机的变换矩阵, 4*4
                cur_T_base_end (np.ndarray): 当前从机械臂末端到基座的变换矩阵, 4*4
                cur_T_cam_model (np.ndarray): 当前从物体到相机的变换矩阵, 4*4
            Returns:
                (Tuple[np.ndarray, float]): 计算得到的机械臂位姿 T_base_end , 机械臂 ready 和 target 位姿之间的旋转角度差
        """

        N=sym_tfs.shape[0]

        ready_T_model_cam = inv_tf(ready_T_cam_model)
        ready_T_end_base = inv_tf(ready_T_base_end)
        T_cam_end = inv_tf(T_end_cam)

        target_pose=np.zeros((N,4,4),dtype=np.float32) #初始化末端目标位姿集
        delta_angles=[np.pi*2]*N #初始化旋转角
        for i in range(N):
            sym=sym_tfs[i]
            target_T_base_end=cur_T_base_end @ T_end_cam @ cur_T_cam_model@ sym @ ready_T_model_cam @ T_cam_end
            target_pose[i]=target_T_base_end

            #目标（候选）位姿到模板ready位姿的旋转角,由于物体是刚体，
            #所有候选位姿与模板ready位姿之间的旋转代价都包含物体的在base中的旋转变换这一代价

            delta_T_base_end=ready_T_end_base @ target_T_base_end
            delta_R=delta_T_base_end[:3,:3]
            cosine=(np.trace(delta_R)-1)/2
            cosine=np.clip(cosine,-1,1) #数值稳定性保护
            angle = abs(np.arcoss(cosine))

            # 计算 ready_T_base_end 和 target_T_base_end 之间的 Z 轴夹角,位姿保护，避免末端发生翻转
            ready_z_dir=ready_T_base_end[:3,2]
            target_z_dir=target_T_base_end[:3,2]
            cosine=np.dot(ready_z_dir,target_z_dir)/(np.linalg.norm(ready_z_dir)*np.linalg.norm(target_z_dir))
            cosine=np.clip(cosine,-1,1)
            angle_z=abs(np.arccos(cosine))

            if angle_z > self.ro_th_angle_z:
                angle =np.pi*2+angle_z #超出阈值，增加一个惩罚值

            delta_angles[i]=angle

        #计算delta_angle里最小值的索引和值及对应的位姿
        min_idx=int(np.argmin(delta_angles))
        target_T_base_end=target_pose[min_idx]
        delta_angle=delta_angles[min_idx]

        return target_T_base_end, delta_angle
    
    def compute_ready_pose(self,
                            tmpl_state_list: list[dict],
                            sym_tfs: np.ndarray,
                            T_end_cam: np.ndarray,
                            cur_T_base_end: np.ndarray,
                            cur_T_cam_model: np.ndarray,
                            tmpl_idx: int ) -> tuple[np.ndarray, int]:
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

        best_target_T_base_end = None
        best_tmpl_idx = -1
    
        # 计算最佳模板
        if tmpl_idx >= 0:  # 已经指定了模板索引
            grasp_tmpl = tmpl_state_list[tmpl_idx]
    
            ready_T_base_end = grasp_tmpl['ready_T_base_end']
            ready_T_cam_model = grasp_tmpl['ready_T_cam_model']
            target_T_base_end, _ = self.compute_tmpl_ready_pose(self,sym_tfs,
                                                            T_end_cam,
                                                            ready_T_base_end,
                                                            ready_T_cam_model,
                                                            cur_T_base_end,
                                                            cur_T_cam_model)
            best_target_T_base_end = target_T_base_end
            best_tmpl_idx = tmpl_idx
        else:  # 未指定模板索引, 计算最佳模板
            tmpl_num = len(tmpl_state_list)
            min_delta_angle = 1000.0
            for i in range(tmpl_num):
                grasp_tmpl = tmpl_state_list[i]
    
                ready_T_base_end = grasp_tmpl['ready_T_base_end']
                ready_T_cam_model = grasp_tmpl['ready_T_cam_model']
                target_T_base_end, delta_angle = self.compute_tmpl_ready_pose(sym_tfs,
                                                                            T_end_cam,
                                                                            ready_T_base_end,
                                                                            ready_T_cam_model,
                                                                            cur_T_base_end,
                                                                            cur_T_cam_model)
                if delta_angle < min_delta_angle:
                    min_delta_angle = delta_angle
                    best_tmpl_idx = i
                    best_target_T_base_end = target_T_base_end
                # end if
            # end for
        # end if
    
        return best_target_T_base_end, best_tmpl_idx
    # end def compute_ready_pose
    def skew(self,v):
        """3 维向量 → 反对称矩阵，满足 skew(a) @ b == np.cross(a, b)"""
        return np.array([[    0, -v[2],  v[1]],
                        [ v[2],     0, -v[0]],
                        [-v[1],  v[0],     0]], dtype=float)

    def rotmated_ref_pose(self,
                        axis: np.ndarray=self.axis,
                        offest: np.ndarray=self.offset,
                        discrete_martix_list: np.ndarray = self.discrete_martix_list,
                        ready_T_cam_model: np.ndarray = None,
                        cur_T_cam_model: np.ndarray = None,
                        ):

        '''对于连续对称性，采用连续旋转位姿计算目标矩阵，并于离散对称矩阵组合反回完整的对称矩阵'''

        if axis.shape[0] != offest.shape[0]:
            raise ValueError('axis and offset must have the same shape')

        compute_T_cam_model = []
        for discrete_martix in discrete_martix_list:
            for i in range(axis.shape[0]):

                ready_T_cam_model= ready_T_cam_model@discrete_martix  #先你离散对称处理模板位姿

                axis_n=axis[i]
                offset_n=offest[i]

                k = np.asarray(axis_n, float)
                k = k / np.linalg.norm(k)

                R = np.asarray(cur_T_cam_model[:3, :3], float)
                t = np.asarray(cur_T_cam_model[:3, 3], float)

                M = R.T @ np.asarray(ready_T_cam_model[:3, :3], float)

                a = np.trace(M) - k @ M @ k
                b = -np.trace(self.skew(k) @ M)

                if a * a + b * b < 1e-18:
                    alpha = 0.0                 # 目标函数对 α 恒定（完全对称），任取
                else:
                    alpha = np.arctan2(b, a)
                # end if

                Rs = transforms3d.axangles.axangle2mat(k, alpha)
                t_s = offset_n - Rs @ offset_n

                T = np.eye(4)
                T[:3, :3] = R @ Rs
                T[:3, 3] = R @ t_s + t

                compute_T_cam_model.append(T)

        return np.array(compute_T_cam_model)

    def compute_continuous_ready_pose(self,
                                        T_cam_model: np.ndarray,
                                        ready_T_cam_model: np.ndarray,                                  
                                        cur_T_base_end: np.ndarray,
                                        cur_T_cam_model: np.ndarray,
                                        axis: np.ndarray=self.axis,
                                        offest: np.ndarray=self.offset,
                                        discrete_martix_list: np.ndarray =self.discrete_martix_list,
                                        ) -> tuple[np.ndarray, float]:
                
        '''用连续旋转获得末端目标位姿'''
    


    

 














import argparse
import glob
import json
import logging
import os
import sys

import cv2
import numpy as np
import transforms3d

#导入本地工程模块
core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{core_dir}/../../")
sys.path.append(root_dir)

import core.common_utils.utils as utils
from core.board_utils import apriltag_helper
from core.common_utils.utils import Common_data

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
            T01s[:, :3, :3],       #旋转矩阵部分
            T01s[:, :3, 3],        #平移向量部分
            T23s[:, :3, :3],
            T23s[:, :3, 3],
            method=cv2.CALIB_HAND_EYE_PARK
            )

        T12s = np.eye(4, dtype=np.float64)
        T12s[:3, :3] = R
        T12s[:3, 3] = t.ravel()

        return T12s

    def compute_error(T01s:np.ndarray, T23s:np.ndarray, T12s:np.ndarray)->float:
        """ 计算AX=XB问题的误差 """

        assert T01s.shape[0] == T23s.shape[0], ("T01s和T23s的数量不一致")

        N=T01s.shape[0]
        if N < 2:
            logger.warning("AX=XB问题的数量小于2, 无法计算误差")
            return 0.0,0.0

        T03s=T01s @ T12s @ T23s
        sum_angle = 0.0
        sum_translation = 0.0   
        cont=0

        for i in range(N):
            for j in range(i+1,N):

                #第i次和第j次计算的标定板坐标系在基座坐标系的位姿的的相对误差，理想状态应是单位矩阵
                delta_T =  T03s[j] @ np.linalg.inv(T03s[i]) 

                #计算旋转误差
                R = delta_T[:3,:3]
                cos_angle= (np.trace(R) - 1.0) / 2.0
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                #计算平移误差
                t_error = np.linalg.norm(delta_T[:3,3])

                sum_angle += angle
                sum_translation += t_error 
                cont+=1

        error_angle = sum_angle / cont*180/np.pi          #角度误差，单位为度
        error_translation = sum_translation / cont*1000.0 # 平移误差，单位为毫米    

        return error_angle, error_translation
    
    
    """
    将字典转换为列表, 并按键排序
    """

    T_cam_board_list=[]
    T_base_end__list=[]

    for img_id in range(len(T_base_end__dict)):

        if img_id not in T_cam_board_dict:
            logger.warning(f"采集序号 {img_id} 在 T_cam_board_dict 中不存在, 将跳过该序号")
            continue

        T_base_end__list.append(T_base_end__dict[img_id])
        T_cam_board_list.append(T_cam_board_dict[img_id])   

    if len(T_base_end__list) < 4:
        logger.error("手眼标定需要至少4组数据, 当前数据量不足")
        return None

    """
    计算眼在手上标定矩阵
    """
    if eye_in_hand:
        T01s = np.array(T_base_end__list, dtype=np.float64)
        T23s = np.array(T_cam_board_list, dtype=np.float64)

        T_end_cam = save_AXXB(T01s, T23s)
        error_angle, error_translation = compute_error(T01s, T23s, T_end_cam)

        logger.info(f"{Common_data.GREEN}眼在手上标定矩阵:{Common_data.RESET}")
        logger.info(f"{Common_data.GREEN}手眼标定误差 - 角度误差: {error_angle:.2f}°, 平移: {error_translation:.2f}mm{Common_data.RESET}")

        return T_end_cam

    else:
        T01s=np.array(T_base_end__list, dtype=np.float64)  
        T23s=np.array(T_cam_board_list, dtype=np.float64)

        for i in range(len(T01s)):
            T01s[i] = utils.inv_tf(T01s[i])
            T23s[i] = utils.inv_tf(T23s[i])

        T_cam_base = save_AXXB(T01s, T23s)
        error_angle, error_translation = compute_error(T01s, T23s, T_cam_base)

        logger.info(f"{Common_data.GREEN}眼在手外标定矩阵:{Common_data.RESET}")
        logger.info(f"{Common_data.GREEN}手眼标定误差 - 角度误差: {error_angle:.2f}°, 平移: {error_translation:.2f}mm{Common_data.RESET}")

        return T_cam_base
    

def main():
    parser=argparse.ArgumentParser(description="camera calibration using apriltag")

    parser.add_argument("--cam_param_path", type=str, required=True, help="相机参数文件路径, json格式")
    parser.add_argument("--board_info", type=str, required=True, help="标定板信息 [tag_size, space_size, tag_rows, tag_cols]")
    parser.add_argument("--image_dir", type=str, required=True, help="标定板图像目录")
    parser.add_argument("--arm_pose_path",type=str,default="/home/i4/桌面/test_location/calib/collect_image_handeye/arm_pose.json",help="机械臂末端位姿文件的路径",)
    parser.add_argument("--eye_in_hand",type=bool,default=True,help="是否为手眼标定( True )或固定相机标定( False )")

    args=parser.parse_args()

    image_dir=args.image_dir   
    arm_pose_path=args.arm_pose_path 
    cam_param_path=args.cam_param_path
    board_info=json.loads(args.board_info)
    eye_in_hand=args.eye_in_hand

    print(f"\n图像目录={image_dir}\n机械臂位姿文件={arm_pose_path}\n相机参数文件={cam_param_path}\n标定板信息={board_info}\n")

    #加载相机参数文件
    intrinsic, distortion=utils.read_cam_params(cam_param_path)
    if intrinsic is None or distortion is None:  
        print(f"{Common_data.RED}相机参数文件错误！{Common_data.RESET}")
        sys.exit(1)

    K=np.array([[intrinsic[0],0,intrinsic[2]],[0,intrinsic[1],intrinsic[3]],[0,0,1]]) #创建相机内参矩阵
    D=np.array(distortion)                                                            #创建相机畸变参数向量

    #加载机械臂末端位姿文件
    with open(arm_pose_path, 'r') as f:
        arm_pose_dict = json.load(f)
        if len(arm_pose_dict) <1:
            logger.error(f"{Common_data.RED}机械臂末端位姿文件为空，请检查文件内容{Common_data.RESET}")
            sys.exit(1)
    
    #创建标定板
    tag3d_list=apriltag_helper.create_board_3d_points(board_info[0], board_info[1], board_info[2], board_info[3])

    #创建apriltag检测器
    detector=apriltag_helper.Detector(tag_family="tag36h11")

    #检测图像中的tag
    cam_pose_dict={}  #存储每张图像对应的标定板位姿  
    arm_pose_dict={}  #存储每张图像对应的机械臂末端位姿

    image_path_list=glob.glob(f"{image_dir}/*.png")
    image_path_list.sort()

    if len(image_path_list)<1:
        logger.warning(f"目录：{image_dir}不包含任何图片")
        sys.exit(1)
    
    for image_path in image_path_list:
        image_name=os.path.basename(image_path)
        image_id=os.path.splitext(image_name)[0] 

        if f"{image_id}" not in arm_pose_dict:
            logger.warning(f"采集序号 {image_id} 在机械臂末端位姿文件中不存在, 将跳过该序号")
            continue 

        arm_pose=arm_pose_dict[f"{image_id}"]

        img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"无法读取图片{image_path}")
            continue
        tag2d_list=detector.detect(img,-1)

        cam_pose=apriltag_helper.locate_calib_board(tag3d_list, tag2d_list, K, D) #检测计算T_cam_board

        if cam_pose is None:
            logger.warning(f"{Common_data.YELLOW}无法在{image_name}中定位标定板，跳过.{Common_data.RESET}")
            continue


        """机械臂位姿需要转换为4x4的变换矩阵"""
        arm_pose_dict[f"{image_id}"]=np.array(arm_pose)
        cam_pose_dict[f"{image_id}"]=cam_pose

        #执行手眼标定
        result_T=calib_handeye(arm_pose_dict, cam_pose_dict,eye_in_hand)

        #保存结果
        save_path=os.path.join(os.path.dirname(cam_param_path),"calib_handeye.json")
        result={}

        if eye_in_hand:
            result["T_end_cam"]=utils.matrix_to_array(result_T)
        else:
            result["T_base_cam"]=utils.matrix_to_array(result_T)

        with open(save_path, 'w') as f:
            json.dump(result, f,indent=4)
        logger.info(f"结果保存到{save_path}") 

if __name__=="__main__":
    main()











    




    





   
        
    
  


        



    

    

        











     

    


        

    




    

    


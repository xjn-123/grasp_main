import argparse
import glob
import json
import logging
import os
import sys
from tkinter import image_names

import cv2
import numpy as np
from numpy.random import f

#导入本地工程模块
core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(f"{core_dir}/../../")
sys.path.append(root_dir)

from core.board_utils import apriltag_helper

logger = logging.getLogger(__name__)


def calib_camera(
        trags_3d_list:list[apriltag_helper.Tag3D],
        trag_2d_list_list:list[list[apriltag_helper.Tag2D]],
        image_size:tuple[int,int]        
)->tuple[list[float],list[float]]:
    
    pts3d_list=[]
    pts2d_list=[]

    for tag2d_list in trag_2d_list_list:

        pts2d=[]
        pts3d=[]

        for tag2d in tag2d_list:
            tag2d_id=tag2d.id
            tag3d=next(t for t in trags_3d_list if t.id==tag2d_id)
            if tag3d is None:
                logger.warning(
                    f"tagid={tag2d_id} only detected 2d_id skipping it"
                )
                continue

            for i in range(4):
                pts3d.append(np.array(tag3d.corners[i]))
                pts2d.append(np.array(tag2d.corners[i]))
        if len(pts3d)<4:
            logger.warning("valid 3D-2D point pairs found")
            continue

        pts3d_list.append(np.array(pts3d))
        pts2d_list.append(np.array(pts2d))

    if len(pts3d_list)<10:
        logger.warning("valid images with sufficient 3D-2D point pairs found")
        return None,None

    K=np.eye(3)
    D=np.zeros(5)

    ret,K,D,rvecs,tvecs=cv2.calibrateCamera(pts3d_list,
                                            pts2d_list,
                                            image_size,
                                            K,
                                            D)

    intrinsic=[K[0,0],K[1,1],K[0,2],K[1,2]]
    distortion=D.tolist()

    logger.info(f"calibration result: intrinsic={intrinsic}, distortion={distortion}")
    logger.info(f"calibration result: rvecs={rvecs}, tvecs={tvecs}")
    logger.info(f"calibration result: ret={ret}")

    return intrinsic,distortion

def main():
    parser=argparse.ArgumentParser(description="camera calibration using apriltag")

    parser.add_argument("--image_dir",type=str,required=True,help="图像文件夹路径")
    parser.add_argument("--calib_board_info",type=str,required=True,help="标定板信息 [tag_size,tag_space,tag_rows,tag_cols]")
    parser.add_argument("--start_id",type=int,default=0,help="标定板起始id")

    args=parser.parse_args()

    image_dir=args.image_dir
    calib_board_info=json.loads(args.calib_board_info)
    start_id=args.start_id

    print(f"\n图像目录={image_dir}\n标定板信息={calib_board_info}\n")

    tag3d_list=apriltag_helper.create_calib_board_3d(calib_board_info[0],
                                                    calib_board_info[1],
                                                    calib_board_info[2],
                                                    calib_board_info[3],
                                                    start_tag_id=start_id)

    detector=apriltag_helper.Detector(tag_family="tag36h11")

    image_path_list=glob.glob(f"{image_dir}/*.png")
    image_path_list.sort()
    if len(image_path_list)<1:
        logger.warning(f"目录：{image_dir}不包含任何图片")
        sys.exit(1)

    tag2d_list_list=[]
    image_size=(0,0)
    for image_path in image_path_list:
        image_name=os.path.basename(image_path)
        image_id=os.path.splitext(image_name)[0]

        img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

        if img is None:
            logger.warning(f"图像路径:{image_path}解析图像失败")
            continue

        img_size=(img.shape[1],img.shape[0])

        tag2d_list=detector.detect(img,-1)
        if len(tag2d_list)<1:
            logger.warning(f"图片{image_path}未检测到tag")
            continue

        tag2d_list_list.append(tag2d_list)

    intrinsic,distortion=calib_camera(tag3d_list,tag2d_list_list,image_size)
    if intrinsic is None or distortion is None:
        logger.warning(f"图片{image_path}标定失败")
        sys.exit(1)
    print(f"intrinsic={intrinsic},distortion={distortion}")

    save_path=os.path.join(os.path.dirname(image_dir),"cam_params.json")
    result_dict={
        "camera_type":"Pinhole",
        "IntrinsicFormat":"fx,fy,cx,cy",
        "DistortionFormat":"k1,k2,p1,p2,k3",
        "resolution":[img_size[0],img_size[1]],
        "intrinsic":intrinsic,
        "distortion":distortion
    }
    with open(save_path,"w") as f:  # noqa: F811
        json.dump(result_dict,f,indent=4)

    logger.info(f"保存标定结果到{save_path}")

if __name__=="__main__":
    main()


    
    



    
        
    
        








    














    






    

        


        

    

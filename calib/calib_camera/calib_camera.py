import argparse
import glob
import json
import logging
import os
import sys

import cv2
import numpy as np
import pyapriltags

logger = logging.getLogger(__name__)


def calib_camera(
        trags_3d_list:list[pyapriltags.Tag3D],
        trag_2d_list_list:list[list[pyapriltags.Tag2D]],
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
                pts2d.append(np.array(tag2d.conners[i]))
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

        ret,K,D,rvecs,tvecs=cv2.calibrateCamera(objectionspoints=pts3d_list,
                                                imagepoints=pts2d_list,
                                                image_size=image_size,
                                                cameraMatrix=K,
                                                distCoeffs=D)

        intrinsic=[K[0,0],K[1,1],K[0,2],K[1,2]]
        distortion=D.tolist()

        logger.info(f"calibration result: intrinsic={intrinsic}, distortion={distortion}")
        logger.info(f"calibration result: rvecs={rvecs}, tvecs={tvecs}")
        logger.info(f"calibration result: ret={ret}")

        return intrinsic,distortion

def create_calib_board(tag_size:float,tag_space:float,tag_rows:int,tag_cols:int,start_id:int)->list[pyapriltags.Tag3D]:
    tags_3d_list=[]
    for i in range(tag_rows):
        for j in range(tag_cols):
            tag_id=start_id+i*tag_cols+j
            tag_center_x=j*(tag_size+tag_space)
            tag_center_y=i*(tag_size+tag_space)
            tag_center_z=0.0

            tag_corners=[
                [tag_center_x-tag_size/2,tag_center_y-tag_size/2,tag_center_z],
                [tag_center_x+tag_size/2,tag_center_y-tag_size/2,tag_center_z],
                [tag_center_x+tag_size/2,tag_center_y+tag_size/2,tag_center_z],
                [tag_center_x-tag_size/2,tag_center_y+tag_size/2,tag_center_z]
            ]

            tag3d=pyapriltags.Tag3D(id=tag_id,corners=tag_corners)
            tags_3d_list.append(tag3d)

    return tags_3d_list

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

    tag3d_list=create_calib_board(calib_board_info[0],calib_board_info[1],calib_board_info[2],calib_board_info[3],start_id=start_id)

    detector=pyapriltags.Detector(families="tag36h11")

    






    

        


        

    
